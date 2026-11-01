"""Backbone-independent, zone-shared semantic FiLM correction.

Inputs: global feature [B,H], local features [B,N,6], semantics [N,4].
Output: correction [B,N] in the same scaled-temperature units as the decoder.
"""

import math

import torch
from torch import nn


class ZoneFiLMAdapter(nn.Module):
    def __init__(self, feature_dim=128, adapter_dim=32, semantic_hidden=16,
                 strength=0.1):
        super().__init__()
        if not math.isfinite(strength) or not 0.0 <= strength < 1.0:
            raise ValueError("FiLM strength must be finite and in [0, 1).")
        self.feature_dim = int(feature_dim)
        self.adapter_dim = int(adapter_dim)
        self.register_buffer("strength", torch.tensor(float(strength), dtype=torch.float64))
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim + 6, adapter_dim), nn.SiLU(),
        )
        self.semantic_net = nn.Sequential(
            nn.Linear(4, semantic_hidden), nn.SiLU(),
            nn.Linear(semantic_hidden, 2 * adapter_dim),
        )
        self.readout = nn.Sequential(
            nn.Linear(adapter_dim, 16), nn.SiLU(), nn.Linear(16, 1),
        )
        self.double()
        # Only zero the final correction layer. This preserves the base output
        # at initialization without a zero gate that could freeze the branch.
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def modulation(self, semantics):
        if semantics.ndim != 2 or semantics.shape[-1] != 4:
            raise ValueError("Expected semantics [zones, 4].")
        raw_gamma, raw_beta = self.semantic_net(2.0 * semantics - 1.0).chunk(2, dim=-1)
        return (1.0 + self.strength * torch.tanh(raw_gamma),
                self.strength * torch.tanh(raw_beta))

    def forward(self, feature, local_features, semantics, modulation=None):
        if feature.ndim != 2 or feature.shape[-1] != self.feature_dim:
            raise ValueError("Unexpected backbone feature dimensions.")
        if local_features.ndim != 3 or local_features.shape[-1] != 6:
            raise ValueError("Expected local features [batch, zones, 6].")
        if local_features.shape[0] != feature.shape[0]:
            raise ValueError("Global and local batch sizes differ.")
        if semantics.shape[0] != local_features.shape[1]:
            raise ValueError("Semantic and dynamic zone counts differ.")
        global_by_zone = feature.unsqueeze(1).expand(-1, local_features.shape[1], -1)
        value = self.projection(torch.cat([global_by_zone, local_features], dim=-1))
        gamma, beta = self.modulation(semantics) if modulation is None else modulation
        return self.readout(value * gamma.unsqueeze(0) + beta.unsqueeze(0)).squeeze(-1)
