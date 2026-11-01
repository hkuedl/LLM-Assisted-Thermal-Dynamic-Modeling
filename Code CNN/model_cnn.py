"""Lightweight causal CNN models aligned with the MLP/GRU experiments."""

from typing import Tuple

import torch
import torch.nn as nn

import config
from prior_utils import get_semantic_tensor


def _initial_prediction_frame(y_state: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(y_state) if config.PREDICT_RESIDUAL_DELTA else y_state


def _advance_prediction_state(
    y_state: torch.Tensor,
    decoder_output: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if config.PREDICT_RESIDUAL_DELTA:
        if config.RESIDUAL_ROLLOUT_STATE_MODE == "delta":
            return decoder_output, decoder_output
        return y_state + decoder_output, decoder_output
    return decoder_output, decoder_output


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.left_padding,
        ).double()

    def forward(self, values):
        output = self.conv(values)
        return output[:, :, :-self.left_padding] if self.left_padding else output


class CausalCNNEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        channels = int(getattr(config, "CNN_CHANNELS", 64))
        kernel_size = int(getattr(config, "CNN_KERNEL_SIZE", 3))
        dropout = float(getattr(config, "CNN_DROPOUT_P", 0.0))
        layers = [
            CausalConv1d(input_dim, channels, kernel_size),
            nn.SiLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([
            CausalConv1d(channels, channels, kernel_size),
            nn.SiLU(),
        ])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.convolutions = nn.Sequential(*layers).double()
        self.projection = nn.Linear(channels, latent_dim).double()

    def forward(self, sequence):
        # sequence: [batch, history, features]
        hidden = self.convolutions(sequence.transpose(1, 2))
        return self.projection(hidden[:, :, -1])


def build_decoder(latent_dim, y_dim):
    dropout = float(getattr(config, "CNN_DROPOUT_P", 0.0))
    output = nn.Linear(latent_dim, y_dim)
    layers = [
        nn.Linear(latent_dim, latent_dim),
        nn.SiLU(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(output)
    decoder = nn.Sequential(*layers).double()
    if getattr(config, "CNN_ZERO_INIT_DECODER_OUTPUT", True):
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
    return decoder


def build_fusion(latent_dim, spatial_dim):
    dropout = float(getattr(config, "CNN_DROPOUT_P", 0.0))
    layers = [
        nn.Linear(latent_dim + spatial_dim, latent_dim),
        nn.SiLU(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers).double()


class Baseline_CNN(nn.Module):
    def __init__(self, y_dim, p_dim, x_env_dim, priors=None, latent_dim=None):
        super().__init__()
        latent_dim = int(latent_dim or getattr(config, "CNN_HIDDEN_DIM", 128))
        self.y_dim, self.p_dim, self.latent_dim = y_dim, p_dim, latent_dim
        # Same convention as History MLP: four past samples plus current.
        self.history_window = max(1, int(getattr(config, "CNN_HISTORY_STEPS", 4)))
        self.history_steps = self.history_window + 1
        self.W_graph = nn.Parameter(
            torch.randn(y_dim, y_dim, dtype=torch.float64) * 0.05
        )
        input_dim = y_dim + p_dim + x_env_dim
        self.encoder = CausalCNNEncoder(input_dim, latent_dim)
        self.fusion = build_fusion(latent_dim, y_dim)
        self.decoder = build_decoder(latent_dim, y_dim)

    def _history_tensor(self, history):
        recent = history[-self.history_steps:]
        if len(recent) < self.history_steps:
            recent = [recent[0]] * (self.history_steps - len(recent)) + recent
        return torch.stack(recent, dim=1)

    def build_step_frame(self, y_curr, u_j, effective_priors=None):
        p_curr = u_j[:, self.y_dim:self.y_dim + self.p_dim]
        x_curr = u_j[:, self.y_dim + self.p_dim:]
        return torch.cat([y_curr, p_curr, x_curr], dim=-1)

    def build_current_spatial(self, y_curr, effective_priors=None):
        return y_curr @ self.W_graph.t()

    def _context_frames(self, history_context, effective_priors=None):
        if history_context is None:
            return []
        return [
            self.build_step_frame(
                history_context[j, :, :self.y_dim],
                history_context[j],
                effective_priors,
            )
            for j in range(history_context.shape[0])
        ]

    def forward(self, u, history_context=None):
        y_curr = u[0, :, :self.y_dim]
        history = self._context_frames(history_context)
        preds = [_initial_prediction_frame(y_curr)]
        for j in range(u.shape[0]):
            history.append(self.build_step_frame(y_curr, u[j]))
            temporal = self.encoder(self._history_tensor(history))
            spatial = self.build_current_spatial(y_curr)
            feature = self.fusion(torch.cat([temporal, spatial], dim=-1))
            y_next = self.decoder(feature)
            preds.append(y_next)
            y_curr, _ = _advance_prediction_state(y_curr, y_next)
        return torch.stack(preds, dim=0)


class Physical_Informed_CNN(Baseline_CNN):
    def __init__(self, y_dim, p_dim, x_env_dim, priors, latent_dim=None):
        super().__init__(y_dim, p_dim, x_env_dim, priors, latent_dim)
        semantics = get_semantic_tensor(priors, y_dim, required_dim=5)
        self.register_buffer("inertia_prior", semantics[:, 0])
        self.register_buffer("occ_prior", semantics[:, 1])
        self.register_buffer("env_prior", semantics[:, 2])
        self.register_buffer("trust_prior", semantics[:, 3])
        self.register_buffer("sens_prior", semantics[:, 4])

        adjacency = (
            torch.tensor(priors["adjacency_matrix"], dtype=torch.float64)
            if priors and "adjacency_matrix" in priors
            else torch.eye(y_dim, dtype=torch.float64)
        )
        self.register_buffer("A_mask", torch.clamp(adjacency, 0.0, 1.0))
        self.register_buffer(
            "off_diag_mask", 1.0 - torch.eye(y_dim, dtype=torch.float64),
        )
        self.topo_scale = nn.Parameter(torch.tensor([0.05], dtype=torch.float64))
        self.W_graph_delta = nn.Parameter(torch.zeros(y_dim, y_dim, dtype=torch.float64))
        self.inertia_scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        self.gain_inertia_delta = nn.Parameter(torch.zeros(y_dim, dtype=torch.float64))
        self.occ_scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        self.gain_occ_delta = nn.Parameter(torch.zeros(y_dim, dtype=torch.float64))
        self.env_scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        self.gain_env_delta = nn.Parameter(torch.zeros(y_dim, dtype=torch.float64))

    def _effective_priors(self):
        trust = 1.0 - 0.5 * self.trust_prior
        graph_delta = self.W_graph_delta * self.off_diag_mask * trust.unsqueeze(1)
        graph = torch.clamp(
            (self.topo_scale * self.A_mask + graph_delta) * self.off_diag_mask,
            -0.2,
            0.2,
        )
        inertia = 1.0 + 0.2 * torch.tanh(
            self.inertia_scale * (self.inertia_prior - 0.5)
            + self.gain_inertia_delta
        )
        occupancy = 1.0 + 0.2 * torch.tanh(
            self.occ_scale * (self.occ_prior - 0.5) + self.gain_occ_delta
        )
        env_prior = torch.sqrt(torch.clamp(self.env_prior * self.sens_prior, min=0.0))
        environment = 1.0 + 0.2 * torch.tanh(
            self.env_scale * (env_prior - 0.5) + self.gain_env_delta
        )
        return graph, inertia, occupancy, environment

    def build_step_frame(self, y_curr, u_j, effective_priors=None):
        p_curr = u_j[:, self.y_dim:self.y_dim + self.p_dim]
        x_curr = u_j[:, self.y_dim + self.p_dim:]
        return torch.cat([y_curr, p_curr, x_curr], dim=-1)

    def build_current_spatial(self, y_curr, effective_priors=None):
        graph = (
            effective_priors[0]
            if effective_priors is not None
            else self._effective_priors()[0]
        )
        return torch.sum(
            graph.unsqueeze(0) * (y_curr.unsqueeze(1) - y_curr.unsqueeze(2)),
            dim=2,
        )

    def forward(self, u, history_context=None):
        y_curr = u[0, :, :self.y_dim]
        priors = self._effective_priors()
        history = self._context_frames(history_context, priors)
        preds = [_initial_prediction_frame(y_curr)]
        for j in range(u.shape[0]):
            history.append(self.build_step_frame(y_curr, u[j], priors))
            temporal = self.encoder(self._history_tensor(history))
            spatial = self.build_current_spatial(y_curr, priors)
            feature = self.fusion(torch.cat([temporal, spatial], dim=-1))
            y_next = self.decoder(feature)
            preds.append(y_next)
            y_curr, _ = _advance_prediction_state(y_curr, y_next)
        return torch.stack(preds, dim=0)


class Cluster_Physical_Informed_CNN(Physical_Informed_CNN):
    cluster_source = "data_shape_auto"
    cluster_max_groups_config = "DATA_CLUSTER_MAX_GROUPS"

    def __init__(self, y_dim, p_dim, x_env_dim, priors, latent_dim=None, n_clusters=None):
        super().__init__(y_dim, p_dim, x_env_dim, priors, latent_dim)
        if n_clusters is None:
            raise ValueError(
                "Cluster CNN requires n_clusters from the finalized zone assignment."
            )
        max_clusters = max(1, int(getattr(config, self.cluster_max_groups_config, 5)))
        self.n_clusters = int(n_clusters)
        if not 1 <= self.n_clusters <= max_clusters:
            raise ValueError(
                f"n_clusters must be in [1, {max_clusters}], got {self.n_clusters}"
            )
        self.register_buffer("zone_cluster_ids", torch.zeros(y_dim, dtype=torch.long))
        input_dim = y_dim + p_dim + x_env_dim
        self.cluster_encoders = nn.ModuleList([
            CausalCNNEncoder(input_dim, self.latent_dim)
            for _ in range(self.n_clusters)
        ])
        self.cluster_fusions = nn.ModuleList([
            build_fusion(self.latent_dim, y_dim)
            for _ in range(self.n_clusters)
        ])
        self.cluster_decoders = nn.ModuleList([
            build_decoder(self.latent_dim, y_dim)
            for _ in range(self.n_clusters)
        ])
        # Match the independent-expert design used by the GRU/MLP cluster models.
        del self.encoder
        del self.fusion
        del self.decoder

    def set_zone_cluster_ids(self, cluster_ids):
        ids = torch.as_tensor(cluster_ids, dtype=torch.long)
        if ids.numel() != self.y_dim:
            raise ValueError(
                f"cluster_ids length {ids.numel()} does not match y_dim {self.y_dim}"
            )
        unique = torch.unique(ids, sorted=True)
        expected = torch.arange(self.n_clusters, dtype=torch.long)
        if not torch.equal(unique.cpu(), expected):
            raise ValueError(
                f"cluster IDs must use every contiguous ID in [0, {self.n_clusters - 1}], "
                f"got {unique.tolist()}"
            )
        self.zone_cluster_ids.copy_(ids.to(self.zone_cluster_ids.device))

    def forward(self, u, history_context=None):
        y_curr = u[0, :, :self.y_dim]
        priors = self._effective_priors()
        history = self._context_frames(history_context, priors)
        preds = [_initial_prediction_frame(y_curr)]
        cluster_ids = self.zone_cluster_ids.to(u.device)
        masks = [
            (cluster_ids == k).to(u.dtype).view(1, self.y_dim)
            for k in range(self.n_clusters)
        ]
        for j in range(u.shape[0]):
            history.append(self.build_step_frame(y_curr, u[j], priors))
            sequence = self._history_tensor(history)
            spatial = self.build_current_spatial(y_curr, priors)
            y_next = torch.zeros_like(y_curr)
            for k, mask in enumerate(masks):
                if torch.any(mask > 0):
                    temporal = self.cluster_encoders[k](sequence)
                    feature = self.cluster_fusions[k](
                        torch.cat([temporal, spatial], dim=-1)
                    )
                    y_next = y_next + self.cluster_decoders[k](feature) * mask
            preds.append(y_next)
            y_curr, _ = _advance_prediction_state(y_curr, y_next)
        return torch.stack(preds, dim=0)


class Data_Shape_AutoCluster_CNN(Cluster_Physical_Informed_CNN):
    cluster_source = "data_shape_auto"
    cluster_max_groups_config = "DATA_CLUSTER_MAX_GROUPS"


class LLM_Cluster_CNN(Cluster_Physical_Informed_CNN):
    cluster_source = "llm"
    cluster_max_groups_config = "CNN_LLM_CLUSTER_MAX_GROUPS"


MODEL_REGISTRY = {
    "Baseline CNN": {"class": Baseline_CNN},
    "PI CNN": {"class": Physical_Informed_CNN},
    "PI + Data Shape AutoCluster CNN": {
        "class": Data_Shape_AutoCluster_CNN,
    },
    "PI + LLM Cluster CNN": {"class": LLM_Cluster_CNN},
}
