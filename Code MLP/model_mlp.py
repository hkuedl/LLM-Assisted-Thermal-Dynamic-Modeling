
import torch
import torch.nn as nn
from typing import Tuple

import config
from prior_utils import get_semantic_tensor


class BoundedResidualOutput(nn.Module):
    """Smoothly bound residual deltas while remaining linear near zero."""

    def __init__(self, max_delta):
        super().__init__()
        max_delta_tensor = torch.as_tensor(
            max_delta, dtype=torch.float64,
        ).reshape(-1)
        if max_delta_tensor.numel() == 0 or not torch.all(
            torch.isfinite(max_delta_tensor) & (max_delta_tensor > 0)
        ):
            raise ValueError("every per-zone max_delta must be positive and finite")
        self.register_buffer("max_delta", max_delta_tensor)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.max_delta * torch.tanh(value / self.max_delta)


def build_mlp(input_dim: int, latent_dim: int, output_dim: int):
    layers = [
        nn.Linear(input_dim, latent_dim),
        nn.SiLU(),
    ]
    if config.USE_DECODER_DROPOUT and config.DECODER_DROPOUT_P > 0:
        layers.append(nn.Dropout(config.DECODER_DROPOUT_P))
    layers.extend([
        nn.Linear(latent_dim, latent_dim),
        nn.SiLU(),
    ])
    if config.USE_DECODER_DROPOUT and config.DECODER_DROPOUT_P > 0:
        layers.append(nn.Dropout(config.DECODER_DROPOUT_P))
    layers.append(nn.Linear(latent_dim, output_dim))

    if getattr(config, "MLP_BOUND_ENCODER_LATENT", False):
        layers.append(nn.Tanh())

    return nn.Sequential(*layers).double()


def build_decoder(latent_dim: int, y_dim: int):
    layers = [nn.Linear(latent_dim, latent_dim), nn.SiLU()]
    if config.USE_DECODER_DROPOUT and config.DECODER_DROPOUT_P > 0:
        layers.append(nn.Dropout(config.DECODER_DROPOUT_P))
    output_layer = nn.Linear(latent_dim, y_dim)
    layers.append(output_layer)

    if getattr(config, "MLP_ZERO_INIT_DECODER_OUTPUT", False):
        with torch.no_grad():
            output_layer.weight.zero_()
            if output_layer.bias is not None:
                output_layer.bias.zero_()

    if (
        getattr(config, "MLP_BOUND_RESIDUAL_OUTPUT", False)
        and config.PREDICT_RESIDUAL_DELTA
    ):
        max_delta = getattr(config, "MLP_MAX_RESIDUAL_DELTA", None)
        if max_delta is None:
            raise RuntimeError(
                "MLP_MAX_RESIDUAL_DELTA must be estimated from the training "
                "set before constructing a bounded MLP decoder."
            )
        max_delta_tensor = torch.as_tensor(max_delta).reshape(-1)
        if max_delta_tensor.numel() not in (1, y_dim):
            raise ValueError(
                f"MLP_MAX_RESIDUAL_DELTA has {max_delta_tensor.numel()} values; "
                f"expected 1 or {y_dim}."
            )
        layers.append(BoundedResidualOutput(max_delta))

    return nn.Sequential(*layers).double()


def _initial_prediction_frame(y_state: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(y_state) if config.PREDICT_RESIDUAL_DELTA else y_state


def _advance_prediction_state(y_state: torch.Tensor, decoder_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if config.PREDICT_RESIDUAL_DELTA:
        if config.RESIDUAL_ROLLOUT_STATE_MODE == "delta":
            return decoder_output, decoder_output
        return y_state + decoder_output, decoder_output
    return decoder_output, decoder_output


def _recent_history_flat(history, window: int):
    pieces = []
    for offset in range(window + 1):
        idx = len(history) - 1 - offset
        if idx < 0:
            idx = 0
        pieces.append(history[idx])
    return torch.cat(pieces, dim=-1)


def _recent_u_flat(u, j: int, start: int, end: int, window: int):
    pieces = []
    for offset in range(window + 1):
        idx = max(0, j - offset)
        pieces.append(u[idx, :, start:end])
    return torch.cat(pieces, dim=-1)


def _recent_u_with_context_flat(
    history_context,
    u,
    j: int,
    start: int,
    end: int,
    window: int,
):
    if history_context is None:
        return _recent_u_flat(u, j, start, end, window)
    available = torch.cat([history_context, u[:j + 1]], dim=0)
    pieces = []
    for offset in range(window + 1):
        idx = max(0, available.shape[0] - 1 - offset)
        pieces.append(available[idx, :, start:end])
    return torch.cat(pieces, dim=-1)


class Pure_Dense_MLP(nn.Module):
    def __init__(self, y_dim, p_dim, x_env_dim, priors=None, latent_dim=128):
        super().__init__()
        self.y_dim, self.p_dim, self.latent_dim = y_dim, p_dim, latent_dim
        self.W_graph = nn.Parameter(torch.randn(y_dim, y_dim, dtype=torch.float64) * 0.05)
        input_dim = y_dim * 2 + p_dim + x_env_dim
        self.encoder = build_mlp(input_dim, latent_dim, latent_dim)
        self.decoder = build_decoder(latent_dim, y_dim)

    def encode_step(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def forward(self, u):
        y_curr = u[0, :, :self.y_dim]
        preds = [_initial_prediction_frame(y_curr)]
        for j in range(u.shape[0]):
            y_spatial = y_curr @ self.W_graph.t()
            inputs = torch.cat([
                y_curr,
                y_spatial,
                u[j, :, self.y_dim:self.y_dim + self.p_dim],
                u[j, :, self.y_dim + self.p_dim:],
            ], dim=-1)
            feat = self.encode_step(inputs)
            y_next = self.decoder(feat)
            preds.append(y_next)
            y_curr, _ = _advance_prediction_state(y_curr, y_next)
        return torch.stack(preds, dim=0)


class Gaussian_Input_Noise_MLP(Pure_Dense_MLP):
    enable_gaussian_input_noise_training = True


class History_Dense_MLP(nn.Module):
    def __init__(self, y_dim, p_dim, x_env_dim, priors=None, latent_dim=128):
        super().__init__()
        self.y_dim, self.p_dim, self.latent_dim = y_dim, p_dim, latent_dim
        self.history_window = max(1, int(getattr(config, "HISTORY_MLP_WINDOW_STEPS", 4)))
        self.W_graph = nn.Parameter(torch.randn(y_dim, y_dim, dtype=torch.float64) * 0.05)
        input_dim = (
            y_dim * (self.history_window + 1)
            + y_dim
            + (p_dim + x_env_dim) * (self.history_window + 1)
        )
        self.encoder = build_mlp(input_dim, latent_dim, latent_dim)
        self.decoder = build_decoder(latent_dim, y_dim)

    def encode_step(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def forward(self, u, history_context=None):
        y_curr = u[0, :, :self.y_dim]
        y_history = (
            [history_context[j, :, :self.y_dim] for j in range(history_context.shape[0])]
            if history_context is not None
            else []
        )
        y_history.append(y_curr)
        preds = [_initial_prediction_frame(y_curr)]
        for j in range(u.shape[0]):
            y_spatial = y_curr @ self.W_graph.t()
            y_hist = _recent_history_flat(y_history, self.history_window)
            p_hist = _recent_u_with_context_flat(
                history_context, u, j, self.y_dim,
                self.y_dim + self.p_dim, self.history_window,
            )
            x_hist = _recent_u_with_context_flat(
                history_context, u, j, self.y_dim + self.p_dim,
                u.shape[-1], self.history_window,
            )
            inputs = torch.cat([y_hist, y_spatial, p_hist, x_hist], dim=-1)
            feat = self.encode_step(inputs)
            y_next = self.decoder(feat)
            preds.append(y_next)
            y_curr, _ = _advance_prediction_state(y_curr, y_next)
            y_history.append(y_curr)
        return torch.stack(preds, dim=0)


class Physical_Informed_MLP(nn.Module):
    def __init__(self, y_dim, p_dim, x_env_dim, priors, latent_dim=128):
        super().__init__()
        self.y_dim, self.p_dim, self.latent_dim = y_dim, p_dim, latent_dim
        semantics = get_semantic_tensor(priors, y_dim, required_dim=5)
        self.register_buffer("inertia_prior", semantics[:, 0])
        self.register_buffer("occ_prior", semantics[:, 1])
        self.register_buffer("env_prior", semantics[:, 2])
        self.register_buffer("trust_prior", semantics[:, 3])
        self.register_buffer("sens_prior", semantics[:, 4])

        adj_raw = torch.tensor(priors["adjacency_matrix"], dtype=torch.float64) if priors else torch.eye(y_dim, dtype=torch.float64)
        self.register_buffer("A_mask", torch.clamp(adj_raw, 0.0, 1.0))
        self.register_buffer("off_diag_mask", 1.0 - torch.eye(y_dim, dtype=torch.float64))

        self.topo_scale = nn.Parameter(torch.tensor([0.05], dtype=torch.float64))
        self.W_graph_delta = nn.Parameter(torch.zeros(y_dim, y_dim, dtype=torch.float64))
        self.inertia_scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        self.gain_inertia_delta = nn.Parameter(torch.zeros(y_dim, dtype=torch.float64))
        self.occ_scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        self.gain_occ_delta = nn.Parameter(torch.zeros(y_dim, dtype=torch.float64))
        self.env_scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        self.gain_env_delta = nn.Parameter(torch.zeros(y_dim, dtype=torch.float64))

        input_dim = y_dim * 2 + p_dim + x_env_dim
        self.encoder = build_mlp(input_dim, latent_dim, latent_dim)
        self.decoder = build_decoder(latent_dim, y_dim)

    def apply_zone_output_adapter(self, y_next: torch.Tensor) -> torch.Tensor:
        return y_next

    def encode_step(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def _effective_priors(self):
        trust_factor = 1.0 - 0.5 * self.trust_prior
        delta_eff = (self.W_graph_delta * self.off_diag_mask) * trust_factor.unsqueeze(1)
        W_graph_eff = torch.clamp((self.topo_scale * self.A_mask + delta_eff) * self.off_diag_mask, -0.2, 0.2)

        gain_inertia = 1.0 + 0.2 * torch.tanh(self.inertia_scale * (self.inertia_prior - 0.5) + self.gain_inertia_delta)
        gain_occ = 1.0 + 0.2 * torch.tanh(self.occ_scale * (self.occ_prior - 0.5) + self.gain_occ_delta)
        env_eff_prior = torch.sqrt(torch.clamp(self.env_prior * self.sens_prior, min=0.0))
        gain_env = 1.0 + 0.2 * torch.tanh(self.env_scale * (env_eff_prior - 0.5) + self.gain_env_delta)
        return W_graph_eff, gain_inertia, gain_occ, gain_env

    def build_step_inputs(self, u, j, y_curr, W_graph_eff, gain_inertia, gain_occ, gain_env):
        p_curr = u[j, :, self.y_dim:self.y_dim + self.p_dim]
        x_curr = u[j, :, self.y_dim + self.p_dim:]

        y_curr_mod = y_curr * gain_inertia.unsqueeze(0)
        y_spatial = torch.sum(W_graph_eff.unsqueeze(0) * (y_curr.unsqueeze(1) - y_curr.unsqueeze(2)), dim=2)
        p_curr_mod = p_curr * gain_occ.unsqueeze(0) if self.p_dim == self.y_dim else p_curr

        x_curr_mod = x_curr.clone()
        if x_curr.shape[-1] == 1 + 2 * self.y_dim:
            x_curr_mod[:, 1:1 + self.y_dim] *= gain_env.unsqueeze(0)
            x_curr_mod[:, 1 + self.y_dim:] *= gain_occ.unsqueeze(0)

        return torch.cat([y_curr_mod, y_spatial, p_curr_mod, x_curr_mod], dim=-1)

    def forward(self, u):
        y_curr = u[0, :, :self.y_dim]
        preds = [_initial_prediction_frame(y_curr)]
        W_graph_eff, gain_inertia, gain_occ, gain_env = self._effective_priors()

        for j in range(u.shape[0]):
            inputs = self.build_step_inputs(u, j, y_curr, W_graph_eff, gain_inertia, gain_occ, gain_env)
            feat = self.encode_step(inputs)
            y_next = self.apply_zone_output_adapter(self.decoder(feat))
            preds.append(y_next)
            y_curr, _ = _advance_prediction_state(y_curr, y_next)
        return torch.stack(preds, dim=0)


class LLM_Zone_Adapter_Physical_Informed_MLP(Physical_Informed_MLP):
    def __init__(self, y_dim, p_dim, x_env_dim, priors, latent_dim=128):
        super().__init__(y_dim, p_dim, x_env_dim, priors, latent_dim=latent_dim)
        semantics = get_semantic_tensor(priors, y_dim, required_dim=5)
        self.register_buffer("Zone_Adapter_Semantics", semantics)
        self.zone_adapter_gain = nn.Linear(5, 1).double()
        self.zone_adapter_bias = nn.Linear(5, 1).double()
        nn.init.zeros_(self.zone_adapter_gain.weight)
        nn.init.zeros_(self.zone_adapter_gain.bias)
        nn.init.zeros_(self.zone_adapter_bias.weight)
        nn.init.zeros_(self.zone_adapter_bias.bias)

    def apply_zone_output_adapter(self, y_next: torch.Tensor) -> torch.Tensor:
        sem = self.Zone_Adapter_Semantics.to(y_next.device)
        gain = 1.0 + config.ZONE_ADAPTER_GAIN_RHO * torch.tanh(self.zone_adapter_gain(sem)).view(1, self.y_dim)
        y_adapted = y_next * gain
        if not config.PREDICT_RESIDUAL_DELTA and config.ZONE_ADAPTER_BIAS_RHO > 0:
            bias = config.ZONE_ADAPTER_BIAS_RHO * torch.tanh(self.zone_adapter_bias(sem)).view(1, self.y_dim)
            y_adapted = y_adapted + bias
        return y_adapted


class Residual_Gated_Physical_Informed_MLP(Physical_Informed_MLP):
    def build_step_inputs(self, u, j, y_curr, W_graph_eff, gain_inertia, gain_occ, gain_env):
        p_curr = u[j, :, self.y_dim:self.y_dim + self.p_dim]
        x_curr = u[j, :, self.y_dim + self.p_dim:]
        y_spatial = torch.sum(W_graph_eff.unsqueeze(0) * (y_curr.unsqueeze(1) - y_curr.unsqueeze(2)), dim=2)
        return torch.cat([y_curr, y_spatial, p_curr, x_curr], dim=-1)

    def residual_response_gate(self):
        rho_inertia = float(getattr(config, "PI_MLP_RESIDUAL_INERTIA_RHO", 0.10))
        rho_occ = float(getattr(config, "PI_MLP_RESIDUAL_OCC_RHO", 0.05))
        rho_env = float(getattr(config, "PI_MLP_RESIDUAL_ENV_RHO", 0.05))

        gate = torch.ones(self.y_dim, dtype=torch.float64, device=self.inertia_prior.device)
        if rho_inertia > 0:
            inertia_term = torch.tanh(self.inertia_scale * (self.inertia_prior - 0.5) + self.gain_inertia_delta)
            gate = gate * (1.0 - rho_inertia * inertia_term)
        if rho_occ > 0:
            occ_term = torch.tanh(self.occ_scale * (self.occ_prior - 0.5) + self.gain_occ_delta)
            gate = gate * (1.0 + rho_occ * occ_term)
        if rho_env > 0:
            env_eff_prior = torch.sqrt(torch.clamp(self.env_prior * self.sens_prior, min=0.0))
            env_term = torch.tanh(self.env_scale * (env_eff_prior - 0.5) + self.gain_env_delta)
            gate = gate * (1.0 + rho_env * env_term)
        return torch.clamp(gate, 0.75, 1.25)

    def apply_zone_output_adapter(self, y_next: torch.Tensor) -> torch.Tensor:
        return y_next * self.residual_response_gate().to(y_next.device).unsqueeze(0)


class History_Physical_Informed_MLP(Physical_Informed_MLP):
    def __init__(self, y_dim, p_dim, x_env_dim, priors, latent_dim=128):
        super().__init__(y_dim, p_dim, x_env_dim, priors, latent_dim=latent_dim)
        self.history_window = max(1, int(getattr(config, "HISTORY_MLP_WINDOW_STEPS", 4)))
        input_dim = (
            y_dim * (self.history_window + 1)
            + y_dim
            + (p_dim + x_env_dim) * (self.history_window + 1)
        )
        self.encoder = build_mlp(input_dim, latent_dim, latent_dim)

    def build_history_inputs(self, u, j, y_curr, y_history, W_graph_eff, history_context=None):
        y_spatial = torch.sum(W_graph_eff.unsqueeze(0) * (y_curr.unsqueeze(1) - y_curr.unsqueeze(2)), dim=2)
        y_hist = _recent_history_flat(y_history, self.history_window)
        p_hist = _recent_u_with_context_flat(
            history_context, u, j, self.y_dim,
            self.y_dim + self.p_dim, self.history_window,
        )
        x_hist = _recent_u_with_context_flat(
            history_context, u, j, self.y_dim + self.p_dim,
            u.shape[-1], self.history_window,
        )
        return torch.cat([y_hist, y_spatial, p_hist, x_hist], dim=-1)

    def forward(self, u, history_context=None):
        y_curr = u[0, :, :self.y_dim]
        y_history = (
            [history_context[j, :, :self.y_dim] for j in range(history_context.shape[0])]
            if history_context is not None
            else []
        )
        y_history.append(y_curr)
        preds = [_initial_prediction_frame(y_curr)]
        W_graph_eff, _, _, _ = self._effective_priors()

        for j in range(u.shape[0]):
            inputs = self.build_history_inputs(
                u, j, y_curr, y_history, W_graph_eff,
                history_context=history_context,
            )
            feat = self.encode_step(inputs)
            y_next = self.apply_zone_output_adapter(self.decoder(feat))
            preds.append(y_next)
            y_curr, _ = _advance_prediction_state(y_curr, y_next)
            y_history.append(y_curr)
        return torch.stack(preds, dim=0)


class Data_Cluster_Head_Physical_Informed_MLP(Physical_Informed_MLP):
    def __init__(self, y_dim, p_dim, x_env_dim, priors, latent_dim=128):
        super().__init__(y_dim, p_dim, x_env_dim, priors, latent_dim=latent_dim)
        self.n_clusters = max(1, int(getattr(config, "DATA_CLUSTER_NUM_GROUPS", 3)))
        self.register_buffer("zone_cluster_ids", torch.zeros(y_dim, dtype=torch.long))
        hidden_dim = max(16, latent_dim // 2)
        self.cluster_heads = nn.ModuleList()
        for _ in range(self.n_clusters):
            head = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, y_dim),
            ).double()
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.cluster_heads.append(head)

    def set_zone_cluster_ids(self, cluster_ids):
        ids = torch.as_tensor(cluster_ids, dtype=torch.long)
        if ids.numel() != self.y_dim:
            raise ValueError(f"cluster_ids length {ids.numel()} does not match y_dim {self.y_dim}")
        ids = torch.clamp(ids, 0, self.n_clusters - 1)
        self.zone_cluster_ids.data.copy_(ids.to(self.zone_cluster_ids.device))

    def apply_cluster_head(self, feat: torch.Tensor, base_delta: torch.Tensor) -> torch.Tensor:
        rho = float(getattr(config, "DATA_CLUSTER_HEAD_RHO", 0.03))
        if rho <= 0 or self.n_clusters <= 1:
            return base_delta

        cluster_ids = self.zone_cluster_ids.to(base_delta.device)
        correction = torch.zeros_like(base_delta)
        for k, head in enumerate(self.cluster_heads):
            mask = (cluster_ids == k).to(base_delta.dtype).view(1, self.y_dim)
            if torch.any(mask > 0):
                correction = correction + head(feat) * mask
        return base_delta + rho * correction

    def forward(self, u):
        y_curr = u[0, :, :self.y_dim]
        preds = [_initial_prediction_frame(y_curr)]
        W_graph_eff, gain_inertia, gain_occ, gain_env = self._effective_priors()

        for j in range(u.shape[0]):
            inputs = self.build_step_inputs(u, j, y_curr, W_graph_eff, gain_inertia, gain_occ, gain_env)
            feat = self.encode_step(inputs)
            base_delta = self.decoder(feat)
            y_next = self.apply_cluster_head(feat, base_delta)
            preds.append(y_next)
            y_curr, _ = _advance_prediction_state(y_curr, y_next)
        return torch.stack(preds, dim=0)


class Data_Cluster_MLP_Physical_Informed_MLP(Physical_Informed_MLP):
    cluster_source = "data"
    auto_cluster = False
    enable_shared_mix = False

    def __init__(self, y_dim, p_dim, x_env_dim, priors, latent_dim=128):
        super().__init__(y_dim, p_dim, x_env_dim, priors, latent_dim=latent_dim)
        if hasattr(self, "cluster_max_groups_config"):
            cluster_cfg = getattr(
                config,
                self.cluster_max_groups_config,
                getattr(config, "DATA_CLUSTER_MAX_GROUPS", 5),
            )
        else:
            cluster_cfg = getattr(
                config,
                "DATA_CLUSTER_MAX_GROUPS" if getattr(self, "auto_cluster", False) else "DATA_CLUSTER_NUM_GROUPS",
                3,
            )
        if str(cluster_cfg).lower() == "auto":
            cluster_cfg = getattr(config, "DATA_CLUSTER_MAX_GROUPS", 4)
        self.n_clusters = max(1, int(cluster_cfg))
        self.register_buffer("zone_cluster_ids", torch.zeros(y_dim, dtype=torch.long))
        input_dim = y_dim * 2 + p_dim + x_env_dim
        self.cluster_encoders = nn.ModuleList([
            build_mlp(input_dim, latent_dim, latent_dim)
            for _ in range(self.n_clusters)
        ])
        self.cluster_decoders = nn.ModuleList([
            build_decoder(latent_dim, y_dim)
            for _ in range(self.n_clusters)
        ])

    def set_zone_cluster_ids(self, cluster_ids):
        ids = torch.as_tensor(cluster_ids, dtype=torch.long)
        if ids.numel() != self.y_dim:
            raise ValueError(f"cluster_ids length {ids.numel()} does not match y_dim {self.y_dim}")
        ids = torch.clamp(ids, 0, self.n_clusters - 1)
        self.zone_cluster_ids.data.copy_(ids.to(self.zone_cluster_ids.device))

    def forward(self, u):
        batch_size = u.shape[1]
        y_curr = u[0, :, :self.y_dim]
        preds = [_initial_prediction_frame(y_curr)]
        W_graph_eff, gain_inertia, gain_occ, gain_env = self._effective_priors()

        cluster_ids = self.zone_cluster_ids.to(u.device)
        cluster_masks = [
            (cluster_ids == k).to(dtype=torch.float64, device=u.device).view(1, self.y_dim)
            for k in range(self.n_clusters)
        ]

        for j in range(u.shape[0]):
            inputs = self.build_step_inputs(u, j, y_curr, W_graph_eff, gain_inertia, gain_occ, gain_env)
            y_next = torch.zeros(batch_size, self.y_dim, dtype=torch.float64, device=u.device)
            for k in range(self.n_clusters):
                mask = cluster_masks[k]
                if torch.any(mask > 0):
                    feat = self.cluster_encoders[k](inputs)
                    y_next = y_next + self.cluster_decoders[k](feat) * mask
            if getattr(self, "enable_shared_mix", False):
                rho = float(getattr(config, "DATA_AUTOCLUSTER_SHARED_MIX_RHO", 0.03))
                if rho > 0:
                    rho = max(0.0, min(1.0, rho))
                    y_shared = self.decoder(self.encoder(inputs))
                    y_next = (1.0 - rho) * y_next + rho * y_shared
            preds.append(y_next)
            y_curr, _ = _advance_prediction_state(y_curr, y_next)
        return torch.stack(preds, dim=0)


class Prior_Cluster_MLP_Physical_Informed_MLP(Data_Cluster_MLP_Physical_Informed_MLP):
    cluster_source = "prior_name"


class Data_AutoCluster_MLP_Physical_Informed_MLP(Data_Cluster_MLP_Physical_Informed_MLP):
    cluster_source = "data_auto"
    auto_cluster = True


class Data_AutoCluster_DecoderShrink_MLP_Physical_Informed_MLP(Data_AutoCluster_MLP_Physical_Informed_MLP):
    cluster_source = "data_auto"
    auto_cluster = True
    enable_cluster_decoder_shrink_regularization = True

    def decoder_shrink_regularization_loss(self):
        if not hasattr(self, "cluster_decoders") or len(self.cluster_decoders) <= 1:
            device = next(self.parameters()).device
            return torch.tensor(0.0, dtype=torch.float64, device=device)

        loss_terms = []
        for param_group in zip(*[decoder.parameters() for decoder in self.cluster_decoders]):
            stacked = torch.stack([p for p in param_group], dim=0)
            center = stacked.mean(dim=0, keepdim=True).detach()
            loss_terms.append(torch.mean((stacked - center) ** 2))

        if not loss_terms:
            device = next(self.parameters()).device
            return torch.tensor(0.0, dtype=torch.float64, device=device)
        return torch.stack(loss_terms).mean()


class Data_AutoCluster_SharedMix_MLP_Physical_Informed_MLP(Data_AutoCluster_MLP_Physical_Informed_MLP):
    cluster_source = "data_auto"
    auto_cluster = True
    enable_shared_mix = True


class Data_Shape_Cluster_MLP_Physical_Informed_MLP(Data_Cluster_MLP_Physical_Informed_MLP):
    cluster_source = "data_shape"


class Data_Shape_AutoCluster_MLP_Physical_Informed_MLP(Data_Cluster_MLP_Physical_Informed_MLP):
    cluster_source = "data_shape_auto"
    auto_cluster = True


class Prior_AutoCluster_MLP_Physical_Informed_MLP(Data_Cluster_MLP_Physical_Informed_MLP):
    cluster_source = "prior_name_auto"
    auto_cluster = True

class Semantic_AutoCluster_MLP_Physical_Informed_MLP(Data_Cluster_MLP_Physical_Informed_MLP):
    cluster_source = "semantic_auto"
    auto_cluster = True


class LLM_Cluster_MLP_Physical_Informed_MLP(Data_Cluster_MLP_Physical_Informed_MLP):
    cluster_source = "llm"
    auto_cluster = True


class LLM_Data_Arbitrated_Cluster_MLP_Physical_Informed_MLP(Data_Cluster_MLP_Physical_Informed_MLP):
    cluster_source = "llm_data_arbitrated"
    auto_cluster = True


class LLM_Data_Outlier_Cluster_MLP_Physical_Informed_MLP(Data_Cluster_MLP_Physical_Informed_MLP):
    cluster_source = "llm_data_outlier"
    auto_cluster = True


class LLM_Within_Data_Cluster_MLP_Physical_Informed_MLP(Data_Cluster_MLP_Physical_Informed_MLP):
    cluster_source = "llm_within_data"
    auto_cluster = True


class LLM_Data_Refinement_Cluster_MLP_Physical_Informed_MLP(Data_Cluster_MLP_Physical_Informed_MLP):
    cluster_source = "llm_data_refinement"
    auto_cluster = True


class History_Cluster_MLP_Physical_Informed_MLP(History_Physical_Informed_MLP):
    cluster_source = "data"
    auto_cluster = False
    enable_shared_mix = False

    def __init__(self, y_dim, p_dim, x_env_dim, priors, latent_dim=128):
        super().__init__(y_dim, p_dim, x_env_dim, priors, latent_dim=latent_dim)
        if hasattr(self, "cluster_max_groups_config"):
            cluster_cfg = getattr(
                config,
                self.cluster_max_groups_config,
                getattr(config, "DATA_CLUSTER_MAX_GROUPS", 5),
            )
        else:
            cluster_cfg = getattr(
                config,
                "DATA_CLUSTER_MAX_GROUPS" if getattr(self, "auto_cluster", False) else "DATA_CLUSTER_NUM_GROUPS",
                3,
            )
        if str(cluster_cfg).lower() == "auto":
            cluster_cfg = getattr(config, "DATA_CLUSTER_MAX_GROUPS", 4)
        self.n_clusters = max(1, int(cluster_cfg))
        self.register_buffer("zone_cluster_ids", torch.zeros(y_dim, dtype=torch.long))
        input_dim = (
            y_dim * (self.history_window + 1)
            + y_dim
            + (p_dim + x_env_dim) * (self.history_window + 1)
        )
        self.cluster_encoders = nn.ModuleList([
            build_mlp(input_dim, latent_dim, latent_dim)
            for _ in range(self.n_clusters)
        ])
        self.cluster_decoders = nn.ModuleList([
            build_decoder(latent_dim, y_dim)
            for _ in range(self.n_clusters)
        ])

    def set_zone_cluster_ids(self, cluster_ids):
        ids = torch.as_tensor(cluster_ids, dtype=torch.long)
        if ids.numel() != self.y_dim:
            raise ValueError(f"cluster_ids length {ids.numel()} does not match y_dim {self.y_dim}")
        ids = torch.clamp(ids, 0, self.n_clusters - 1)
        self.zone_cluster_ids.data.copy_(ids.to(self.zone_cluster_ids.device))

    def forward(self, u, history_context=None):
        batch_size = u.shape[1]
        y_curr = u[0, :, :self.y_dim]
        y_history = (
            [history_context[j, :, :self.y_dim] for j in range(history_context.shape[0])]
            if history_context is not None
            else []
        )
        y_history.append(y_curr)
        preds = [_initial_prediction_frame(y_curr)]
        W_graph_eff, _, _, _ = self._effective_priors()

        cluster_ids = self.zone_cluster_ids.to(u.device)
        cluster_masks = [
            (cluster_ids == k).to(dtype=torch.float64, device=u.device).view(1, self.y_dim)
            for k in range(self.n_clusters)
        ]

        for j in range(u.shape[0]):
            inputs = self.build_history_inputs(
                u, j, y_curr, y_history, W_graph_eff,
                history_context=history_context,
            )
            y_next = torch.zeros(batch_size, self.y_dim, dtype=torch.float64, device=u.device)
            for k in range(self.n_clusters):
                mask = cluster_masks[k]
                if torch.any(mask > 0):
                    feat = self.cluster_encoders[k](inputs)
                    y_next = y_next + self.cluster_decoders[k](feat) * mask
            if getattr(self, "enable_shared_mix", False):
                rho = float(getattr(config, "DATA_AUTOCLUSTER_SHARED_MIX_RHO", 0.03))
                if rho > 0:
                    rho = max(0.0, min(1.0, rho))
                    y_shared = self.decoder(self.encoder(inputs))
                    y_next = (1.0 - rho) * y_next + rho * y_shared
            preds.append(y_next)
            y_curr, _ = _advance_prediction_state(y_curr, y_next)
            y_history.append(y_curr)
        return torch.stack(preds, dim=0)


class Data_AutoCluster_History_MLP_Physical_Informed_MLP(History_Cluster_MLP_Physical_Informed_MLP):
    cluster_source = "data_auto"
    auto_cluster = True


class Data_Shape_AutoCluster_History_MLP_Physical_Informed_MLP(History_Cluster_MLP_Physical_Informed_MLP):
    cluster_source = "data_shape_auto"
    auto_cluster = True


class LLM_Cluster_History_MLP_Physical_Informed_MLP(History_Cluster_MLP_Physical_Informed_MLP):
    cluster_source = "llm"
    auto_cluster = True


class Semantic_Initialized_ROF_Cluster_History_MLP_Physical_Informed_MLP(History_Cluster_MLP_Physical_Informed_MLP):
    cluster_source = "rof_semantic"
    auto_cluster = True
    cluster_max_groups_config = "LLM_DATA_OUTLIER_MAX_GROUPS"
    semantic_parent_cluster_source = "llm"
    semantic_parent_model_name = "PI + LLM Cluster History MLP"


class Sensitivity_Aligned_Physical_Informed_MLP(Physical_Informed_MLP):
    enable_prior_sensitivity_alignment_regularization = True


MODEL_REGISTRY = {
    "Baseline MLP": {"class": Pure_Dense_MLP},
    "Baseline MLP + Gaussian Input Noise": {"class": Gaussian_Input_Noise_MLP},
    "Baseline History MLP": {"class": History_Dense_MLP},
    "PI MLP": {"class": Physical_Informed_MLP},
    "PI Residual-Gated MLP": {"class": Residual_Gated_Physical_Informed_MLP},
    "PI History MLP": {"class": History_Physical_Informed_MLP},
    "PI + LLM Zone Adapter MLP": {"class": LLM_Zone_Adapter_Physical_Informed_MLP},
    "PI + Data Cluster Head MLP": {"class": Data_Cluster_Head_Physical_Informed_MLP},
    "PI + Data Cluster MLP": {"class": Data_Cluster_MLP_Physical_Informed_MLP},
    "PI + Data AutoCluster MLP": {"class": Data_AutoCluster_MLP_Physical_Informed_MLP},
    "PI + Data AutoCluster History MLP": {"class": Data_AutoCluster_History_MLP_Physical_Informed_MLP},
    "PI + Data Shape AutoCluster History MLP": {"class": Data_Shape_AutoCluster_History_MLP_Physical_Informed_MLP},
    "PI + Data AutoCluster MLP + Decoder Shrink": {"class": Data_AutoCluster_DecoderShrink_MLP_Physical_Informed_MLP},
    "PI + Data AutoCluster MLP + Shared Mix": {"class": Data_AutoCluster_SharedMix_MLP_Physical_Informed_MLP},
    "PI + Data Shape Cluster MLP": {"class": Data_Shape_Cluster_MLP_Physical_Informed_MLP},
    "PI + Data Shape AutoCluster MLP": {"class": Data_Shape_AutoCluster_MLP_Physical_Informed_MLP},
    "PI + Prior Cluster MLP": {"class": Prior_Cluster_MLP_Physical_Informed_MLP},
    "PI + Prior AutoCluster MLP": {"class": Prior_AutoCluster_MLP_Physical_Informed_MLP},
    "PI + Semantic AutoCluster MLP": {"class": Semantic_AutoCluster_MLP_Physical_Informed_MLP},
    "PI + LLM Cluster MLP": {"class": LLM_Cluster_MLP_Physical_Informed_MLP},
    "PI + LLM Cluster History MLP": {"class": LLM_Cluster_History_MLP_Physical_Informed_MLP},
    "PI + Semantic-Initialized ROF Cluster History MLP": {"class": Semantic_Initialized_ROF_Cluster_History_MLP_Physical_Informed_MLP},
    "PI + LLM Data-Arbitrated Cluster MLP": {"class": LLM_Data_Arbitrated_Cluster_MLP_Physical_Informed_MLP},
    "PI + LLM Cluster + Data Outlier MLP": {"class": LLM_Data_Outlier_Cluster_MLP_Physical_Informed_MLP},
    "PI + LLM Cluster + Within DataCluster MLP": {"class": LLM_Within_Data_Cluster_MLP_Physical_Informed_MLP},
    "PI + LLM Cluster + Data Refinement MLP": {"class": LLM_Data_Refinement_Cluster_MLP_Physical_Informed_MLP},
    "PI + SensAlign MLP": {"class": Sensitivity_Aligned_Physical_Informed_MLP},
}
