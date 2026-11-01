"""Aligned direct-temperature GRU models used by the public experiment.

The backbone predicts next-step temperature directly and uses four real
preceding frames to warm up the recurrent hidden state. Cluster models create
exactly one recurrent encoder and one decoder per active cluster; the configured
maximum is only a clustering/search guard.
"""

from typing import Tuple

import torch
import torch.nn as nn

import config
from prior_utils import get_semantic_tensor


def build_decoder(latent_dim: int, y_dim: int) -> nn.Module:
    layers = [nn.Linear(latent_dim, latent_dim), nn.SiLU()]
    if config.USE_DECODER_DROPOUT and config.DECODER_DROPOUT_P > 0:
        layers.append(nn.Dropout(config.DECODER_DROPOUT_P))
    layers.append(nn.Linear(latent_dim, y_dim))
    return nn.Sequential(*layers).double()


def init_gru_cell(cell: nn.GRUCell) -> nn.GRUCell:
    if not bool(getattr(config, "GRU_ZERO_GATE_BIAS", False)):
        return cell
    with torch.no_grad():
        for name, parameter in cell.named_parameters():
            if "bias" in name:
                nn.init.zeros_(parameter)
    return cell


def _normalize_cluster_ids(
    cluster_ids,
    y_dim: int,
    max_groups: int,
) -> Tuple[torch.Tensor, int, list]:
    """Normalize category labels without silently clipping or merging them."""
    ids = torch.as_tensor(cluster_ids, dtype=torch.long).reshape(-1)
    if ids.numel() != y_dim:
        raise ValueError(
            f"cluster_ids length {ids.numel()} does not match y_dim {y_dim}"
        )

    unique_ids = torch.unique(ids, sorted=True)
    actual_groups = int(unique_ids.numel())
    if actual_groups < 1:
        raise ValueError("At least one cluster is required.")
    if actual_groups > int(max_groups):
        raise ValueError(
            f"Found {actual_groups} clusters after normalization, but the "
            f"shared maximum is {max_groups}. Original labels: "
            f"{unique_ids.tolist()}"
        )

    normalized = torch.empty_like(ids)
    for new_id, old_id in enumerate(unique_ids.tolist()):
        normalized[ids == int(old_id)] = int(new_id)
    counts = torch.bincount(normalized, minlength=actual_groups).tolist()
    return normalized, actual_groups, counts


def _checked_history_context(history_context, u, history_window: int):
    if history_context is None:
        return None
    if history_context.ndim != 3:
        raise ValueError(
            "history_context must have shape [history_steps, batch, features]"
        )
    if history_context.shape[1:] != u.shape[1:]:
        raise ValueError(
            f"history_context batch/features {tuple(history_context.shape[1:])} "
            f"do not match input {tuple(u.shape[1:])}"
        )
    return history_context[-max(1, int(history_window)):]


class Pure_Dense_GRU(nn.Module):
    """Baseline GRU with a learned dense spatial transform."""

    def __init__(
        self,
        y_dim,
        p_dim,
        x_env_dim,
        priors=None,
        latent_dim=128,
    ):
        super().__init__()
        self.y_dim = y_dim
        self.p_dim = p_dim
        self.latent_dim = latent_dim
        self.history_window = int(
            getattr(config, "GRU_HISTORY_CONTEXT_STEPS", 4)
        )
        self.W_graph = nn.Parameter(
            torch.randn(y_dim, y_dim, dtype=torch.float64) * 0.05
        )
        input_dim = y_dim * 2 + p_dim + x_env_dim
        self.gru_cell = init_gru_cell(
            nn.GRUCell(input_dim, latent_dim).double()
        )
        self.decoder = build_decoder(latent_dim, y_dim)

    def _step_inputs(self, y_state, frame):
        y_spatial = y_state @ self.W_graph.t()
        return torch.cat([
            y_state,
            y_spatial,
            frame[:, self.y_dim:self.y_dim + self.p_dim],
            frame[:, self.y_dim + self.p_dim:],
        ], dim=-1)

    def forward(self, u, history_context=None):
        batch_size = u.shape[1]
        h_state = torch.zeros(
            batch_size,
            self.latent_dim,
            dtype=torch.float64,
            device=u.device,
        )

        context = _checked_history_context(
            history_context, u, self.history_window,
        )
        if context is not None:
            for frame in context:
                y_history = frame[:, :self.y_dim]
                h_state = self.gru_cell(
                    self._step_inputs(y_history, frame), h_state,
                )

        y_state = u[0, :, :self.y_dim]
        predictions = [y_state]
        for frame in u:
            h_state = self.gru_cell(
                self._step_inputs(y_state, frame), h_state,
            )
            y_state = self.decoder(h_state)
            predictions.append(y_state)
        return torch.stack(predictions, dim=0)


class Physical_Informed_GRU(nn.Module):
    """GRU using IDF topology while preserving the raw measured inputs."""

    def __init__(
        self,
        y_dim,
        p_dim,
        x_env_dim,
        priors,
        latent_dim=128,
    ):
        super().__init__()
        self.y_dim = y_dim
        self.p_dim = p_dim
        self.latent_dim = latent_dim
        self.history_window = int(
            getattr(config, "GRU_HISTORY_CONTEXT_STEPS", 4)
        )

        semantics = get_semantic_tensor(priors, y_dim, required_dim=5)
        self.register_buffer("trust_prior", semantics[:, 3])
        adjacency = torch.as_tensor(
            priors["adjacency_matrix"], dtype=torch.float64,
        )
        if adjacency.shape != (y_dim, y_dim):
            raise ValueError(
                f"adjacency_matrix shape {tuple(adjacency.shape)} does not "
                f"match ({y_dim}, {y_dim})"
            )
        self.register_buffer("A_mask", torch.clamp(adjacency, 0.0, 1.0))
        self.register_buffer(
            "off_diag_mask",
            1.0 - torch.eye(y_dim, dtype=torch.float64),
        )
        self.topo_scale = nn.Parameter(
            torch.tensor([0.05], dtype=torch.float64)
        )
        self.W_graph_delta = nn.Parameter(
            torch.zeros(y_dim, y_dim, dtype=torch.float64)
        )

        input_dim = y_dim * 2 + p_dim + x_env_dim
        self.gru_cell = init_gru_cell(
            nn.GRUCell(input_dim, latent_dim).double()
        )
        self.decoder = build_decoder(latent_dim, y_dim)

    def _effective_graph(self):
        trust_factor = 1.0 - 0.5 * self.trust_prior
        graph_delta = (
            self.W_graph_delta * self.off_diag_mask
        ) * trust_factor.unsqueeze(1)
        return torch.clamp(
            (self.topo_scale * self.A_mask + graph_delta)
            * self.off_diag_mask,
            -0.2,
            0.2,
        )

    def _step_inputs(self, y_state, frame, graph):
        temperature_difference = (
            y_state.unsqueeze(1) - y_state.unsqueeze(2)
        )
        y_spatial = torch.sum(
            graph.unsqueeze(0) * temperature_difference,
            dim=2,
        )
        return torch.cat([
            y_state,
            y_spatial,
            frame[:, self.y_dim:self.y_dim + self.p_dim],
            frame[:, self.y_dim + self.p_dim:],
        ], dim=-1)

    def forward(self, u, history_context=None):
        batch_size = u.shape[1]
        h_state = torch.zeros(
            batch_size,
            self.latent_dim,
            dtype=torch.float64,
            device=u.device,
        )
        graph = self._effective_graph()

        context = _checked_history_context(
            history_context, u, self.history_window,
        )
        if context is not None:
            for frame in context:
                y_history = frame[:, :self.y_dim]
                h_state = self.gru_cell(
                    self._step_inputs(y_history, frame, graph), h_state,
                )

        y_state = u[0, :, :self.y_dim]
        predictions = [y_state]
        for frame in u:
            h_state = self.gru_cell(
                self._step_inputs(y_state, frame, graph), h_state,
            )
            y_state = self.decoder(h_state)
            predictions.append(y_state)
        return torch.stack(predictions, dim=0)


class Cluster_GRU_Physical_Informed_GRU(Physical_Informed_GRU):
    """Independent GRU experts routed by an automatically obtained grouping."""

    cluster_source = "data_shape_auto"

    def __init__(
        self,
        y_dim,
        p_dim,
        x_env_dim,
        priors,
        latent_dim=128,
        n_clusters=None,
    ):
        super().__init__(
            y_dim, p_dim, x_env_dim, priors, latent_dim=latent_dim,
        )
        self.max_clusters = max(
            1, int(getattr(config, "DATA_CLUSTER_MAX_GROUPS", 5))
        )
        if n_clusters is None:
            raise ValueError(
                "Cluster GRU construction requires the actual n_clusters. "
                "The pipeline must resolve labels before model initialization."
            )
        self.n_clusters = int(n_clusters)
        if not 1 <= self.n_clusters <= self.max_clusters:
            raise ValueError(
                f"n_clusters must be in [1, {self.max_clusters}], got "
                f"{self.n_clusters}."
            )
        self.register_buffer(
            "zone_cluster_ids", torch.zeros(y_dim, dtype=torch.long),
        )

        input_dim = y_dim * 2 + p_dim + x_env_dim
        # Construct one complete expert at a time. With the same seed, expert
        # 0..min(K1,K2)-1 then have identical initial parameters even when two
        # grouping methods choose different K.
        cluster_gru_cells = []
        cluster_decoders = []
        for _ in range(self.n_clusters):
            cluster_gru_cells.append(
                init_gru_cell(nn.GRUCell(input_dim, latent_dim).double())
            )
            cluster_decoders.append(build_decoder(latent_dim, y_dim))
        self.cluster_gru_cells = nn.ModuleList(cluster_gru_cells)
        self.cluster_decoders = nn.ModuleList(cluster_decoders)

        # The shared PI recurrent branch is not part of the cluster methods.
        del self.gru_cell
        del self.decoder

    def set_zone_cluster_ids(self, cluster_ids):
        ids, actual_groups, counts = _normalize_cluster_ids(
            cluster_ids, self.y_dim, self.max_clusters,
        )
        if actual_groups != self.n_clusters:
            raise ValueError(
                f"Model was built for K={self.n_clusters}, but labels contain "
                f"K={actual_groups}."
            )
        self.zone_cluster_ids.data.copy_(
            ids.to(self.zone_cluster_ids.device)
        )
        print(
            f"  [ClusterConfig] {self.__class__.__name__}: "
            f"K={actual_groups}, counts={counts}, experts={self.n_clusters}, "
            f"search_max={self.max_clusters}"
        )

    def forward(self, u, history_context=None):
        batch_size = u.shape[1]
        graph = self._effective_graph()
        hidden_states = [
            torch.zeros(
                batch_size,
                self.latent_dim,
                dtype=torch.float64,
                device=u.device,
            )
            for _ in range(self.n_clusters)
        ]
        cluster_ids = self.zone_cluster_ids.to(u.device)
        cluster_masks = [
            (cluster_ids == cluster_id).to(
                dtype=torch.float64, device=u.device,
            ).view(1, self.y_dim)
            for cluster_id in range(self.n_clusters)
        ]

        context = _checked_history_context(
            history_context, u, self.history_window,
        )
        if context is not None:
            for frame in context:
                y_history = frame[:, :self.y_dim]
                inputs = self._step_inputs(y_history, frame, graph)
                for cluster_id in range(self.n_clusters):
                    hidden_states[cluster_id] = self.cluster_gru_cells[
                        cluster_id
                    ](inputs, hidden_states[cluster_id])

        y_state = u[0, :, :self.y_dim]
        predictions = [y_state]
        for frame in u:
            inputs = self._step_inputs(y_state, frame, graph)
            y_next = torch.zeros(
                batch_size,
                self.y_dim,
                dtype=torch.float64,
                device=u.device,
            )
            for cluster_id in range(self.n_clusters):
                hidden_states[cluster_id] = self.cluster_gru_cells[
                    cluster_id
                ](inputs, hidden_states[cluster_id])
                y_next = y_next + self.cluster_decoders[cluster_id](
                    hidden_states[cluster_id]
                ) * cluster_masks[cluster_id]
            y_state = y_next
            predictions.append(y_state)
        return torch.stack(predictions, dim=0)


class Data_Shape_AutoCluster_GRU_Physical_Informed_GRU(
    Cluster_GRU_Physical_Informed_GRU
):
    cluster_source = "data_shape_auto"


class LLM_Cluster_GRU_Physical_Informed_GRU(
    Cluster_GRU_Physical_Informed_GRU
):
    cluster_source = "llm"


MODEL_REGISTRY = {
    "Baseline GRU": {"class": Pure_Dense_GRU},
    "PI": {"class": Physical_Informed_GRU},
    "PI + Data Shape AutoCluster GRU": {
        "class": Data_Shape_AutoCluster_GRU_Physical_Informed_GRU,
    },
    "PI + LLM Cluster GRU": {
        "class": LLM_Cluster_GRU_Physical_Informed_GRU,
    },
}
