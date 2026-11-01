"""Direct-temperature LSTM models with the shared zone-level FiLM adapter.

The FiLM path matches the MLP/CNN experiment: one adapter is shared across
zones and cluster experts, semantic scores modulate a small latent projection,
and the final correction layer starts at zero so the backbone output is
preserved at initialization.
"""

import torch

import config
from film_adapter import ZoneFiLMAdapter
from model_lstm import (
    Physical_Informed_LSTM,
    Cluster_LSTM_Physical_Informed_LSTM,
)
from prior_utils import get_semantic_tensor


def _validated_film_semantics(priors, y_dim):
    expected = [
        "Thermal_Inertia",
        "Occupancy_Volatility",
        "Env_Exposure",
        "Base_Physics_Trust",
        "Env_Sensitivity",
    ]
    semantics = get_semantic_tensor(priors, y_dim, required_dim=5)
    semantics = torch.as_tensor(semantics, dtype=torch.float64)
    names = list(priors.get("semantic_names", expected))
    if names != expected:
        raise ValueError(
            "Semantic columns do not match the five-column experiment schema."
        )
    if semantics.shape != (y_dim, 5) or not torch.isfinite(semantics).all():
        raise ValueError("Expected finite semantic_matrix [zones, 5].")
    if not ((semantics >= 0) & (semantics <= 1)).all():
        raise ValueError("Semantic scores must be in [0, 1].")
    return semantics[:, [0, 1, 2, 4]].clone()


class _ZoneFiLMMixin:
    def _setup_zone_film(self, priors):
        if bool(getattr(config, "PREDICT_RESIDUAL_DELTA", False)):
            raise ValueError(
                "The aligned LSTM FiLM experiment requires direct-temperature "
                "prediction."
            )
        if self.p_dim != self.y_dim:
            raise ValueError("Expected one HVAC input per zone.")
        self.register_buffer(
            "film_semantics",
            _validated_film_semantics(priors, self.y_dim),
        )

        # Keep adapter initialization independent of the number of experts.
        # This lets Data Shape and LLM runs with different K use the same FiLM
        # starting point for the same experiment seed.
        experiment_seed = int(getattr(config, "EXPERIMENT_SEED", 42))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(experiment_seed + 1_000_003)
            self.film_adapter = ZoneFiLMAdapter(
                self.latent_dim,
                int(getattr(config, "ZONE_FILM_DIM", 32)),
                int(getattr(config, "ZONE_FILM_SEMANTIC_HIDDEN", 16)),
                float(getattr(config, "ZONE_FILM_STRENGTH", 0.1)),
            )

    def _local_features(self, spatial, frame, y_state):
        n = self.y_dim
        expected_x_env = 1 + 2 * n
        actual_x_env = frame.shape[-1] - 2 * n
        if actual_x_env != expected_x_env:
            raise ValueError(
                "Expected [temperature, HVAC, outdoor, solar, internal] "
                f"inputs, got x_env_dim={actual_x_env}."
            )
        return torch.stack([
            y_state,
            frame[:, n:2 * n],
            frame[:, 3 * n + 1:4 * n + 1],
            frame[:, 2 * n + 1:3 * n + 1],
            frame[:, 2 * n:2 * n + 1].expand(-1, n),
            spatial,
        ], dim=-1)


class FiLM_Physical_Informed_LSTM(_ZoneFiLMMixin, Physical_Informed_LSTM):
    def __init__(
        self, y_dim, p_dim, x_env_dim, priors, latent_dim=128,
    ):
        super().__init__(
            y_dim, p_dim, x_env_dim, priors, latent_dim=latent_dim,
        )
        self._setup_zone_film(priors)

    def forward(self, u, history_context=None):
        batch_size = u.shape[1]
        hidden = torch.zeros(
            batch_size,
            self.latent_dim,
            dtype=u.dtype,
            device=u.device,
        )
        cell = torch.zeros_like(hidden)
        graph = self._effective_graph()

        context = None
        if history_context is not None:
            context = history_context[-max(1, int(self.history_window)):]
        if context is not None:
            for frame in context:
                y_history = frame[:, :self.y_dim]
                hidden, cell = self.lstm_cell(
                    self._step_inputs(y_history, frame, graph),
                    (hidden, cell),
                )

        y_state = u[0, :, :self.y_dim]
        predictions = [y_state]
        modulation = self.film_adapter.modulation(self.film_semantics)
        for frame in u:
            inputs = self._step_inputs(y_state, frame, graph)
            hidden, cell = self.lstm_cell(inputs, (hidden, cell))
            spatial = inputs[:, self.y_dim:2 * self.y_dim]
            local = self._local_features(spatial, frame, y_state)
            y_state = self.decoder(hidden) + self.film_adapter(
                hidden,
                local,
                self.film_semantics,
                modulation=modulation,
            )
            predictions.append(y_state)
        return torch.stack(predictions, dim=0)


class _ClusterFiLMLSTM(_ZoneFiLMMixin, Cluster_LSTM_Physical_Informed_LSTM):
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
            y_dim,
            p_dim,
            x_env_dim,
            priors,
            latent_dim=latent_dim,
            n_clusters=n_clusters,
        )
        self._setup_zone_film(priors)

    def forward(self, u, history_context=None):
        batch_size = u.shape[1]
        graph = self._effective_graph()
        hidden_states = [
            torch.zeros(
                batch_size,
                self.latent_dim,
                dtype=u.dtype,
                device=u.device,
            )
            for _ in range(self.n_clusters)
        ]
        cell_states = [torch.zeros_like(hidden) for hidden in hidden_states]
        cluster_ids = self.zone_cluster_ids.to(u.device)
        selected_by_cluster = [
            torch.nonzero(cluster_ids == k, as_tuple=False).flatten()
            for k in range(self.n_clusters)
        ]

        context = None
        if history_context is not None:
            context = history_context[-max(1, int(self.history_window)):]
        if context is not None:
            for frame in context:
                y_history = frame[:, :self.y_dim]
                inputs = self._step_inputs(y_history, frame, graph)
                for k in range(self.n_clusters):
                    hidden_states[k], cell_states[k] = self.cluster_lstm_cells[k](
                        inputs, (hidden_states[k], cell_states[k]),
                    )

        y_state = u[0, :, :self.y_dim]
        predictions = [y_state]
        gamma, beta = self.film_adapter.modulation(self.film_semantics)
        for frame in u:
            inputs = self._step_inputs(y_state, frame, graph)
            spatial = inputs[:, self.y_dim:2 * self.y_dim]
            local = self._local_features(spatial, frame, y_state)
            y_next = torch.zeros_like(y_state)
            for k, selected in enumerate(selected_by_cluster):
                if selected.numel() == 0:
                    raise RuntimeError(f"Cluster {k} has no assigned zones.")
                hidden_states[k], cell_states[k] = self.cluster_lstm_cells[k](
                    inputs, (hidden_states[k], cell_states[k]),
                )
                base = self.cluster_decoders[k](hidden_states[k])
                correction = self.film_adapter(
                    hidden_states[k],
                    local.index_select(1, selected),
                    self.film_semantics.index_select(0, selected),
                    modulation=(
                        gamma.index_select(0, selected),
                        beta.index_select(0, selected),
                    ),
                )
                y_next[:, selected] = (
                    base.index_select(1, selected) + correction
                )
            y_state = y_next
            predictions.append(y_state)
        return torch.stack(predictions, dim=0)


class FiLM_Data_Shape_Cluster_LSTM(_ClusterFiLMLSTM):
    cluster_source = "data_shape_auto"


class FiLM_LLM_Cluster_LSTM(_ClusterFiLMLSTM):
    cluster_source = "llm"
