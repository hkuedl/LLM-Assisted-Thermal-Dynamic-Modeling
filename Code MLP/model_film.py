"""0927 History PI MLP plus a zone-shared semantic FiLM correction."""

import torch

import config
from film_adapter import ZoneFiLMAdapter
from model_mlp import (
    History_Physical_Informed_MLP,
    Data_Shape_AutoCluster_History_MLP_Physical_Informed_MLP,
    LLM_Cluster_History_MLP_Physical_Informed_MLP,
    _advance_prediction_state, _initial_prediction_frame,
)
class _FiLMSetup:
    def __init__(self, y_dim, p_dim, x_env_dim, priors, latent_dim=128):
        # Construct the complete original PI before adding any new parameters:
        # resetting to the same seed therefore gives identical base weights.
        super().__init__(y_dim, p_dim, x_env_dim, priors, latent_dim=latent_dim)
        if config.PREDICT_RESIDUAL_DELTA:
            raise ValueError("This FiLM experiment requires direct-temperature prediction.")
        if p_dim != y_dim or x_env_dim != 1 + 2 * y_dim:
            raise ValueError("Expected [temperature, HVAC, outdoor, solar, internal] inputs.")
        semantics = torch.as_tensor(priors["semantic_matrix"], dtype=torch.float64)
        expected = ["Thermal_Inertia", "Occupancy_Volatility", "Env_Exposure",
                    "Base_Physics_Trust", "Env_Sensitivity"]
        if list(priors.get("semantic_names", expected)) != expected:
            raise ValueError("Semantic columns do not match the 0927 schema.")
        if semantics.shape != (y_dim, 5) or not torch.isfinite(semantics).all():
            raise ValueError("Expected finite semantic_matrix [zones, 5].")
        if not ((semantics >= 0) & (semantics <= 1)).all():
            raise ValueError("Semantic scores must be in [0, 1].")
        self.register_buffer("film_semantics", semantics[:, [0, 1, 2, 4]].clone())
        self.film_adapter = ZoneFiLMAdapter(
            latent_dim, int(getattr(config, "ZONE_FILM_DIM", 32)),
            int(getattr(config, "ZONE_FILM_SEMANTIC_HIDDEN", 16)),
            float(getattr(config, "ZONE_FILM_STRENGTH", 0.1)),
        )

    def _local_features(self, inputs, frame, y_curr):
        n = self.y_dim
        spatial = inputs[:, n * (self.history_window + 1):n * (self.history_window + 2)]
        return torch.stack([
            y_curr, frame[:, n:2*n], frame[:, 3*n+1:4*n+1],
            frame[:, 2*n+1:3*n+1], frame[:, 2*n:2*n+1].expand(-1, n), spatial,
        ], dim=-1)


class FiLM_History_Physical_Informed_MLP(_FiLMSetup, History_Physical_Informed_MLP):

    def forward(self, u, history_context=None):
        y_curr = u[0, :, :self.y_dim]
        y_history = ([] if history_context is None else
                     [frame[:, :self.y_dim] for frame in history_context])
        y_history.append(y_curr)
        predictions = [_initial_prediction_frame(y_curr)]
        graph, _, _, _ = self._effective_priors()
        # Parameters are constant during one forward; retain the shared
        # autograd graph while avoiding repeated semantic-network evaluation.
        modulation = self.film_adapter.modulation(self.film_semantics)
        for j, frame in enumerate(u):
            inputs = self.build_history_inputs(
                u, j, y_curr, y_history, graph, history_context=history_context,
            )
            feature = self.encode_step(inputs)
            # Read the spatial block built by the original PI; do not change
            # its temperature normalization or topology during this experiment.
            local_features = self._local_features(inputs, frame, y_curr)
            y_next = self.decoder(feature) + self.film_adapter(
                feature, local_features, self.film_semantics, modulation=modulation,
            )
            predictions.append(y_next)
            # Feed the FULL corrected prediction back, with gradients intact.
            y_curr, _ = _advance_prediction_state(y_curr, y_next)
            y_history.append(y_curr)
        return torch.stack(predictions, dim=0)


class _ClusterFiLM(_FiLMSetup):
    def forward(self, u, history_context=None):
        y_curr = u[0, :, :self.y_dim]
        y_history = ([] if history_context is None else
                     [frame[:, :self.y_dim] for frame in history_context])
        y_history.append(y_curr)
        predictions = [_initial_prediction_frame(y_curr)]
        graph, _, _, _ = self._effective_priors()
        gamma, beta = self.film_adapter.modulation(self.film_semantics)
        cluster_ids = self.zone_cluster_ids.to(u.device)
        indices = [torch.nonzero(cluster_ids == k, as_tuple=False).flatten()
                   for k in range(self.n_clusters)]
        for j, frame in enumerate(u):
            inputs = self.build_history_inputs(
                u, j, y_curr, y_history, graph, history_context=history_context,
            )
            local = self._local_features(inputs, frame, y_curr)
            y_next = torch.zeros_like(y_curr)
            for k, selected in enumerate(indices):
                if selected.numel() == 0:
                    continue
                feature = self.cluster_encoders[k](inputs)
                base_prediction = self.cluster_decoders[k](feature)
                # Preserve original masked decoder summation. The same FiLM
                # module serves all clusters; evaluate only their selected zones.
                mask = (cluster_ids == k).to(u.dtype).unsqueeze(0)
                correction = self.film_adapter(
                    feature, local.index_select(1, selected),
                    self.film_semantics.index_select(0, selected),
                    modulation=(gamma.index_select(0, selected), beta.index_select(0, selected)),
                )
                expanded = torch.zeros_like(y_curr).index_copy(1, selected, correction)
                y_next = y_next + base_prediction * mask + expanded
            predictions.append(y_next)
            y_curr, _ = _advance_prediction_state(y_curr, y_next)
            y_history.append(y_curr)
        return torch.stack(predictions, dim=0)


class FiLM_Data_Shape_Cluster_MLP(_ClusterFiLM, Data_Shape_AutoCluster_History_MLP_Physical_Informed_MLP):
    """Original Data Shape groups/experts plus the shared FiLM correction."""


class FiLM_LLM_Cluster_MLP(_ClusterFiLM, LLM_Cluster_History_MLP_Physical_Informed_MLP):
    """Original LLM groups/experts plus the shared FiLM correction."""
