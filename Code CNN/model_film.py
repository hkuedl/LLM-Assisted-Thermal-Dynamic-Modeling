"""0927 PI CNN plus the same zone-shared semantic FiLM correction as MLP."""

import torch

import config
from film_adapter import ZoneFiLMAdapter
from model_cnn import (
    Physical_Informed_CNN, Data_Shape_AutoCluster_CNN, LLM_Cluster_CNN,
    _advance_prediction_state, _initial_prediction_frame,
)


class _FiLMSetup:
    def __init__(self, y_dim, p_dim, x_env_dim, priors, latent_dim=None, n_clusters=None):
        # Finish the original model first, preserving every base parameter's
        # initialization for an identical seed and the same finalized K.
        cluster_kwargs = {} if n_clusters is None else {"n_clusters": n_clusters}
        super().__init__(
            y_dim, p_dim, x_env_dim, priors,
            latent_dim=latent_dim, **cluster_kwargs,
        )
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
            self.latent_dim, int(getattr(config, "ZONE_FILM_DIM", 32)),
            int(getattr(config, "ZONE_FILM_SEMANTIC_HIDDEN", 16)),
            float(getattr(config, "ZONE_FILM_STRENGTH", 0.1)),
        )

    def _local_features(self, spatial, frame, y_curr):
        n = self.y_dim
        return torch.stack([
            y_curr, frame[:, n:2*n], frame[:, 3*n+1:4*n+1],
            frame[:, 2*n+1:3*n+1], frame[:, 2*n:2*n+1].expand(-1, n), spatial,
        ], dim=-1)


class FiLM_Physical_Informed_CNN(_FiLMSetup, Physical_Informed_CNN):
    def forward(self, u, history_context=None):
        y_curr = u[0, :, :self.y_dim]
        priors = self._effective_priors()
        history = self._context_frames(history_context, priors)
        predictions = [_initial_prediction_frame(y_curr)]
        # Reuse within this forward only; keep the semantic network's gradients.
        modulation = self.film_adapter.modulation(self.film_semantics)
        for frame in u:
            history.append(self.build_step_frame(y_curr, frame, priors))
            temporal = self.encoder(self._history_tensor(history))
            spatial = self.build_current_spatial(y_curr, priors)
            feature = self.fusion(torch.cat([temporal, spatial], dim=-1))
            local = self._local_features(spatial, frame, y_curr)
            y_next = self.decoder(feature) + self.film_adapter(
                feature, local, self.film_semantics, modulation=modulation,
            )
            predictions.append(y_next)
            # The complete corrected temperature feeds the next CNN history.
            y_curr, _ = _advance_prediction_state(y_curr, y_next)
        return torch.stack(predictions, dim=0)


class _ClusterFiLM(_FiLMSetup):
    def forward(self, u, history_context=None):
        y_curr = u[0, :, :self.y_dim]
        priors = self._effective_priors()
        history = self._context_frames(history_context, priors)
        predictions = [_initial_prediction_frame(y_curr)]
        gamma, beta = self.film_adapter.modulation(self.film_semantics)
        cluster_ids = self.zone_cluster_ids.to(u.device)
        indices = [torch.nonzero(cluster_ids == k, as_tuple=False).flatten()
                   for k in range(self.n_clusters)]
        for frame in u:
            history.append(self.build_step_frame(y_curr, frame, priors))
            sequence = self._history_tensor(history)
            spatial = self.build_current_spatial(y_curr, priors)
            local = self._local_features(spatial, frame, y_curr)
            y_next = torch.zeros_like(y_curr)
            for k, selected in enumerate(indices):
                if selected.numel() == 0:
                    continue
                temporal = self.cluster_encoders[k](sequence)
                feature = self.cluster_fusions[k](torch.cat([temporal, spatial], dim=-1))
                base_prediction = self.cluster_decoders[k](feature)
                mask = (cluster_ids == k).to(u.dtype).unsqueeze(0)
                # One shared adapter across experts; each zone is corrected once.
                correction = self.film_adapter(
                    feature, local.index_select(1, selected),
                    self.film_semantics.index_select(0, selected),
                    modulation=(gamma.index_select(0, selected), beta.index_select(0, selected)),
                )
                expanded = torch.zeros_like(y_curr).index_copy(1, selected, correction)
                y_next = y_next + base_prediction * mask + expanded
            predictions.append(y_next)
            y_curr, _ = _advance_prediction_state(y_curr, y_next)
        return torch.stack(predictions, dim=0)


class FiLM_Data_Shape_Cluster_CNN(_ClusterFiLM, Data_Shape_AutoCluster_CNN):
    """Original Data Shape groups/experts plus the shared FiLM correction."""


class FiLM_LLM_Cluster_CNN(_ClusterFiLM, LLM_Cluster_CNN):
    """Original LLM groups/experts plus the shared FiLM correction."""
