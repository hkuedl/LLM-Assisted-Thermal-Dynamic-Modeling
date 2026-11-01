"""Utilities for loading the offline IDF-derived modeling priors."""

import glob
import json
import os
from typing import List, Optional

import numpy as np
import torch

import config


def get_semantic_tensor(
    priors: Optional[dict],
    y_dim: int,
    required_dim: int = 3,
    default_value: float = 0.5,
) -> torch.Tensor:
    if priors and "semantic_matrix" in priors and len(priors["semantic_matrix"]) == y_dim:
        semantic = np.asarray(priors["semantic_matrix"], dtype=float)
    else:
        semantic = np.full((y_dim, required_dim), default_value, dtype=float)

    if semantic.ndim != 2 or semantic.shape[0] != y_dim:
        semantic = np.full((y_dim, required_dim), default_value, dtype=float)
    if semantic.shape[1] < required_dim:
        padding = np.full(
            (y_dim, required_dim - semantic.shape[1]),
            default_value,
            dtype=float,
        )
        semantic = np.hstack([semantic, padding])
    return torch.tensor(semantic[:, :required_dim], dtype=torch.float64)


def is_degenerate_semantic_prior(priors: Optional[dict]) -> bool:
    if not priors or "semantic_matrix" not in priors:
        return True
    try:
        semantic = np.asarray(priors["semantic_matrix"], dtype=float)
        finite = semantic[np.isfinite(semantic)]
        return semantic.ndim != 2 or finite.size == 0 or len(np.unique(finite)) <= 1
    except (TypeError, ValueError):
        return True


def find_stability_prior_path(building_name: str) -> Optional[str]:
    aliases = {"67zone_HotelSmall_1": "67zone_HotelSmall"}
    stability_name = aliases.get(building_name, building_name)
    candidates = [
        os.path.join(config.PRIOR_STABILITY_ROOT, f"{building_name}.json"),
        os.path.join(
            config.PRIOR_STABILITY_ROOT,
            f"Result_LLM_Stability_{stability_name}",
            "median_quantized_prior.json",
        ),
    ]
    candidates.extend(
        glob.glob(
            os.path.join(
                config.PRIOR_STABILITY_ROOT,
                f"llm_priors_{building_name}_*_median_quantized.json",
            )
        )
    )
    return next((path for path in candidates if path and os.path.exists(path)), None)


def load_prior_json(path: str, zones: List[str], label: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            priors = json.load(handle)
        prior_zones = priors.get("zones", zones)
        if len(prior_zones) != len(zones):
            print(f"[Prior Warning] {label} zone count mismatch: {path}")
            return None
        priors["semantic_names"] = priors.get(
            "semantic_names", config.SEMANTIC_NAMES_DEFAULT
        )
        priors["zones"] = list(zones)
        if label == "result-cache" and is_degenerate_semantic_prior(priors):
            print(f"[Prior Warning] Degenerate result cache ignored: {path}")
            return None
        return priors
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[Prior Warning] Failed to read {label}: {path} | {exc}")
        return None


def load_or_copy_existing_prior(
    cache_path: str,
    building_name: str,
    zones: List[str],
    force_regenerate: bool = False,
) -> dict:
    if os.path.exists(cache_path) and not force_regenerate:
        cached = load_prior_json(cache_path, zones, "result-cache")
        if cached is not None:
            return cached

    source_path = find_stability_prior_path(building_name)
    if source_path:
        priors = load_prior_json(source_path, zones, "stability-source")
        if priors is not None:
            if config.PRIOR_COPY_TO_RESULT_DIR:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as handle:
                    json.dump(priors, handle, indent=2)
                print(f"[Prior Copied] {source_path} -> {cache_path}")
            return priors

    print(f"[Prior Warning] No prior found for {building_name}; using neutral values.")
    return {
        "zones": list(zones),
        "semantic_names": list(config.SEMANTIC_NAMES_DEFAULT),
        "semantic_matrix": np.full((len(zones), 5), 0.5, dtype=float).tolist(),
    }
