import os
import glob
import json
import torch
import random
import re
import pandas as pd
import numpy as np

import config
from data_utils import (
    build_multizone_timeseries,
    pack_dataset_author_style,
    pack_history_context_author_style,
    persistence_predict_from_frame,
)
from prior_utils import load_or_copy_existing_prior
from model_mlp import MODEL_REGISTRY
from train import (
    get_model_checkpoint_path,
    run_unified_training,
    predict_and_unscale_unified,
    print_and_log_metrics,
    print_first_step_residual_diagnostics,
    print_raw_rollout_zone_diagnostics,
    semantic_initialize_cluster_model,
    target_mode_model_name,
)

def _instantiate_model_class(m_cls, zones_count, x_env_dim, priors, device):
    return (
        m_cls(zones_count, zones_count, x_env_dim, priors).to(device)
        if "priors" in m_cls.__init__.__code__.co_varnames
        else m_cls(zones_count, zones_count, x_env_dim).to(device)
    )

def ensure_semantic_parent_checkpoint(
    parent_model_name,
    zone_cluster_cache,
    zones,
    x_env_dim,
    priors,
    device,
    seed,
    result_dir,
    csv_p,
    few_shot_days,
    train_data,
    train_label,
    SS_Y,
    train_history_context=None,
):
    parent_ckpt = get_model_checkpoint_path(
        result_dir, csv_p, len(zones),
        parent_model_name, few_shot_days, seed,
    )
    if os.path.exists(parent_ckpt):
        print(f"  [SemanticInit Parent] Found LLM parent checkpoint: {parent_ckpt}")
        return True
    if parent_model_name not in MODEL_REGISTRY or "llm" not in zone_cluster_cache:
        print("  [SemanticInit Parent Skip] Missing parent model or LLM labels.")
        return False

    print(f"  [SemanticInit Parent] Training missing parent: {parent_model_name}")
    set_seed(seed)
    parent_cls = MODEL_REGISTRY[parent_model_name]["class"]
    parent_model = _instantiate_model_class(
        parent_cls, len(zones), x_env_dim, priors, device,
    )
    if hasattr(parent_model, "set_zone_cluster_ids"):
        parent_model.set_zone_cluster_ids(zone_cluster_cache["llm"])
    trained_parent = run_unified_training(
        target_mode_model_name(parent_model_name),
        parent_model,
        train_data,
        train_label,
        SS_Y,
        seed,
        dataset_file=csv_p,
        day_tag=few_shot_days,
        result_dir=result_dir,
        train_history_context=train_history_context,
    )
    return trained_parent is not None and os.path.exists(parent_ckpt)

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def infer_numbered_external_tests(csv_path: str, variant_ids=(1, 2, 3), suffix: str = "_new"):
    data_dir = os.path.dirname(csv_path) or "."
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    prefix = re.sub(r"_\d+$", "", stem)
    external_specs = []
    for variant_id in variant_ids:
        ext_csv = os.path.join(data_dir, f"{prefix}_{variant_id}{suffix}.csv")
        ext_idf = os.path.join(data_dir, f"{prefix}_{variant_id}{suffix}.idf")
        if os.path.exists(ext_csv):
            external_specs.append((ext_csv, ext_idf if os.path.exists(ext_idf) else None))
    return external_specs

def run_persistence(df_test, ext_tests, zones, csv_p, few_shot_days, result_dir):
    if not config.RUN_PERSISTENCE_BASELINE: return
    print(f"\n--- Evaluating Persistence baseline ---")
    pers_preds = persistence_predict_from_frame(df_test, zones)
    print_and_log_metrics(pers_preds, df_test, zones, "Persistence", csv_p, few_shot_days, result_dir)
    for ext in ext_tests:
        ext_preds = persistence_predict_from_frame(ext["df"], zones)
        ext_model_name = f"Persistence | Evaluated on {ext['name']}"
        print_and_log_metrics(ext_preds, ext["df"], zones, ext_model_name, csv_p, few_shot_days, result_dir, is_external=True)

def print_zone_clusters(labels, zones, raw_features, feature_names, cluster_name):
    n_groups = int(np.max(labels)) + 1 if len(labels) else 0
    counts = np.bincount(labels, minlength=n_groups)
    print(f"  [{cluster_name}] K={n_groups}, counts={counts.tolist()}")
    for k in range(n_groups):
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            print(f"    [Cluster {k}] empty")
            continue
        means = raw_features[idx].mean(axis=0)
        summary = ", ".join(f"{name}={val:.3f}" for name, val in zip(feature_names, means))
        zone_list = [zones[i] for i in idx]
        print(f"    [Cluster {k}] n={len(zone_list)} | {summary}")
        chunk_size = 4
        for start in range(0, len(zone_list), chunk_size):
            chunk = zone_list[start:start + chunk_size]
            print(f"      zones[{start:02d}:{start + len(chunk):02d}]: {', '.join(chunk)}")

def get_cluster_seed(seed):
    return int(getattr(config, "CLUSTER_RANDOM_SEED", seed))


def run_kmeans_or_quantile(features, n_groups, seed, cluster_name, fallback_col=0):
    X = np.nan_to_num(np.asarray(features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    try:
        from sklearn.cluster import KMeans
        cluster_seed = get_cluster_seed(seed)
        return KMeans(n_clusters=n_groups, random_state=cluster_seed, n_init=10).fit_predict(X_scaled).astype(int)
    except Exception as err:
        print(f"  [{cluster_name} Warning] KMeans failed ({err}); using quantile fallback.")
        order_feature = X_scaled[:, fallback_col]
        edges = np.quantile(order_feature, np.linspace(0, 1, n_groups + 1)[1:-1])
        return np.digitize(order_feature, edges).astype(int)

def _clean_zone_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())

def resolve_cluster_count(features, seed, cluster_name, auto_override=None):
    X = np.nan_to_num(np.asarray(features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    n_zones = X.shape[0]
    if n_zones <= 1:
        return 1

    requested = getattr(config, "DATA_CLUSTER_NUM_GROUPS", 3)
    auto_k = (
        bool(getattr(config, "DATA_CLUSTER_AUTO_K", False)) or str(requested).lower() == "auto"
        if auto_override is None
        else bool(auto_override)
    )
    if not auto_k:
        return max(1, min(int(requested), n_zones))

    k_min = max(2, int(getattr(config, "DATA_CLUSTER_MIN_GROUPS", 2)))
    k_max = min(int(getattr(config, "DATA_CLUSTER_MAX_GROUPS", 4)), n_zones - 1)
    if k_max < k_min:
        return max(1, min(2, n_zones))

    X_scaled = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    best_k, best_score = k_min, -np.inf
    scores = []
    cluster_seed = get_cluster_seed(seed)
    tol = float(getattr(config, "DATA_CLUSTER_AUTO_K_TOL", 0.0))
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        for k in range(k_min, k_max + 1):
            labels = KMeans(n_clusters=k, random_state=cluster_seed, n_init=10).fit_predict(X_scaled)
            if len(np.unique(labels)) < 2:
                continue
            score = float(silhouette_score(X_scaled, labels))
            scores.append((k, score))
            if score > best_score:
                best_k, best_score = k, score
        selected_score = best_score
        if scores and tol > 0:
            eligible = [(k, score) for k, score in scores if score >= best_score - tol]
            best_k, selected_score = min(eligible, key=lambda item: item[0])
        print(
            f"  [{cluster_name}] Auto-K selected K={best_k} "
            f"by silhouette={selected_score:.3f} "
            f"(best={best_score:.3f}, tol={tol:.3f}, cluster_seed={cluster_seed})"
        )
    except Exception as err:
        print(f"  [{cluster_name} Warning] Auto-K failed ({err}); using K={best_k}.")
    return best_k

def infer_training_curve_clusters(df_train, zones, seed, auto_k=False, cluster_name="DataCluster"):
    n_zones = len(zones)

    raw_features, feature_names = compute_training_curve_features(df_train, zones)
    n_groups = resolve_cluster_count(raw_features, seed, cluster_name, auto_override=auto_k)
    if n_groups <= 1:
        return np.zeros(n_zones, dtype=int)
    labels = run_kmeans_or_quantile(raw_features, n_groups, seed, cluster_name, fallback_col=1)
    print_zone_clusters(labels, zones, raw_features, feature_names, cluster_name)
    return labels.astype(int)

def compute_training_curve_features(df_train, zones):
    n_zones = len(zones)
    temps = np.column_stack([df_train[f"T_in__{z}"].values.astype(float) for z in zones])
    hvac = np.column_stack([df_train[f"Q_hvac__{z}"].values.astype(float) for z in zones])
    occ = np.column_stack([df_train[f"Q_occ__{z}"].values.astype(float) for z in zones])
    tout = df_train["T_out"].values.astype(float)
    dtemp = np.diff(temps, axis=0)

    def safe_corr(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 3:
            return 0.0
        a = a[mask]
        b = b[mask]
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    feats = []
    for i in range(n_zones):
        ti = temps[:, i]
        di = dtemp[:, i] if dtemp.shape[0] > 0 else np.zeros(1)
        feats.append([
            float(np.nanstd(ti)),
            float(np.nanpercentile(ti, 95) - np.nanpercentile(ti, 5)),
            float(np.nanmean(np.abs(di))),
            float(np.nanstd(di)),
            safe_corr(ti, tout),
            safe_corr(ti, hvac[:, i]),
            safe_corr(ti, occ[:, i]),
        ])

    feature_names = ["stdT", "amp95_5", "meanAbsDelta", "stdDelta", "corrTout", "corrHVAC", "corrOcc"]
    raw_features = np.nan_to_num(np.asarray(feats, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return raw_features, feature_names

def _compact_cluster_labels(labels):
    labels = np.asarray(labels, dtype=int)
    unique = sorted(np.unique(labels).tolist())
    remap = {old: new for new, old in enumerate(unique)}
    return np.asarray([remap[int(v)] for v in labels], dtype=int)

def _merge_labels_to_target_k(labels, merge_idx, merge_features, target_k, cluster_name):
    labels = np.asarray(labels, dtype=int).copy()
    merge_idx = np.asarray(merge_idx, dtype=int)
    active_groups = sorted(np.unique(labels[merge_idx]).tolist()) if merge_idx.size else []
    target_k = max(1, min(int(target_k), len(active_groups))) if active_groups else 1
    features = np.nan_to_num(np.asarray(merge_features, dtype=float))
    features = (features - features.mean(axis=0, keepdims=True)) / (
        features.std(axis=0, keepdims=True) + 1e-8
    )
    groups = {
        int(g): set(np.where(labels == int(g))[0].tolist()) & set(merge_idx.tolist())
        for g in active_groups
    }
    groups = {g: idxs for g, idxs in groups.items() if idxs}
    while len(groups) > target_k:
        centroids = {
            g: features[sorted(idxs)].mean(axis=0)
            for g, idxs in groups.items()
        }
        best_pair, best_dist = None, float("inf")
        gids = sorted(groups)
        for i, gi in enumerate(gids):
            for gj in gids[i + 1:]:
                dist = float(np.linalg.norm(centroids[gi] - centroids[gj]))
                if dist < best_dist:
                    best_pair, best_dist = (gi, gj), dist
        if best_pair is None:
            break
        keep, drop = best_pair
        print(f"  [{cluster_name}] Merge normal LLM groups {drop} -> {keep}.")
        labels[list(groups[drop])] = keep
        groups[keep].update(groups.pop(drop))
    return _compact_cluster_labels(labels)

def infer_response_outlier_filtered_llm_clusters(
    df_train,
    zones,
    llm_labels,
    priors=None,
    seed=42,
    cluster_name="ResponseOutlierFilteredLLMCluster",
):
    base_labels = _compact_cluster_labels(np.asarray(llm_labels, dtype=int))
    n_zones = len(zones)
    if base_labels.size != n_zones:
        raise ValueError("LLM label count does not match zone count.")

    raw_features, feature_names = compute_training_curve_features(df_train, zones)
    data_k = resolve_cluster_count(
        raw_features, seed, f"{cluster_name}_DataOutlierProbe", auto_override=True,
    )
    data_labels = _compact_cluster_labels(
        run_kmeans_or_quantile(
            raw_features, data_k, seed,
            f"{cluster_name}_DataOutlierProbe", fallback_col=1,
        )
    )
    data_counts = np.bincount(data_labels)
    max_abs = int(getattr(config, "ROF_SEMANTIC_OUTLIER_MAX_CLUSTER_SIZE", 2))
    max_frac = float(getattr(config, "ROF_SEMANTIC_OUTLIER_MAX_CLUSTER_FRAC", 0.10))
    cutoff = max(1, min(max_abs, int(np.floor(max_frac * n_zones))))
    max_groups = int(getattr(config, "ROF_SEMANTIC_MAX_OUTLIER_GROUPS", 2))
    keep_budget = bool(getattr(config, "ROF_SEMANTIC_KEEP_LLM_K_BUDGET", True))
    min_normal = int(getattr(config, "ROF_SEMANTIC_MIN_NORMAL_ZONES", 2))

    X = (raw_features - raw_features.mean(axis=0, keepdims=True)) / (
        raw_features.std(axis=0, keepdims=True) + 1e-8
    )
    center = np.median(X, axis=0, keepdims=True)
    candidates = []
    for gid, count in enumerate(data_counts.tolist()):
        idx = np.where(data_labels == gid)[0]
        if idx.size and count <= cutoff:
            dist = float(np.mean(np.linalg.norm(X[idx] - center, axis=1)))
            candidates.append((count, -dist, gid, idx))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    selected, selected_mask = [], np.zeros(n_zones, dtype=bool)
    for _, neg_dist, gid, idx in candidates:
        if len(selected) >= max_groups:
            break
        if n_zones - int(selected_mask.sum()) - idx.size < min_normal:
            continue
        selected.append((gid, idx, -neg_dist))
        selected_mask[idx] = True

    if not selected:
        print(f"  [{cluster_name}] No small response-outlier groups; keep LLM labels.")
        return base_labels.astype(int)

    normal_idx = np.where(~selected_mask)[0]
    normal_labels = base_labels.copy()
    if keep_budget:
        target_normal_k = max(1, len(np.unique(base_labels)) - len(selected))
        normal_labels[selected_mask] = -1
        merge_features = raw_features
        if priors is not None:
            sem = np.asarray(priors.get("semantic_matrix", []), dtype=float)
            if sem.ndim == 2 and sem.shape[0] == n_zones:
                merge_features = sem
        normal_labels = _merge_labels_to_target_k(
            normal_labels, normal_idx, merge_features,
            target_normal_k, cluster_name,
        )

    labels = np.full(n_zones, -1, dtype=int)
    normal_values = sorted(np.unique(normal_labels[normal_idx]).tolist())
    normal_remap = {old: new for new, old in enumerate(normal_values)}
    for idx in normal_idx:
        labels[idx] = normal_remap[int(normal_labels[idx])]
    next_gid = len(normal_values)
    for data_gid, idx, dist in selected:
        labels[idx] = next_gid
        print(
            f"  [{cluster_name}] Split data_group={data_gid}, n={idx.size}, "
            f"distance={dist:.3f}, new_group={next_gid}, "
            f"zones={', '.join(zones[i] for i in idx)}"
        )
        next_gid += 1

    labels = _compact_cluster_labels(labels)
    raw_print = np.hstack([
        base_labels[:, None], data_labels[:, None], labels[:, None], raw_features,
    ])
    print_zone_clusters(
        labels, zones, raw_print,
        ["llm_group", "data_probe_group", "final_group"] + feature_names,
        cluster_name,
    )
    return labels.astype(int)

def infer_llm_cluster_with_data_outliers(df_train, zones, llm_labels, cluster_name="LLMDataOutlierCluster"):
    labels = np.asarray(llm_labels, dtype=int).copy()
    n_zones = len(zones)
    if labels.size != n_zones:
        return labels

    max_groups = int(getattr(config, "LLM_DATA_OUTLIER_MAX_GROUPS", getattr(config, "DATA_CLUSTER_MAX_GROUPS", 5)))
    if len(np.unique(labels)) >= max_groups:
        print(f"  [{cluster_name}] LLM K already reaches max_groups={max_groups}; no outlier split.")
        raw_features = labels.reshape(-1, 1).astype(float)
        print_zone_clusters(labels, zones, raw_features, ["llm_group"], cluster_name)
        return labels.astype(int)

    raw_features, feature_names = compute_training_curve_features(df_train, zones)
    X = (raw_features - raw_features.mean(axis=0, keepdims=True)) / (raw_features.std(axis=0, keepdims=True) + 1e-8)

    min_group_size = int(getattr(config, "LLM_DATA_OUTLIER_MIN_BASE_GROUP_SIZE", 4))
    threshold = float(getattr(config, "LLM_DATA_OUTLIER_DISTANCE_THRESHOLD", 3.0))
    max_frac = float(getattr(config, "LLM_DATA_OUTLIER_MAX_ZONE_FRAC", 0.05))
    max_abs = int(getattr(config, "LLM_DATA_OUTLIER_MAX_ZONES", 2))
    max_outliers = max(1, min(max_abs, int(np.ceil(max_frac * n_zones))))

    candidates = []
    for gid in sorted(np.unique(labels).tolist()):
        idx = np.where(labels == gid)[0]
        if idx.size < min_group_size + 1:
            continue
        group_X = X[idx]
        center = np.median(group_X, axis=0, keepdims=True)
        mad = np.median(np.abs(group_X - center), axis=0, keepdims=True)
        scale = 1.4826 * mad + 1e-6
        robust_z = (group_X - center) / scale
        distances = np.sqrt(np.mean(robust_z ** 2, axis=1))
        for local_i, dist in enumerate(distances):
            if float(dist) >= threshold:
                zone_i = int(idx[local_i])
                candidates.append((float(dist), zone_i, int(gid)))

    if not candidates:
        print(f"  [{cluster_name}] No data-stat outliers found (threshold={threshold:.2f}).")
        raw_print = np.hstack([labels.reshape(-1, 1).astype(float), raw_features])
        print_zone_clusters(labels, zones, raw_print, ["llm_group"] + feature_names, cluster_name)
        return labels.astype(int)

    candidates.sort(reverse=True)
    selected = []
    selected_by_group = {}
    for dist, zone_i, gid in candidates:
        if len(selected) >= max_outliers:
            break
        if selected_by_group.get(gid, 0) >= int(getattr(config, "LLM_DATA_OUTLIER_MAX_PER_BASE_GROUP", 1)):
            continue
        if np.sum(labels == gid) - selected_by_group.get(gid, 0) <= min_group_size:
            continue
        selected.append((dist, zone_i, gid))
        selected_by_group[gid] = selected_by_group.get(gid, 0) + 1

    if not selected:
        print(f"  [{cluster_name}] Candidate outliers rejected by group-size constraints.")
        raw_print = np.hstack([labels.reshape(-1, 1).astype(float), raw_features])
        print_zone_clusters(labels, zones, raw_print, ["llm_group"] + feature_names, cluster_name)
        return labels.astype(int)

    outlier_gid = int(labels.max()) + 1
    for dist, zone_i, gid in selected:
        labels[zone_i] = outlier_gid
        print(
            f"  [{cluster_name}] Split outlier: zone={zones[zone_i]}, "
            f"from_llm_group={gid}, distance={dist:.3f}, new_group={outlier_gid}"
        )

    raw_print = np.hstack([labels.reshape(-1, 1).astype(float), raw_features])
    print_zone_clusters(labels, zones, raw_print, ["final_group"] + feature_names, cluster_name)
    return labels.astype(int)

def infer_llm_cluster_within_data_clusters(df_train, zones, llm_labels, seed, cluster_name="LLMWithinDataCluster"):
    base_labels = np.asarray(llm_labels, dtype=int)
    n_zones = len(zones)
    if base_labels.size != n_zones:
        return base_labels.astype(int)

    raw_features, feature_names = compute_training_curve_features(df_train, zones)
    X_all = (raw_features - raw_features.mean(axis=0, keepdims=True)) / (raw_features.std(axis=0, keepdims=True) + 1e-8)

    max_groups = int(getattr(config, "LLM_WITHIN_DATA_MAX_GROUPS", getattr(config, "DATA_CLUSTER_MAX_GROUPS", 5)))
    min_group_size = int(getattr(config, "LLM_WITHIN_DATA_MIN_GROUP_SIZE", 6))
    max_subgroups = int(getattr(config, "LLM_WITHIN_DATA_MAX_SUBGROUPS", 2))
    min_silhouette = float(getattr(config, "LLM_WITHIN_DATA_MIN_SILHOUETTE", 0.45))
    min_gain = float(getattr(config, "LLM_WITHIN_DATA_MIN_SILHOUETTE_GAIN", 0.05))

    labels = np.full(n_zones, -1, dtype=int)
    next_gid = 0
    base_gids = sorted(np.unique(base_labels).tolist())

    for base_pos, base_gid in enumerate(base_gids):
        idx = np.where(base_labels == base_gid)[0]
        if idx.size == 0:
            continue

        remaining_base_groups = len(base_gids) - base_pos - 1
        remaining_budget = max_groups - next_gid
        available_for_current = remaining_budget - remaining_base_groups
        if idx.size < min_group_size or available_for_current <= 1:
            labels[idx] = next_gid
            print(
                f"  [{cluster_name}] Keep LLM group {base_gid} as one group "
                f"(n={idx.size}, available_for_current={available_for_current}, "
                f"remaining_base_groups={remaining_base_groups})."
            )
            next_gid += 1
            continue

        X = X_all[idx]
        k_max = min(max_subgroups, idx.size - 1, available_for_current)
        best_k, best_score = 1, -np.inf
        one_group_score = 0.0

        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            for k in range(2, k_max + 1):
                sub = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
                if len(np.unique(sub)) < 2:
                    continue
                score = float(silhouette_score(X, sub))
                if score > best_score:
                    best_k, best_score = k, score
                    best_sub = sub
        except Exception as err:
            print(f"  [{cluster_name} Warning] Within-group split failed for LLM group {base_gid}: {err}")
            best_k, best_score = 1, -np.inf

        should_split = (
            best_k > 1
            and best_score >= min_silhouette
            and best_score >= one_group_score + min_gain
        )
        if not should_split:
            labels[idx] = next_gid
            print(
                f"  [{cluster_name}] Keep LLM group {base_gid} as one group "
                f"(n={idx.size}, best_split_k={best_k}, silhouette={best_score:.3f}, "
                f"threshold={min_silhouette:.3f})."
            )
            next_gid += 1
            continue

        print(
            f"  [{cluster_name}] Split LLM group {base_gid}: "
            f"n={idx.size}, sub_k={best_k}, silhouette={best_score:.3f}"
        )
        for sub_gid in sorted(np.unique(best_sub).tolist()):
            sub_idx = idx[np.where(best_sub == sub_gid)[0]]
            labels[sub_idx] = next_gid
            next_gid += 1

    unique = sorted(np.unique(labels).tolist())
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.asarray([remap[int(k)] for k in labels], dtype=int)
    raw_print = np.hstack([base_labels.reshape(-1, 1).astype(float), labels.reshape(-1, 1).astype(float), raw_features])
    print_zone_clusters(labels, zones, raw_print, ["llm_group", "final_group"] + feature_names, cluster_name)
    return labels.astype(int)

def infer_llm_cluster_with_data_refinement(df_train, zones, llm_labels, seed, cluster_name="LLMDataRefinementCluster"):
    base_labels = np.asarray(llm_labels, dtype=int)
    n_zones = len(zones)
    if base_labels.size != n_zones:
        return base_labels.astype(int)

    raw_features, feature_names = compute_training_curve_features(df_train, zones)
    X_all = (raw_features - raw_features.mean(axis=0, keepdims=True)) / (raw_features.std(axis=0, keepdims=True) + 1e-8)

    max_groups = int(getattr(config, "LLM_DATA_REFINEMENT_MAX_GROUPS", 6))
    min_group_size = int(getattr(config, "LLM_DATA_REFINEMENT_MIN_GROUP_SIZE", 6))
    max_subgroups = int(getattr(config, "LLM_DATA_REFINEMENT_MAX_SUBGROUPS", 2))
    min_silhouette = float(getattr(config, "LLM_DATA_REFINEMENT_MIN_SILHOUETTE", 0.42))
    min_gain = float(getattr(config, "LLM_DATA_REFINEMENT_MIN_SILHOUETTE_GAIN", 0.05))

    labels = base_labels.copy()
    split_base_groups = set()

    for base_gid in sorted(np.unique(base_labels).tolist()):
        idx = np.where(base_labels == base_gid)[0]
        current_k = len(np.unique(labels))
        available_extra_groups = max_groups - current_k
        if idx.size < min_group_size or available_extra_groups <= 0:
            print(
                f"  [{cluster_name}] Keep LLM group {base_gid} before outlier check "
                f"(n={idx.size}, available_extra_groups={available_extra_groups})."
            )
            continue

        X = X_all[idx]
        k_max = min(max_subgroups, idx.size - 1, available_extra_groups + 1)
        best_k, best_score, best_sub = 1, -np.inf, None

        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            for k in range(2, k_max + 1):
                sub = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
                if len(np.unique(sub)) < 2:
                    continue
                score = float(silhouette_score(X, sub))
                if score > best_score:
                    best_k, best_score, best_sub = k, score, sub
        except Exception as err:
            print(f"  [{cluster_name} Warning] Data refinement split failed for LLM group {base_gid}: {err}")
            best_k, best_score, best_sub = 1, -np.inf, None

        should_split = (
            best_sub is not None
            and best_k > 1
            and best_score >= min_silhouette
            and best_score >= min_gain
        )
        if not should_split:
            print(
                f"  [{cluster_name}] No stable subgroup in LLM group {base_gid} "
                f"(n={idx.size}, best_split_k={best_k}, silhouette={best_score:.3f}, "
                f"threshold={min_silhouette:.3f})."
            )
            continue

        print(
            f"  [{cluster_name}] Split LLM group {base_gid} by data structure: "
            f"n={idx.size}, sub_k={best_k}, silhouette={best_score:.3f}"
        )
        sub_values = sorted(np.unique(best_sub).tolist())
        first_sub = sub_values[0]
        for sub_gid in sub_values:
            sub_idx = idx[np.where(best_sub == sub_gid)[0]]
            if sub_gid == first_sub:
                labels[sub_idx] = base_gid
            else:
                labels[sub_idx] = int(labels.max()) + 1
        split_base_groups.add(int(base_gid))

    current_k = len(np.unique(labels))
    if current_k < max_groups:
        threshold = float(getattr(config, "LLM_DATA_REFINEMENT_OUTLIER_DISTANCE_THRESHOLD", 3.0))
        max_frac = float(getattr(config, "LLM_DATA_REFINEMENT_OUTLIER_MAX_ZONE_FRAC", 0.05))
        max_abs = int(getattr(config, "LLM_DATA_REFINEMENT_OUTLIER_MAX_ZONES", 2))
        max_outliers = max(1, min(max_abs, int(np.ceil(max_frac * n_zones))))
        max_per_group = int(getattr(config, "LLM_DATA_REFINEMENT_OUTLIER_MAX_PER_BASE_GROUP", 1))
        min_base_group_size = int(getattr(config, "LLM_DATA_REFINEMENT_OUTLIER_MIN_BASE_GROUP_SIZE", 3))
        min_remaining = int(getattr(config, "LLM_DATA_REFINEMENT_OUTLIER_MIN_REMAINING_IN_GROUP", 2))

        candidates = []
        for base_gid in sorted(np.unique(base_labels).tolist()):
            if int(base_gid) in split_base_groups:
                continue
            idx = np.where(base_labels == base_gid)[0]
            if idx.size < min_base_group_size:
                continue
            group_X = X_all[idx]
            center = np.median(group_X, axis=0, keepdims=True)
            mad = np.median(np.abs(group_X - center), axis=0, keepdims=True)
            scale = 1.4826 * mad + 1e-6
            robust_z = (group_X - center) / scale
            distances = np.sqrt(np.mean(robust_z ** 2, axis=1))
            for local_i, dist in enumerate(distances):
                if float(dist) >= threshold:
                    zone_i = int(idx[local_i])
                    candidates.append((float(dist), zone_i, int(base_gid)))

        candidates.sort(reverse=True)
        selected = []
        selected_by_group = {}
        for dist, zone_i, base_gid in candidates:
            if len(selected) >= max_outliers:
                break
            if selected_by_group.get(base_gid, 0) >= max_per_group:
                continue
            remaining = int(np.sum(base_labels == base_gid)) - selected_by_group.get(base_gid, 0) - 1
            if remaining < min_remaining:
                continue
            selected.append((dist, zone_i, base_gid))
            selected_by_group[base_gid] = selected_by_group.get(base_gid, 0) + 1

        if selected:
            outlier_gid = int(labels.max()) + 1
            for dist, zone_i, base_gid in selected:
                labels[zone_i] = outlier_gid
                print(
                    f"  [{cluster_name}] Split sparse outlier: zone={zones[zone_i]}, "
                    f"from_llm_group={base_gid}, distance={dist:.3f}, new_group={outlier_gid}"
                )
        else:
            print(f"  [{cluster_name}] No sparse outliers after subgroup check (threshold={threshold:.2f}).")
    else:
        print(f"  [{cluster_name}] Reached max_groups={max_groups}; skip sparse outlier check.")

    unique = sorted(np.unique(labels).tolist())
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.asarray([remap[int(k)] for k in labels], dtype=int)
    raw_print = np.hstack([base_labels.reshape(-1, 1).astype(float), labels.reshape(-1, 1).astype(float), raw_features])
    print_zone_clusters(labels, zones, raw_print, ["llm_group", "final_group"] + feature_names, cluster_name)
    return labels.astype(int)

def infer_training_shape_clusters(df_train, zones, seed, auto_k=False, cluster_name="DataShapeCluster", period=None):
    n_zones = len(zones)
    period = int(period or getattr(config, "DATA_SHAPE_CLUSTER_PERIOD", 96))
    if period <= 1:
        period = 96

    def get_zone_matrix(prefix, default=0.0):
        cols = []
        for z in zones:
            col = f"{prefix}__{z}"
            if col in df_train.columns:
                cols.append(df_train[col].values.astype(float))
            else:
                cols.append(np.full(len(df_train), default, dtype=float))
        return np.column_stack(cols) if cols else np.zeros((len(df_train), 0), dtype=float)

    def robust_daily_shape(values, target_period):
        values = np.asarray(values, dtype=float)
        values = np.nan_to_num(values, nan=np.nanmedian(values) if np.isfinite(values).any() else 0.0)
        if values.size == 0:
            return np.zeros(target_period, dtype=float)
        n_days = values.size // target_period
        if n_days <= 0:
            padded = np.full(target_period, float(np.nanmedian(values)), dtype=float)
            padded[:values.size] = values
            days = padded.reshape(1, target_period)
        else:
            days = values[:n_days * target_period].reshape(n_days, target_period)
        agg = str(getattr(config, "DATA_SHAPE_CLUSTER_AGG", "mean")).lower()
        if agg == "median":
            shape = np.nanmedian(days, axis=0)
        else:
            shape = np.nanmean(days, axis=0)
        shape = np.nan_to_num(shape, nan=float(np.nanmedian(shape)) if np.isfinite(shape).any() else 0.0)
        shape = shape - np.mean(shape)
        scale = np.std(shape)
        if scale < 1e-8:
            scale = np.max(np.abs(shape))
        if scale < 1e-8:
            return np.zeros_like(shape)
        return shape / (scale + 1e-8)

    def summarize_shape(shape):
        if shape.size == 0:
            return [0.0, 0.0, 0.0]
        peak = int(np.argmax(shape)) / max(1, shape.size - 1)
        trough = int(np.argmin(shape)) / max(1, shape.size - 1)
        amp = float(np.nanpercentile(shape, 95) - np.nanpercentile(shape, 5))
        return [amp, peak, trough]

    temps = get_zone_matrix("T_in")

    feats = []
    raw_summaries = []
    for i in range(n_zones):
        temp_shape = robust_daily_shape(temps[:, i], period)
        feats.append(temp_shape)
        raw_summaries.append(summarize_shape(temp_shape))

    shape_features = np.nan_to_num(np.asarray(feats, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    raw_features = np.nan_to_num(np.asarray(raw_summaries, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    feature_names = ["TinAmp", "TinPeak", "TinTrough"]
    n_groups = resolve_cluster_count(shape_features, seed, cluster_name, auto_override=auto_k)
    if n_groups <= 1:
        return np.zeros(n_zones, dtype=int)
    labels = run_kmeans_or_quantile(shape_features, n_groups, seed, cluster_name, fallback_col=0)
    print_zone_clusters(labels, zones, raw_features, feature_names, cluster_name)
    return labels.astype(int)

def infer_semantic_auto_clusters(priors, zones, seed, auto_k=True, cluster_name="SemanticAutoCluster"):
    n_zones = len(zones)

    sem = np.asarray(priors.get("semantic_matrix", np.ones((n_zones, 5)) * 0.5), dtype=float)
    if sem.ndim != 2 or sem.shape[0] != n_zones:
        sem = np.ones((n_zones, 5), dtype=float) * 0.5
    if sem.shape[1] < 5:
        sem = np.hstack([sem, np.ones((n_zones, 5 - sem.shape[1])) * 0.5])
    sem = np.nan_to_num(sem[:, :5], nan=0.5, posinf=1.0, neginf=0.0)

    feature_names = list(getattr(config, "SEMANTIC_NAMES_DEFAULT", []))
    if len(feature_names) < sem.shape[1]:
        feature_names += [f"semantic_{i}" for i in range(len(feature_names), sem.shape[1])]

    n_groups = resolve_cluster_count(sem, seed, cluster_name, auto_override=auto_k)
    if n_groups <= 1:
        return np.zeros(n_zones, dtype=int)
    labels = run_kmeans_or_quantile(sem, n_groups, seed, cluster_name, fallback_col=0)
    print_zone_clusters(labels, zones, sem, feature_names[:sem.shape[1]], cluster_name)
    return labels.astype(int)

def infer_prior_name_clusters(priors, zones, seed, auto_k=False, cluster_name="PriorNameCluster"):
    n_zones = len(zones)

    sem = np.asarray(priors.get("semantic_matrix", np.ones((n_zones, 5)) * 0.5), dtype=float)
    if sem.ndim != 2 or sem.shape[0] != n_zones:
        sem = np.ones((n_zones, 5), dtype=float) * 0.5
    if sem.shape[1] < 5:
        sem = np.hstack([sem, np.ones((n_zones, 5 - sem.shape[1])) * 0.5])
    sem = sem[:, :5]

    keyword_groups = [
        ("class", ["class", "pod", "lecture", "school"]),
        ("office", ["office", "work", "conf", "meeting"]),
        ("high_gain", ["kitchen", "cafeteria", "dining", "gym", "server", "equip", "laundry"]),
        ("transition", ["corridor", "lobby", "hall", "stair", "atrium", "entrance"]),
        ("service", ["storage", "closet", "toilet", "restroom", "mechanical", "core", "utility"]),
        ("perimeter", ["corner", "perimeter", "exterior", "window", "roof"]),
    ]
    name_feats = []
    for zone in zones:
        z = zone.lower()
        name_feats.append([1.0 if any(token in z for token in tokens) else 0.0 for _, tokens in keyword_groups])
    name_feats = np.asarray(name_feats, dtype=float)

    raw_features = np.hstack([sem, name_feats])
    feature_names = (
        ["inertia", "occVol", "envExp", "trust", "envSens"]
        + [name for name, _ in keyword_groups]
    )
    n_groups = resolve_cluster_count(raw_features, seed, cluster_name, auto_override=auto_k)
    if n_groups <= 1:
        return np.zeros(n_zones, dtype=int)
    labels = run_kmeans_or_quantile(raw_features, n_groups, seed, cluster_name, fallback_col=2)
    print_zone_clusters(labels, zones, raw_features, feature_names, cluster_name)
    return labels.astype(int)

def _candidate_llm_cluster_paths(result_dir, base_name, run_tag):
    roots = []
    for root in [
        getattr(config, "LLM_CLUSTER_PRIOR_ROOT", None),
        result_dir,
        getattr(config, "PRIOR_STABILITY_ROOT", None),
        os.path.dirname(result_dir),
    ]:
        if root and root not in roots:
            roots.append(root)

    template = getattr(
        config,
        "LLM_CLUSTER_PRIOR_FILENAME_TEMPLATE",
        "llm_cluster_prior_{building_name}_{data_tag}.json",
    )
    filename = template.format(building_name=base_name, data_tag=run_tag)
    candidates = []
    for root in roots:
        candidates.append(os.path.join(root, filename))
        candidates.append(os.path.join(root, f"Result_LLM_Stability_{base_name}", filename))
        candidates.append(os.path.join(root, f"Result_LLM_Stability_{base_name}", "llm_cluster_prior.json"))
    if bool(getattr(config, "LLM_CLUSTER_RECURSIVE_SEARCH", False)):
        for root in roots:
            candidates.extend(glob.glob(os.path.join(root, "**", filename), recursive=True))
    return list(dict.fromkeys(candidates))

def _candidate_llm_data_arbitrated_cluster_paths(result_dir, base_name, run_tag):
    roots = []
    for root in [
        getattr(config, "LLM_DATA_ARBITRATED_CLUSTER_PRIOR_ROOT", None),
        getattr(config, "LLM_CLUSTER_PRIOR_ROOT", None),
        result_dir,
        getattr(config, "PRIOR_STABILITY_ROOT", None),
        os.path.dirname(result_dir),
    ]:
        if root and root not in roots:
            roots.append(root)

    template = getattr(
        config,
        "LLM_DATA_ARBITRATED_CLUSTER_PRIOR_FILENAME_TEMPLATE",
        "llm_data_arbitrated_cluster_prior_{building_name}_{data_tag}.json",
    )
    filename = template.format(building_name=base_name, data_tag=run_tag)
    candidates = []
    for root in roots:
        candidates.append(os.path.join(root, filename))
        candidates.append(os.path.join(root, f"Result_LLM_Stability_{base_name}", filename))
        candidates.append(os.path.join(root, f"Result_LLM_Stability_{base_name}", "llm_data_arbitrated_cluster_prior.json"))
    if bool(getattr(config, "LLM_CLUSTER_RECURSIVE_SEARCH", False)):
        for root in roots:
            candidates.extend(glob.glob(os.path.join(root, "**", filename), recursive=True))
    return list(dict.fromkeys(candidates))

def load_llm_cluster_ids(result_dir, base_name, run_tag, zones):
    cluster_path = None
    for candidate in _candidate_llm_cluster_paths(result_dir, base_name, run_tag):
        if os.path.exists(candidate):
            cluster_path = candidate
            break
    if cluster_path is None:
        print(
            f"  [LLMCluster Warning] No llm_cluster_prior JSON found for "
            f"{base_name}/{run_tag}; falling back to PriorNameAutoCluster."
        )
        return None

    with open(cluster_path, "r", encoding="utf-8") as f:
        cluster_prior = json.load(f)

    print(f"  [LLMCluster] Loaded: {cluster_path}")
    labels = np.full(len(zones), -1, dtype=int)

    raw_ids = cluster_prior.get("zone_cluster_ids")
    raw_zones = cluster_prior.get("zones")
    if isinstance(raw_ids, list) and isinstance(raw_zones, list) and len(raw_ids) == len(raw_zones):
        id_by_clean = {_clean_zone_name(z): int(k) for z, k in zip(raw_zones, raw_ids)}
        for i, zone in enumerate(zones):
            if _clean_zone_name(zone) in id_by_clean:
                labels[i] = id_by_clean[_clean_zone_name(zone)]

    zone_records = cluster_prior.get("zone_records", {})
    if isinstance(zone_records, dict):
        rec_by_clean = {_clean_zone_name(z): rec for z, rec in zone_records.items()}
        for i, zone in enumerate(zones):
            rec = rec_by_clean.get(_clean_zone_name(zone))
            if isinstance(rec, dict) and "group_id" in rec:
                labels[i] = int(rec["group_id"])

    groups = cluster_prior.get("groups", [])
    if isinstance(groups, list):
        for group in groups:
            if not isinstance(group, dict):
                continue
            gid = int(group.get("group_id", group.get("id", 0)))
            for zone in group.get("zones", []):
                for i, real_zone in enumerate(zones):
                    if _clean_zone_name(zone) == _clean_zone_name(real_zone):
                        labels[i] = gid

    if np.any(labels < 0):
        missing = [zones[i] for i in np.where(labels < 0)[0]]
        print(f"  [LLMCluster Warning] Missing zones in cluster prior: {missing[:8]}")
        fallback_gid = int(labels[labels >= 0].max() + 1) if np.any(labels >= 0) else 0
        labels[labels < 0] = fallback_gid

    unique = sorted(np.unique(labels).tolist())
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.asarray([remap[int(k)] for k in labels], dtype=int)
    raw_features = labels.reshape(-1, 1).astype(float)
    print_zone_clusters(labels, zones, raw_features, ["llm_group"], "LLMCluster")
    return labels

def load_llm_data_arbitrated_cluster_ids(result_dir, base_name, run_tag, zones):
    cluster_path = None
    for candidate in _candidate_llm_data_arbitrated_cluster_paths(result_dir, base_name, run_tag):
        if os.path.exists(candidate):
            cluster_path = candidate
            break
    if cluster_path is None:
        print(
            f"  [LLMDataArbitratedCluster Warning] No llm_data_arbitrated_cluster_prior JSON found for "
            f"{base_name}/{run_tag}; falling back to LLMCluster."
        )
        return None

    with open(cluster_path, "r", encoding="utf-8") as f:
        cluster_prior = json.load(f)

    print(f"  [LLMDataArbitratedCluster] Loaded: {cluster_path}")
    labels = np.full(len(zones), -1, dtype=int)

    raw_ids = cluster_prior.get("zone_cluster_ids")
    raw_zones = cluster_prior.get("zones")
    if isinstance(raw_ids, list) and isinstance(raw_zones, list) and len(raw_ids) == len(raw_zones):
        id_by_clean = {_clean_zone_name(z): int(k) for z, k in zip(raw_zones, raw_ids)}
        for i, zone in enumerate(zones):
            if _clean_zone_name(zone) in id_by_clean:
                labels[i] = id_by_clean[_clean_zone_name(zone)]

    zone_records = cluster_prior.get("zone_records", {})
    if isinstance(zone_records, dict):
        rec_by_clean = {_clean_zone_name(z): rec for z, rec in zone_records.items()}
        for i, zone in enumerate(zones):
            rec = rec_by_clean.get(_clean_zone_name(zone))
            if isinstance(rec, dict) and "group_id" in rec:
                labels[i] = int(rec["group_id"])

    groups = cluster_prior.get("groups", [])
    if isinstance(groups, list):
        for group in groups:
            if not isinstance(group, dict):
                continue
            gid = int(group.get("group_id", group.get("id", 0)))
            for zone in group.get("zones", []):
                for i, real_zone in enumerate(zones):
                    if _clean_zone_name(zone) == _clean_zone_name(real_zone):
                        labels[i] = gid

    if np.any(labels < 0):
        missing = [zones[i] for i in np.where(labels < 0)[0]]
        print(f"  [LLMDataArbitratedCluster Warning] Missing zones in cluster prior: {missing[:8]}")
        fallback_gid = int(labels[labels >= 0].max() + 1) if np.any(labels >= 0) else 0
        labels[labels < 0] = fallback_gid

    unique = sorted(np.unique(labels).tolist())
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.asarray([remap[int(k)] for k in labels], dtype=int)
    raw_features = labels.reshape(-1, 1).astype(float)
    print_zone_clusters(labels, zones, raw_features, ["arbitrated_group"], "LLMDataArbitratedCluster")
    return labels

def run_single_dataset(csv_p, idf_p, FOCUSED_MODEL_FLAGS, seed, result_dir, predict_period=96, few_shot_days=14, device="cuda", external_variant_ids=None):
    print(f"\n>>> 处理建筑: {os.path.basename(csv_p)}")
    df_raw = pd.read_csv(csv_p)
    df_multi, zones = build_multizone_timeseries(df_raw)
    
    dt = pd.to_datetime(df_multi["datetime"], errors="coerce")
    train_mask = (dt.dt.month == 6) | (dt.dt.month == 7)
    test_mask = dt.dt.month == 8
    train_indices = np.flatnonzero(train_mask.to_numpy())
    test_indices = np.flatnonzero(test_mask.to_numpy())
    df_train = df_multi[train_mask].reset_index(drop=True)
    df_test = df_multi[test_mask].reset_index(drop=True)

    test_start_dt, test_end_dt = dt[test_mask].min(), dt[test_mask].max()

    if few_shot_days > 0:
        rows_to_keep = few_shot_days * predict_period
        if rows_to_keep < len(df_train):
            df_train = df_train.iloc[:rows_to_keep].reset_index(drop=True)
            train_indices = train_indices[:rows_to_keep]

    train_data, train_label, SS_X, SS_Y = pack_dataset_author_style(df_train, zones, predict_period, device=device, fit_scaler=True)
    test_data, test_label, _, _ = pack_dataset_author_style(df_test, zones, predict_period, device=device, fit_scaler=False, scaler_X=SS_X, scaler_Y=SS_Y)

    if (
        getattr(config, "MLP_BOUND_RESIDUAL_OUTPUT", False)
        and config.PREDICT_RESIDUAL_DELTA
    ):
        quantile = float(
            getattr(config, "MLP_RESIDUAL_DELTA_QUANTILE", 0.999)
        )
        if not 0.0 < quantile <= 1.0:
            raise ValueError(
                "MLP_RESIDUAL_DELTA_QUANTILE must be in (0, 1]."
            )
        delta_abs = train_label.detach().abs().cpu()
        max_delta = torch.quantile(
            delta_abs.reshape(-1, delta_abs.shape[-1]),
            quantile,
            dim=0,
        )
        if not torch.all(torch.isfinite(max_delta) & (max_delta > 0.0)):
            raise RuntimeError(
                "Could not estimate positive finite per-zone MLP residual "
                "bounds from the training labels."
            )
        config.MLP_MAX_RESIDUAL_DELTA = max_delta.tolist()
        print(
            f"  [MLPResidualBound] q={quantile:.4f}, "
            f"per_zone_min={max_delta.min().item():.6f}, "
            f"per_zone_median={max_delta.median().item():.6f}, "
            f"per_zone_max={max_delta.max().item():.6f} "
            "(standardized delta units)"
        )

    history_steps = max(1, int(getattr(config, "HISTORY_MLP_WINDOW_STEPS", 4)))
    train_history_context = pack_history_context_author_style(
        df_multi, train_indices, zones, predict_period, history_steps,
        SS_X, SS_Y, device=device,
    )
    test_history_context = pack_history_context_author_style(
        df_multi, test_indices, zones, predict_period, history_steps,
        SS_X, SS_Y, device=device,
    )

    external_tests = []
    if external_variant_ids:
        ext_specs = infer_numbered_external_tests(csv_p, variant_ids=external_variant_ids)
        for ext_csv, _ in ext_specs:
            df_ext = pd.read_csv(ext_csv)
            df_ext_m, _ = build_multizone_timeseries(df_ext)
            ext_dt = pd.to_datetime(df_ext_m["datetime"], errors="coerce")
            
            ext_test_mask = (ext_dt >= test_start_dt) & (ext_dt <= test_end_dt)
            ext_test_indices = np.flatnonzero(ext_test_mask.to_numpy())
            df_ext_target = df_ext_m[ext_test_mask].reset_index(drop=True)
            if len(df_ext_target) > 0:
                ext_data, _, _, _ = pack_dataset_author_style(df_ext_target, zones, predict_period, device=device, fit_scaler=False, scaler_X=SS_X, scaler_Y=SS_Y)
                ext_history_context = pack_history_context_author_style(
                    df_ext_m, ext_test_indices, zones, predict_period,
                    history_steps, SS_X, SS_Y, device=device,
                )
                external_tests.append({
                    "name": os.path.basename(ext_csv),
                    "data": ext_data,
                    "df": df_ext_target,
                    "history_context": ext_history_context,
                })

    base_name = os.path.basename(idf_p).split('.')[0]
    run_tag = "few_shot" if few_shot_days > 0 else "fulldata"
    cache_path = os.path.join(result_dir, f"llm_priors_{base_name}_{run_tag}.json")
    try: priors = load_or_copy_existing_prior(cache_path, base_name, zones)
    except FileNotFoundError as e:
        print(f"[Warning] {e} Using default priors.")
        priors = {"semantic_matrix": np.ones((len(zones), 5))*0.5}
    enabled_names = [name for name, enabled in FOCUSED_MODEL_FLAGS.items() if enabled]
    needs_data_clusters = any("Data Cluster" in name for name in enabled_names)
    needs_data_auto_clusters = any("Data AutoCluster" in name for name in enabled_names)
    needs_data_shape_clusters = any("Data Shape Cluster" in name for name in enabled_names)
    needs_data_shape_auto_clusters = any("Data Shape AutoCluster" in name for name in enabled_names)
    needs_prior_clusters = any("Prior Cluster" in name for name in enabled_names)
    needs_prior_auto_clusters = any("Prior AutoCluster" in name for name in enabled_names)
    needs_semantic_auto_clusters = any("Semantic AutoCluster" in name for name in enabled_names)
    needs_semantic_initialized_rof_clusters = any(
        "Semantic-Initialized ROF Cluster" in name for name in enabled_names
    )
    needs_llm_clusters = (
        any("LLM Cluster" in name for name in enabled_names)
        or needs_semantic_initialized_rof_clusters
    )
    needs_llm_data_outlier_clusters = any("LLM Cluster + Data Outlier" in name for name in enabled_names)
    needs_llm_within_data_clusters = any("LLM Cluster + Within DataCluster" in name for name in enabled_names)
    needs_llm_data_refinement_clusters = any("LLM Cluster + Data Refinement" in name for name in enabled_names)
    needs_llm_data_arbitrated_clusters = any("LLM Data-Arbitrated Cluster" in name for name in enabled_names)
    needs_rof_semantic_clusters = needs_semantic_initialized_rof_clusters
    zone_cluster_cache = {}
    if needs_data_clusters:
        zone_cluster_cache["data"] = infer_training_curve_clusters(
            df_train, zones, seed, auto_k=False, cluster_name="DataCluster"
        )
    if needs_data_auto_clusters:
        zone_cluster_cache["data_auto"] = infer_training_curve_clusters(
            df_train, zones, seed, auto_k=True, cluster_name="DataAutoCluster"
        )
    if needs_data_shape_clusters:
        zone_cluster_cache["data_shape"] = infer_training_shape_clusters(
            df_train, zones, seed, auto_k=False, cluster_name="DataShapeCluster", period=predict_period
        )
    if needs_data_shape_auto_clusters:
        zone_cluster_cache["data_shape_auto"] = infer_training_shape_clusters(
            df_train, zones, seed, auto_k=True, cluster_name="DataShapeAutoCluster", period=predict_period
        )
    if needs_prior_clusters:
        zone_cluster_cache["prior_name"] = infer_prior_name_clusters(
            priors, zones, seed, auto_k=False, cluster_name="PriorNameCluster"
        )
    if needs_prior_auto_clusters:
        zone_cluster_cache["prior_name_auto"] = infer_prior_name_clusters(
            priors, zones, seed, auto_k=True, cluster_name="PriorNameAutoCluster"
        )
    if needs_semantic_auto_clusters:
        zone_cluster_cache["semantic_auto"] = infer_semantic_auto_clusters(
            priors, zones, seed, auto_k=True, cluster_name="SemanticAutoCluster"
        )
    if needs_llm_clusters:
        llm_labels = load_llm_cluster_ids(result_dir, base_name, run_tag, zones)
        if llm_labels is None:
            llm_labels = infer_prior_name_clusters(
                priors, zones, seed, auto_k=True, cluster_name="LLMClusterFallback"
            )
        zone_cluster_cache["llm"] = llm_labels
    if needs_llm_data_outlier_clusters:
        llm_labels = zone_cluster_cache.get("llm")
        if llm_labels is None:
            llm_labels = load_llm_cluster_ids(result_dir, base_name, run_tag, zones)
        if llm_labels is None:
            llm_labels = infer_prior_name_clusters(
                priors, zones, seed, auto_k=True, cluster_name="LLMDataOutlierFallback"
            )
        zone_cluster_cache["llm_data_outlier"] = infer_llm_cluster_with_data_outliers(
            df_train, zones, llm_labels, cluster_name="LLMDataOutlierCluster"
        )
    if needs_llm_within_data_clusters:
        llm_labels = zone_cluster_cache.get("llm")
        if llm_labels is None:
            llm_labels = load_llm_cluster_ids(result_dir, base_name, run_tag, zones)
        if llm_labels is None:
            llm_labels = infer_prior_name_clusters(
                priors, zones, seed, auto_k=True, cluster_name="LLMWithinDataFallback"
            )
        zone_cluster_cache["llm_within_data"] = infer_llm_cluster_within_data_clusters(
            df_train, zones, llm_labels, seed, cluster_name="LLMWithinDataCluster"
        )
    if needs_llm_data_refinement_clusters:
        llm_labels = zone_cluster_cache.get("llm")
        if llm_labels is None:
            llm_labels = load_llm_cluster_ids(result_dir, base_name, run_tag, zones)
        if llm_labels is None:
            llm_labels = infer_prior_name_clusters(
                priors, zones, seed, auto_k=True, cluster_name="LLMDataRefinementFallback"
            )
        zone_cluster_cache["llm_data_refinement"] = infer_llm_cluster_with_data_refinement(
            df_train, zones, llm_labels, seed, cluster_name="LLMDataRefinementCluster"
        )
    if needs_llm_data_arbitrated_clusters:
        arbitrated_labels = load_llm_data_arbitrated_cluster_ids(result_dir, base_name, run_tag, zones)
        if arbitrated_labels is None:
            arbitrated_labels = load_llm_cluster_ids(result_dir, base_name, run_tag, zones)
        if arbitrated_labels is None:
            arbitrated_labels = infer_prior_name_clusters(
                priors, zones, seed, auto_k=True, cluster_name="LLMDataArbitratedFallback"
            )
        zone_cluster_cache["llm_data_arbitrated"] = arbitrated_labels

    if needs_rof_semantic_clusters:
        llm_labels = zone_cluster_cache.get("llm")
        if llm_labels is None:
            llm_labels = load_llm_cluster_ids(
                result_dir, base_name, run_tag, zones,
            )
        if llm_labels is None:
            llm_labels = infer_prior_name_clusters(
                priors, zones, seed, auto_k=True,
                cluster_name="ROFSemanticFallback",
            )
        zone_cluster_cache["rof_semantic"] = (
            infer_response_outlier_filtered_llm_clusters(
                df_train,
                zones,
                llm_labels,
                priors=priors,
                seed=seed,
                cluster_name="ResponseOutlierFilteredLLMCluster",
            )
        )

    run_persistence(df_test, external_tests, zones, csv_p, few_shot_days, result_dir)

    x_env_dim = train_data.shape[-1] - 2 * len(zones)
    selected_models = [name for name in MODEL_REGISTRY if FOCUSED_MODEL_FLAGS.get(name, False)]
    requested_missing = [name for name, enabled in FOCUSED_MODEL_FLAGS.items() if enabled and name not in MODEL_REGISTRY]
    if not selected_models:
        print(f"  [Model Selection Warning] No enabled models matched MODEL_REGISTRY.")
        print(f"  [Model Selection Warning] Requested: {[name for name, enabled in FOCUSED_MODEL_FLAGS.items() if enabled]}")
        print(f"  [Model Selection Warning] Available: {list(MODEL_REGISTRY.keys())}")
    elif requested_missing:
        print(f"  [Model Selection Warning] Requested models not found: {requested_missing}")
    else:
        print(f"  [Model Selection] Enabled models: {selected_models}")

    for moniker, flags in MODEL_REGISTRY.items():
        if not FOCUSED_MODEL_FLAGS.get(moniker, False): continue
        result_moniker = target_mode_model_name(moniker)
        m_cls = flags["class"]
        set_seed(seed)
        model = _instantiate_model_class(
            m_cls, len(zones), x_env_dim, priors, device,
        )
        if hasattr(model, "set_zone_cluster_ids"):
            cluster_source = getattr(model, "cluster_source", "data")
            if cluster_source in zone_cluster_cache:
                model.set_zone_cluster_ids(zone_cluster_cache[cluster_source])
        if hasattr(model, "semantic_parent_model_name"):
            parent_model_name = getattr(model, "semantic_parent_model_name")
            ensure_semantic_parent_checkpoint(
                parent_model_name,
                zone_cluster_cache,
                zones,
                x_env_dim,
                priors,
                device,
                seed,
                result_dir,
                csv_p,
                few_shot_days,
                train_data,
                train_label,
                SS_Y,
                train_history_context=(
                    train_history_context if hasattr(model, "history_window") else None
                ),
            )
            child_source = getattr(model, "cluster_source", None)
            parent_source = getattr(
                model, "semantic_parent_cluster_source", "llm",
            )
            child_labels = zone_cluster_cache.get(child_source)
            parent_labels = zone_cluster_cache.get(parent_source)
            if child_labels is None or parent_labels is None:
                print(
                    f"  [SemanticInit Skip] Missing labels for child={child_source} "
                    f"or parent={parent_source}."
                )
            else:
                semantic_initialize_cluster_model(
                    model,
                    result_dir=result_dir,
                    dataset_file=csv_p,
                    zones_count=len(zones),
                    seed=seed,
                    day_tag=few_shot_days,
                    child_labels=child_labels,
                    parent_labels=parent_labels,
                    parent_model_name=parent_model_name,
                    device=device,
                )
                 
        trained_model = run_unified_training(
            result_moniker, model, train_data, train_label, SS_Y, seed,
            dataset_file=csv_p, day_tag=few_shot_days, result_dir=result_dir,
            train_history_context=(
                train_history_context if hasattr(model, "history_window") else None
            ),
        )
        if trained_model is None:
            continue
        model_test_history = (
            test_history_context if hasattr(trained_model, "history_window") else None
        )
        print_first_step_residual_diagnostics(
            trained_model, test_data, SS_Y, zones, result_moniker, "ID",
            history_context=model_test_history,
        )
        print_raw_rollout_zone_diagnostics(
            trained_model, test_data, SS_Y, zones, result_moniker, "ID",
            dataset_file=csv_p, history_context=model_test_history,
        )
        preds = predict_and_unscale_unified(
            trained_model, test_data, SS_Y, zones,
            history_context=(
                model_test_history
            ),
        )
        print_and_log_metrics(preds, df_test, zones, result_moniker, csv_p, few_shot_days, result_dir)
        
        for ext in external_tests:
            ext_model_name = f"{result_moniker} | Evaluated on {ext['name']}"
            model_ext_history = (
                ext["history_context"] if hasattr(trained_model, "history_window") else None
            )
            print_first_step_residual_diagnostics(
                trained_model, ext["data"], SS_Y, zones, ext_model_name, "OOD",
                history_context=model_ext_history,
            )
            print_raw_rollout_zone_diagnostics(
                trained_model, ext["data"], SS_Y, zones, ext_model_name, "OOD",
                dataset_file=csv_p, history_context=model_ext_history,
            )
            ext_preds = predict_and_unscale_unified(
                trained_model, ext["data"], SS_Y, zones,
                history_context=(
                    model_ext_history
                ),
            )
            print_and_log_metrics(ext_preds, ext["df"], zones, ext_model_name, csv_p, few_shot_days, result_dir, is_external=True)

def run_all_experiments(building_datasets, days_to_test, model_flags, result_dir, seed, external_variant_ids, device):
    os.makedirs(result_dir, exist_ok=True)
    for days in days_to_test:
        print("\n" + "="*50)
        print(f" 🚀 大循环启动：{f'{days} 天 Few-Shot' if days > 0 else 'Full Data 完整数据'} 模式")
        print("="*50 + "\n")
        config.FEW_SHOT_MODE = (days > 0)
        
        for csv_path, idf_path in building_datasets:
            run_single_dataset(csv_path, idf_path, model_flags, seed, result_dir=result_dir, few_shot_days=days, device=device, external_variant_ids=external_variant_ids)
