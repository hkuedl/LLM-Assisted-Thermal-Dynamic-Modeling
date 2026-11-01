"""Run matched final no-FiLM MLP cluster ablations without touching 0927 results."""

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

# Preserve the original 0927 entrypoint's import order. On Linux Conda,
# importing torch first can affect which C++ runtime pandas subsequently uses.
import torch
import numpy as np
import pandas as pd

import config
import pipeline as modeling_pipeline
from data_utils import build_multizone_timeseries
from eval_utils import generate_publication_table
from model_mlp import (
    MODEL_REGISTRY,
    Data_Shape_AutoCluster_History_MLP_Physical_Informed_MLP,
    LLM_Cluster_History_MLP_Physical_Informed_MLP,
)
from model_film import FiLM_History_Physical_Informed_MLP, FiLM_Data_Shape_Cluster_MLP, FiLM_LLM_Cluster_MLP
from experiment_spec import BUILDINGS, DEFAULT_BUILDINGS, METHODS
from pipeline import run_all_experiments, set_seed, _candidate_llm_cluster_paths, infer_prior_name_clusters
from prior_utils import find_stability_prior_path


# The historical experiment_spec contains the FiLM variants.  Extend a local
# copy so this runner remains a drop-in file in the existing Code MLP_film
# directory and does not modify any shared source file.
METHODS = dict(METHODS)
METHODS.update({
    "data": "PI + Data Shape AutoCluster History MLP",
    "llm": "PI + LLM Cluster History MLP",
})

MODEL_REGISTRY[METHODS["pi-film"]] = {"class": FiLM_History_Physical_Informed_MLP}
MODEL_REGISTRY[METHODS["data-film"]] = {"class": FiLM_Data_Shape_Cluster_MLP}
MODEL_REGISTRY[METHODS["llm-film"]] = {"class": FiLM_LLM_Cluster_MLP}
MODEL_REGISTRY[METHODS["data"]] = {
    "class": Data_Shape_AutoCluster_History_MLP_Physical_Informed_MLP,
}
MODEL_REGISTRY[METHODS["llm"]] = {
    "class": LLM_Cluster_History_MLP_Physical_Informed_MLP,
}

NO_FILM_METHODS = {"data", "llm"}


def configure_0927(strength=0.1, epochs=15000, patience=150):
    """Preserve effective settings from Code MLP_clean/run_modeling.py."""
    config.BUILDING_DAY_LR_OVERRIDES = {}
    config.BUILDING_LR_OVERRIDES = {"118zone_Healthcare_1": 1e-3}
    config.NN_LR = 1e-3
    config.NN_EPOCHS, config.NN_PATIENCE = epochs, patience
    config.ROLLOUT_EVAL_ONLY = False
    config.SAVE_MODEL_CHECKPOINTS = True
    config.FORCE_RETRAIN_EXISTING_CHECKPOINTS = False
    config.LOAD_MODEL_CHECKPOINTS = False
    config.RUN_PERSISTENCE_BASELINE = False
    config.HISTORY_MLP_WINDOW_STEPS = 4
    config.PREDICT_RESIDUAL_DELTA = False
    config.MLP_ZERO_INIT_DECODER_OUTPUT = True
    config.MLP_BOUND_ENCODER_LATENT = True
    config.MLP_BOUND_RESIDUAL_OUTPUT = True
    config.MLP_RESIDUAL_DELTA_QUANTILE = 1.0
    config.RESIDUAL_ROLLOUT_STATE_MODE = "temperature"
    config.TEMP_SCALER_MODE = "per_zone"
    config.RESIDUAL_TRAIN_USE_ROLLOUT_LOSS = True
    config.RESIDUAL_TRAIN_DELTA_LAMBDA = 0
    config.RESIDUAL_TRAIN_ROLLOUT_LAMBDA = 1.0
    config.RESIDUAL_TRAIN_BIAS_LAMBDA = 0.0
    config.ROLLOUT_EVAL_CLOSED_LOOP_RESIDUAL = True
    config.ROLLOUT_EVAL_ANCHOR_INTERVAL = 96
    config.ROLLOUT_EVAL_SEGMENTED_FORWARD = False
    config.ROLLOUT_EVAL_FORWARD_CONTEXT_STEPS = 96
    config.ROLLOUT_EVAL_USE_HISTORY_CONTEXT = True
    config.DATA_CLUSTER_MIN_GROUPS = 2
    config.DATA_CLUSTER_MAX_GROUPS = 5
    config.LLM_CLUSTER_MIN_GROUPS = 1
    config.LLM_DATA_OUTLIER_MAX_GROUPS = 6
    config.ROF_SEMANTIC_KEEP_LLM_K_BUDGET = False
    config.ZONE_FILM_DIM = 32
    config.ZONE_FILM_SEMANTIC_HIDDEN = 16
    config.ZONE_FILM_STRENGTH = strength
    config.MODEL_CHECKPOINT_VERSION = "v1_0927_history_mlp_zone_film_direct_temperature"


def validate_prior(prior, zones, source):
    # Original loader checks only length. Reject a mismatch here instead of
    # silently reordering and changing the 0927 experiment's interpretation.
    names = prior.get("zones")
    clean = lambda name: re.sub(r"[^a-z0-9]", "", str(name).lower())
    if names is None or list(map(clean, names)) != list(map(clean, zones)):
        raise ValueError(f"Prior zone order differs from CSV: {source}")
    if len(set(map(clean, zones))) != len(zones):
        raise ValueError("CSV contains ambiguous duplicate zone names.")
    semantic = np.asarray(prior.get("semantic_matrix"), dtype=float)
    graph = np.asarray(prior.get("adjacency_matrix"), dtype=float)
    if semantic.shape != (len(zones), 5) or not np.isfinite(semantic).all():
        raise ValueError(f"Missing/invalid semantic_matrix: {source}")
    if np.any((semantic < 0) | (semantic > 1)):
        raise ValueError(f"Semantic values outside [0,1]: {source}")
    if np.unique(semantic).size <= 1:
        raise ValueError(f"Degenerate semantic prior would be rejected by the original loader: {source}")
    if list(prior.get("semantic_names", config.SEMANTIC_NAMES_DEFAULT)) != config.SEMANTIC_NAMES_DEFAULT:
        raise ValueError(f"Unexpected semantic column order: {source}")
    if graph.shape != (len(zones), len(zones)) or not np.isfinite(graph).all():
        raise ValueError(f"Missing/invalid adjacency_matrix: {source}")


def prepare_priors(data_root, archive_root, seed, buildings, days, require_archive=False):
    prepared = []
    for building in buildings:
        _, zones = build_multizone_timeseries(pd.read_csv(data_root / f"{building}.csv", nrows=1))
        if not zones:
            raise ValueError(f"No temperature zones found in {building} CSV.")
        tags = list(dict.fromkeys("few_shot" if day > 0 else "fulldata" for day in days))
        for tag in tags:
            filename = f"llm_priors_{building}_{tag}.json"
            archived = archive_root / f"Result_MLP_seed{seed}_0927" / filename
            if archived.is_file():
                source, origin = archived, "0927-archive"
            else:
                if require_archive:
                    raise FileNotFoundError(f"0927 semantic cache missing: {archived}")
                current = find_stability_prior_path(building)
                if current is None:
                    raise FileNotFoundError(f"No modeling prior for {building}.")
                source, origin = Path(current), "current-prior-fallback"
                print(f"[Prior] Archived cache missing: {archived}; using {source}.")
            raw = source.read_bytes()
            payload = json.loads(raw)
            validate_prior(payload, zones, source)
            prepared.append((filename, raw, {
                "building": building, "tag": tag, "source": str(source), "origin": origin,
                "zones": len(zones),
            }))
    return prepared


def resolve_0927_cluster_labels(prior, zones):
    """Match the label handling in the original 0927 load_llm_cluster_ids."""
    clean = lambda name: re.sub(r"[^a-z0-9]", "", str(name).lower())
    labels = np.full(len(zones), -1, dtype=int)
    raw_ids, raw_zones = prior.get("zone_cluster_ids"), prior.get("zones")
    if isinstance(raw_ids, list) and isinstance(raw_zones, list) and len(raw_ids) == len(raw_zones):
        by_name = {clean(zone): int(gid) for zone, gid in zip(raw_zones, raw_ids)}
        for i, zone in enumerate(zones):
            if clean(zone) in by_name:
                labels[i] = by_name[clean(zone)]
    records = prior.get("zone_records", {})
    if isinstance(records, dict):
        by_name = {clean(zone): record for zone, record in records.items()}
        for i, zone in enumerate(zones):
            record = by_name.get(clean(zone))
            if isinstance(record, dict) and "group_id" in record:
                labels[i] = int(record["group_id"])
    groups = prior.get("groups", [])
    if isinstance(groups, list):
        for group in groups:
            if not isinstance(group, dict):
                continue
            gid = int(group.get("group_id", group.get("id", 0)))
            for zone in group.get("zones", []):
                for i, real_zone in enumerate(zones):
                    if clean(zone) == clean(real_zone):
                        labels[i] = gid
    missing = [zones[i] for i in np.where(labels < 0)[0]]
    if missing:
        fallback_gid = int(labels[labels >= 0].max() + 1) if np.any(labels >= 0) else 0
        labels[labels < 0] = fallback_gid
    unique = sorted(np.unique(labels).tolist())
    return [unique.index(int(gid)) for gid in labels], missing


def prepare_cluster_priors(data_root, archive_root, result_dir, seed, buildings, days, cluster_root=None,
                           require_archive=False):
    """Use 0927 fallback rules only in old-prior mode; other modes stay strict."""
    prepared = []
    clean = lambda name: re.sub(r"[^a-z0-9]", "", str(name).lower())
    for building in buildings:
        _, zones = build_multizone_timeseries(pd.read_csv(data_root / f"{building}.csv", nrows=1))
        for tag in dict.fromkeys("few_shot" if day > 0 else "fulldata" for day in days):
            filename = f"llm_cluster_prior_{building}_{tag}.json"
            archive = archive_root / f"Result_MLP_seed{seed}_0927"
            if require_archive:
                # Historical runs can read shared cluster JSONs without caching
                # them in the seed folder. Only consult an explicitly selected
                # old cluster directory when the archived cache is absent.
                candidates = [archive / filename,
                              archive / f"Result_LLM_Stability_{building}" / filename,
                              archive / f"Result_LLM_Stability_{building}" / "llm_cluster_prior.json"]
                if cluster_root is not None:
                    root = Path(cluster_root)
                    candidates += [root / filename,
                                   root / f"Result_LLM_Stability_{building}" / filename,
                                   root / f"Result_LLM_Stability_{building}" / "llm_cluster_prior.json"]
            elif cluster_root is not None:
                # Explicit experimental source takes precedence over 0927 archives.
                # Missing files here are an error; do not fall back to old groups.
                root = Path(cluster_root)
                candidates = [root / filename,
                              root / f"Result_LLM_Stability_{building}" / filename,
                              root / f"Result_LLM_Stability_{building}" / "llm_cluster_prior.json"]
            else:
                candidates = [archive / filename,
                              archive / f"Result_LLM_Stability_{building}" / "llm_cluster_prior.json"]
                candidates += [Path(p) for p in _candidate_llm_cluster_paths(str(result_dir), building, tag)]
            source = next((p for p in candidates if p.is_file()), None)
            if require_archive:
                # Preserve the old experiment's failure handling, not the later
                # strict validation used by repaired-prior experiments.
                if source is None:
                    semantic_path = archive / f"llm_priors_{building}_{tag}.json"
                    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
                    ids = np.asarray(infer_prior_name_clusters(
                        semantic, zones, seed, auto_k=True, cluster_name="LLMClusterFallback"
                    ), dtype=int).tolist()
                    missing = list(zones)
                    handling = "missing_file_prior_name_fallback"
                    origin = "0927-prior-name-fallback"
                    source_name = str(semantic_path)
                    # Save the fallback assignment for inspection. During training
                    # the original pipeline still takes its no-file fallback path.
                    fallback = {"source": origin, "zones": list(zones), "zone_cluster_ids": ids}
                    raw = (json.dumps(fallback, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
                else:
                    raw = source.read_bytes()
                    ids, missing = resolve_0927_cluster_labels(json.loads(raw), zones)
                    handling = ("empty_to_single_group" if len(missing) == len(zones)
                                else "missing_zones_to_extra_group" if missing else "saved_groups")
                    origin = "0927-archive" if archive in source.parents else "explicit-old-cluster-root"
                    source_name = str(source)
                # The original model clamps labels to its allocated branch count.
                effective_ids = np.clip(ids, 0, config.DATA_CLUSTER_MAX_GROUPS - 1).tolist()
                actual_k = len(set(effective_ids))
                print(f"[0927 cluster] {building}/{tag} handling={handling} "
                      f"K={actual_k} missing={len(missing)}/{len(zones)} source={source_name}", flush=True)
                prepared.append((filename, raw, {
                    "building": building, "tag": tag, "source": source_name, "origin": origin,
                    "handling": handling,
                    "missing_zones": missing, "raw_k": len(set(ids)), "actual_k": actual_k,
                    "zone_cluster_ids": effective_ids,
                }))
                continue
            if source is None:
                raise FileNotFoundError(f"LLM grouping missing for {building}; no fallback allowed.")
            raw = source.read_bytes(); prior = json.loads(raw)
            mapping = {}
            if isinstance(prior.get('zones'), list) and isinstance(prior.get('zone_cluster_ids'), list):
                if len(prior['zones']) == len(prior['zone_cluster_ids']):
                    mapping.update({clean(z): int(k) for z, k in zip(prior['zones'], prior['zone_cluster_ids'])})
            for zone, record in prior.get('zone_records', {}).items():
                if isinstance(record, dict) and 'group_id' in record:
                    mapping[clean(zone)] = int(record['group_id'])
            for group in prior.get('groups', []):
                for zone in group.get('zones', []):
                    mapping[clean(zone)] = int(group.get('group_id', group.get('id', 0)))
            missing = [zone for zone in zones if clean(zone) not in mapping or mapping[clean(zone)] < 0]
            if missing:
                raise ValueError(f"LLM prior missing zones in {source}: {missing[:8]}")
            ids = [mapping[clean(zone)] for zone in zones]
            unique = sorted(set(ids))
            if len(unique) > config.DATA_CLUSTER_MAX_GROUPS:
                raise ValueError(f"LLM K={len(unique)} exceeds the unchanged 0927 capacity 5: {source}")
            normalized = [unique.index(k) for k in ids]
            prepared.append((filename, raw, {
                "building": building, "tag": tag, "source": str(source),
                "actual_k": len(unique),
                "zone_cluster_ids": normalized,
                "origin": (("0927-archive" if archive in source.parents else "explicit-old-cluster-root")
                           if require_archive else
                           ("explicit-cluster-root" if cluster_root is not None else
                            ("0927-archive" if archive in source.parents else "current-prior"))),
            }))
    return prepared


def write_json_atomic(path, payload):
    """Replace a metadata JSON without leaving a partial file."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def merge_manifest(result_dir, run_record):
    """Keep staged methods and training-day settings in one seed directory."""
    manifest_path = result_dir / "run_manifest.json"
    if manifest_path.is_file():
        aggregate = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(aggregate.get("seed", run_record["seed"])) != run_record["seed"]:
            raise ValueError(f"Seed mismatch in existing manifest: {manifest_path}")
    else:
        aggregate = {"seed": run_record["seed"], "runs": [], "methods": [], "completed_methods": []}

    aggregate.setdefault("runs", []).append(run_record)
    aggregate["methods"] = list(dict.fromkeys(aggregate.get("methods", []) + run_record["methods"]))
    for key in (
        "buildings", "days", "device", "data_root", "archive_root",
        "prior_sources", "cluster_prior_sources", "torch_version", "target",
        "history_frames", "early_stopping", "film", "config",
    ):
        aggregate[key] = run_record[key]
    aggregate["last_updated"] = run_record["started_at"]
    write_json_atomic(manifest_path, aggregate)
    return aggregate


def completed_id_methods(result_dir, csv_path, day, methods):
    """Find saved ID metrics for this exact building/day/method, not OOD rows."""
    building = Path(csv_path).stem
    suffix = f"{day}d" if day > 0 else "full"
    path = Path(result_dir) / f"{building}_results_{suffix}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    if not isinstance(payload, dict):
        return set()
    marker = f"[FewShot_{day}d]" if day > 0 else "[FullData]"
    completed = set()
    for dataset_key, records in payload.items():
        if (not dataset_key.startswith(marker + " ")
                or not dataset_key.endswith(" " + Path(csv_path).name)
                or not isinstance(records, dict)):
            continue
        for method in methods:
            item = records.get(method)
            if not isinstance(item, dict) or not isinstance(item.get("Global_Metrics"), dict):
                continue
            metrics = item["Global_Metrics"]
            try:
                values = [float(metrics[key]) for key in ("RMSE", "MAE")]
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(value) and value >= 0 for value in values):
                completed.add(method)
    return completed


def run_remaining_experiments(datasets, days, model_flags, result_dir, seed, device):
    """Skip saved ID results; incomplete combinations start ordinary training."""
    requested = [name for name, enabled in model_flags.items() if enabled]
    skipped = []
    for day in days:
        for dataset in datasets:
            completed = completed_id_methods(result_dir, dataset[0], day, requested)
            for name in requested:
                if name in completed:
                    record = {"building": Path(dataset[0]).stem, "day": day, "method": name}
                    skipped.append(record)
                    print(f"[Skip completed ID] seed={seed} building={record['building']} "
                          f"days={day if day > 0 else 'Full'} method={name}", flush=True)
            remaining = {name: enabled and name not in completed for name, enabled in model_flags.items()}
            if any(remaining.values()):
                # The pipeline reseeds each model and retains the normal day/building order.
                run_all_experiments([dataset], [day], remaining, str(result_dir), seed, [1, 2, 3], device)
    return skipped


def select_result_directory(root, seed, use_0927_priors=False, repaired_priors=False,
                            repaired_result_subdir="repaired_prior", methods=(),
                            run_tag="film_v1"):
    """Keep historical, FiLM, and final no-FiLM result series isolated."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", repaired_result_subdir):
        raise ValueError("--repaired-result-subdir must be a single directory name.")
    if repaired_result_subdir == "old_prior":
        raise ValueError("old_prior is reserved for the historical comparison.")
    if repaired_result_subdir != "repaired_prior" and not repaired_priors:
        raise ValueError("--repaired-result-subdir requires --repaired-priors.")
    if any(method in NO_FILM_METHODS for method in methods):
        return (
            Path(root)
            / "MLP_FiLM_pilot"
            / f"Result_MLP_FiLM_seed{seed}_{run_tag}"
        )
    folder = Path(root) / f"Result_MLP_FiLM_seed{seed}"
    if use_0927_priors:
        return folder / "old_prior"
    return folder / repaired_result_subdir if repaired_priors else folder


def main(args):
    use_0927_priors = getattr(args, "use_0927_priors", False)
    repaired_priors = getattr(args, "repaired_priors", False)
    skip_completed = getattr(args, "skip_completed", False)
    if repaired_priors and use_0927_priors:
        raise ValueError("Choose --repaired-priors or --use-0927-priors, not both.")
    if repaired_priors and not getattr(args, "cluster_prior_root", None):
        raise ValueError("--repaired-priors requires an explicit --cluster-prior-root.")
    if any(method in NO_FILM_METHODS for method in args.models):
        if any(method not in NO_FILM_METHODS for method in args.models):
            raise ValueError(
                "Run no-FiLM and FiLM methods separately so their result series stay isolated."
            )
        if args.run_tag == "film_v1":
            raise ValueError(
                "No-FiLM runs require an explicit --run-tag, e.g. "
                "mlp_nofilm_final."
            )
    if not math.isfinite(args.film_strength) or not 0 <= args.film_strength < 1:
        raise ValueError("--film-strength must be finite and in [0,1).")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("--epochs and --patience must be positive.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_tag):
        raise ValueError("--run-tag must be a simple directory suffix.")
    buildings = list(dict.fromkeys(args.buildings))
    days = list(dict.fromkeys(args.days))
    data_root = Path(args.data_root).expanduser().resolve()
    result_root = Path(args.result_root).expanduser().resolve()
    prior_root = Path(args.prior_root).expanduser().resolve()
    archive_root = Path(args.archive_root or args.result_root).expanduser().resolve()
    for path in ((data_root,) if use_0927_priors or repaired_priors else (data_root, prior_root)):
        if not path.is_dir():
            raise FileNotFoundError(path)
    # Match the 0927 layout: one stable top-level directory per seed. Different
    # models and training-day settings use distinct JSON keys/files inside it.
    result_dir = select_result_directory(
        result_root, args.seed, use_0927_priors, repaired_priors,
        getattr(args, "repaired_result_subdir", "repaired_prior"),
        args.models, args.run_tag)
    if result_dir.exists() and not result_dir.is_dir():
        raise FileExistsError(f"Result path is not a directory: {result_dir}")
    datasets = [(str(data_root / f"{name}.csv"), str(data_root / f"{name}.idf")) for name in buildings]
    missing = [p for pair in datasets for p in pair if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError("Missing building inputs: " + ", ".join(missing))
    configure_0927(args.film_strength, args.epochs, args.patience)
    no_film_run = any(method in NO_FILM_METHODS for method in args.models)
    if no_film_run:
        config.MODEL_CHECKPOINT_VERSION = (
            "final_mlp_nofilm_direct_temperature"
        )
    config.DATA_ROOT, config.RESULT_ROOT = str(data_root), str(result_root)
    config.PRIOR_STABILITY_ROOT = str(prior_root)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    # Fail before creating a result directory if the requested device is unusable.
    torch.empty(0, device=device)
    priors = prepare_priors(data_root, archive_root, args.seed, buildings, days,
                            require_archive=use_0927_priors or repaired_priors)
    explicit_cluster_root = getattr(args, "cluster_prior_root", None)
    if explicit_cluster_root is not None:
        explicit_cluster_root = Path(explicit_cluster_root).expanduser().resolve()
        if not explicit_cluster_root.is_dir():
            raise FileNotFoundError(explicit_cluster_root)
    cluster_priors = (prepare_cluster_priors(data_root, archive_root, result_dir, args.seed, buildings, days,
                                           cluster_root=explicit_cluster_root,
                                           require_archive=use_0927_priors)
                      if any(method in args.models for method in ('llm', 'llm-film')) else [])
    if use_0927_priors:
        print(f"[Prior mode] 0927 semantic cache and legacy cluster handling; "
              f"empty/partial JSON uses the old extra-group rule; missing JSON uses PriorNameAutoCluster; "
              f"result={result_dir}", flush=True)
    elif repaired_priors:
        if any(record["actual_k"] < 2 for _, _, record in cluster_priors):
            raise ValueError("Repaired cluster prior still has only one group; check --cluster-prior-root.")
        print(f"[Prior mode] 0927 semantic cache; explicitly selected repaired clusters; result={result_dir}", flush=True)
    for _, _, record in cluster_priors:
        print(f"[Cluster source] {record['building']} K={record['actual_k']} source={record['source']}", flush=True)
    if cluster_priors:
        config.LLM_CLUSTER_PRIOR_ROOT = str(result_dir)
    model_flags = {name: key in args.models for key, name in METHODS.items()}
    manifest = {
        "seed": args.seed, "buildings": buildings, "days": days, "device": device,
        "methods": [name for name, enabled in model_flags.items() if enabled],
        "run_tag": args.run_tag, "status": "running",
        "use_0927_priors": use_0927_priors,
        "cluster_loading_policy": "0927-legacy" if use_0927_priors else "strict",
        "repaired_priors": repaired_priors,
        "repaired_result_subdir": result_dir.name if repaired_priors else None,
        "skip_completed": skip_completed,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(data_root), "archive_root": str(archive_root),
        "prior_sources": [record for _, _, record in priors],
        "cluster_prior_sources": [record for _, _, record in cluster_priors],
        "torch_version": torch.__version__, "target": "direct_temperature",
        "history_frames": 5, "early_stopping": "training_mse_as_in_0927",
        "film": (
            {"enabled": False, "extra_parameters": 0}
            if no_film_run
            else {
                "enabled": True,
                "dimension": 32,
                "semantic_hidden": 16,
                "strength": args.film_strength,
                "semantic_columns": [0, 1, 2, 4],
                "extra_parameters": 6033,
            }
        ),
        "cluster_experts": {
            "assignment": "final_repaired_grouping",
            "allocated_capacity": 5,
            "active_groups": "derived from the saved assignment",
        },
        "film_enabled_by_method": {
            name: key not in NO_FILM_METHODS
            for key, name in METHODS.items()
            if key in args.models
        },
        "config": {key: value for key, value in vars(config).items()
                   if key.isupper() and isinstance(value, (str, int, float, bool, list, type(None)))},
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    explicit_cluster_files = {filename for filename, _, _ in cluster_priors} if explicit_cluster_root else set()
    # Resume assumes the same priors. Reuse the existing cache comparison so
    # saved metrics cannot be kept while silently replacing their input prior.
    for filename, raw, _ in priors + cluster_priors:
        destination = result_dir / filename
        if destination.is_file() and destination.read_bytes() != raw:
            if skip_completed:
                raise ValueError(f"Cannot resume with a changed prior: {destination}. "
                                 "Use the prior from the interrupted run.")
            if filename not in explicit_cluster_files:
                raise ValueError(f"Prior content conflict in shared seed directory: {destination}")
            print(f"[Cluster cache] Updating {destination} from the explicitly selected prior.", flush=True)
    aggregate_manifest = merge_manifest(result_dir, manifest)
    for filename, raw, _ in priors + cluster_priors:
        destination = result_dir / filename
        destination.write_bytes(raw)
    print(f"[Run] seed={args.seed} device={device} buildings={buildings} days={days}")
    film_label = "disabled" if no_film_run else str(args.film_strength)
    print(f"[Run] methods={manifest['methods']} lr=1e-3 FiLM={film_label} result={result_dir}")
    set_seed(args.seed)
    original_candidates = modeling_pipeline._candidate_llm_cluster_paths
    if use_0927_priors and cluster_priors:
        old_paths = {
            (record["building"], record["tag"]): (
                [] if record["handling"] == "missing_file_prior_name_fallback"
                else [str(result_dir / filename)]
            ) for filename, _, record in cluster_priors
        }
        # Reuse the original loader and fallback, but restrict file selection to
        # this experiment's OLD inputs. A missing old file must remain missing,
        # even if a current/repaired cache exists elsewhere in the search roots.
        def old_candidates(directory, building, tag):
            key = (building, tag)
            return old_paths[key] if key in old_paths else original_candidates(directory, building, tag)
        modeling_pipeline._candidate_llm_cluster_paths = old_candidates
    try:
        if skip_completed:
            manifest["skipped_completed_id"] = run_remaining_experiments(
                datasets, days, model_flags, result_dir, args.seed, device)
            print(f"[Resume] Skipped {len(manifest['skipped_completed_id'])} completed ID combinations.", flush=True)
        else:
            run_all_experiments(datasets, days, model_flags, str(result_dir), args.seed, [1, 2, 3], device)
    finally:
        modeling_pipeline._candidate_llm_cluster_paths = original_candidates
    generate_publication_table(str(result_dir))
    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
    aggregate_manifest["runs"][-1] = manifest
    aggregate_manifest["completed_methods"] = list(dict.fromkeys(
        aggregate_manifest.get("completed_methods", []) + manifest["methods"]
    ))
    aggregate_manifest["last_updated"] = manifest["completed_at"]
    write_json_atomic(result_dir / "run_manifest.json", aggregate_manifest)
    write_json_atomic(result_dir / "RUN_COMPLETED.json", {
        "completed": True,
        "completed_methods": aggregate_manifest["completed_methods"],
        "last_updated": manifest["completed_at"],
    })


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=range(38, 43), required=True)
    parser.add_argument("--buildings", nargs="+", choices=BUILDINGS, default=list(DEFAULT_BUILDINGS))
    parser.add_argument("--days", nargs="+", type=int, choices=[0, 3, 7, 14, 21, 28], default=[7])
    parser.add_argument("--models", nargs="+", choices=METHODS, default=["pi-film"])
    parser.add_argument("--data-root", default=config.DATA_ROOT)
    parser.add_argument("--result-root", default=config.RESULT_ROOT)
    parser.add_argument("--prior-root", default=config.PRIOR_STABILITY_ROOT)
    parser.add_argument("--cluster-prior-root", default=config.LLM_CLUSTER_PRIOR_ROOT,
                        help="Explicit LLM grouping directory. With --use-0927-priors, this must be the unchanged OLD shared directory and is used only if an archived cluster is absent. Otherwise it overrides archived clusters.")
    parser.add_argument("--use-0927-priors", action="store_true",
                        help="Use per-seed 0927 semantics and legacy cluster rules: missing zones go to one extra group (all missing -> K=1); a missing JSON uses PriorNameAutoCluster. Archived clusters precede an explicitly selected old shared root. Save under Result_MLP_FiLM_seed{seed}/old_prior/.")
    parser.add_argument("--repaired-priors", action="store_true",
                        help="Keep per-seed 0927 semantic caches but use the explicitly selected repaired cluster root. Save under Result_MLP_FiLM_seed{seed}/repaired_prior/. Do not combine with --use-0927-priors.")
    parser.add_argument("--repaired-result-subdir", default="repaired_prior",
                        help="Subdirectory inside each seed folder for --repaired-priors. Default: repaired_prior. Use repaired_prior_v2 to isolate the new repair experiment.")
    parser.add_argument("--skip-completed", action="store_true",
                        help="Resume unchanged experiments: skip building/day/method combinations with finite saved ID RMSE and MAE in this seed result directory. Does not resume epochs or fill missing OOD evaluations for skipped ID results.")
    parser.add_argument("--archive-root", default=None, help="Parent of Result_MLP_seed*_0927; defaults to result-root.")
    parser.add_argument(
        "--run-tag",
        default="film_v1",
        help=(
            "FiLM: metadata label. No-FiLM: required result-series suffix under "
            "MLP_FiLM_pilot/."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--film-strength", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=15000)
    parser.add_argument("--patience", type=int, default=150)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
