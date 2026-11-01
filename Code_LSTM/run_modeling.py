import argparse
from pathlib import Path

import torch

import config
from eval_utils import generate_publication_table
from model_lstm import MODEL_REGISTRY
from model_film import (
    FiLM_Physical_Informed_LSTM,
    FiLM_Data_Shape_Cluster_LSTM,
    FiLM_LLM_Cluster_LSTM,
)
from pipeline import run_all_experiments, set_seed


METHODS = {
    "pi-film": "PI + Zone FiLM LSTM",
    "data-film": "PI + Data Shape AutoCluster + Zone FiLM LSTM",
    "llm-film": "PI + LLM Cluster + Zone FiLM LSTM",
}
MODEL_REGISTRY[METHODS["pi-film"]] = {
    "class": FiLM_Physical_Informed_LSTM,
}
MODEL_REGISTRY[METHODS["data-film"]] = {
    "class": FiLM_Data_Shape_Cluster_LSTM,
}
MODEL_REGISTRY[METHODS["llm-film"]] = {
    "class": FiLM_LLM_Cluster_LSTM,
}


BUILDING_NAMES = (
    "118zone_Healthcare_1",
    "67zone_HotelSmall_1",
    "46zone_SchoolSecondary_1",
    "25zone_SchoolPrimary_1",
    "22zone_HotelLarge_B_1",
    "18zone_OfficeMedium_B_1",
    "10zone_RetailStripmall_C_1",
)
DEFAULT_DAYS_TO_TEST = (0, 3, 7, 14, 21, 28)


def main(
    seed: int,
    data_root: str,
    result_root: str,
    prior_root: str,
    cluster_prior_root: str = None,
    run_tag: str = "direct_t_film_dynamic_k",
    device_override: str = None,
    overwrite_existing: bool = False,
    building: str = "all",
    days_to_test=None,
    models_to_run=None,
) -> None:
    set_seed(seed)

    config.BUILDING_DAY_LR_OVERRIDES = {}
    config.BUILDING_LR_OVERRIDES = {}

    data_root = Path(data_root).expanduser().resolve()
    result_root = Path(result_root).expanduser().resolve()
    prior_root = Path(prior_root).expanduser().resolve()
    cluster_prior_root = (
        Path(cluster_prior_root).expanduser().resolve()
        if cluster_prior_root else prior_root
    )
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not prior_root.is_dir():
        raise FileNotFoundError(f"Modeling prior root does not exist: {prior_root}")
    if not cluster_prior_root.is_dir():
        raise FileNotFoundError(
            f"Cluster prior root does not exist: {cluster_prior_root}"
        )
    result_root.mkdir(parents=True, exist_ok=True)

    config.DATA_ROOT = str(data_root)
    config.RESULT_ROOT = str(result_root)
    config.PRIOR_STABILITY_ROOT = str(prior_root)
    config.LLM_CLUSTER_PRIOR_ROOT = str(cluster_prior_root)
    config.LLM_CLUSTER_STRICT_ROOT = cluster_prior_root != prior_root
    suffix = f"_{run_tag}" if run_tag else ""
    result_dir = str(result_root / f"Result_LSTM_FiLM_seed{seed}{suffix}")
    result_path = Path(result_dir)
    if result_path.is_dir() and any(result_path.iterdir()) and not overwrite_existing:
        raise FileExistsError(
            f"Result directory is not empty: {result_path}. "
            "Use a different --run-tag, or pass --overwrite-existing explicitly."
        )
    device = device_override or ("cuda:0" if torch.cuda.is_available() else "cpu")

    config.ZONE_ADAPTER_GAIN_RHO = 0.03
    config.ZONE_ADAPTER_BIAS_RHO = 0.00
    config.LLM_CLUSTER_MIN_GROUPS = 1
    config.ROF_SEMANTIC_KEEP_LLM_K_BUDGET = False

    # Preserve PyTorch's default LSTMCell gate-bias initialization.
    config.LSTM_ZERO_GATE_BIAS = False
    config.LSTM_ZERO_INIT_DECODER_OUTPUT = False
    config.EXPERIMENT_SEED = int(seed)

    config.ROLLOUT_EVAL_ONLY = False
    config.SAVE_MODEL_CHECKPOINTS = True
    config.FORCE_RETRAIN_EXISTING_CHECKPOINTS = overwrite_existing
    config.LOAD_MODEL_CHECKPOINTS = False
    config.RUN_PERSISTENCE_BASELINE = False

    config.DATA_CLUSTER_MAX_GROUPS = 5
    config.MULTI_OOD_LLM_CLUSTER_MAX_GROUPS = 5
    config.OOD_LLM_CLUSTER_PRIOR_ROOT = (
        f"{result_root}/OOD_LLM_Cluster_Priors"
    )

    config.RESIDUAL_ROLLOUT_STATE_MODE = "temperature"
    config.TEMP_SCALER_MODE = "per_zone"
    config.PREDICT_RESIDUAL_DELTA = False
    config.RESIDUAL_TRAIN_USE_ROLLOUT_LOSS = False
    config.RESIDUAL_TRAIN_DELTA_LAMBDA = 0.0
    config.RESIDUAL_TRAIN_ROLLOUT_LAMBDA = 0.0
    config.RESIDUAL_TRAIN_BIAS_LAMBDA = 0.0
    config.PI_FILM_RESIDUAL_RHO = 0.2

    config.BUILDING_LR_OVERRIDES = {}
    config.NN_LR = 1e-3

    config.ROLLOUT_EVAL_CLOSED_LOOP_RESIDUAL = False
    config.ROLLOUT_EVAL_ANCHOR_INTERVAL = 96
    config.ROLLOUT_EVAL_SEGMENTED_FORWARD = False
    config.ROLLOUT_EVAL_FORWARD_CONTEXT_STEPS = 96
    config.ROLLOUT_EVAL_USE_HISTORY_CONTEXT = True

    config.LSTM_HISTORY_CONTEXT_STEPS = 4
    config.ZONE_FILM_STRENGTH = 0.1
    config.ZONE_FILM_DIM = 32
    config.ZONE_FILM_SEMANTIC_HIDDEN = 16
    config.MODEL_CHECKPOINT_VERSION = (
        "v2_lstm_direct_temperature_history4_dynamic_k_zone_film"
    )

    config.CLUSTER_DECODER_SHRINK_LAMBDA = 1e-2
    config.CLUSTER_DECODER_SHRINK_START_EPOCH = 200

    config.LLM_DATA_OUTLIER_MAX_GROUPS = 6
    config.LLM_DATA_OUTLIER_MAX_ZONES = 2
    config.LLM_DATA_OUTLIER_MAX_PER_BASE_GROUP = 1
    config.LLM_DATA_OUTLIER_MIN_BASE_GROUP_SIZE = 3
    config.LLM_DATA_OUTLIER_DISTANCE_THRESHOLD = 3.0

    model_flags = {
        "Baseline LSTM": True,
        "PI": False,
        "PI + Data Shape AutoCluster LSTM": False,
        "PI + LLM Cluster LSTM": False,
        METHODS["pi-film"]: True,
        METHODS["data-film"]: True,
        METHODS["llm-film"]: True,
    }
    if models_to_run is not None:
        requested_models = set(models_to_run)
        unknown_models = requested_models.difference(model_flags)
        if unknown_models:
            raise ValueError(f"Unknown LSTM model(s): {sorted(unknown_models)}")
        model_flags = {name: name in requested_models for name in model_flags}

    selected_buildings = BUILDING_NAMES if building == "all" else (building,)
    selected_days = list(DEFAULT_DAYS_TO_TEST if days_to_test is None else days_to_test)
    building_datasets = [
        (str(data_root / f"{name}.csv"), str(data_root / f"{name}.idf"))
        for name in selected_buildings
    ]
    missing_inputs = [
        path
        for dataset_pair in building_datasets
        for path in dataset_pair
        if not Path(path).is_file()
    ]
    if missing_inputs:
        raise FileNotFoundError(
            "Missing building input file(s): " + ", ".join(missing_inputs)
        )

    print(
        f"[Run] seed={seed}, device={device}, "
        f"target=direct_temperature, raw_input_modulation=False, "
        f"history_context={config.LSTM_HISTORY_CONTEXT_STEPS}, "
        f"dynamic_cluster_experts=True, zone_film=True, "
        f"cluster_prior_root={cluster_prior_root}, "
        f"LSTM_ZERO_GATE_BIAS={config.LSTM_ZERO_GATE_BIAS}, "
        f"buildings={list(selected_buildings)}, days={selected_days}, "
        f"result_dir={result_dir}"
    )

    run_all_experiments(
        building_datasets=building_datasets,
        days_to_test=selected_days,
        model_flags=model_flags,
        result_dir=result_dir,
        seed=seed,
        external_variant_ids=[1, 2, 3],
        device=device,
    )

    print("\n[Analysis] Generating Publication Table Excel...")
    generate_publication_table(result_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        choices=range(38, 43),
        required=True,
    )
    parser.add_argument("--data-root", default=config.DATA_ROOT)
    parser.add_argument("--result-root", default=config.RESULT_ROOT)
    parser.add_argument(
        "--prior-root",
        default=config.PRIOR_STABILITY_ROOT,
    )
    parser.add_argument(
        "--cluster-prior-root",
        default=config.LLM_CLUSTER_PRIOR_ROOT,
        help=(
            "LLM cluster-prior directory. Semantic priors continue to come "
            "from --prior-root."
        ),
    )
    parser.add_argument("--run-tag", default="direct_t_film_dynamic_k")
    parser.add_argument("--device", default=None, help="For example: cpu, cuda:0, or cuda:1")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Allow writing into a non-empty result directory and retrain checkpoints.",
    )
    parser.add_argument(
        "--building",
        choices=("all",) + BUILDING_NAMES,
        default="all",
    )
    parser.add_argument(
        "--days",
        type=int,
        nargs="+",
        default=None,
        help="Training-day settings; use 0 for the full June-July training set.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional exact model names from the LSTM model list.",
    )
    args = parser.parse_args()
    main(
        args.seed,
        args.data_root,
        args.result_root,
        args.prior_root,
        args.cluster_prior_root,
        args.run_tag,
        args.device,
        args.overwrite_existing,
        args.building,
        args.days,
        args.models,
    )
