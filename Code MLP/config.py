import os

# =========================================================================
# 1. 基础路径配置
# =========================================================================
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_CODE_DIR)
_RELEASE_ROOT = os.path.dirname(_CODE_ROOT)

# Portable public-release layout:
#   release_root/public code/
#   release_root/Data/
#   release_root/Result/
DATA_ROOT = os.environ.get(
    "BUILDING_LLM_DATA_ROOT", os.path.join(_RELEASE_ROOT, "Data")
)
RESULT_ROOT = os.environ.get(
    "BUILDING_LLM_RESULT_ROOT", os.path.join(_RELEASE_ROOT, "Result")
)
_DEFAULT_PRIOR_ROOT = os.path.join(
    RESULT_ROOT, "priors", "semantic_generation", "selected"
)
_DEFAULT_CLUSTER_PRIOR_ROOT = os.path.join(
    RESULT_ROOT, "priors", "llm_cluster", "runtime"
)
PRIOR_STABILITY_ROOT = os.environ.get(
    "BUILDING_LLM_PRIOR_STABILITY_ROOT", _DEFAULT_PRIOR_ROOT
)
MODEL_CHECKPOINT_DIR_NAME = "Model_Checkpoints"
MODEL_CHECKPOINT_VERSION = "v6_mlp_residual_rolloutloss_prior_variants"

# =========================================================================
# 2. 仿真设置 & 先验
# =========================================================================
DT_HOURS = 0.25
HEAT_SCALE = 1000.0
READ_EXISTING_PRIORS_ONLY = True
PRIOR_COPY_TO_RESULT_DIR = True
SEMANTIC_NAMES_DEFAULT = ["Thermal_Inertia", "Occupancy_Volatility", "Env_Exposure", "Base_Physics_Trust", "Env_Sensitivity"]

# =========================================================================
# 3. 神经网路训练及残差设定
# =========================================================================
NN_EPOCHS = 15000
NN_LR = 5e-3
NN_VERBOSE_EVERY = 50
NN_PATIENCE = 150
NN_MIN_DELTA = 1e-5

BUILDING_LR_OVERRIDES = {
    "118zone_Healthcare_1": 1e-3,
}
BUILDING_DAY_LR_OVERRIDES = {}

USE_DECODER_DROPOUT = True
DECODER_DROPOUT_P = 0.1

# Optional History MLP stability controls. Defaults preserve prior experiments.
MLP_ZERO_INIT_DECODER_OUTPUT = False
MLP_BOUND_ENCODER_LATENT = False
MLP_BOUND_RESIDUAL_OUTPUT = False
MLP_RESIDUAL_DELTA_QUANTILE = 1.0
MLP_MAX_RESIDUAL_DELTA = None

GAUSSIAN_INPUT_NOISE_STD = 0.01
GAUSSIAN_INPUT_NOISE_TEMP_ONLY = True

PREDICT_RESIDUAL_DELTA = True
RESIDUAL_ROLLOUT_STATE_MODE = "temperature"  # 或 "delta"
RESIDUAL_TRAIN_USE_ROLLOUT_LOSS = True
RESIDUAL_TRAIN_DELTA_LAMBDA = 0.0
RESIDUAL_TRAIN_ROLLOUT_LAMBDA = 1.0
RESIDUAL_TRAIN_BIAS_LAMBDA = 0.0

USE_PADDED_TEMP_SCALER = False
TEMP_SCALER_PAD_C = 0
TEMP_SCALER_MODE = "per_zone"

ROLLOUT_EVAL_CLOSED_LOOP_RESIDUAL = True
ROLLOUT_EVAL_ANCHOR_INTERVAL = 96
ROLLOUT_EVAL_SEGMENTED_FORWARD = False
ROLLOUT_EVAL_FORWARD_CONTEXT_STEPS = 96
ROLLOUT_EVAL_USE_HISTORY_CONTEXT = True
EVAL_FIRST_STEP_DIAGNOSTICS = False
EVAL_EXCLUDE_FIRST_STEPS = 1
EVAL_DIAGNOSTIC_TOP_ZONES = 3
RAW_ROLLOUT_DEBUG = False
RAW_ROLLOUT_DEBUG_BUILDING = ""
RAW_ROLLOUT_DEBUG_ZONE = ""
RAW_ROLLOUT_DEBUG_MODEL_KEYWORD = ""
RAW_ROLLOUT_DEBUG_SPLITS = ("ID", "OOD")
RAW_ROLLOUT_DEBUG_BATCH_INDEX = 0
RAW_ROLLOUT_DEBUG_STEPS = 24
RAW_ROLLOUT_DEBUG_BATCH_SUMMARY = 8

# =========================================================================
# 4. 各种正则化与辅助 Loss 阈值 (无损保留)
# =========================================================================
FILM_STRENGTH = 0.03
PI_FILM_RESIDUAL_RHO = 0.2

CLUSTER_DECODER_SHRINK_LAMBDA = 1e-4
CLUSTER_DECODER_SHRINK_START_EPOCH = 200

TOPOLOGY_DELTA_REG_START_EPOCH = 200
TOPOLOGY_DELTA_REG_ALPHA = 1e-5
PRIOR_SENS_ALIGN_LAMBDA = 1e-3
PRIOR_SENS_ALIGN_START_EPOCH = 100
PRIOR_SENS_ALIGN_EPS = 0.02
ZONE_ADAPTER_GAIN_RHO = 0.20
ZONE_ADAPTER_BIAS_RHO = 0.00
PI_MLP_RESIDUAL_INERTIA_RHO = 0.10
PI_MLP_RESIDUAL_OCC_RHO = 0.05
PI_MLP_RESIDUAL_ENV_RHO = 0.05
HISTORY_MLP_WINDOW_STEPS = 4
DATA_CLUSTER_NUM_GROUPS = 3
DATA_CLUSTER_AUTO_K = False
DATA_CLUSTER_MIN_GROUPS = 2
DATA_CLUSTER_MAX_GROUPS = 4
CLUSTER_RANDOM_SEED = 42
DATA_CLUSTER_AUTO_K_TOL = 0.0
DATA_CLUSTER_HEAD_RHO = 0.03
DATA_AUTOCLUSTER_SHARED_MIX_RHO = 0.03
DATA_SHAPE_CLUSTER_PERIOD = 96
DATA_SHAPE_CLUSTER_AGG = "mean"  # "mean" or "median"

# Separate JSON prior for LLM-defined zone clusters. This is intentionally kept
# outside the original llm_priors_*.json files so existing PI priors are unchanged.
LLM_CLUSTER_PRIOR_ROOT = os.environ.get(
    "BUILDING_LLM_CLUSTER_PRIOR_ROOT", _DEFAULT_CLUSTER_PRIOR_ROOT
)
LLM_CLUSTER_PRIOR_FILENAME_TEMPLATE = "llm_cluster_prior_{building_name}_{data_tag}.json"
LLM_DATA_ARBITRATED_CLUSTER_PRIOR_ROOT = None
LLM_DATA_ARBITRATED_CLUSTER_PRIOR_FILENAME_TEMPLATE = "llm_data_arbitrated_cluster_prior_{building_name}_{data_tag}.json"
LLM_CLUSTER_RECURSIVE_SEARCH = False
LLM_DATA_OUTLIER_MAX_GROUPS = 5
LLM_DATA_OUTLIER_DISTANCE_THRESHOLD = 3.0
LLM_DATA_OUTLIER_MAX_ZONES = 2
LLM_DATA_OUTLIER_MAX_ZONE_FRAC = 0.05
LLM_DATA_OUTLIER_MAX_PER_BASE_GROUP = 1
LLM_DATA_OUTLIER_MIN_BASE_GROUP_SIZE = 4
LLM_WITHIN_DATA_MAX_GROUPS = 5
LLM_WITHIN_DATA_MIN_GROUP_SIZE = 6
LLM_WITHIN_DATA_MAX_SUBGROUPS = 2
LLM_WITHIN_DATA_MIN_SILHOUETTE = 0.45
LLM_WITHIN_DATA_MIN_SILHOUETTE_GAIN = 0.05
LLM_DATA_REFINEMENT_MAX_GROUPS = 6
LLM_DATA_REFINEMENT_MIN_GROUP_SIZE = 6
LLM_DATA_REFINEMENT_MAX_SUBGROUPS = 2
LLM_DATA_REFINEMENT_MIN_SILHOUETTE = 0.42
LLM_DATA_REFINEMENT_MIN_SILHOUETTE_GAIN = 0.05
LLM_DATA_REFINEMENT_OUTLIER_DISTANCE_THRESHOLD = 3.0
LLM_DATA_REFINEMENT_OUTLIER_MAX_ZONES = 2
LLM_DATA_REFINEMENT_OUTLIER_MAX_ZONE_FRAC = 0.05
LLM_DATA_REFINEMENT_OUTLIER_MAX_PER_BASE_GROUP = 1
LLM_DATA_REFINEMENT_OUTLIER_MIN_BASE_GROUP_SIZE = 3
LLM_DATA_REFINEMENT_OUTLIER_MIN_REMAINING_IN_GROUP = 2

# =========================================================================
# 5. Runtime 开关 (通常在 main.py 参数中改写)
# =========================================================================
SAVE_RESULTS = True
SAVE_MODEL_CHECKPOINTS = True
LOAD_MODEL_CHECKPOINTS = False
FORCE_RETRAIN_EXISTING_CHECKPOINTS = True
ROLLOUT_EVAL_ONLY = False
RUN_PERSISTENCE_BASELINE = False
