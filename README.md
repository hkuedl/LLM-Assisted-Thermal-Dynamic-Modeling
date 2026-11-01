# LLM-Assisted Thermal Dynamic Modeling

Official reproduction code for LLM-assisted multizone building thermal-dynamics
modeling. The repository contains four forecasting backbones (MLP, CNN, GRU,
and LSTM), semantic and topology prior generation, and the OOD event-and-phase
post-processing workflow.

> **Data and results are distributed separately.** Download the complete
> [`Data` and `Result` folders from Google Drive](https://drive.google.com/drive/folders/1Ar-OhwFWcuEEXwX_Amf26GFJR2aphw3-?usp=drive_link)
> before running the experiments. Large datasets, trained checkpoints, and
> complete prediction outputs are not stored in this Git repository.

## Contents

- [1. Repository structure](#1-repository-structure)
- [2. Installation](#2-installation)
- [3. Download the data and results](#3-download-the-data-and-results)
- [4. Run the forecasting experiments](#4-run-the-forecasting-experiments)
- [5. Regenerate the LLM priors](#5-regenerate-the-llm-priors)
- [6. Run OOD post-processing](#6-run-ood-post-processing)
- [7. Paths and environment variables](#7-paths-and-environment-variables)
- [8. Reproducibility notes](#8-reproducibility-notes)

## 1. Repository structure

```text
LLM-Assisted-Thermal-Dynamic-Modeling/
├── Code CNN/             # CNN forecasting experiments
├── Code MLP/             # MLP forecasting experiments
├── Code_GRU/             # GRU forecasting experiments
├── Code_LSTM/            # LSTM forecasting experiments
├── prior_generation/     # semantic and zone-cluster prior generation
├── post_processing/      # OOD event-and-phase correction and evaluation
└── README.md
```

The code expects the cloned repository, `Data`, and `Result` to be sibling
directories:

```text
release_root/
├── LLM-Assisted-Thermal-Dynamic-Modeling/
├── Data/
└── Result/
```

Default paths are resolved from the source-file locations, so no
machine-specific absolute path is required.

## 2. Installation

Python 3.10 or later is recommended. Create an isolated environment from the
repository root:

```bash
git clone https://github.com/hkuedl/LLM-Assisted-Thermal-Dynamic-Modeling.git
cd LLM-Assisted-Thermal-Dynamic-Modeling
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

Install a PyTorch build appropriate for your CPU or CUDA platform by following
the [official PyTorch installation guide](https://pytorch.org/get-started/locally/),
then install the shared experiment dependencies:

```bash
python -m pip install numpy pandas scikit-learn openpyxl
python -m pip install -r prior_generation/requirements.txt
python -m pip install -r post_processing/requirements.txt
```

The LLM client dependency is needed only when regenerating priors. Existing
priors and all offline post-processing steps can be used without an API key.

## 3. Download the data and results

1. Open the [Google Drive release folder](https://drive.google.com/drive/folders/1Ar-OhwFWcuEEXwX_Amf26GFJR2aphw3-?usp=drive_link).
2. Download both `Data` and `Result`.
3. Place them beside the cloned repository, using the exact folder names shown
   below.

```text
release_root/
├── LLM-Assisted-Thermal-Dynamic-Modeling/
├── Data/
│   ├── 118zone_Healthcare_1.csv
│   ├── 118zone_Healthcare_1.idf
│   └── ...
└── Result/
    ├── priors/
    ├── OOD/
    └── ...
```

The downloaded `Result` directory supplies the frozen priors, checkpoints,
manifests, and prediction outputs used by the reproduction workflows. New runs
also write their outputs under this directory.

## 4. Run the forecasting experiments

All forecasting entry points require `--seed`. The runners accept seeds 38-42,
but the released paper-result matrix uses seeds 38, 39, 41, and 42. Use
`--device cpu`, `--device cuda:0`, or another valid PyTorch device when you need
an explicit device selection.

### 4.1 CNN

Run one CNN experiment for one building and one training-data setting. This
7-day command is a quick functional example, not a paper-matrix setting:

```bash
cd "Code CNN"
python run_modeling.py \
  --seed 38 \
  --buildings 10zone_RetailStripmall_C_1 \
  --days 7 \
  --models pi-film \
  --device cuda:0
```

Run the five variants exposed by the current public CNN runner for all seven
buildings and the four paper training-data settings:

```bash
cd "Code CNN"
python run_modeling.py \
  --seed 38 \
  --buildings 118zone_Healthcare_1 67zone_HotelSmall_1 \
              46zone_SchoolSecondary_1 25zone_SchoolPrimary_1 \
              22zone_HotelLarge_B_1 18zone_OfficeMedium_B_1 \
              10zone_RetailStripmall_C_1 \
  --days 14 21 28 0 \
  --models data llm pi-film data-film llm-film \
  --device cuda:0
```

CNN model keys exposed by this runner:

| Key | Model |
| --- | --- |
| `data` | PI + Data Shape AutoCluster CNN |
| `llm` | PI + LLM Cluster CNN |
| `pi-film` | PI + Zone FiLM CNN |
| `data-film` | PI + Data Shape AutoCluster + Zone FiLM CNN |
| `llm-film` | PI + LLM Cluster + Zone FiLM CNN |

### 4.2 MLP

Run one MLP experiment. This 7-day command is a quick functional example, not
a paper-matrix setting:

```bash
cd "Code MLP"
python run_modeling.py \
  --seed 38 \
  --buildings 10zone_RetailStripmall_C_1 \
  --days 7 \
  --models pi-film \
  --device cuda:0
```

Run the five variants exposed by the current public MLP runner for all seven
buildings and the four paper training-data settings:

```bash
cd "Code MLP"
python run_modeling.py \
  --seed 38 \
  --buildings 118zone_Healthcare_1 67zone_HotelSmall_1 \
              46zone_SchoolSecondary_1 25zone_SchoolPrimary_1 \
              22zone_HotelLarge_B_1 18zone_OfficeMedium_B_1 \
              10zone_RetailStripmall_C_1 \
  --days 14 21 28 0 \
  --models data llm pi-film data-film llm-film \
  --device cuda:0
```

MLP uses the same five model keys as CNN, mapped to the corresponding MLP
backbones. The downloaded `Result` folder contains the complete frozen
seven-method paper results; the current public CNN/MLP runners do not expose
standalone Baseline and PI as `--models` choices.

### 4.3 GRU

Run one GRU experiment:

```bash
cd Code_GRU
python run_modeling.py \
  --seed 38 \
  --building 10zone_RetailStripmall_C_1 \
  --days 7 \
  --models "PI + Zone FiLM GRU" \
  --device cuda:0
```

Run the default public GRU sweep (all buildings; all six code-supported day
settings; Baseline plus the three FiLM variants):

```bash
cd Code_GRU
python run_modeling.py --seed 38 --building all --device cuda:0
```

This default is not the complete paper matrix. To reproduce one paper seed
with all seven methods and the four paper training-data settings, run:

```bash
python run_modeling.py \
  --seed 38 \
  --building all \
  --days 14 21 28 0 \
  --models "Baseline GRU" "PI" \
           "PI + Data Shape AutoCluster GRU" \
           "PI + LLM Cluster GRU" \
           "PI + Zone FiLM GRU" \
           "PI + Data Shape AutoCluster + Zone FiLM GRU" \
           "PI + LLM Cluster + Zone FiLM GRU" \
  --device cuda:0
```

Repeat with seeds 39, 41, and 42. When `--models` is supplied, GRU accepts
these exact names (including spaces):

```text
Baseline GRU
PI
PI + Data Shape AutoCluster GRU
PI + LLM Cluster GRU
PI + Zone FiLM GRU
PI + Data Shape AutoCluster + Zone FiLM GRU
PI + LLM Cluster + Zone FiLM GRU
```

### 4.4 LSTM

Run one LSTM experiment:

```bash
cd Code_LSTM
python run_modeling.py \
  --seed 38 \
  --building 10zone_RetailStripmall_C_1 \
  --days 7 \
  --models "PI + Zone FiLM LSTM" \
  --device cuda:0
```

Run the default public LSTM sweep (all buildings; all six code-supported day
settings; Baseline plus the three FiLM variants):

```bash
cd Code_LSTM
python run_modeling.py --seed 38 --building all --device cuda:0
```

This default is not the complete paper matrix. To reproduce one paper seed
with all seven methods and the four paper training-data settings, run:

```bash
python run_modeling.py \
  --seed 38 \
  --building all \
  --days 14 21 28 0 \
  --models "Baseline LSTM" "PI" \
           "PI + Data Shape AutoCluster LSTM" \
           "PI + LLM Cluster LSTM" \
           "PI + Zone FiLM LSTM" \
           "PI + Data Shape AutoCluster + Zone FiLM LSTM" \
           "PI + LLM Cluster + Zone FiLM LSTM" \
  --device cuda:0
```

Repeat with seeds 39, 41, and 42. When `--models` is supplied, LSTM accepts
these exact names (including spaces):

```text
Baseline LSTM
PI
PI + Data Shape AutoCluster LSTM
PI + LLM Cluster LSTM
PI + Zone FiLM LSTM
PI + Data Shape AutoCluster + Zone FiLM LSTM
PI + LLM Cluster + Zone FiLM LSTM
```

### 4.5 Paper-result matrix versus code-supported settings

The released paper-result matrix contains 3,136 combinations:

| Dimension | Paper-result setting |
| --- | --- |
| Backbones | MLP, CNN, GRU, LSTM |
| Methods | Baseline, PI, Data Shape AutoCluster, LLM Cluster, and the three corresponding FiLM variants |
| Seeds | 38, 39, 41, 42 |
| Buildings | all seven listed below |
| Training quantities | 14 days, 21 days, 28 days, and Full (`--days 0`) |

The count is 4 backbones x 7 methods x 4 seeds x 7 buildings x 4 training
quantities = **3,136**. The code additionally accepts seed 40 and 3-day/7-day
training settings, but these are not part of the released paper-result matrix.

There is an important CLI difference: GRU and LSTM can select all seven methods
directly, whereas the current public CNN and MLP runners expose five method
keys. Use the complete frozen results in the downloaded `Result` folder when
examining the full seven-method CNN/MLP paper matrix.

### 4.6 Buildings and training-data settings

The released forecasting code supports seven buildings:

```text
118zone_Healthcare_1
67zone_HotelSmall_1
46zone_SchoolSecondary_1
25zone_SchoolPrimary_1
22zone_HotelLarge_B_1
18zone_OfficeMedium_B_1
10zone_RetailStripmall_C_1
```

`--days 0` selects the full June-July training set. The paper-result settings
are `14`, `21`, `28`, and `0`; the additional code-supported settings are `3`
and `7` days. CNN and MLP use `--buildings` (one or more names); GRU and LSTM
use `--building` (one name or `all`).

To inspect every option without launching training:

```bash
python run_modeling.py --help
```

For interrupted CNN/MLP sweeps, `--skip-completed` skips completed
building/day/method combinations. GRU/LSTM refuse to write into a non-empty run
directory. Prefer a new `--run-tag` or a separate `--result-root` rather than
`--overwrite-existing`; use the overwrite option only when replacing existing
outputs is intentional.

To keep reproduced outputs separate from the downloaded release results,
create a sibling output directory and pass it to any forecasting runner:

```bash
mkdir -p ../../Reproduced_Results
python run_modeling.py --seed 38 \
  --result-root ../../Reproduced_Results \
  --prior-root ../../Result/priors/semantic_generation/selected \
  --cluster-prior-root ../../Result/priors/llm_cluster/runtime
```

This example is run from one of the four model directories. Add the desired
building, day, model, and device options described above.

## 5. Regenerate the LLM priors

The downloaded `Result` folder already contains the frozen priors used by the
forecasting experiments. This section is needed only to reproduce semantic and
zone-cluster prior generation from the source data.

Prior generation uses the Together API. Set the key in the shell so it is not
written to the repository:

```bash
export TOGETHER_API_KEY="your_api_key_here"
```

Run the complete three-stage reproduction pipeline:

```bash
cd prior_generation
python run_all.py \
  --model meta-models/Muse-Glimmer-30B \
  --runs 10
```

The command reads the sibling `Data/` directory and writes a new run to:

```text
Result/prior_reproduction/
├── semantic_generation/
├── cluster_generation/
└── cluster_generation_config.json
```

It does not overwrite the frozen experiment priors under `Result/priors/`.
Custom locations can be supplied explicitly:

```bash
python run_all.py \
  --data-root /path/to/Data \
  --output-root /path/to/new_prior_run \
  --model meta-models/Muse-Glimmer-30B \
  --runs 10
```

Newly generated priors are not selected automatically by the forecasting
runners. To test them, explicitly point a runner to the reproduction outputs
(the following paths are relative to a model directory):

```bash
python run_modeling.py --seed 38 \
  --prior-root ../../Result/prior_reproduction/semantic_generation \
  --cluster-prior-root ../../Result/prior_reproduction/cluster_generation/priors
```

See [`prior_generation/README.md`](prior_generation/README.md) for the prior
selection and cluster-repair stages.

## 6. Run OOD post-processing

The post-processing workflow keeps forecasting models frozen. It reads the
complete downloaded `Result` directory, including:

```text
Result/OOD/postprocessing/case_manifest.csv
Result/OOD/postprocessing/run_manifest.csv
```

The public configuration evaluates OOD-2 through OOD-5 at 14-day, 21-day,
28-day, and Full training quantities, using 10 LLM samples per prior.

Run from `post_processing/`:

```bash
cd post_processing

# 1. Preview the LLM requests without making API calls.
python generate_priors.py

# 2. Generate or resume the public mean-10 event-and-phase priors.
#    Requires TOGETHER_API_KEY (or OPENAI_API_KEY).
python generate_priors.py --run

# 3. Apply the raw, fixed-gate, and LLM event-and-phase corrections.
python apply_priors.py

# 4. Produce paired summaries and figures.
python visualize_results.py
```

If the downloaded `Result` folder already includes the public mean-10 priors,
skip step 2 and run only the offline application and visualization commands.
New public outputs are written by default to:

```text
Result/OOD/postprocessing_public/public_prompt/
```

The same workflow is available interactively in
`post_processing/Run_Postprocessing.ipynb`; real LLM generation is disabled by
default. See [`post_processing/README.md`](post_processing/README.md) for the
configuration and evaluation conventions.

Run the offline regression tests with:

```bash
cd post_processing
python -m unittest discover -s tests -v
```

## 7. Paths and environment variables

The forecasting scripts accept the following path overrides:

```text
--data-root
--result-root
--prior-root
--cluster-prior-root
```

Equivalent forecasting environment variables are:

```text
BUILDING_LLM_DATA_ROOT
BUILDING_LLM_RESULT_ROOT
BUILDING_LLM_PRIOR_STABILITY_ROOT
BUILDING_LLM_CLUSTER_PRIOR_ROOT
```

Post-processing uses:

```text
BUILDING_DATA_ROOT
BUILDING_RESULT_ROOT
```

Example for a non-default layout:

```bash
export BUILDING_LLM_DATA_ROOT=/path/to/Data
export BUILDING_LLM_RESULT_ROOT=/path/to/Result
export BUILDING_LLM_PRIOR_STABILITY_ROOT=/path/to/Result/priors/semantic_generation/selected
export BUILDING_LLM_CLUSTER_PRIOR_ROOT=/path/to/Result/priors/llm_cluster/runtime
```

## 8. Reproducibility notes

- The runners accept seeds 38-42; the released paper-result matrix uses seeds
  38, 39, 41, and 42.
- `Data` contains the building CSV and IDF inputs; `Result` contains the frozen
  priors, checkpoints, manifests, predictions, and released evaluation outputs.
- LLM-based forecasting variants require the semantic and cluster priors from
  `Result/priors/`, but do not make API calls during model training.
- Prior regeneration always writes to a new reproduction location and does not
  silently replace the priors used in the released experiments.
- Do not commit API keys. Local key files and generated data/result folders are
  excluded from version control.
