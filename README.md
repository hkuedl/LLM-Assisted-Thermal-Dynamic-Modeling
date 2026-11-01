# LLM-Assisted Thermal Dynamic Modeling

This repository contains the official code for the work **“Interactive
Building Thermal Dynamics Modeling with Large Language Models”** by Yangze
Zhou, Yu Zuo, Xueyuan Cui, Claudio Canizares, and Yi Wang.

The repository includes MLP, CNN, GRU, and LSTM backbones, LLM prior generation,
and OOD event-and-phase post-processing.

> **Data and results are distributed separately.** Download the complete
> [`Data` and `Result` folders from Google Drive](https://drive.google.com/drive/folders/1Ar-OhwFWcuEEXwX_Amf26GFJR2aphw3-?usp=drive_link)
> before running the experiments. Datasets, checkpoints, and full prediction
> outputs are not stored in this repository.

## Repository layout

```text
release_root/
├── LLM-Assisted-Thermal-Dynamic-Modeling/
│   ├── Code CNN/
│   ├── Code MLP/
│   ├── Code_GRU/
│   ├── Code_LSTM/
│   ├── prior_generation/
│   └── post_processing/
├── Data/
└── Result/
```

Download `Data` and `Result` from Google Drive and place them beside the cloned
repository using these exact folder names. Default paths are resolved from this
layout; machine-specific absolute paths are unnecessary.

## Installation

Create the provided Conda environment under a dedicated name:

```bash
git clone https://github.com/hkuedl/LLM-Assisted-Thermal-Dynamic-Modeling.git
cd LLM-Assisted-Thermal-Dynamic-Modeling

conda env create --name thermal-llm --file environment.yml
conda activate thermal-llm
```

Install the small workflow-specific dependency sets after activation:

```bash
python -m pip install -r prior_generation/requirements.txt
python -m pip install -r post_processing/requirements.txt
```

An API key is needed only to regenerate LLM priors. Model training and offline
post-processing can use the priors already included in the downloaded
`Result` folder.

## Experiment settings

The released paper-result matrix contains **3,136 combinations**:

| Dimension | Paper-result setting |
| --- | --- |
| Backbones | MLP, CNN, GRU, LSTM |
| Methods | Baseline, PI, Data Shape AutoCluster, LLM Cluster, and their three FiLM variants |
| Seeds | 38, 39, 41, 42 |
| Buildings | 7 buildings |
| Training quantities | 14, 21, 28 days, and Full (`--days 0`) |

The runners also accept seed 40 and 3-day/7-day settings, but these are not in
the released paper-result matrix.

Supported buildings:

```text
118zone_Healthcare_1
67zone_HotelSmall_1
46zone_SchoolSecondary_1
25zone_SchoolPrimary_1
22zone_HotelLarge_B_1
18zone_OfficeMedium_B_1
10zone_RetailStripmall_C_1
```

## Run the forecasting models

Use `--device cpu`, `--device cuda:0`, or another valid PyTorch device. Run
`python run_modeling.py --help` inside a model directory for all options.

### CNN and MLP

Quick functional example (7 days is not a paper-matrix setting):

```bash
cd "Code CNN"                         # use "Code MLP" for MLP
python run_modeling.py \
  --seed 38 \
  --buildings 10zone_RetailStripmall_C_1 \
  --days 7 \
  --models pi-film \
  --device cuda:0
```

Paper-setting sweep for one seed (run once in `Code CNN` and once in
`Code MLP`):

```bash
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

Repeat for seeds 39, 41, and 42. The public CNN/MLP runners expose these five
keys:

| Key | Variant |
| --- | --- |
| `data` | PI + Data Shape AutoCluster |
| `llm` | PI + LLM Cluster |
| `pi-film` | PI + Zone FiLM |
| `data-film` | PI + Data Shape AutoCluster + Zone FiLM |
| `llm-film` | PI + LLM Cluster + Zone FiLM |

Standalone Baseline and PI are not exposed by the current CNN/MLP
`--models` interface. Their complete frozen paper results are included in the
downloaded `Result` folder.

### GRU

Complete paper-setting sweep for one seed:

```bash
cd Code_GRU
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

### LSTM

Complete paper-setting sweep for one seed:

```bash
cd Code_LSTM
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

Repeat the GRU and LSTM commands for seeds 39, 41, and 42.

### Output paths and resuming

To avoid mixing new runs with downloaded results, use a separate sibling
output directory. From any model directory:

```bash
mkdir -p ../../Reproduced_Results
python run_modeling.py --seed 38 \
  --result-root ../../Reproduced_Results \
  --prior-root ../../Result/priors/semantic_generation/selected \
  --cluster-prior-root ../../Result/priors/llm_cluster/runtime

# Add the desired building, day, model, and device options shown above.
```

CNN/MLP support `--skip-completed`. For GRU/LSTM, use a new `--run-tag` or
`--result-root` if a run directory already contains files; use
`--overwrite-existing` only when replacement is intentional.

## Regenerate LLM priors

The downloaded `Result` folder already contains the frozen priors used by the
released experiments. To regenerate them with the Together API:

```bash
cd prior_generation
export TOGETHER_API_KEY="your_api_key_here"
python run_all.py --model meta-models/Muse-Glimmer-30B --runs 10
```

This reads the sibling `Data` directory and writes a new run under
`Result/prior_reproduction/`; it does not overwrite frozen priors. To use the
new priors, pass these paths from a model directory:

```bash
python run_modeling.py --seed 38 \
  --prior-root ../../Result/prior_reproduction/semantic_generation \
  --cluster-prior-root ../../Result/prior_reproduction/cluster_generation/priors

# Add the desired building, day, model, and device options shown above.
```

See [`prior_generation/README.md`](prior_generation/README.md) for details.

## OOD post-processing

The workflow reads the complete downloaded `Result` directory, including:

```text
Result/OOD/postprocessing/case_manifest.csv
Result/OOD/postprocessing/run_manifest.csv
```

The public configuration evaluates OOD-2 through OOD-5 at 14-day, 21-day,
28-day, and Full training quantities with 10 LLM samples per prior.

```bash
cd post_processing

python generate_priors.py          # preview only; no API call
python generate_priors.py --run    # generate/resume mean-10 priors; API key required
python apply_priors.py             # apply corrections and compute metrics
python visualize_results.py        # generate summaries and figures
```

If the downloaded `Result` folder already contains the public mean-10 priors,
skip `generate_priors.py --run`. Outputs are written to
`Result/OOD/postprocessing_public/public_prompt/` by default. The same workflow
is available in `post_processing/Run_Postprocessing.ipynb`.

Run the offline tests with:

```bash
cd post_processing
python -m unittest discover -s tests -v
```

## Path overrides

Forecasting scripts support `--data-root`, `--result-root`, `--prior-root`, and
`--cluster-prior-root`. Equivalent environment variables are:

```text
BUILDING_LLM_DATA_ROOT
BUILDING_LLM_RESULT_ROOT
BUILDING_LLM_PRIOR_STABILITY_ROOT
BUILDING_LLM_CLUSTER_PRIOR_ROOT
```

Post-processing uses `BUILDING_DATA_ROOT` and `BUILDING_RESULT_ROOT`.

Do not commit API keys, datasets, checkpoints, or generated results.
