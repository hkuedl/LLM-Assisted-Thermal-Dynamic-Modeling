# LLM-Assisted Thermal Dynamic Modeling

Official code for **“Interactive Building Thermal Dynamics Modeling with Large
Language Models”** by Yangze Zhou, Yu Zuo, Xueyuan Cui, Claudio Canizares, and
Yi Wang.

This repository provides the MLP, CNN, GRU, and LSTM implementations used in
the study, together with LLM prior generation and OOD post-processing code.

The released materials support three main tasks:

- training and evaluating the four thermal-dynamics backbones;
- generating semantic and topology priors with an LLM;
- reproducing the OOD post-processing analysis and figures.

## Repository structure

```text
LLM-Assisted-Thermal-Dynamic-Modeling/
├── Code CNN/             # CNN experiments
├── Code MLP/             # MLP experiments
├── Code_GRU/             # GRU experiments
├── Code_LSTM/            # LSTM experiments
├── prior_generation/     # semantic and topology prior generation
└── post_processing/      # OOD post-processing and visualization
```

## Data and results

The datasets, generated priors, checkpoints, and complete results are available
from [Google Drive](https://drive.google.com/drive/folders/1Ar-OhwFWcuEEXwX_Amf26GFJR2aphw3-?usp=drive_link).

After downloading, keep `Data` and `Result` beside the cloned repository:

```text
release_root/
├── LLM-Assisted-Thermal-Dynamic-Modeling/
├── Data/
└── Result/
```

## Environment

```bash
git clone https://github.com/hkuedl/LLM-Assisted-Thermal-Dynamic-Modeling.git
cd LLM-Assisted-Thermal-Dynamic-Modeling

conda env create --name thermal-llm --file environment.yml
conda activate thermal-llm
```

## Usage

### Thermal-dynamics models

Each model directory contains a `run_modeling.py` entry point:

| Backbone | Directory |
| --- | --- |
| CNN | `Code CNN/` |
| MLP | `Code MLP/` |
| GRU | `Code_GRU/` |
| LSTM | `Code_LSTM/` |

For example, the following command runs the GRU experiments with seed 42:

```bash
cd Code_GRU
python run_modeling.py --seed 42 --building all
```

Use `python run_modeling.py --help` inside the corresponding model directory to
view the available options.

### LLM prior generation

The `prior_generation/` workflow regenerates semantic and topology priors from
the downloaded building data:

```bash
cd prior_generation
python run_all.py --model meta-models/Muse-Glimmer-30B --runs 10
```

See [`prior_generation/README.md`](prior_generation/README.md) for details.

### OOD post-processing

The `post_processing/` workflow reads the downloaded results and reproduces the
OOD correction metrics and figures:

```bash
cd post_processing
python apply_priors.py
python visualize_results.py
```

See [`post_processing/README.md`](post_processing/README.md) for the complete
workflow, including optional prior regeneration.
