# LLM-Assisted Thermal Dynamic Modeling

This repository contains the public reproduction code for the four thermal
dynamic model backbones and the OOD post-processing workflow.

Keep the cloned repository, `Data`, and `Result` at the same level:

```text
release_root/
├── LLM-Assisted-Thermal-Dynamic-Modeling/
├── Data/
└── Result/
```

The code resolves all defaults relative to its own location:

- input CSV and IDF files: `../Data/`
- results and checkpoints: `../Result/`
- semantic priors: `../Result/priors/semantic_generation/selected/`
- final cluster priors: `../Result/priors/llm_cluster/runtime/`

For example:

```bash
cd "public code/Code_GRU"
python run_modeling.py --seed 38 --building all
```

The same layout is used by the CNN, MLP, GRU, and LSTM runners. The path
arguments (`--data-root`, `--result-root`, `--prior-root`, and
`--cluster-prior-root`) remain available when a custom location is needed.

The four corresponding environment variables are:

```text
BUILDING_LLM_DATA_ROOT
BUILDING_LLM_RESULT_ROOT
BUILDING_LLM_PRIOR_STABILITY_ROOT
BUILDING_LLM_CLUSTER_PRIOR_ROOT
```

`prior_generation/` regenerates priors into a new output directory. It does
not overwrite the frozen priors under `Result/priors/`. Keep `apikey.txt`
local and remove it before publishing the repository.

`post_processing/` reads the complete `Result/` folder,
including `Result/OOD/postprocessing/case_manifest.csv` and
`run_manifest.csv`. Its public outputs are written under
`Result/OOD/postprocessing_public/`. See that directory's README and main
notebook for the mean-10 event-and-phase workflow.

Large datasets, trained checkpoints, and complete prediction outputs are not
stored in this Git repository. Place the separately released `Data/` and
`Result/` directories beside the cloned repository before running the code.
