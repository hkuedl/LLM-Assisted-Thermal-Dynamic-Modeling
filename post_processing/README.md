# OOD event-and-phase post-processing

This directory contains the post-processing workflow used to evaluate fixed
correction gates and LLM-derived event-and-phase gates. Forecasting models are
frozen. The code does not train or alter MLP, CNN, GRU, or LSTM checkpoints.

## Release layout

Place the cloned repository, `Data`, and `Result` at the same level:

```text
release_root/
├── Data/
├── Result/
└── LLM-Assisted-Thermal-Dynamic-Modeling/
    └── post_processing/
```

`Result/` is the complete experimental `Building_Temperature_Final_Results`
directory renamed for release. The workflow reads:

```text
Result/OOD/postprocessing/case_manifest.csv
Result/OOD/postprocessing/run_manifest.csv
```

Manifest paths should be relative to either `Result/` or the directory that
contains the manifest. Raw predictions, targets, control priors, and event
evidence remain in `Result/`; source CSV and IDF files remain in `Data/`.

By default, new public outputs are written to:

```text
Result/OOD/postprocessing_public/public_prompt/
```

The default configuration uses ten LLM samples, the public prompt wording,
all four training quantities, and source labels `OOD-2` through `OOD-5`. These
source labels correspond to the four OOD scenarios reported in the paper; the
code does not silently relabel them.

## Run

The notebook `Run_Postprocessing.ipynb` is the main entry point. Its generation
switch is off by default.

Equivalent command-line steps are:

```bash
# Preview LLM calls. No API request is made.
python generate_priors.py

# Generate or resume the public mean-10 priors.
python generate_priors.py --run

# Apply Raw, the 0.05 fixed-gate grid, and the LLM event-and-phase gate.
python apply_priors.py

# Create paired summaries and figures.
python visualize_results.py
```

Saved valid requests are reused when the same command is resumed. If public
mean-10 priors are already present in the complete `Result/` directory, keep
generation disabled and run only the offline application and visualization
steps.

For a frozen historical prior set, set `PRIOR_SOURCE='existing'` and point
`EXISTING_PRIORS` to its directory. Historical experiment directory names are
not defaults in this release.

## Files

- `prompt.py`: original prompt and the public wording with version labels
  removed; numerical evidence and output structure are shared.
- `generate_priors.py`: LLM sampling, validation, caching, and mean aggregation.
- `apply_priors.py`: deterministic correction operator, gate application, and
  RMSE/MAE evaluation.
- `visualize_results.py`: building-equal summaries and standard figures.
- `Paper_Figures.ipynb`: paper-oriented mean-10 tables and figures.
- `tests/`: small offline tests for prompt and numerical behavior.

The LLM supplies event strength and phase-local adjustments. The deterministic
operator supplies correction direction, support, and magnitude limits. For ten
samples, event gates and phase adjustments are averaged separately and combined
as `clip(mean(event_gate) + mean(phase_adjustment), 0, 1)`.

