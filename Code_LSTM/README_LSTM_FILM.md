# LSTM experiments

The release uses this directory layout:

```text
release_root/
├── public code/
├── Data/
└── Result/
```

Default paths are resolved from the code location. No server-specific absolute
path is required:

- data: `../../Data/`
- output: `../../Result/`
- semantic priors: `../../Result/priors/semantic_generation/selected/`
- cluster priors: `../../Result/priors/llm_cluster/runtime/`

The four path arguments can still override these defaults when needed.

Example (all buildings and configured training-day settings):

```bash
python run_modeling.py --seed 38 --building all
```
