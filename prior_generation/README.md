# Prior generation

The scripts preserve the original three-stage method:

1. Generate ten semantic/topological prior candidates and select the median-quantized semantic prior.
2. Generate ten zone-cluster candidates.
3. Complete unresolved assignments locally while preserving valid assignments; introduce a new expert group only when supported by the supplied evidence.

The semantic, initial clustering, and repair prompts are kept in `prior_generation_core.py` and `generate_cluster_priors.py`.

Install dependencies and place the Together API key in `apikey.txt` beside these scripts. Do not include that file in a public release.

Run the complete reproduction pipeline:

With the public layout (`public code/`, `Data/`, and `Result/` at the same
level), the defaults are already portable:

```bash
python run_all.py --model meta-models/Muse-Glimmer-30B --runs 10
```

This reads `../Data/` relative to `public code/` and writes a new run under
`../Result/prior_reproduction/`. Use `--data-root` or `--output-root` only
when intentionally using another location.

A reproduction run always writes to a new output directory and does not overwrite the frozen experiment priors in `../llm_cluster/runtime/`.
