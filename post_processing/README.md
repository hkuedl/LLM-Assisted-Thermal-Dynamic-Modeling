# OOD Post-Processing

This directory contains the OOD event-and-phase post-processing workflow used
in **“Interactive Building Thermal Dynamics Modeling with Large Language
Models.”** It applies correction gates to frozen forecasting outputs and
generates the corresponding metrics, tables, and figures.

The workflow uses the `Data` and `Result` folders distributed with the project.
Place both folders beside the cloned repository as described in the main
README.

## Run

From the `post_processing` directory. LLM generation requires the relevant API
key to be available in the environment.

```bash
# Preview prior generation without making API calls
python generate_priors.py

# Generate or resume the LLM priors
python generate_priors.py --run

# Apply the priors and compute evaluation metrics
python apply_priors.py

# Generate summary tables and figures
python visualize_results.py
```

If the downloaded results already contain the generated priors, the second
step can be skipped. Outputs are saved under:

```text
Result/OOD/postprocessing_public/
```

The same workflow can be run interactively with
`Run_Postprocessing.ipynb`.

## Main files

- `generate_priors.py`: generates and caches LLM event-and-phase priors.
- `apply_priors.py`: applies the correction rules and evaluates predictions.
- `visualize_results.py`: creates summary tables and figures.
- `Run_Postprocessing.ipynb`: notebook interface for the complete workflow.
- `Paper_Figures.ipynb`: paper-oriented tables and visualizations.

## Tests

```bash
python -m unittest discover -s tests -v
```
