"""Generate the five-dimensional semantic priors for all buildings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import prior_generation_core as core


DEFAULT_MODEL = "meta-models/Muse-Glimmer-30B"
SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SCRIPT_DIR / "buildings.json"))
    parser.add_argument("--output", default=str(SCRIPT_DIR / "outputs" / "semantic"))
    parser.add_argument(
        "--data-root",
        help="Directory containing the building CSV and IDF files.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--sleep-s", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    items = config["buildings"] if isinstance(config, dict) else config
    output_root = Path(args.output).expanduser().resolve()
    for item in items:
        csv_path = Path(item["csv"])
        idf_path = Path(item["idf"])
        if not csv_path.is_absolute():
            csv_path = ((data_root or config_path.parent) / csv_path).resolve()
        if not idf_path.is_absolute():
            idf_path = ((data_root or config_path.parent) / idf_path).resolve()
        building = item.get("building") or core.building_name_from_csv(str(csv_path))
        building_output = output_root / f"Result_LLM_Stability_{building}"
        run_args = argparse.Namespace(
            csv=str(csv_path),
            idf=str(idf_path),
            reference_idf=None,
            out_dir=str(building_output),
            runs=args.runs,
            model=args.model,
            temperature=args.temperature,
            seed=args.seed,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            sleep_s=args.sleep_s,
            max_zones=None,
            include_connections=False,
            numeric_output=False,
            single_agent=False,
            multi_agent=True,
            thermal_context_chars=20000,
            schedule_context_chars=20000,
            resume=args.resume,
        )
        result = core.run_stability_experiment(run_args)
        final_path = output_root / f"llm_priors_{building}_{item.get('tag', 'fulldata')}_median_quantized.json"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_text(
            json.dumps(result["median_quantized_prior"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Generated: {final_path}")


if __name__ == "__main__":
    main()
