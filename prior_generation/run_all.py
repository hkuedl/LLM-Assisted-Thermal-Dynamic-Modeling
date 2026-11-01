"""Regenerate semantic priors and cluster priors with one command."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE_ROOT = HERE.parent.parent
DEFAULT_DATA_ROOT = RELEASE_ROOT / "Data"
DEFAULT_OUTPUT_ROOT = RELEASE_ROOT / "Result" / "prior_reproduction"


def run(command):
    print("Running:", " ".join(map(str, command)))
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Directory containing CSV and IDF files",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="New directory for this reproduction run",
    )
    parser.add_argument("--model", default="meta-models/Muse-Glimmer-30B")
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    output = Path(args.output_root).expanduser().resolve()
    semantic_output = output / "semantic_generation"
    cluster_output = output / "cluster_generation"
    output.mkdir(parents=True, exist_ok=True)

    run([
        sys.executable, str(HERE / "generate_semantic_priors.py"),
        "--config", str(HERE / "buildings.json"),
        "--data-root", str(Path(args.data_root).expanduser().resolve()),
        "--output", str(semantic_output),
        "--model", args.model, "--runs", str(args.runs),
    ])

    base = json.loads((HERE / "buildings.json").read_text(encoding="utf-8"))
    cluster_items = []
    for item in base["buildings"]:
        building = item["building"]
        cluster_items.append({
            "building": building,
            "idf": item["idf"],
            "semantic_prior": str(
                semantic_output / f"llm_priors_{building}_{item.get('tag', 'fulldata')}_median_quantized.json"
            ),
        })
    cluster_config = output / "cluster_generation_config.json"
    cluster_config.write_text(
        json.dumps({"buildings": cluster_items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    run([
        sys.executable, str(HERE / "generate_cluster_priors.py"),
        "--config", str(cluster_config),
        "--data-root", str(Path(args.data_root).expanduser().resolve()),
        "--output", str(cluster_output),
        "--model", args.model, "--runs", str(args.runs),
    ])
    print("Completed reproduction run:", output)


if __name__ == "__main__":
    main()
