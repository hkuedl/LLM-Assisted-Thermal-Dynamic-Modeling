"""Generate zone-cluster priors from the fixed semantic priors.

The semantic prior and the cluster prior are separate artifacts:

* ``semantic_prior`` contains one five-dimensional semantic vector per zone.
* ``cluster_prior`` assigns each zone to a shared expert branch.

This module reproduces the cluster-completion policy. Existing, unambiguous
assignments are preserved. Only unresolved zones are completed, and an
evidence-supported new group may be introduced when none of the existing
groups is suitable.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from together import Together
except ImportError:  # Pre-generated priors can be inspected without the API client.
    Together = None

import prior_generation_core as core
from api_key import load_api_key


DEFAULT_MODEL = "meta-models/Muse-Glimmer-30B"
SCRIPT_DIR = Path(__file__).resolve().parent
PROTOCOL = "local_completion_allow_new_groups"


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _group_id(value) -> int:
    if isinstance(value, bool):
        raise ValueError("Boolean group IDs are not valid.")
    return int(value)


def _canonical_zone(name: str, zones: Iterable[str]) -> Optional[str]:
    return core.match_zone_name(str(name), list(zones))


def cluster_state(raw: dict, zones: List[str]) -> dict:
    """Inspect a raw answer without silently assigning missing zones."""
    groups: Dict[int, dict] = {}
    assignments: Dict[str, List[dict]] = {zone: [] for zone in zones}
    unknown: List[str] = []

    for item in raw.get("groups", []) if isinstance(raw.get("groups"), list) else []:
        if not isinstance(item, dict):
            continue
        gid = _group_id(item.get("group_id", item.get("id", 0)))
        groups[gid] = copy.deepcopy(item)
        for supplied in item.get("zones", []):
            zone = _canonical_zone(supplied, zones)
            if zone is None:
                unknown.append(str(supplied))
            else:
                assignments[zone].append({"group_id": gid, "source": "groups"})

    records = raw.get("zone_records", {})
    if isinstance(records, dict):
        for supplied, record in records.items():
            if not isinstance(record, dict) or "group_id" not in record:
                continue
            zone = _canonical_zone(supplied, zones)
            if zone is None:
                unknown.append(str(supplied))
                continue
            assignments[zone].append(
                {"group_id": _group_id(record["group_id"]), "source": "zone_records", **record}
            )

    fixed, conflicts, duplicates = {}, [], []
    for zone, rows in assignments.items():
        gids = [int(row["group_id"]) for row in rows]
        unique = sorted(set(gids))
        if len(unique) == 1:
            fixed[zone] = unique[0]
            if len(gids) > 1:
                duplicates.append(zone)
        elif len(unique) > 1:
            conflicts.append(zone)

    return {
        "groups": groups,
        "records": assignments,
        "fixed": fixed,
        "missing": [zone for zone in zones if not assignments[zone]],
        "conflicts": conflicts,
        "same_group_duplicates": duplicates,
        "unknown": sorted(set(unknown)),
    }


def validate_complete(raw: dict, zones: List[str]) -> dict:
    state = cluster_state(raw, zones)
    unresolved = state["missing"] + state["conflicts"]
    if unresolved:
        raise ValueError(f"Unresolved zones: {unresolved}")
    normalized = core.validate_llm_cluster_prior(raw, zones)
    if not 2 <= int(normalized["n_groups"]) <= 5:
        raise ValueError(f"Expected 2-5 nonempty groups; got {normalized['n_groups']}")
    return normalized


def _required_text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must contain a short explanation.")
    return value.strip()


def apply_local_completion(raw: dict, completion: dict, zones: List[str]) -> dict:
    state = cluster_state(raw, zones)
    pending = [zone for zone in zones if zone not in state["fixed"]]
    if not state["fixed"] or not pending:
        raise ValueError("Local completion requires fixed and unresolved zones.")
    if not isinstance(completion, dict):
        raise ValueError("Repair must be a JSON object.")
    if completion.get("cannot_repair"):
        raise ValueError(f"LLM cannot repair: {completion['cannot_repair']}")
    if not isinstance(completion.get("assignments"), list):
        raise ValueError("Repair must contain assignments.")

    definitions = copy.deepcopy(state["groups"])
    new_groups = completion.get("new_groups", [])
    if not isinstance(new_groups, list):
        raise ValueError("new_groups must be a list.")
    new_ids = set()
    for item in new_groups:
        gid = _group_id(item.get("group_id"))
        if gid in definitions:
            raise ValueError(f"New group ID already exists: {gid}")
        definition = copy.deepcopy(item)
        for field in ("label", "thermal_hypothesis", "why_existing_groups_unsuitable"):
            definition[field] = _required_text(item.get(field), field)
        evidence = item.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("A new group needs evidence from the supplied input.")
        definition["evidence"] = [_required_text(x, "evidence") for x in evidence]
        definition["zones"] = []
        definition["reason"] = definition["why_existing_groups_unsuitable"]
        definitions[gid] = definition
        new_ids.add(gid)

    completed, reasons = {}, {}
    for row in completion["assignments"]:
        zone = row.get("zone") if isinstance(row, dict) else None
        if zone not in pending or zone in completed:
            raise ValueError(f"Only unresolved zones may be assigned once: {zone}")
        gid = _group_id(row.get("group_id"))
        if gid not in definitions:
            raise ValueError(f"Undefined group: {gid}")
        completed[zone] = gid
        reasons[zone] = _required_text(row.get("reason"), "assignment reason")

    absent = [zone for zone in pending if zone not in completed]
    if absent:
        raise ValueError(f"Completion still misses: {absent}")

    mapping = dict(state["fixed"], **completed)
    used = set(mapping.values())
    if not 2 <= len(used) <= 5:
        raise ValueError(f"Final group count must be 2-5; got {len(used)}")
    if new_ids - used:
        raise ValueError("New groups must not be empty.")

    result = copy.deepcopy(raw)
    result["groups"] = []
    for gid, definition in definitions.items():
        members = [zone for zone in zones if mapping[zone] == gid]
        if members:
            result["groups"].append(dict(definition, group_id=gid, zones=members))
    result["n_groups"] = len(used)
    result["zone_records"] = {}
    for zone in zones:
        original = next(iter(state["records"].get(zone, [])), {})
        record = copy.deepcopy(original)
        record["group_id"] = mapping[zone]
        if zone in completed:
            record["evidence"] = [reasons[zone]]
            record.pop("confidence", None)
        result["zone_records"][zone] = record
    return result


def build_completion_prompt(raw: dict, zones: List[str], evidence: List[dict], building: str) -> str:
    state = cluster_state(raw, zones)
    pending = [zone for zone in zones if zone not in state["fixed"]]
    lookup = {row["zone"]: row for row in evidence}
    groups = []
    for gid, definition in state["groups"].items():
        members = [zone for zone in zones if state["fixed"].get(zone) == gid]
        groups.append(
            {
                "group_id": gid,
                "label": definition.get("label", ""),
                "thermal_hypothesis": definition.get("thermal_hypothesis", ""),
                "mechanism_profile": definition.get("mechanism_profile", {}),
                "fixed_zones": members,
                "member_evidence_examples": [lookup[z] for z in members[:3] if z in lookup],
            }
        )
    context = {
        "building": building,
        "existing_groups": groups,
        "unresolved_zones": pending,
        "unresolved_evidence": [lookup[z] for z in pending],
        "issues": {key: state[key] for key in ("missing", "conflicts", "unknown")},
    }
    instruction = """Complete ONLY the unresolved zone assignments in this building.
Keep every fixed zone assignment and existing group definition unchanged.
Use the supplied semantic/IDF evidence. Priorities are internal-gain/schedule
volatility, envelope exposure, control response, thermal inertia, then name hints.
Do not use prediction errors, test data, or a desired historical partition.

For each unresolved zone, choose an existing group if the evidence supports it.
You MAY instead define a new group containing ONLY unresolved zones if you can
explain both their shared mechanism and why the existing groups are unsuitable.
New groups are optional, never mandatory. Names alone are insufficient grounds
for a new group; do not invent occupancy schedules, controls, or physical values
absent from the evidence. The member examples are partial evidence, not every
member's full data. Similar names/counts do not prove similar dynamics.
Do not create a catch-all group simply because zones are missing.
Allow a singleton only for a clear special-control/high-gain outlier, as in the
original clustering prompt. Keep the final number of nonempty groups within 2-5.
Every new group_id must differ from all existing IDs. Do not renumber existing IDs.
Include every unresolved zone exactly once and NO fixed zones in assignments.
If there is insufficient evidence for a defensible complete assignment, return
{"cannot_repair": "brief reason"} rather than forcing an assignment.

Return strict JSON with this structure (new_groups can be []):
{"new_groups": [{"group_id": 4, "label": "short name",
 "thermal_hypothesis": "shared mechanism supported by the input",
 "why_existing_groups_unsuitable": "specific differences from candidate groups",
 "evidence": ["zone name + input field/value supporting the decision"]}],
 "assignments": [{"zone": "exact unresolved zone name", "group_id": 4,
 "reason": "brief input-based justification"}]}
The number 4 above is an example, not a preferred ID or a required new group.
"""
    return instruction + "\nInput:\n" + json.dumps(context, ensure_ascii=False, indent=2)


def call_json(client, model: str, system: str, prompt: str, output_dir: Path, stem: str,
              temperature: float, seed: Optional[int], max_tokens: int) -> dict:
    return core.call_llm_json(
        client=client,
        model=model,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
        system_prompt=system,
        user_prompt=prompt,
        raw_output_dir=str(output_dir),
        run_idx=1,
        batch_idx=1,
        agent_name=stem,
        allow_empty_fallback=False,
    )


def generate_candidate(raw: Optional[dict], zones: List[str], evidence: List[dict],
                       building: str, client, model: str, output_dir: Path,
                       temperature: float, seed: Optional[int], max_tokens: int,
                       max_completion_fraction: float) -> dict:
    system, full_prompt = core.build_llm_zone_cluster_prompts(building, zones, evidence)
    attempts = ([0] if isinstance(raw, dict) else []) + [1, 2, 3]
    retry_prompt = full_prompt
    last_error = None
    for attempt in attempts:
        candidate = raw if attempt == 0 else None
        stem = "saved" if attempt == 0 else f"full_{attempt:02d}"
        try:
            answer = copy.deepcopy(candidate) if candidate is not None else call_json(
                client, model, system, retry_prompt, output_dir,
                stem, temperature,
                None if seed is None else seed + attempt - 1,
                max_tokens,
            )
            try:
                normalized = validate_complete(answer, zones)
            except ValueError:
                state = cluster_state(answer, zones)
                pending = [zone for zone in zones if zone not in state["fixed"]]
                if not state["fixed"] or not pending or not 2 <= len(state["groups"]) <= 5:
                    raise
                if len(pending) > max(1, math.ceil(len(zones) * max_completion_fraction)):
                    raise ValueError(f"Too many unresolved zones for local completion: {len(pending)}")
                prompt = build_completion_prompt(answer, zones, evidence, building)
                retry_completion = prompt
                normalized = None
                for completion_attempt in (1, 2):
                    try:
                        completion = call_json(
                            client, model, system, retry_completion, output_dir,
                            f"{stem}_repair_{completion_attempt:02d}",
                            temperature, seed, max_tokens,
                        )
                        merged = apply_local_completion(answer, completion, zones)
                        normalized = validate_complete(merged, zones)
                        normalized["raw_llm_cluster_output"] = answer
                        normalized["repair_output"] = completion
                        normalized["repair_summary"] = {
                            "fixed_zone_count": len(state["fixed"]),
                            "repaired_zones": pending,
                            "missing_zones": state["missing"],
                            "conflicting_zones": state["conflicts"],
                            "same_group_duplicate_zones": state["same_group_duplicates"],
                            "removed_unknown_names": state["unknown"],
                            "new_group_count": len(completion.get("new_groups", [])),
                            "new_group_ids": [g["group_id"] for g in completion.get("new_groups", [])],
                        }
                        break
                    except ValueError as error:
                        last_error = error
                        retry_completion = prompt + "\nPrevious repair error: " + str(error)
                if normalized is None:
                    raise ValueError(f"Local completion failed: {last_error}")
            normalized["protocol"] = PROTOCOL
            normalized["generation_action"] = (
                "repair_same_raw" if "repair_output" in normalized else "reuse_complete_raw"
            ) if attempt == 0 else (
                "regenerate_then_repair" if "repair_output" in normalized else "regenerate_complete"
            )
            normalized["full_regeneration_attempts"] = attempt
            return normalized
        except ValueError as error:
            last_error = error
            retry_prompt = (
                full_prompt
                + f"\nPrevious invalid response: {error}\nAssign every zone exactly once."
            )
    raise ValueError(f"Could not produce a complete cluster candidate: {last_error}")


def generate_building(item: dict, output_root: Path, model: str, runs: int,
                      temperature: float, seed: Optional[int], max_tokens: int,
                      threshold: float, max_completion_fraction: float) -> Path:
    building = item["building"]
    semantic = read_json(Path(item["semantic_prior"]))
    zones = semantic.get("zones", [])
    matrix = semantic.get("semantic_matrix", [])
    if not zones or len(matrix) != len(zones) or any(len(row) != 5 for row in matrix):
        raise ValueError(f"Invalid semantic prior for {building}")
    idf_text = Path(item["idf"]).read_text(encoding="utf-8")
    evidence = core.build_zone_cluster_evidence(idf_text, zones, semantic)

    if Together is None:
        raise RuntimeError("Install the 'together' package before regenerating cluster priors.")
    client = Together(api_key=load_api_key())
    building_dir = Path(output_root) / building
    run_dir = building_dir / "cluster_prior_runs"
    raw_dir = building_dir / "raw_llm_outputs"
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    for index in range(1, runs + 1):
        path = run_dir / f"cluster_run_{index:03d}.json"
        if path.is_file():
            candidate = read_json(path)
            validate_complete(candidate, zones)
        else:
            run_seed = None if seed is None else seed + index - 1
            candidate = generate_candidate(
                None, zones, evidence, building, client, model,
                raw_dir, temperature, run_seed, max_tokens, max_completion_fraction,
            )
            write_json(path, candidate)
        completed.append(candidate)

    final = core.build_consensus_cluster_prior(completed, zones, building, threshold)
    if not 2 <= int(final["n_groups"]) <= 5:
        raise ValueError(f"Final cluster count must be 2-5; got {final['n_groups']}")
    final["cluster_runs"] = completed
    final["generation"] = {
        "protocol": PROTOCOL,
        "model": model,
        "temperature": temperature,
        "runs": runs,
        "semantic_prior": "median_quantized",
        "input_policy": "fresh generation with local completion of unresolved zones",
    }
    write_json(building_dir / "llm_cluster_prior.json", final)
    for tag in ("few_shot", "fulldata"):
        write_json(output_root / "priors" / f"llm_cluster_prior_{building}_{tag}.json", final)
    return building_dir / "llm_cluster_prior.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SCRIPT_DIR / "buildings.json"))
    parser.add_argument("--output", default=str(SCRIPT_DIR / "outputs" / "cluster"))
    parser.add_argument(
        "--data-root",
        help="Directory containing the building IDF files. Relative IDF paths are resolved here.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--consensus-threshold", type=float, default=0.6)
    parser.add_argument("--max-completion-fraction", type=float, default=0.20)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    items = config["buildings"] if isinstance(config, dict) else config
    for item in items:
        for key in ("idf", "semantic_prior"):
            if item.get(key):
                path = Path(item[key])
                if key == "idf" and data_root is not None and not path.is_absolute():
                    path = data_root / path
                elif not path.is_absolute():
                    path = config_path.parent / path
                item[key] = str(path.resolve())
        path = generate_building(
            item=item,
            output_root=Path(args.output).expanduser().resolve(),
            model=args.model,
            runs=args.runs,
            temperature=args.temperature,
            seed=args.seed,
            max_tokens=args.max_tokens,
            threshold=args.consensus_threshold,
            max_completion_fraction=args.max_completion_fraction,
        )
        print(f"Generated: {path}")


if __name__ == "__main__":
    main()
