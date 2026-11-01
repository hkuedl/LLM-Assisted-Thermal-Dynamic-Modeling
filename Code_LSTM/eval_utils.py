"""Evaluation output needed by the modeling pipeline.

Post-calibration comparisons and control-shift plotting intentionally live
outside this clean modeling baseline.
"""

import glob
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

import config


TRAIN_DAYS_ORDER = [3, 7, 14, 21, 28, "Full"]


@dataclass
class ModelName:
    raw: str
    clean: str
    family: str
    is_control_invariant: bool


def normalize_base_model_name(raw_model_name: str) -> ModelName:
    clean = raw_model_name.strip()
    for suffix in [
        "+ DeltaTarget",
        "(Proposed Core)",
        "(Proposed Hybrid)",
        "(Proposed C1)",
        "(Proposed)",
    ]:
        clean = clean.replace(suffix, "").strip()
    clean = " ".join(clean.split())
    is_control_invariant = (
        "ControlInvariant" in clean
        or "Control-Invariant" in clean
        or clean.endswith("+ CI")
        or clean.startswith("CI ")
    )
    family = clean.replace("+ ControlInvariant", "").replace("+ CI", "").strip()
    return ModelName(raw_model_name, clean, family, is_control_invariant)


def load_existing_result_record(log_path: str) -> dict:
    if not os.path.exists(log_path):
        return {}
    try:
        with open(log_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def atomic_json_dump(payload: dict, log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    temp_path = f"{log_path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
    os.replace(temp_path, log_path)


def log_result_to_json(
    log_path,
    dataset_name,
    zones_count,
    model_name,
    day_tag,
    gl_rmse,
    gl_mae,
    gl_mape,
    gl_r2,
    zone_metrics,
):
    if not config.SAVE_RESULTS:
        return
    records = load_existing_result_record(log_path)
    run_tag = f"[FewShot_{day_tag}d]" if day_tag > 0 else "[FullData]"
    dataset_key = f"{run_tag} [{zones_count} Zones] {os.path.basename(dataset_name)}"
    records.setdefault(dataset_key, {})[model_name] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Global_Metrics": {
            "RMSE": round(gl_rmse, 4),
            "MAE": round(gl_mae, 4),
            "MAPE": round(gl_mape, 4),
            "R2": round(gl_r2, 4),
        },
        "Zone_Metrics": zone_metrics,
    }
    atomic_json_dump(records, log_path)


def save_prediction_csv(
    y_pred,
    df,
    zones,
    model_name,
    dataset_file,
    day_tag,
    result_dir,
):
    if y_pred is None or not config.SAVE_RESULTS:
        return None

    from data_utils import build_next_step_eval_frame

    target_df = build_next_step_eval_frame(df)
    min_len = min(len(y_pred), len(target_df))
    y_array = np.asarray(y_pred)
    output = {}
    if "Date/Time" in target_df.columns:
        output["Date/Time"] = target_df["Date/Time"].iloc[:min_len].values
    if "datetime" in target_df.columns:
        output["datetime"] = target_df["datetime"].iloc[:min_len].values
    for zone_index, zone_name in enumerate(zones):
        output[f"T_in__{zone_name}"] = y_array[:min_len, zone_index]

    day_text = f"{day_tag}d" if day_tag > 0 else "full"
    dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
    output_dir = os.path.join(
        result_dir, "Prediction_Curves", dataset_name, day_text
    )
    os.makedirs(output_dir, exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")
    output_path = os.path.join(output_dir, f"{safe_model}.csv")
    pd.DataFrame(output).to_csv(output_path, index=False)
    print(f"  [Prediction CSV] Saved to {output_path}")
    return output_path


def parse_model_eval_name(model_name: str):
    clean_name = model_name.strip()
    if "| Evaluated on" in clean_name:
        return clean_name.split("| Evaluated on", 1)[0].strip(), "OOD_ControlShift"
    return clean_name, "ID_Original"


def generate_publication_table(result_dir: str):
    rows = []
    paths = glob.glob(os.path.join(result_dir, "*_results_*d.json"))
    paths += glob.glob(os.path.join(result_dir, "*_results_full.json"))

    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                records = json.load(handle)
            basename = os.path.basename(path)
            if "_results_" not in basename:
                continue
            building = basename.split("_results_", 1)[0]
            suffix = basename.split("_results_", 1)[1]
            train_days = "Full" if "full" in suffix else int(suffix.replace("d.json", ""))
            for methods in records.values():
                for model_name, info in methods.items():
                    metrics = info.get("Global_Metrics")
                    if not metrics:
                        continue
                    base_model, eval_set = parse_model_eval_name(model_name)
                    rows.append(
                        {
                            "Building": building,
                            "Train_Days": train_days,
                            "Eval_Set": eval_set,
                            "Model": base_model,
                            "RMSE": metrics.get("RMSE"),
                            "MAE": metrics.get("MAE"),
                            "MAPE": metrics.get("MAPE"),
                        }
                    )
        except (OSError, ValueError, json.JSONDecodeError):
            continue

    if not rows:
        print("[Table] No records found.")
        return None

    frame = pd.DataFrame(rows)
    day_order = sorted(
        frame["Train_Days"].unique(), key=lambda value: 99999 if value == "Full" else value
    )
    output_path = os.path.join(result_dir, "Publication_Tables_All.xlsx")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for eval_set in ["ID_Original", "OOD_ControlShift"]:
            subset = frame[frame["Eval_Set"] == eval_set]
            if subset.empty:
                continue
            average = subset.pivot_table(
                index="Model", columns="Train_Days", values="RMSE", aggfunc="mean"
            ).reindex(columns=day_order)
            average.to_excel(writer, sheet_name=f"{eval_set[:24]}_RMSE")
    print(f"[Table] Publication Excel saved to: {output_path}")
    return frame


def collect_result_records(
    result_dirs,
    exclude_keywords=None,
    exclude_buildings_exact=None,
    include_base_models=None,
    include_strategies=None,
):
    """Collect ID and raw OOD modeling metrics across seed result folders."""
    if isinstance(result_dirs, str):
        result_dirs = [result_dirs]

    excluded = {str(name).strip().lower() for name in (exclude_buildings_exact or [])}
    allowed_models = {
        normalize_base_model_name(name).clean for name in (include_base_models or [])
    }
    allowed_strategies = set(include_strategies or ["ID", "OOD"])
    rows = []

    for seed_index, result_dir in enumerate(result_dirs):
        paths = glob.glob(os.path.join(result_dir, "*_results_*d.json"))
        paths += glob.glob(os.path.join(result_dir, "*_results_full.json"))
        for path in paths:
            basename = os.path.basename(path)
            if "_results_" not in basename:
                continue
            building, suffix = basename.split("_results_", 1)
            if building.lower() in excluded:
                continue
            if exclude_keywords and any(
                keyword.lower() in building.lower() for keyword in exclude_keywords
            ):
                continue
            train_days = "Full" if "full" in suffix else int(suffix.replace("d.json", ""))
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    records = json.load(handle)
            except (OSError, ValueError, json.JSONDecodeError):
                continue

            for methods in records.values():
                for raw_model_name, info in methods.items():
                    metrics = info.get("Global_Metrics", {})
                    if metrics.get("RMSE") is None:
                        continue
                    if "| Evaluated on" in raw_model_name:
                        base_model, external = raw_model_name.split("| Evaluated on", 1)
                        eval_set, strategy = "OOD_ControlShift", "OOD"
                        external_set = os.path.splitext(os.path.basename(external.strip()))[0]
                    else:
                        base_model = raw_model_name
                        eval_set, strategy, external_set = "ID_Original", "ID", "ID"

                    model_name = normalize_base_model_name(base_model)
                    if allowed_models and model_name.clean not in allowed_models:
                        continue
                    if strategy not in allowed_strategies and strategy != "ID":
                        continue
                    variant_match = re.search(r"_new(\d+)$", external_set)
                    external_variant = (
                        f"New_{variant_match.group(1)}" if variant_match else external_set
                    )
                    rows.append(
                        {
                            "Seed": seed_index,
                            "Source_File": path,
                            "Raw_Model_Name": raw_model_name,
                            "Building": building,
                            "Result_Building": building,
                            "Test_Building": external_set if strategy == "OOD" else building,
                            "Train_Days": train_days,
                            "Base_Model": base_model.strip(),
                            "Model_Clean": model_name.clean,
                            "Model_Family": model_name.family,
                            "ControlInvariant": model_name.is_control_invariant,
                            "Eval_Set": eval_set,
                            "Strategy": strategy,
                            "External_Set": external_set,
                            "External_Variant": external_variant,
                            "RMSE": metrics.get("RMSE"),
                            "MAE": metrics.get("MAE"),
                            "MAPE": metrics.get("MAPE"),
                            "R2": metrics.get("R2"),
                        }
                    )

    if not rows:
        print("No matching result records were found.")
        return None
    return pd.DataFrame(rows)


def summarize_result_records(
    raw_df,
    output_path,
    include_strategies=None,
    **_,
):
    """Aggregate duplicate runs within seeds, then report mean/std across seeds."""
    group_columns = [
        "Seed",
        "Building",
        "Train_Days",
        "Base_Model",
        "Model_Clean",
        "Model_Family",
        "ControlInvariant",
        "Eval_Set",
        "Strategy",
        "External_Set",
        "External_Variant",
    ]
    seed_grouped = raw_df.groupby(group_columns, as_index=False).agg(
        RMSE=("RMSE", "mean"),
        MAE=("MAE", "mean"),
    )
    summary_columns = group_columns[1:]
    grouped = seed_grouped.groupby(summary_columns, as_index=False).agg(
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        MAE_mean=("MAE", "mean"),
        Seed_Count=("RMSE", "count"),
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name="Raw", index=False)
        grouped.to_excel(writer, sheet_name="Summary", index=False)
        for eval_set, sheet_name in [
            ("ID_Original", "ID_Macro"),
            ("OOD_ControlShift", "OOD_Macro"),
        ]:
            subset = seed_grouped[seed_grouped["Eval_Set"] == eval_set]
            if subset.empty:
                continue
            table = subset.pivot_table(
                index="Base_Model",
                columns="Train_Days",
                values="RMSE",
                aggfunc="mean",
            )
            table.to_excel(writer, sheet_name=sheet_name)
    print(f"[Summary] Saved to: {output_path}")
    return grouped
