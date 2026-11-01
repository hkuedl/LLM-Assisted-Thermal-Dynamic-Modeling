import argparse
import glob
import json
import os
import re
import shutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from api_key import load_api_key

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from together import Together
except ImportError:
    Together = None


SEMANTIC_NAMES = [
    "Thermal_Inertia",
    "Occupancy_Volatility",
    "Env_Exposure",
    "Base_Physics_Trust",
    "Env_Sensitivity",
]

CONTROL_NAMES = [
    "Cooling_Setpoint_Mean",
    "Cooling_Setpoint_Range",
    "Heating_Setpoint_Mean",
    "Heating_Setpoint_Range",
    "Setpoint_Switch_Count",
    "Comfort_Band_Mean",
    "Cooling_Shift_MeanAbs",
    "Heating_Shift_MeanAbs",
    "Schedule_Distance",
    "Control_Changed_Flag",
]

LLM_CONTROL_CALIBRATION_NAMES = [
    "Response_Sensitivity",
    "Center_Shift_Weight",
    "Band_Mapping_Weight",
    "Heating_Boundary_Weight",
    "Cooling_Boundary_Weight",
    "Active_Window_Weight",
    "Control_Calibration_Trust",
    "Setpoint_Tracking_Gain",
    "Baseline_Shift_Strength",
    "Load_Coupling",
    "Thermal_Memory",
]

LABEL_TO_VALUE = {
    "very_low": 0.1,
    "low": 0.3,
    "medium": 0.5,
    "medium_high": 0.7,
    "high": 0.9,
}

VALUE_TO_LABEL_THRESHOLDS = [
    (0.2, "very_low"),
    (0.4, "low"),
    (0.6, "medium"),
    (0.8, "medium_high"),
    (1.01, "high"),
]


def ultra_clean(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def normalize_label(value) -> Optional[str]:
    if value is None:
        return None

    label = re.sub(r"[^a-z_]", "_", str(value).strip().lower())
    label = re.sub(r"_+", "_", label).strip("_")

    aliases = {
        "verylow": "very_low",
        "very_low": "very_low",
        "low": "low",
        "med_low": "low",
        "medium_low": "low",
        "mid": "medium",
        "moderate": "medium",
        "medium": "medium",
        "med": "medium",
        "mediumhigh": "medium_high",
        "medium_high": "medium_high",
        "med_high": "medium_high",
        "high": "high",
        "very_high": "high",
        "veryhigh": "high",
    }
    return aliases.get(label)


def value_to_label(value: float) -> str:
    x = max(0.0, min(1.0, float(value)))
    for threshold, label in VALUE_TO_LABEL_THRESHOLDS:
        if x < threshold:
            return label
    return "high"


def labels_to_values(labels: List[str]) -> Optional[List[float]]:
    if len(labels) < len(SEMANTIC_NAMES):
        return None

    values = []
    for label in labels[:len(SEMANTIC_NAMES)]:
        normalized = normalize_label(label)
        if normalized is None:
            return None
        values.append(LABEL_TO_VALUE[normalized])
    return values


def semantic_item_to_values(item) -> Optional[List[float]]:
    if isinstance(item, dict):
        if "scores" in item and item["scores"] is not None:
            vals = [float(x) for x in item["scores"]]
            vals = vals[:5] if len(vals) >= 5 else vals + [0.5] * (5 - len(vals))
            return [max(0.0, min(1.0, x)) for x in vals]

        if "labels" in item and item["labels"] is not None:
            return labels_to_values(item["labels"])

        label_keys = [
            "thermal_inertia",
            "occupancy_volatility",
            "env_exposure",
            "base_physics_trust",
            "env_sensitivity",
        ]
        if all(key in item for key in label_keys):
            return labels_to_values([item[key] for key in label_keys])

    if isinstance(item, list):
        if not item:
            return None
        if all(isinstance(x, str) for x in item):
            return labels_to_values(item)
        vals = [float(x) for x in item]
        vals = vals[:5] if len(vals) >= 5 else vals + [0.5] * (5 - len(vals))
        return [max(0.0, min(1.0, x)) for x in vals]

    return None


def build_multizone_timeseries(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    def norm_c(c: str):
        return re.sub(r"\s+", " ", str(c).strip().lower())

    def clean_z(s: str):
        return re.sub(r"\s+", " ", str(s).strip())

    def extr(c: str, kw: str):
        r = re.sub(r"\(.*?\)", "", c.strip()).strip()
        if ":" in r and norm_c(kw) in norm_c(r.split(":", 1)[1]):
            return clean_z(r.split(":", 1)[0])
        return clean_z(r)

    c_norm = {c: norm_c(c) for c in df_raw.columns}
    time_c = next((c for c in df_raw.columns if "date/time" in c_norm[c]), None)
    out_c = next(
        (
            c
            for c in df_raw.columns
            if "site outdoor air drybulb temperature" in c_norm[c]
        ),
        None,
    )

    if out_c is None:
        raise ValueError("Cannot find outdoor temperature column.")

    z_map = {}
    for c in df_raw.columns:
        nc = c_norm[c]
        if "zone air temperature [c]" in nc:
            z_map.setdefault(extr(c, "temperature"), {})["temp"] = c
        elif "sensible heating" in nc:
            z_map.setdefault(extr(c, "heating"), {})["heat"] = c
        elif "sensible cooling" in nc:
            z_map.setdefault(extr(c, "cooling"), {})["cool"] = c
        elif "internal total heating" in nc:
            z_map.setdefault(extr(c, "internal"), {})["internal"] = c
        elif "zone windows total heat gain rate" in nc:
            z_map.setdefault(extr(c, "windows total heat gain"), {})["rad"] = c

    rows = {}
    if time_c is not None:
        out = []
        for s in df_raw[time_c].astype(str):
            m = re.match(
                r"^\s*(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2}):(\d{2})",
                s.strip(),
            )
            if m:
                mo, d, h, mi, sec = map(int, m.groups())
                if h == 24:
                    out.append(datetime(2023, mo, d, 0, mi, sec) + timedelta(days=1))
                else:
                    out.append(datetime(2023, mo, d, h, mi, sec))
            else:
                try:
                    if " 24:" in s:
                        out.append(
                            pd.to_datetime(
                                s.strip().split(" 24:")[0]
                                + " 00:"
                                + s.strip().split(" 24:")[1]
                            )
                            + pd.Timedelta(days=1)
                        )
                    else:
                        out.append(pd.to_datetime(s.strip()))
                except Exception:
                    out.append(pd.NaT)
        rows["datetime"] = pd.to_datetime(out)

    rows["T_out"] = pd.to_numeric(df_raw[out_c], errors="coerce").astype(float)
    zones = []

    for z, info in z_map.items():
        if "temp" not in info:
            continue
        rows[f"T_in__{z}"] = pd.to_numeric(
            df_raw[info["temp"]],
            errors="coerce",
        ).astype(float).values
        t_len = len(rows[f"T_in__{z}"])
        rows[f"Q_rad__{z}"] = (
            pd.to_numeric(df_raw[info["rad"]], errors="coerce").astype(float).values
            if "rad" in info else np.zeros(t_len)
        )
        zones.append(z)

    return pd.DataFrame(rows).fillna(0).reset_index(drop=True), zones


def parse_idf_context(idf_text: str, zones: List[str]):
    clean_to_info = {
        ultra_clean(z): {
            "real": z,
            "idx": i,
        }
        for i, z in enumerate(zones)
    }

    idf_clean = re.sub(r"!.*", "", idf_text)
    idf_clean = re.sub(r"[\r\n]+", " ", idf_clean)
    objects = idf_clean.split(";")
    raw_blocks = idf_text.split(";")
    parsed_blocks = []

    for block in raw_blocks:
        if any(
            keyword in block
            for keyword in [
                "BuildingSurface:Detailed",
                "Zone,",
                "FenestrationSurface:Detailed",
                "Material",
                "Construction",
                "WindowMaterial",
                "People",
                "Lights",
                "ElectricEquipment",
                "Schedule",
                "ZoneHVAC",
                "AirTerminal",
            ]
        ):
            bc = re.sub(
                r"(?m)^\s*[-\d\.Ee\+]+\s*[,;]\s*!-.*?(Coordinate|Vertex|X|Y|Z).*$",
                "",
                block,
            )
            parsed_blocks.append(
                os.linesep.join([s for s in bc.splitlines() if s.strip()]).strip()
                + ";"
            )

    surface_to_zone = {}
    internal_connections = []

    for obj in objects:
        obj = obj.strip()
        if not obj:
            continue

        fields = [field.strip() for field in obj.split(",")]
        if not fields:
            continue

        if fields[0].lower() == "buildingsurface:detailed" and len(fields) >= 5:
            surf_name = ultra_clean(fields[1])
            zone_name = ultra_clean(fields[4]) if len(fields) > 4 else ""
            outside_boundary = ultra_clean(fields[5]) if len(fields) > 5 else ""
            outside_boundary_object = ultra_clean(fields[6]) if len(fields) > 6 else ""

            surface_to_zone[surf_name] = zone_name

            if outside_boundary == "surface" and outside_boundary_object:
                internal_connections.append((surf_name, outside_boundary_object))

    valid_pairs_set = set()
    for src_surf, tgt_surf in internal_connections:
        src_z = surface_to_zone.get(src_surf)
        tgt_z = surface_to_zone.get(tgt_surf)

        if (
            src_z
            and tgt_z
            and src_z in clean_to_info
            and tgt_z in clean_to_info
            and src_z != tgt_z
        ):
            valid_pairs_set.add(
                tuple(
                    sorted(
                        (
                            clean_to_info[src_z]["real"],
                            clean_to_info[tgt_z]["real"],
                        )
                    )
                )
            )

    pair_confidence_map = {f"{a}||{b}": 1.0 for a, b in valid_pairs_set}
    return clean_to_info, parsed_blocks, valid_pairs_set, pair_confidence_map


def parse_idf_object_index(idf_text: str):
    raw_blocks = [blk.strip() + ";" for blk in idf_text.split(";") if blk.strip()]
    object_blocks = {}
    construction_layers = {}
    surface_records = []
    fenestration_records = []
    people_records = []
    lights_records = []
    electric_equipment_records = []
    schedule_blocks = {}

    for raw_block in raw_blocks:
        no_comments = re.sub(r"!.*", "", raw_block)
        flat = re.sub(r"[\r\n]+", " ", no_comments).strip()
        if not flat:
            continue

        fields = [field.strip() for field in flat.rstrip(";").split(",")]
        if len(fields) < 2:
            continue

        obj_type = fields[0].lower()
        obj_name = fields[1]
        object_blocks[(obj_type, ultra_clean(obj_name))] = raw_block

        if obj_type.startswith("schedule:"):
            schedule_blocks[ultra_clean(obj_name)] = raw_block

        if obj_type == "construction" and len(fields) >= 3:
            construction_layers[ultra_clean(obj_name)] = [
                field
                for field in fields[2:]
                if field.strip()
            ]

        if obj_type == "buildingsurface:detailed" and len(fields) >= 6:
            surface_records.append(
                {
                    "name": fields[1],
                    "surface_type": fields[2] if len(fields) > 2 else "",
                    "construction": fields[3] if len(fields) > 3 else "",
                    "zone": fields[4] if len(fields) > 4 else "",
                    "boundary": fields[5] if len(fields) > 5 else "",
                    "boundary_object": fields[6] if len(fields) > 6 else "",
                    "sun": fields[7] if len(fields) > 7 else "",
                    "wind": fields[8] if len(fields) > 8 else "",
                }
            )

        if obj_type == "fenestrationsurface:detailed" and len(fields) >= 5:
            fenestration_records.append(
                {
                    "name": fields[1],
                    "surface_type": fields[2] if len(fields) > 2 else "",
                    "construction": fields[3] if len(fields) > 3 else "",
                    "parent_surface": fields[4] if len(fields) > 4 else "",
                }
            )

        if obj_type == "people" and len(fields) >= 4:
            people_records.append(
                {
                    "name": fields[1],
                    "zone": fields[2] if len(fields) > 2 else "",
                    "number_schedule": fields[3] if len(fields) > 3 else "",
                    "activity_schedule": fields[7] if len(fields) > 7 else "",
                }
            )

        if obj_type == "lights" and len(fields) >= 4:
            lights_records.append(
                {
                    "name": fields[1],
                    "zone": fields[2] if len(fields) > 2 else "",
                    "schedule": fields[3] if len(fields) > 3 else "",
                }
            )

        if obj_type == "electricequipment" and len(fields) >= 4:
            electric_equipment_records.append(
                {
                    "name": fields[1],
                    "zone": fields[2] if len(fields) > 2 else "",
                    "schedule": fields[3] if len(fields) > 3 else "",
                }
            )

    return {
        "object_blocks": object_blocks,
        "construction_layers": construction_layers,
        "surface_records": surface_records,
        "fenestration_records": fenestration_records,
        "people_records": people_records,
        "lights_records": lights_records,
        "electric_equipment_records": electric_equipment_records,
        "schedule_blocks": schedule_blocks,
    }


def build_batch_thermal_context(
    idf_index: dict,
    batch_zones: List[str],
    max_chars: int,
) -> str:
    batch_clean = {ultra_clean(z) for z in batch_zones}
    object_blocks = idf_index["object_blocks"]
    construction_layers = idf_index["construction_layers"]
    surface_records = idf_index["surface_records"]
    fenestration_records = idf_index["fenestration_records"]

    related_surfaces = [
        rec
        for rec in surface_records
        if ultra_clean(rec.get("zone", "")) in batch_clean
    ]
    related_surface_names = {
        ultra_clean(rec.get("name", ""))
        for rec in related_surfaces
    }
    related_fenestration = [
        rec
        for rec in fenestration_records
        if ultra_clean(rec.get("parent_surface", "")) in related_surface_names
    ]
    construction_names = {
        ultra_clean(rec.get("construction", ""))
        for rec in related_surfaces + related_fenestration
        if rec.get("construction")
    }

    material_names = set()
    for construction_name in construction_names:
        for layer_name in construction_layers.get(construction_name, []):
            material_names.add(ultra_clean(layer_name))

    lines = []
    lines.append("THERMAL_INERTIA_EVIDENCE")
    lines.append(
        "Use this evidence for Thermal_Inertia. High inertia usually means thick/dense/high-specific-heat layers "
        "such as concrete, brick, heavyweight mass, or multiple massive opaque layers. Low inertia usually means "
        "lightweight/no-mass/insulation-dominated/glazing-dominated constructions."
    )

    lines.append("\nBATCH_ZONE_SURFACES")
    for rec in related_surfaces:
        lines.append(
            "- Zone={zone}; Surface={name}; Type={surface_type}; Construction={construction}; "
            "Boundary={boundary}; Sun={sun}; Wind={wind}".format(**rec)
        )

    if related_fenestration:
        lines.append("\nBATCH_FENESTRATION")
        for rec in related_fenestration:
            lines.append(
                "- ParentSurface={parent_surface}; Window={name}; Type={surface_type}; Construction={construction}".format(**rec)
            )

    lines.append("\nRELATED_CONSTRUCTIONS")
    for construction_name in sorted(construction_names):
        block = object_blocks.get(("construction", construction_name))
        if block:
            lines.append(block)

    lines.append("\nRELATED_MATERIALS")
    material_type_prefixes = [
        "material",
        "material:nomass",
        "material:airgap",
        "windowmaterial",
        "windowmaterial:simpleglazingsystem",
        "windowmaterial:glazing",
        "windowmaterial:gas",
        "windowmaterial:shade",
        "windowmaterial:blind",
    ]
    for material_name in sorted(material_names):
        for (obj_type, obj_name), block in object_blocks.items():
            if obj_name != material_name:
                continue
            if obj_type == "material" or any(obj_type.startswith(prefix) for prefix in material_type_prefixes):
                lines.append(block)
                break

    context = "\n".join(lines)
    return context[:max_chars]


def build_batch_schedule_context(
    idf_index: dict,
    batch_zones: List[str],
    max_chars: int,
) -> str:
    batch_clean = {ultra_clean(z) for z in batch_zones}
    people_records = idf_index["people_records"]
    lights_records = idf_index["lights_records"]
    electric_equipment_records = idf_index["electric_equipment_records"]
    schedule_blocks = idf_index["schedule_blocks"]

    related_people = [
        rec
        for rec in people_records
        if ultra_clean(rec.get("zone", "")) in batch_clean
    ]
    related_lights = [
        rec
        for rec in lights_records
        if ultra_clean(rec.get("zone", "")) in batch_clean
    ]
    related_equipment = [
        rec
        for rec in electric_equipment_records
        if ultra_clean(rec.get("zone", "")) in batch_clean
    ]

    schedule_names = set()
    for rec in related_people:
        for key in ["number_schedule", "activity_schedule"]:
            if rec.get(key):
                schedule_names.add(ultra_clean(rec[key]))
    for rec in related_lights + related_equipment:
        if rec.get("schedule"):
            schedule_names.add(ultra_clean(rec["schedule"]))

    expanded_schedule_names = set(schedule_names)
    for schedule_name in list(schedule_names):
        block = schedule_blocks.get(schedule_name, "")
        clean_block = ultra_clean(block)
        for candidate_name in schedule_blocks:
            if candidate_name != schedule_name and candidate_name in clean_block:
                expanded_schedule_names.add(candidate_name)

    lines = []
    lines.append("OCCUPANCY_AND_INTERNAL_GAIN_EVIDENCE")
    lines.append(
        "Use this evidence for Occupancy_Volatility and Base_Physics_Trust. "
        "Schedules with strong hour-to-hour or day/night changes imply higher volatility. "
        "Flat, always-on, storage/mechanical, or low-occupancy schedules imply lower volatility. "
        "Highly stochastic human-driven zones should generally have lower Base_Physics_Trust than stable service/storage zones."
    )

    lines.append("\nBATCH_PEOPLE_OBJECTS")
    if related_people:
        for rec in related_people:
            lines.append(
                "- Zone={zone}; People={name}; NumberSchedule={number_schedule}; ActivitySchedule={activity_schedule}".format(**rec)
            )
    else:
        lines.append("- No People objects directly matched to this batch.")

    lines.append("\nBATCH_LIGHTS_OBJECTS")
    if related_lights:
        for rec in related_lights:
            lines.append(
                "- Zone={zone}; Lights={name}; Schedule={schedule}".format(**rec)
            )
    else:
        lines.append("- No Lights objects directly matched to this batch.")

    lines.append("\nBATCH_ELECTRIC_EQUIPMENT_OBJECTS")
    if related_equipment:
        for rec in related_equipment:
            lines.append(
                "- Zone={zone}; ElectricEquipment={name}; Schedule={schedule}".format(**rec)
            )
    else:
        lines.append("- No ElectricEquipment objects directly matched to this batch.")

    lines.append("\nRELATED_SCHEDULE_BLOCKS")
    if expanded_schedule_names:
        for schedule_name in sorted(expanded_schedule_names):
            block = schedule_blocks.get(schedule_name)
            if block:
                lines.append(block)
    else:
        lines.append("- No directly referenced schedule blocks found for this batch.")

    context = "\n".join(lines)
    return context[:max_chars]


def clean_idf_fields(raw_block: str) -> List[str]:
    cleaned_lines = [line.split("!", 1)[0] for line in raw_block.splitlines()]
    cleaned = "\n".join(cleaned_lines).replace(";", ",")
    return [field.strip() for field in cleaned.split(",") if field.strip()]


def idf_name_key(name: str) -> str:
    return ultra_clean(name)


def time_token_to_hour(token: str) -> Optional[float]:
    match = re.match(r"Until:\s*(\d{1,2}):(\d{2})", str(token), re.I)
    if not match:
        return None
    hour = int(match.group(1)) + int(match.group(2)) / 60.0
    return min(24.0, max(0.0, hour))


def token_to_float(token) -> Optional[float]:
    try:
        return float(token)
    except (TypeError, ValueError):
        return None


def schedule_fields_to_daily_profile(fields: List[str], steps_per_hour: int = 4) -> Optional[List[float]]:
    if len(fields) < 4:
        return None

    segments = []
    current_for = "AllDays"
    current_points = []
    i = 3
    while i < len(fields):
        field = fields[i]
        lower = field.lower()

        if lower.startswith("for:"):
            if current_points:
                segments.append((current_for, current_points))
            current_for = field.split(":", 1)[1].strip() if ":" in field else field
            current_points = []
            i += 1
            continue

        hour = time_token_to_hour(field)
        if hour is not None and i + 1 < len(fields):
            value = token_to_float(fields[i + 1])
            if value is not None:
                current_points.append((hour, value))
            i += 2
            continue

        i += 1

    if current_points:
        segments.append((current_for, current_points))

    if not segments:
        return None

    chosen_points = None
    for label, points in segments:
        normalized = ultra_clean(label)
        if normalized in {"alldays", "allotherdays", "weekdays"}:
            chosen_points = points
            break
    if chosen_points is None:
        chosen_points = segments[0][1]

    n_steps = 24 * steps_per_hour
    profile = np.zeros(n_steps, dtype=float)
    last_step = 0
    last_value = chosen_points[0][1]

    for until_hour, value in chosen_points:
        step = int(round(until_hour * steps_per_hour))
        step = min(n_steps, max(last_step, step))
        profile[last_step:step] = value
        last_step = step
        last_value = value

    if last_step < n_steps:
        profile[last_step:] = last_value

    return profile.tolist()


def constant_schedule_profile(fields: List[str], steps_per_hour: int = 4) -> Optional[List[float]]:
    for field in fields[2:]:
        value = token_to_float(field)
        if value is not None:
            return [value] * (24 * steps_per_hour)
    return None


def schedule_profile_from_fields(fields: List[str], steps_per_hour: int = 4) -> Optional[List[float]]:
    if not fields:
        return None
    obj_type = fields[0].lower()
    if obj_type == "schedule:compact":
        return schedule_fields_to_daily_profile(fields, steps_per_hour=steps_per_hour)
    if obj_type == "schedule:constant":
        return constant_schedule_profile(fields, steps_per_hour=steps_per_hour)
    return schedule_fields_to_daily_profile(fields, steps_per_hour=steps_per_hour)


def parse_idf_control_index(idf_text: str, steps_per_hour: int = 4) -> dict:
    raw_blocks = [blk.strip() + ";" for blk in idf_text.split(";") if blk.strip()]
    zone_lists = {}
    zone_controls = {}
    dual_setpoints = {}
    schedules = {}

    for raw_block in raw_blocks:
        fields = clean_idf_fields(raw_block)
        if len(fields) < 2:
            continue

        obj_type = fields[0].lower()
        obj_name = fields[1]

        if obj_type == "zonelist" and len(fields) >= 3:
            zone_lists[idf_name_key(obj_name)] = [z for z in fields[2:] if z]

        elif obj_type == "zonecontrol:thermostat" and len(fields) >= 6:
            control_record = {
                "thermostat": fields[1],
                "zone_or_zonelist": fields[2],
                "control_type_schedule": fields[3],
                "setpoint_object_type": fields[4],
                "setpoint_object": fields[5],
            }
            target_name = fields[2]
            expanded_zones = zone_lists.get(idf_name_key(target_name), [target_name])
            for zone in expanded_zones:
                rec = dict(control_record)
                rec["zone"] = zone
                zone_controls[idf_name_key(zone)] = rec

        elif obj_type == "thermostatsetpoint:dualsetpoint" and len(fields) >= 4:
            dual_setpoints[idf_name_key(obj_name)] = {
                "name": obj_name,
                "heating_schedule": fields[2],
                "cooling_schedule": fields[3],
            }

        elif obj_type.startswith("schedule:"):
            schedules[idf_name_key(obj_name)] = {
                "name": obj_name,
                "object_type": obj_type,
                "schedule_type_limits": fields[2] if len(fields) > 2 else "",
                "fields": fields,
                "profile": schedule_profile_from_fields(fields, steps_per_hour=steps_per_hour),
            }

    for zone_key, rec in zone_controls.items():
        sp = dual_setpoints.get(idf_name_key(rec.get("setpoint_object", "")))
        if sp:
            rec.update(sp)

    return {
        "zone_controls": zone_controls,
        "dual_setpoints": dual_setpoints,
        "schedules": schedules,
        "zone_lists": zone_lists,
        "steps_per_hour": steps_per_hour,
    }


def get_schedule_profile(control_index: dict, schedule_name: Optional[str]) -> Optional[np.ndarray]:
    if not schedule_name:
        return None
    rec = control_index.get("schedules", {}).get(idf_name_key(schedule_name))
    if not rec:
        return None
    profile = rec.get("profile")
    if profile is None:
        return None
    return np.array(profile, dtype=float)


def profile_stats(profile: Optional[np.ndarray]) -> dict:
    if profile is None or len(profile) == 0:
        return {
            "mean": None,
            "min": None,
            "max": None,
            "range": None,
            "switch_count": 0,
        }
    diffs = np.diff(profile)
    return {
        "mean": float(np.mean(profile)),
        "min": float(np.min(profile)),
        "max": float(np.max(profile)),
        "range": float(np.max(profile) - np.min(profile)),
        "switch_count": int(np.sum(np.abs(diffs) > 1e-9)),
    }


def normalize_temperature_mean(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return float(np.clip((value - 10.0) / 25.0, 0.0, 1.0))


def normalize_temperature_delta(value: Optional[float], scale: float = 10.0) -> float:
    if value is None:
        return 0.0
    return float(np.clip(abs(value) / scale, 0.0, 1.0))


def mean_abs_delta(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None or len(a) != len(b):
        return 0.0
    return float(np.mean(np.abs(a - b)))


def build_control_prior_for_zone(
    zone: str,
    target_index: dict,
    reference_index: Optional[dict] = None,
) -> Tuple[List[float], dict]:
    zone_key = idf_name_key(zone)
    target_control = target_index.get("zone_controls", {}).get(zone_key, {})
    reference_control = (
        reference_index.get("zone_controls", {}).get(zone_key, {})
        if reference_index is not None else {}
    )

    cooling_schedule = target_control.get("cooling_schedule")
    heating_schedule = target_control.get("heating_schedule")
    ref_cooling_schedule = reference_control.get("cooling_schedule")
    ref_heating_schedule = reference_control.get("heating_schedule")

    cooling_profile = get_schedule_profile(target_index, cooling_schedule)
    heating_profile = get_schedule_profile(target_index, heating_schedule)
    ref_cooling_profile = (
        get_schedule_profile(reference_index, ref_cooling_schedule)
        if reference_index is not None else None
    )
    ref_heating_profile = (
        get_schedule_profile(reference_index, ref_heating_schedule)
        if reference_index is not None else None
    )

    cooling_stats = profile_stats(cooling_profile)
    heating_stats = profile_stats(heating_profile)

    cooling_shift = mean_abs_delta(cooling_profile, ref_cooling_profile)
    heating_shift = mean_abs_delta(heating_profile, ref_heating_profile)
    schedule_distance = 0.5 * (cooling_shift + heating_shift)
    changed_flag = float(schedule_distance > 1e-9)

    switch_count = cooling_stats["switch_count"] + heating_stats["switch_count"]
    comfort_band_mean = 0.0
    if cooling_profile is not None and heating_profile is not None and len(cooling_profile) == len(heating_profile):
        comfort_band_mean = float(np.mean(np.maximum(cooling_profile - heating_profile, 0.0)))

    vector = [
        normalize_temperature_mean(cooling_stats["mean"]),
        normalize_temperature_delta(cooling_stats["range"]),
        normalize_temperature_mean(heating_stats["mean"]),
        normalize_temperature_delta(heating_stats["range"]),
        float(np.clip(switch_count / 16.0, 0.0, 1.0)),
        normalize_temperature_delta(comfort_band_mean),
        normalize_temperature_delta(cooling_shift),
        normalize_temperature_delta(heating_shift),
        normalize_temperature_delta(schedule_distance),
        changed_flag,
    ]

    record = {
        "zone": zone,
        "thermostat": target_control.get("thermostat"),
        "control_type_schedule": target_control.get("control_type_schedule"),
        "setpoint_object": target_control.get("setpoint_object"),
        "heating_schedule": heating_schedule,
        "cooling_schedule": cooling_schedule,
        "reference_heating_schedule": ref_heating_schedule,
        "reference_cooling_schedule": ref_cooling_schedule,
        "cooling_stats": cooling_stats,
        "heating_stats": heating_stats,
        "cooling_shift_mean_abs_c": cooling_shift,
        "heating_shift_mean_abs_c": heating_shift,
        "schedule_distance_mean_abs_c": schedule_distance,
        "control_changed_flag": bool(changed_flag),
        "control_vector": vector,
    }

    if cooling_profile is not None:
        record["cooling_profile_24h"] = cooling_profile.tolist()
    if heating_profile is not None:
        record["heating_profile_24h"] = heating_profile.tolist()
    if ref_cooling_profile is not None:
        record["reference_cooling_profile_24h"] = ref_cooling_profile.tolist()
    if ref_heating_profile is not None:
        record["reference_heating_profile_24h"] = ref_heating_profile.tolist()

    return vector, record


def extract_control_priors(
    idf_text: str,
    zones: List[str],
    reference_idf_text: Optional[str] = None,
    steps_per_hour: int = 4,
) -> dict:
    target_index = parse_idf_control_index(idf_text, steps_per_hour=steps_per_hour)
    reference_index = (
        parse_idf_control_index(reference_idf_text, steps_per_hour=steps_per_hour)
        if reference_idf_text is not None else None
    )

    matrix = []
    records = {}
    for zone in zones:
        vector, record = build_control_prior_for_zone(
            zone=zone,
            target_index=target_index,
            reference_index=reference_index,
        )
        matrix.append(vector)
        records[zone] = record

    changed_zones = [
        zone for zone, record in records.items()
        if record.get("control_changed_flag")
    ]
    return {
        "control_matrix": matrix,
        "control_names": CONTROL_NAMES,
        "control_records": records,
        "control_summary": {
            "has_reference_idf": reference_idf_text is not None,
            "steps_per_hour": steps_per_hour,
            "n_zones": len(zones),
            "n_control_changed_zones": len(changed_zones),
            "changed_zones": changed_zones,
        },
    }


def calibration_value(value, default: float = 0.5) -> float:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = normalize_label(value)
        if normalized is not None:
            return LABEL_TO_VALUE[normalized]
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except Exception:
        return default


def default_llm_control_calibration_record(zone: str) -> dict:
    return {
        "zone": zone,
        "shift_taxonomy": "unchanged_or_uncertain",
        "adaptation_operator": "weak_or_uncertain",
        "operator_confidence": "medium",
        "dominant_regime": "deadband_coordinate",
        "secondary_regime": "identity_preserving",
        "regime_confidence": "medium",
        "oscillation_preservation": "medium",
        "response_sensitivity": "medium",
        "center_shift_weight": "medium",
        "band_mapping_weight": "medium",
        "heating_boundary_weight": "low",
        "cooling_boundary_weight": "low",
        "active_window_weight": "medium_high",
        "control_calibration_trust": "medium",
        "thermal_response_regime": "mixed_response",
        "setpoint_tracking_gain": "medium",
        "baseline_shift_strength": "medium",
        "load_coupling": "medium",
        "thermal_memory": "medium",
        "correction_shape": "center_shift_with_memory",
        "calibration_mode": "parser_default",
        "expected_temp_direction": "uncertain",
        "reason": "Default calibration because LLM control calibration was unavailable.",
    }


def llm_control_item_to_vector(item: dict) -> List[float]:
    return [
        calibration_value(item.get("response_sensitivity"), 0.5),
        calibration_value(item.get("center_shift_weight"), 0.5),
        calibration_value(item.get("band_mapping_weight"), 0.5),
        calibration_value(item.get("heating_boundary_weight"), 0.3),
        calibration_value(item.get("cooling_boundary_weight"), 0.3),
        calibration_value(item.get("active_window_weight"), 0.7),
        calibration_value(item.get("control_calibration_trust"), 0.5),
        calibration_value(item.get("setpoint_tracking_gain"), 0.5),
        calibration_value(item.get("baseline_shift_strength"), 0.5),
        calibration_value(item.get("load_coupling"), 0.5),
        calibration_value(item.get("thermal_memory"), 0.5),
    ]


def compact_profile_summary(values, max_points: int = 12):
    if values is None:
        return None
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return None
    if arr.size <= max_points:
        return [round(float(x), 3) for x in arr.tolist()]
    idx = np.linspace(0, arr.size - 1, max_points).round().astype(int)
    return [round(float(arr[i]), 3) for i in idx]


def build_llm_control_calibration_prompt(batch_zones: List[str], control_prior: dict, semantic_lookup: dict) -> Tuple[str, str, str]:
    label_rule = (
        'Use only these labels for weights: "very_low", "low", "medium", "medium_high", "high". '
        "Return strictly valid JSON. Do not output markdown or explanations."
    )
    records = control_prior.get("control_records", {})
    evidence = []
    for zone in batch_zones:
        rec = records.get(zone, {})
        evidence.append(
            {
                "zone": zone,
                "heating_schedule": rec.get("heating_schedule"),
                "cooling_schedule": rec.get("cooling_schedule"),
                "reference_heating_schedule": rec.get("reference_heating_schedule"),
                "reference_cooling_schedule": rec.get("reference_cooling_schedule"),
                "heating_shift_mean_abs_c": rec.get("heating_shift_mean_abs_c"),
                "cooling_shift_mean_abs_c": rec.get("cooling_shift_mean_abs_c"),
                "schedule_distance_mean_abs_c": rec.get("schedule_distance_mean_abs_c"),
                "control_changed_flag": rec.get("control_changed_flag"),
                "old_heating_profile_sample": compact_profile_summary(rec.get("reference_heating_profile_24h")),
                "old_cooling_profile_sample": compact_profile_summary(rec.get("reference_cooling_profile_24h")),
                "new_heating_profile_sample": compact_profile_summary(rec.get("heating_profile_24h")),
                "new_cooling_profile_sample": compact_profile_summary(rec.get("cooling_profile_24h")),
                "physical_semantics": semantic_lookup.get(zone),
            }
        )

    system_prompt = f"""You are an HVAC thermostat regime specialist.

Task: infer how a zone temperature trajectory should be constrained by a new thermostat regime under old-to-new IDF heating/cooling setpoint changes.

The deterministic parser already extracted exact old/new heating and cooling setpoint profiles.
Do NOT predict temperatures.
Do NOT invent schedule values.
Do NOT output arbitrary formulas.
You may only select regime labels and adaptation operators from the allowed lists below.

Definitions:
- H = heating setpoint, G = cooling setpoint, C = (H + G) / 2, B = G - H.
- T is the model-predicted zone temperature trained under the old IDF.

Allowed shift_taxonomy values:
- unchanged_or_uncertain
- deadband_translation_up
- deadband_translation_down
- deadband_widening
- deadband_narrowing
- heating_threshold_shift
- cooling_threshold_shift
- phase_or_occupancy_window_shift
- mixed_control_shift

Allowed thermostat regime values:
- heating_boundary_constrained: temperature is mainly lifted by the lower heating setpoint boundary.
- cooling_boundary_constrained: temperature is mainly capped by the upper cooling setpoint boundary.
- deadband_coordinate: temperature should preserve its relative coordinate inside the thermostat deadband.
- center_shift_tracking: temperature mainly follows the deadband center shift.
- free_floating: temperature is weakly controlled and should be corrected conservatively.
- oscillatory_boundary: temperature oscillates near a thermostat boundary and oscillation amplitude should be preserved.
- identity_preserving: leave most of the model prediction unchanged.
- uncertain: insufficient evidence.

Allowed adaptation_operator values:
- center_translation: T' = T + alpha * (C_new - C_old)
- deadband_coordinate_mapping: T' = C_new + ((T - C_old) / B_old) * B_new
- heating_boundary_push: T' = T + alpha * relu(H_new - T)
- cooling_boundary_push: T' = T - alpha * relu(T - G_new)
- mixed_deadband_mapping: blend T with deadband_coordinate_mapping
- weak_or_uncertain: do not apply strong control correction

Allowed thermal_response_regime values:
- setpoint_tracking: zone temperature should move with the old-to-new thermostat center shift.
- boundary_lift: the new heating lower bound is likely to lift the temperature trajectory.
- boundary_cap: the new cooling upper bound is likely to cap or lower the temperature trajectory.
- load_amplified: schedule/control changes likely interact with occupancy or internal loads, producing larger temperature shifts than pure setpoint translation.
- inertia_buffered: high thermal mass or slow HVAC response should damp and delay the correction.
- weak_response: thermostat changes are unlikely to strongly change the zone temperature.
- mixed_response: multiple mechanisms are plausible.

Allowed correction_shape values:
- center_shift
- boundary_projection
- deadband_coordinate
- center_shift_with_memory
- load_amplified_shift
- conservative_identity

Output fields per zone:
- shift_taxonomy: one value from the allowed taxonomy.
- dominant_regime: one value from the allowed thermostat regime values.
- secondary_regime: one value from the allowed thermostat regime values.
- regime_confidence: confidence that the regime assignment is correct.
- oscillation_preservation: whether oscillatory deviations around the boundary should be preserved.
- adaptation_operator: one value from the allowed operator library.
- operator_confidence: confidence that this operator is the right physical mapping.
- response_sensitivity: how strongly this zone temperature should respond to thermostat schedule changes.
- center_shift_weight: how much to apply new_center - old_center translation.
- band_mapping_weight: how much to map old deadband coordinate into new deadband coordinate.
- heating_boundary_weight: how much to softly lift predictions below new heating setpoint.
- cooling_boundary_weight: how much to softly lower predictions above new cooling setpoint.
- active_window_weight: how much calibration should concentrate on changed schedule windows.
- control_calibration_trust: confidence that thermostat-based post-calibration is appropriate.
- thermal_response_regime: one value from the allowed thermal_response_regime values.
- setpoint_tracking_gain: expected fraction of old-to-new deadband center shift visible in zone temperature.
- baseline_shift_strength: expected persistent baseline shift outside instantaneous schedule edges.
- load_coupling: expected amplification due to occupancy/internal-gain coupling rather than thermostat alone.
- thermal_memory: expected temporal persistence after a changed setpoint window.
- correction_shape: one value from the allowed correction_shape values.
- calibration_mode: one of center_shift, deadband_mapping, boundary_correction, mixed, weak_or_uncertain.
- expected_temp_direction: one of upward, downward, mixed, uncertain.
- reason: short evidence-based reason, max 24 words.

Guidance:
- Prefer regime labels over numerical tuning. The downstream model uses these labels to project predictions to heating-boundary, cooling-boundary, deadband, center-shift, or identity manifolds.
- If the old/new setpoints create a new higher lower bound and the zone is likely held up by heating control, use heating_boundary_constrained.
- If the new cooling setpoint is the primary active upper bound, use cooling_boundary_constrained.
- If both H and G move and the zone should keep a similar relative position between H and G, use deadband_coordinate.
- If the entire admissible band translates upward/downward without strong boundary evidence, use center_shift_tracking.
- If zone semantics suggest weak HVAC response or high inertia, use free_floating or identity_preserving as secondary_regime.
- If the zone likely cycles around a control boundary, use oscillatory_boundary and high oscillation_preservation.
- If both heating and cooling setpoints move upward with similar magnitude, use deadband_translation_up and center_translation.
- If both setpoints move and the comfort band changes substantially, use deadband_coordinate_mapping or mixed_deadband_mapping.
- If only heating setpoint changes, use heating_threshold_shift and heating_boundary_push.
- If only cooling setpoint changes, use cooling_threshold_shift and cooling_boundary_push or center_translation depending on direction.
- If changed windows move in time, use phase_or_occupancy_window_shift and weak_or_uncertain unless the setpoint mapping is still clear.
- If schedules barely change or zone sensitivity/trust is low, use low weights.
- Use physical semantics to reduce sensitivity for high-inertia or weakly controlled zones and increase it for schedule-driven zones.
- For kitchens, cafes, banquet rooms, classrooms, clean rooms, operating or medical spaces, reason about whether internal gains, ventilation, or strict control can amplify the visible temperature response.
- If old/new setpoint centers move upward and the zone is likely actively conditioned, prefer setpoint_tracking or boundary_lift over weak_response.
- If the zone has high Thermal_Inertia, use higher thermal_memory but lower setpoint_tracking_gain.
- If Occupancy_Volatility or internal gains are likely high, use load_amplified and medium_high/high load_coupling.
{label_rule}

Output schema:
{{"control_calibration": [{{"zone": "ZoneName", "shift_taxonomy": "deadband_translation_up", "dominant_regime": "heating_boundary_constrained", "secondary_regime": "deadband_coordinate", "regime_confidence": "label", "oscillation_preservation": "label", "adaptation_operator": "heating_boundary_push", "operator_confidence": "label", "response_sensitivity": "label", "center_shift_weight": "label", "band_mapping_weight": "label", "heating_boundary_weight": "label", "cooling_boundary_weight": "label", "active_window_weight": "label", "control_calibration_trust": "label", "thermal_response_regime": "setpoint_tracking", "setpoint_tracking_gain": "label", "baseline_shift_strength": "label", "load_coupling": "label", "thermal_memory": "label", "correction_shape": "center_shift_with_memory", "calibration_mode": "mixed", "expected_temp_direction": "upward", "reason": "short reason"}}]}}"""

    user_prompt = "Zone control-shift evidence:\n" + json.dumps(evidence, indent=2)
    return ("control_calibration", system_prompt, user_prompt)


def extract_llm_control_calibration_priors(
    control_prior: dict,
    zones: List[str],
    semantic_matrix: np.ndarray,
    client: Together,
    model: str,
    temperature: float,
    seed: Optional[int],
    batch_size: int,
    max_tokens: int,
    sleep_s: float,
    raw_output_dir: Optional[str],
    run_idx: int,
) -> dict:
    semantic_lookup = {}
    for z_idx, zone in enumerate(zones):
        vals = semantic_matrix[z_idx].tolist() if z_idx < len(semantic_matrix) else [0.5] * len(SEMANTIC_NAMES)
        semantic_lookup[zone] = {
            name: round(float(vals[i]), 3)
            for i, name in enumerate(SEMANTIC_NAMES[:len(vals)])
        }

    records = {
        zone: default_llm_control_calibration_record(zone)
        for zone in zones
    }
    changed_zones = [
        zone for zone in zones
        if control_prior.get("control_records", {}).get(zone, {}).get("control_changed_flag")
    ]

    if not changed_zones:
        matrix = [llm_control_item_to_vector(records[z]) for z in zones]
        return {
            "llm_control_calibration_matrix": matrix,
            "llm_control_calibration_names": LLM_CONTROL_CALIBRATION_NAMES,
            "llm_control_calibration_records": records,
            "llm_control_calibration_summary": {
                "n_llm_calibrated_zones": 0,
                "mode": "no_changed_control_zones",
            },
        }

    n_batches = int(np.ceil(len(changed_zones) / batch_size))
    for i in range(0, len(changed_zones), batch_size):
        batch_zones = changed_zones[i:i + batch_size]
        batch_idx = i // batch_size + 1
        agent_name, system_prompt, user_prompt = build_llm_control_calibration_prompt(
            batch_zones=batch_zones,
            control_prior=control_prior,
            semantic_lookup=semantic_lookup,
        )
        print(f"     [{agent_name} agent] extracting control calibration {batch_idx}/{n_batches}...")
        result = call_llm_json(
            client=client,
            model=model,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_output_dir=raw_output_dir,
            run_idx=run_idx,
            batch_idx=batch_idx,
            agent_name=agent_name,
        )

        items = result.get("control_calibration", [])
        if isinstance(items, dict):
            items = [
                {"zone": zone, **payload}
                if isinstance(payload, dict) else {"zone": zone}
                for zone, payload in items.items()
            ]
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                matched_zone = match_zone_name(item.get("zone", ""), batch_zones)
                if matched_zone is not None:
                    merged = default_llm_control_calibration_record(matched_zone)
                    merged.update(item)
                    merged["zone"] = matched_zone
                    records[matched_zone] = merged

        if sleep_s > 0:
            time.sleep(sleep_s)

    matrix = [llm_control_item_to_vector(records[z]) for z in zones]
    return {
        "llm_control_calibration_matrix": matrix,
        "llm_control_calibration_names": LLM_CONTROL_CALIBRATION_NAMES,
        "llm_control_calibration_records": records,
        "llm_control_calibration_summary": {
            "n_llm_calibrated_zones": len(changed_zones),
            "mode": "llm_guided_zone_calibration",
        },
    }


def _balanced_json_candidates(text: str) -> List[str]:
    candidates = []
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : idx + 1])
                    break
    return candidates


def parse_llm_json(raw_content: str):
    raw_content = raw_content.strip()
    raw_content = re.sub(r"^```json\s*", "", raw_content)
    raw_content = re.sub(r"^```\s*", "", raw_content)
    raw_content = re.sub(r"\s*```$", "", raw_content)

    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        pass

    best_error = None
    for candidate in _balanced_json_candidates(raw_content):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            best_error = e

    match = re.search(r"\{.*\}", raw_content, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            best_error = e

    if best_error is not None:
        raise best_error
    raise json.JSONDecodeError("No JSON object found", raw_content, 0)


def match_zone_name(z: str, candidate_zones: List[str]) -> Optional[str]:
    if not z:
        return None

    if z in candidate_zones:
        return z

    uz = ultra_clean(z)
    for real_z in candidate_zones:
        ur = ultra_clean(real_z)
        if uz == ur or uz in ur or ur in uz:
            return real_z

    return None


def call_llm_json(
    client: Together,
    model: str,
    temperature: float,
    seed: Optional[int],
    max_tokens: int,
    system_prompt: str,
    user_prompt: str,
    raw_output_dir: Optional[str],
    run_idx: int,
    batch_idx: int,
    agent_name: str,
    allow_empty_fallback: bool = True,
) -> dict:
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        kwargs["seed"] = seed

    resp = client.chat.completions.create(**kwargs)
    raw_content = resp.choices[0].message.content.strip()

    try:
        return parse_llm_json(raw_content)
    except json.JSONDecodeError as e:
        print(f"     [{agent_name}] JSON broken.")
        print(f"     [{agent_name}] JSON error: {e}")

        if raw_output_dir:
            os.makedirs(raw_output_dir, exist_ok=True)
            raw_path = os.path.join(
                raw_output_dir,
                f"bad_json_{agent_name}_run_{run_idx:03d}_batch_{batch_idx:03d}.txt",
            )
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(raw_content)
            print(f"     [{agent_name}] Raw output saved: {raw_path}")

        if not allow_empty_fallback:
            raise
        return {"semantics": []}


def parse_agent_label_items(llm_result: dict, batch_zones: List[str]) -> Dict[str, dict]:
    out = {}
    semantics = llm_result.get("semantics", [])

    if isinstance(semantics, dict):
        semantics = [
            {"zone": zone, **payload}
            if isinstance(payload, dict)
            else {"zone": zone, "label": payload}
            for zone, payload in semantics.items()
        ]

    if not isinstance(semantics, list):
        return out

    for item in semantics:
        if not isinstance(item, dict):
            continue

        matched_zone = match_zone_name(item.get("zone", ""), batch_zones)
        if matched_zone is not None:
            out[matched_zone] = item

    return out


def extract_agent_reason_trace(
    llm_result: dict,
    batch_zones: List[str],
    agent_name: str,
    run_idx: int,
    batch_idx: int,
) -> List[dict]:
    traces = []
    agent_items = parse_agent_label_items(llm_result, batch_zones)

    for zone, item in agent_items.items():
        trace = {
            "run": run_idx,
            "batch": batch_idx,
            "agent": agent_name,
            "zone": zone,
            "basis": item.get("basis"),
            "reason": item.get("reason") or item.get("concise_reason"),
            "confidence": item.get("confidence"),
        }

        for key in [
            "thermal_inertia",
            "env_exposure",
            "env_sensitivity",
            "occupancy_volatility",
            "base_physics_trust",
        ]:
            if key in item:
                trace[key] = item.get(key)

        traces.append(trace)

    return traces


def build_connection_agent_prompt(
    local_edges: List[str],
    thermal_context: str,
    local_snippet: str,
) -> Tuple[str, str, str]:
    label_rule = (
        'Use only these labels: "very_low", "low", "medium", "medium_high", "high". '
        "Return strictly valid JSON. Do not output markdown or explanations."
    )

    connection_system = f"""You are a building thermal-topology specialist.

Task: judge ONLY thermal-coupling confidence for the provided zone-zone edges.

The edge list was parsed deterministically from IDF surface adjacency.
Do NOT invent new edges.
Do NOT remove edges.
Only score the provided edges.

Meaning:
- high: the adjacency likely creates meaningful thermal interaction between zone air temperatures.
- medium_high: clear adjacency with likely coupling, but not enough evidence for strongest confidence.
- medium: adjacency exists but coupling strength is uncertain or ordinary.
- low: adjacency exists geometrically but thermal coupling is likely weak, buffered, or uncertain.
- very_low: adjacency is probably weakly relevant for thermal dynamics despite geometric contact.

Use shared-wall/internal-boundary evidence, surface/construction context, zone function, and ambiguity.
If evidence is insufficient, use medium instead of extreme labels.
{label_rule}

Output schema:
{{"connections": [{{"zone_a": "ZoneA", "zone_b": "ZoneB", "confidence": "label", "basis": "shared_internal_wall|large_shared_boundary|buffered_or_indirect|uncertain", "reason": "short evidence-based reason, max 20 words"}}]}}"""

    connection_user = (
        "Candidate edges to score:\n"
        + json.dumps(local_edges, indent=2)
        + "\n\nThermal/Envelope Context:\n"
        + thermal_context
        + "\n\nOther Local Snippet:\n"
        + local_snippet
    )

    return ("connection", connection_system, connection_user)


def build_connection_discovery_agent_prompt(
    batch_zones: List[str],
    thermal_context: str,
    local_snippet: str,
) -> Tuple[str, str, str]:
    label_rule = (
        'Use only these labels: "very_low", "low", "medium", "medium_high", "high". '
        "Return strictly valid JSON. Do not output markdown or explanations."
    )

    connection_system = f"""You are a building thermal-topology specialist.

Task: infer a sparse zone-zone thermal-coupling graph because explicit Surface-to-Surface adjacency was not available from the IDF parser.

This is NOT a request for a dense graph.
Only propose zone pairs that are likely to have direct or meaningful thermal interaction.
Prefer same-floor, physically adjacent, shared-boundary, core/perimeter, corridor-room, or functionally neighboring zones when evidence supports them.
Do not connect every zone to every other zone.
If evidence is weak, return fewer edges with lower confidence.
As a calibration target, propose at most 2 to 3 meaningful edges per zone unless the evidence clearly supports more.

Meaning of confidence:
- high: strong evidence of meaningful direct thermal coupling.
- medium_high: plausible direct adjacency or strong neighboring relation.
- medium: plausible but uncertain coupling.
- low: weak candidate retained only because topology evidence is sparse.
- very_low: avoid outputting these unless there is a specific reason.

Use zone names, floor/wing/pod identifiers, surface/construction context, and any interior/core/perimeter clues.
{label_rule}

Output schema:
{{"connections": [{{"zone_a": "ZoneA", "zone_b": "ZoneB", "confidence": "label", "basis": "same_floor_neighbor|corridor_room|core_adjacent|shared_context|uncertain", "reason": "short evidence-based reason, max 20 words"}}]}}"""

    connection_user = (
        "Batch zones eligible for sparse connection discovery:\n"
        + json.dumps(batch_zones, indent=2)
        + "\n\nThermal/Envelope Context:\n"
        + thermal_context
        + "\n\nOther Local Snippet:\n"
        + local_snippet
    )

    return ("connection", connection_system, connection_user)


def parse_connection_items(llm_result: dict, local_edges: List[str]) -> List[dict]:
    edge_lookup = {}
    for edge in local_edges:
        if "||" not in edge:
            continue
        a, b = edge.split("||", 1)
        key = tuple(sorted((ultra_clean(a), ultra_clean(b))))
        edge_lookup[key] = edge

    raw_connections = llm_result.get("connections", [])
    if isinstance(raw_connections, dict):
        raw_connections = [
            {"edge": edge, "confidence": label}
            for edge, label in raw_connections.items()
        ]

    parsed = []
    if not isinstance(raw_connections, list):
        return parsed

    for item in raw_connections:
        if not isinstance(item, dict):
            continue

        edge = item.get("edge")
        zone_a = item.get("zone_a")
        zone_b = item.get("zone_b")

        if edge and "||" in edge:
            a, b = edge.split("||", 1)
        elif zone_a and zone_b:
            a, b = zone_a, zone_b
        else:
            continue

        key = tuple(sorted((ultra_clean(a), ultra_clean(b))))
        matched_edge = edge_lookup.get(key)
        if matched_edge is None:
            continue

        confidence_label = normalize_label(item.get("confidence") or item.get("label"))
        if confidence_label is None:
            continue

        parsed.append(
            {
                "edge": matched_edge,
                "confidence_label": confidence_label,
                "confidence_value": LABEL_TO_VALUE[confidence_label],
                "basis": item.get("basis"),
                "reason": item.get("reason") or item.get("concise_reason"),
            }
        )

    return parsed


def parse_discovered_connection_items(
    llm_result: dict,
    batch_zones: List[str],
) -> List[dict]:
    raw_connections = llm_result.get("connections", [])
    if isinstance(raw_connections, dict):
        raw_connections = [
            {"edge": edge, "confidence": label}
            for edge, label in raw_connections.items()
        ]

    parsed = []
    seen_edges = set()
    if not isinstance(raw_connections, list):
        return parsed

    for item in raw_connections:
        if not isinstance(item, dict):
            continue

        edge = item.get("edge")
        zone_a = item.get("zone_a")
        zone_b = item.get("zone_b")

        if edge and "||" in edge:
            raw_a, raw_b = edge.split("||", 1)
        elif zone_a and zone_b:
            raw_a, raw_b = zone_a, zone_b
        else:
            continue

        matched_a = match_zone_name(raw_a, batch_zones)
        matched_b = match_zone_name(raw_b, batch_zones)
        if matched_a is None or matched_b is None or matched_a == matched_b:
            continue

        edge_key = tuple(sorted((matched_a, matched_b)))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        confidence_label = normalize_label(item.get("confidence") or item.get("label"))
        if confidence_label is None:
            continue

        parsed.append(
            {
                "edge": f"{edge_key[0]}||{edge_key[1]}",
                "confidence_label": confidence_label,
                "confidence_value": LABEL_TO_VALUE[confidence_label],
                "basis": item.get("basis"),
                "reason": item.get("reason") or item.get("concise_reason"),
            }
        )

    return parsed


def apply_label_to_semantic(
    semantic_priors: Dict[str, List[float]],
    zone: str,
    semantic_idx: int,
    label,
) -> None:
    normalized = normalize_label(label)
    if normalized is not None:
        semantic_priors[zone][semantic_idx] = LABEL_TO_VALUE[normalized]


def recommend_trust_from_labels(vals: List[float]) -> Tuple[str, str]:
    thermal = vals[0]
    occupancy = vals[1]
    exposure = vals[2]
    sensitivity = vals[4]

    if occupancy >= 0.7 and sensitivity >= 0.7:
        return "low", "high_occupancy_and_weather_sensitivity"
    if occupancy >= 0.7:
        return "low", "high_occupancy_volatility"
    if sensitivity >= 0.7 and thermal <= 0.5:
        return "low", "weather_sensitive_with_limited_thermal_buffer"
    if occupancy <= 0.3 and sensitivity <= 0.5:
        return "medium_high", "stable_loads_and_low_weather_sensitivity"
    if occupancy <= 0.3 and thermal >= 0.7:
        return "medium_high", "stable_loads_and_high_thermal_buffer"
    if exposure >= 0.7 and sensitivity >= 0.7:
        return "medium", "weather_exposed_physics_is_useful_but_uncertain"

    return "medium", "mixed_or_ambiguous_evidence"


def build_trust_agent_prompt(
    batch_zones: List[str],
    local_edges: List[str],
    thermal_context: str,
    schedule_context: str,
    local_snippet: str,
    current_priors: Dict[str, List[float]],
) -> Tuple[str, str, str]:
    label_rule = (
        'Use only these labels: "very_low", "low", "medium", "medium_high", "high". '
        "Return strictly valid JSON. Do not output markdown or explanations."
    )
    batch_header = f"Batch Zones: {batch_zones}\nLocal Edges: {local_edges}\n"

    prior_summary = []
    for zone in batch_zones:
        vals = current_priors[zone]
        recommended_trust, reason = recommend_trust_from_labels(vals)
        prior_summary.append(
            {
                "zone": zone,
                "thermal_inertia": value_to_label(vals[0]),
                "occupancy_volatility": value_to_label(vals[1]),
                "env_exposure": value_to_label(vals[2]),
                "env_sensitivity": value_to_label(vals[4]),
                "recommended_base_physics_trust": recommended_trust,
                "recommendation_reason": reason,
            }
        )

    trust_system = f"""You are a physics-informed modeling specialist.

Task: judge ONLY Base_Physics_Trust for each zone.
Base_Physics_Trust is a derived semantic prior. It MUST be consistent with the preliminary labels from the other agents.

Core meaning:
Base_Physics_Trust indicates how strongly a sparse-data model should rely on stable physical structure versus flexible data-driven adaptation for this zone.

A high Base_Physics_Trust means:
- the zone dynamics are expected to be regular, smooth, and well explained by physical structure, thermal inertia, envelope coupling, adjacency, and stable internal gains;
- sparse training data should not be allowed to strongly override the physical prior;
- physical reasoning should dominate unless data provides strong contrary evidence.

A low Base_Physics_Trust means:
- the zone dynamics are likely affected by stochastic occupancy, irregular schedules, nonlinear internal gains, mixed-use behavior, control complexity, or disturbances not fully captured by simple physical priors;
- adaptive data-driven correction should be allowed more influence;
- physical structure is still relevant, but it may be insufficient by itself.

Important:
Base_Physics_Trust does NOT ask whether physics is true.
It asks whether stable physical priors should dominate over flexible data adaptation for this zone under sparse-data learning.

Rubric:
- high: stable service/storage/core/mechanical zone; low occupancy volatility; low-to-medium weather sensitivity; high or medium_high thermal inertia.
- medium_high: mostly stable zone, but with moderate exposure, moderate schedules, or some uncertainty.
- medium: mixed evidence; neither physical prior nor adaptive correction should dominate strongly.
- low: human-driven or schedule-driven zone; medium_high/high occupancy volatility; low thermal inertia with high sensitivity; mixed-use behavior.
- very_low: highly stochastic occupancy plus high weather sensitivity, complex mixed-use behavior, or strong evidence that simple physical priors are insufficient.

Consistency constraints:
- If Occupancy_Volatility is high or medium_high, do not assign high trust unless there is strong evidence of stable repeated schedules and high thermal buffering.
- If Occupancy_Volatility is low/very_low and Env_Sensitivity is medium or lower, trust should usually be medium_high or high.
- If Env_Sensitivity is high and Thermal_Inertia is low/very_low, trust should usually be low or medium.
- Guest rooms, meeting rooms, lounges, exercise centers, retail/front-office spaces often require more adaptive correction unless schedule evidence is stable.
- Storage, stairs, elevator cores, corridors, and mechanical rooms usually favor stable physical priors unless evidence shows stochastic loads or strong weather sensitivity.
- You may override the recommended trust only when the evidence context clearly justifies it.

Use the recommended trust hints as calibration anchors, not as hidden truth. {label_rule}

Output schema:
{{"semantics": [{{"zone": "ZoneName", "base_physics_trust": "label", "basis": "stable_physics|adaptive_correction_needed|mixed_evidence|weather_sensitive|occupancy_driven|ambiguous", "confidence": "high|medium|low", "reason": "short evidence-based reason, max 20 words"}}]}}"""

    trust_user = (
        batch_header
        + "Preliminary labels and recommended trust hints:\n"
        + json.dumps(prior_summary, indent=2)
        + "\n\n"
        + f"Thermal/Envelope Context:\n{thermal_context}\n\n"
        + f"Schedule/Internal Gain Context:\n{schedule_context}\n\n"
        + f"Other Local Snippet:\n{local_snippet}"
    )

    return ("trust", trust_system, trust_user)

def build_multi_agent_prompts(
    batch_zones: List[str],
    local_edges: List[str],
    thermal_context: str,
    schedule_context: str,
    local_snippet: str,
    current_priors: Dict[str, List[float]],
) -> List[Tuple[str, str, str]]:
    label_rule = (
        'Use only these labels: "very_low", "low", "medium", "medium_high", "high". '
        "Return strictly valid JSON. Do not output markdown or explanations."
    )
    batch_header = f"Batch Zones: {batch_zones}\nLocal Edges: {local_edges}\n"

    thermal_system = f"""You are a building thermal-mass specialist.

Task: judge ONLY Thermal_Inertia for each zone.
Definition: very_low means lightweight/fast response; high means slow/high-mass response.
Use construction, material, surface, and fenestration evidence. {label_rule}

Output schema:
{{"semantics": [{{"zone": "ZoneName", "thermal_inertia": "label", "basis": "heavy_mass|lightweight|insulation_dominated|glazing_dominated|mixed_construction|ambiguous", "confidence": "high|medium|low", "reason": "short evidence-based reason, max 20 words"}}]}}"""

    thermal_user = (
        batch_header
        + f"Thermal/Envelope Context:\n{thermal_context}\n\n"
        + f"Other Local Snippet:\n{local_snippet}"
    )

    envelope_system = f"""You are a building-envelope and exposure specialist.

Task: judge ONLY Env_Exposure and Env_Sensitivity for each zone.
Env_Exposure: low means interior/protected; high means exterior/solar/window/roof exposed.
Env_Sensitivity: low means outdoor extremes weakly affect the zone; high means strong weather sensitivity.
Use surface boundary, sun/wind exposure, window/fenestration, roof/exterior wall evidence. {label_rule}

Important distinction:
- Env_Exposure is about direct contact with outdoor/solar/window/roof conditions.
- Env_Sensitivity is exposure moderated by buffering and thermal inertia.
- A highly exposed but high-mass zone may have medium sensitivity, not automatically high.
- A lightweight perimeter/window/roof zone should have higher sensitivity.
- Interior/core zones surrounded by conditioned zones should have low sensitivity even if nearby exposed surfaces exist.

Sensitivity rubric:
- high: exterior wall/window/roof exposure AND lightweight or limited buffering.
- medium_high: clear exposure with moderate buffering or uncertain mass.
- medium: partial exposure, mixed evidence, or exposed but thermally buffered.
- low/very_low: interior/core/protected zone, no exterior surface/window, or strong buffering.

Output schema:
{{"semantics": [{{"zone": "ZoneName", "env_exposure": "label", "env_sensitivity": "label", "basis": "interior_only|exterior_wall|window|roof_or_solar|buffered_exposure|ambiguous", "confidence": "high|medium|low", "reason": "short evidence-based reason, max 20 words"}}]}}"""

    envelope_user = (
        batch_header
        + f"Thermal/Envelope Context:\n{thermal_context}\n\n"
        + f"Other Local Snippet:\n{local_snippet}"
    )

    schedule_system = f"""You are an occupancy and internal-gain schedule specialist.

Task: judge ONLY Occupancy_Volatility for each zone.
Low volatility means flat/stable/no people/service schedules. High volatility means strong time-varying people, lights, equipment, or activity schedules.
Use People, Lights, ElectricEquipment, and Schedule evidence. {label_rule}

Output schema:
{{"semantics": [{{"zone": "ZoneName", "occupancy_volatility": "label", "basis": "people_schedule|lights_equipment_schedule|flat_schedule|no_people|ambiguous", "confidence": "high|medium|low", "reason": "short evidence-based reason, max 20 words"}}]}}"""

    schedule_user = (
        batch_header
        + f"Schedule/Internal Gain Context:\n{schedule_context}\n\n"
        + f"Other Local Snippet:\n{local_snippet}"
    )

    return [
        ("thermal", thermal_system, thermal_user),
        ("envelope", envelope_system, envelope_user),
        ("schedule", schedule_system, schedule_user),
    ]


def extract_idf_priors_once(
    idf_text: str,
    zones: List[str],
    client: Together,
    model: str,
    temperature: float,
    seed: Optional[int],
    batch_size: int,
    max_tokens: int,
    sleep_s: float,
    include_connections: bool,
    raw_output_dir: Optional[str],
    run_idx: int,
    thermal_context_chars: int,
    schedule_context_chars: int,
    use_label_output: bool,
    multi_agent: bool,
    reference_idf_text: Optional[str] = None,
) -> dict:
    clean_to_info, parsed_blocks, valid_pairs_set, pair_confidence_map = parse_idf_context(
        idf_text,
        zones,
    )
    idf_index = parse_idf_object_index(idf_text)
    semantic_priors = {z: [0.5, 0.5, 0.5, 0.5, 0.5] for z in zones}
    pair_confidence_values = {edge: [] for edge in pair_confidence_map}
    reason_traces = []
    n_batches = int(np.ceil(len(zones) / batch_size))

    print(
        f"[Topology] Explicit IDF surface-adjacency candidate edges: "
        f"{len(pair_confidence_map)}"
    )
    if not pair_confidence_map:
        print(
            "[Topology] No explicit internal Surface edges found. "
            "Connection agent will discover a sparse fallback graph from IDF evidence."
        )

    if include_connections:
        if use_label_output:
            output_schema = """Output STRICTLY valid JSON with EXACTLY TWO KEYS:

1. "connections": { "ZoneA||ZoneB": confidence_score_0_to_1 }

2. "semantics": [
  {
    "zone": "ZoneName",
    "labels": [
      Thermal_Inertia_label,
      Occupancy_Volatility_label,
      Env_Exposure_label,
      Base_Physics_Trust_label,
      Env_Sensitivity_label
    ]
  }
]

Each label MUST be one of: "very_low", "low", "medium", "medium_high", "high"."""
        else:
            output_schema = """Output STRICTLY valid JSON with EXACTLY TWO KEYS:

1. "connections": { "ZoneA||ZoneB": confidence_score_0_to_1 }

2. "semantics": {
  "ZoneName": [
    Thermal_Inertia,
    Occupancy_Volatility,
    Env_Exposure,
    Base_Physics_Trust,
    Env_Sensitivity
  ]
}"""
    else:
        if use_label_output:
            output_schema = """Output STRICTLY valid JSON with EXACTLY ONE KEY:

"semantics": [
  {
    "zone": "ZoneName",
    "labels": [
      Thermal_Inertia_label,
      Occupancy_Volatility_label,
      Env_Exposure_label,
      Base_Physics_Trust_label,
      Env_Sensitivity_label
    ]
  }
]

Each label MUST be one of: "very_low", "low", "medium", "medium_high", "high".
Do not output numeric scores.
Do not output "connections". Zone adjacency is parsed deterministically from the IDF."""
        else:
            output_schema = """Output STRICTLY valid JSON with EXACTLY ONE KEY:

"semantics": [
  {
    "zone": "ZoneName",
    "scores": [
      Thermal_Inertia,
      Occupancy_Volatility,
      Env_Exposure,
      Base_Physics_Trust,
      Env_Sensitivity
    ]
  }
]

Do not output "connections". Zone adjacency is parsed deterministically from the IDF."""

    system_prompt = f"""You are an expert HVAC, building physics, and thermodynamic engineer.

You are helping design a Physics-Informed Neural Network with a Mixture-of-Experts gating mechanism.

Analyze the provided IDF subset and zones. {output_schema}

Definitions:
- Thermal_Inertia: 0 means fast/lightweight thermal response, 1 means slow/high-mass response. Infer it primarily from the provided construction/material/surface evidence.
- Occupancy_Volatility: 0 means stable internal gains, 1 means stochastic occupancy/equipment. Infer it primarily from the provided People/Lights/ElectricEquipment schedule evidence.
- Env_Exposure: 0 means interior/protected zone, 1 means strongly exposed to weather/solar.
- Base_Physics_Trust: 0 means data-driven nonlinear expert is more reliable, 1 means physical prior is more reliable.
- Env_Sensitivity: 0 means outdoor extremes barely affect this zone, 1 means strong sensitivity to outdoor/weather extremes.

Reasoning rules:
- Storage, internal corridors, mechanically stable zones should have high Base_Physics_Trust.
- Lobby, classrooms, healthcare, retail, and highly occupied zones should have lower Base_Physics_Trust.
- Zones with exterior walls, large windows, skylights, or solar exposure should have high Env_Exposure and Env_Sensitivity.
- Internal or well-shielded zones should have lower Env_Sensitivity.
- Zones with strongly time-varying People, Lights, or ElectricEquipment schedules should have higher Occupancy_Volatility.
- Zones with no matched People object and flat/internal service schedules should usually have lower Occupancy_Volatility.

Stability rules:
- Use conservative, repeatable engineering judgment.
- Prefer 0.5 only when the provided evidence is genuinely ambiguous.
- Do not assign all Thermal_Inertia values to 0.5 if construction/material evidence distinguishes zones.
- Do not assign all Occupancy_Volatility values to 0.5 if schedule evidence distinguishes zones.
- Keep scores in [0, 1].
- If outputting labels, use a consistent rubric:
  very_low: strong evidence for minimal effect,
  low: below typical,
  medium: genuinely ambiguous or typical,
  medium_high: above typical,
  high: strong evidence for high effect.

DO NOT use markdown like ```json.
DO NOT output explanations.
Just output the JSON object."""

    for i in range(0, len(zones), batch_size):
        batch_zones = zones[i:i + batch_size]
        batch_idx = i // batch_size + 1
        print(f"  -> batch {batch_idx}/{n_batches}: {len(batch_zones)} zones")

        batch_clean_names = [ultra_clean(bz) for bz in batch_zones]
        local_snippet_blocks = []

        for blk in parsed_blocks:
            clean_blk = ultra_clean(blk)
            if any(cn in clean_blk for cn in batch_clean_names):
                local_snippet_blocks.append(blk)

        local_snippet = "\n\n".join(local_snippet_blocks)[:30000]
        thermal_context = build_batch_thermal_context(
            idf_index,
            batch_zones,
            max_chars=thermal_context_chars,
        )
        schedule_context = build_batch_schedule_context(
            idf_index,
            batch_zones,
            max_chars=schedule_context_chars,
        )
        local_edges = [
            f"{a}||{b}"
            for a, b in valid_pairs_set
            if a in batch_zones or b in batch_zones
        ]

        prompt = (
            f"Batch Zones: {batch_zones}\n"
            f"Local Edges: {local_edges}\n"
            f"Thermal/Envelope Context:\n{thermal_context}\n\n"
            f"Schedule/Internal Gain Context:\n{schedule_context}\n\n"
            f"Other Local Snippet:\n{local_snippet}"
        )

        if multi_agent:
            if local_edges:
                conn_name, conn_system_prompt, conn_user_prompt = build_connection_agent_prompt(
                    local_edges=local_edges,
                    thermal_context=thermal_context,
                    local_snippet=local_snippet,
                )
                print(f"     [{conn_name} agent] scoring thermal coupling edges...")
                conn_result = call_llm_json(
                    client=client,
                    model=model,
                    temperature=temperature,
                    seed=seed,
                    max_tokens=max_tokens,
                    system_prompt=conn_system_prompt,
                    user_prompt=conn_user_prompt,
                    raw_output_dir=raw_output_dir,
                    run_idx=run_idx,
                    batch_idx=batch_idx,
                    agent_name=conn_name,
                )
                conn_items = parse_connection_items(conn_result, local_edges)
                conn_mode = "explicit_surface_adjacency"
            else:
                conn_name, conn_system_prompt, conn_user_prompt = build_connection_discovery_agent_prompt(
                    batch_zones=batch_zones,
                    thermal_context=thermal_context,
                    local_snippet=local_snippet,
                )
                print(f"     [{conn_name} agent] discovering sparse fallback edges...")
                conn_result = call_llm_json(
                    client=client,
                    model=model,
                    temperature=temperature,
                    seed=seed,
                    max_tokens=max_tokens,
                    system_prompt=conn_system_prompt,
                    user_prompt=conn_user_prompt,
                    raw_output_dir=raw_output_dir,
                    run_idx=run_idx,
                    batch_idx=batch_idx,
                    agent_name=conn_name,
                )
                conn_items = parse_discovered_connection_items(conn_result, batch_zones)
                conn_mode = "llm_sparse_discovery"

            for conn_item in conn_items:
                pair_confidence_values.setdefault(conn_item["edge"], []).append(
                    conn_item["confidence_value"]
                )
                reason_traces.append(
                    {
                        "run": run_idx,
                        "batch": batch_idx,
                        "agent": conn_name,
                        "mode": conn_mode,
                        "edge": conn_item["edge"],
                        "confidence": conn_item["confidence_label"],
                        "basis": conn_item.get("basis"),
                        "reason": conn_item.get("reason"),
                    }
                )

            if sleep_s > 0:
                time.sleep(sleep_s)

            agent_prompts = build_multi_agent_prompts(
                batch_zones=batch_zones,
                local_edges=local_edges,
                thermal_context=thermal_context,
                schedule_context=schedule_context,
                local_snippet=local_snippet,
                current_priors=semantic_priors,
            )

            for agent_name, agent_system_prompt, agent_user_prompt in agent_prompts:
                print(f"     [{agent_name} agent] extracting...")
                llm_result = call_llm_json(
                    client=client,
                    model=model,
                    temperature=temperature,
                    seed=seed,
                    max_tokens=max_tokens,
                    system_prompt=agent_system_prompt,
                    user_prompt=agent_user_prompt,
                    raw_output_dir=raw_output_dir,
                    run_idx=run_idx,
                    batch_idx=batch_idx,
                    agent_name=agent_name,
                )

                agent_items = parse_agent_label_items(llm_result, batch_zones)
                reason_traces.extend(
                    extract_agent_reason_trace(
                        llm_result,
                        batch_zones,
                        agent_name,
                        run_idx,
                        batch_idx,
                    )
                )

                for zone, item in agent_items.items():
                    if agent_name == "thermal":
                        apply_label_to_semantic(
                            semantic_priors,
                            zone,
                            0,
                            item.get("thermal_inertia") or item.get("label"),
                        )
                    elif agent_name == "envelope":
                        apply_label_to_semantic(
                            semantic_priors,
                            zone,
                            2,
                            item.get("env_exposure"),
                        )
                        apply_label_to_semantic(
                            semantic_priors,
                            zone,
                            4,
                            item.get("env_sensitivity"),
                        )
                    elif agent_name == "schedule":
                        apply_label_to_semantic(
                            semantic_priors,
                            zone,
                            1,
                            item.get("occupancy_volatility") or item.get("label"),
                        )

                if sleep_s > 0:
                    time.sleep(sleep_s)

            trust_name, trust_system_prompt, trust_user_prompt = build_trust_agent_prompt(
                batch_zones=batch_zones,
                local_edges=local_edges,
                thermal_context=thermal_context,
                schedule_context=schedule_context,
                local_snippet=local_snippet,
                current_priors=semantic_priors,
            )
            print(f"     [{trust_name} agent] extracting with calibrated hints...")
            trust_result = call_llm_json(
                client=client,
                model=model,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
                system_prompt=trust_system_prompt,
                user_prompt=trust_user_prompt,
                raw_output_dir=raw_output_dir,
                run_idx=run_idx,
                batch_idx=batch_idx,
                agent_name=trust_name,
            )

            trust_items = parse_agent_label_items(trust_result, batch_zones)
            reason_traces.extend(
                extract_agent_reason_trace(
                    trust_result,
                    batch_zones,
                    trust_name,
                    run_idx,
                    batch_idx,
                )
            )
            for zone, item in trust_items.items():
                apply_label_to_semantic(
                    semantic_priors,
                    zone,
                    3,
                    item.get("base_physics_trust") or item.get("label"),
                )

            if sleep_s > 0:
                time.sleep(sleep_s)

            continue

        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed

        resp = client.chat.completions.create(**kwargs)
        raw_content = resp.choices[0].message.content.strip()

        try:
            llm_result = parse_llm_json(raw_content)
        except json.JSONDecodeError as e:
            print("     JSON broken; using regex rescue.")
            print(f"     JSON error: {e}")
            if raw_output_dir:
                os.makedirs(raw_output_dir, exist_ok=True)
                raw_path = os.path.join(
                    raw_output_dir,
                    f"bad_json_run_{run_idx:03d}_batch_{batch_idx:03d}.txt",
                )
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(raw_content)
                print(f"     Raw output saved: {raw_path}")
                pos = getattr(e, "pos", 0)
                around_path = os.path.join(
                    raw_output_dir,
                    f"bad_json_run_{run_idx:03d}_batch_{batch_idx:03d}_around_error.txt",
                )
                with open(around_path, "w", encoding="utf-8") as f:
                    f.write(raw_content[max(0, pos - 500): pos + 500])
            llm_result = {"connections": {}, "semantics": {}}

            semantic_pattern = re.compile(
                r'"([^"]+)"\s*:\s*\[\s*'
                r'([0-9\.]+)\s*,\s*'
                r'([0-9\.]+)\s*,\s*'
                r'([0-9\.]+)'
                r'(?:\s*,\s*([0-9\.]+)\s*,\s*([0-9\.]+))?'
                r'\s*\]'
            )

            for match in semantic_pattern.finditer(raw_content):
                groups = match.groups()
                vals = [0.5 if g is None else float(g) for g in groups[1:]]
                llm_result["semantics"][groups[0]] = vals + [0.5] * (5 - len(vals))

            conn_pattern = re.compile(r'"([^"|]+\|\|[^"]+)"\s*:\s*([0-9\.]+)')
            for match in conn_pattern.finditer(raw_content):
                k, v = match.groups()
                llm_result["connections"][k] = float(v)

        if include_connections:
            for k, v in llm_result.get("connections", {}).items():
                try:
                    pair_confidence_map[k] = float(v)
                except Exception:
                    pass

        semantics_result = llm_result.get("semantics", {})
        if isinstance(semantics_result, list):
            semantics_iter = [
                (item.get("zone"), item)
                for item in semantics_result
                if isinstance(item, dict)
            ]
        elif isinstance(semantics_result, dict):
            semantics_iter = semantics_result.items()
        else:
            semantics_iter = []

        for z, v_list in semantics_iter:
            if not z or v_list is None:
                continue
            matched_zone = None
            if z in semantic_priors:
                matched_zone = z
            else:
                uz = ultra_clean(z)
                for real_z in batch_zones:
                    ur = ultra_clean(real_z)
                    if uz == ur or uz in ur or ur in uz:
                        matched_zone = real_z
                        break

            if matched_zone is None:
                continue

            try:
                vals = semantic_item_to_values(v_list)
                if vals is None:
                    continue
                semantic_priors[matched_zone] = [max(0.0, min(1.0, x)) for x in vals]
            except Exception:
                pass

        if sleep_s > 0:
            time.sleep(sleep_s)

    n = len(zones)
    adj_matrix = np.zeros((n, n))
    adj_conf = np.eye(n)

    for edge_id, values in pair_confidence_values.items():
        if values:
            pair_confidence_map[edge_id] = float(np.median(values))

    for edge_id, conf_val in pair_confidence_map.items():
        if "||" not in edge_id:
            continue

        z_a, z_b = edge_id.split("||")

        if ultra_clean(z_a) in clean_to_info and ultra_clean(z_b) in clean_to_info:
            ia = clean_to_info[ultra_clean(z_a)]["idx"]
            ib = clean_to_info[ultra_clean(z_b)]["idx"]
            adj_matrix[ia, ib] = adj_matrix[ib, ia] = 1.0
            adj_conf[ia, ib] = adj_conf[ib, ia] = max(0.01, min(1.0, float(conf_val)))

    sem_matrix = np.array([semantic_priors[z] for z in zones], dtype=float)
    control_prior = extract_control_priors(
        idf_text=idf_text,
        zones=zones,
        reference_idf_text=reference_idf_text,
    )
    if reference_idf_text is not None:
        control_prior.update(
            extract_llm_control_calibration_priors(
                control_prior=control_prior,
                zones=zones,
                semantic_matrix=sem_matrix,
                client=client,
                model=model,
                temperature=temperature,
                seed=seed,
                batch_size=batch_size,
                max_tokens=max_tokens,
                sleep_s=sleep_s,
                raw_output_dir=raw_output_dir,
                run_idx=run_idx,
            )
        )

    prior_out = {
        "zones": zones,
        "adjacency_matrix": adj_matrix.tolist(),
        "adjacency_confidence_matrix": adj_conf.tolist(),
        "semantic_matrix": sem_matrix.tolist(),
        "reason_traces": reason_traces,
    }
    prior_out.update(control_prior)
    return prior_out


def flatten_semantic_runs(runs: List[dict]) -> pd.DataFrame:
    rows = []
    for run_idx, prior in enumerate(runs, start=1):
        sem = np.array(prior["semantic_matrix"], dtype=float)
        for z_idx, zone in enumerate(prior["zones"]):
            for s_idx, semantic_name in enumerate(SEMANTIC_NAMES):
                rows.append(
                    {
                        "run": run_idx,
                        "zone": zone,
                        "semantic": semantic_name,
                        "value": sem[z_idx, s_idx],
                    }
                )
    return pd.DataFrame(rows)


def flatten_reason_traces(runs: List[dict]) -> pd.DataFrame:
    rows = []
    for prior in runs:
        rows.extend(prior.get("reason_traces", []))
    return pd.DataFrame(rows)


def summarize_variability(df_long: pd.DataFrame) -> pd.DataFrame:
    return (
        df_long.groupby(["zone", "semantic"])["value"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .assign(range=lambda d: d["max"] - d["min"], cv=lambda d: d["std"] / d["mean"].abs().clip(lower=1e-9))
    )


def summarize_adjacency_variability(runs: List[dict]) -> pd.DataFrame:
    zones = runs[0]["zones"]
    rows = []
    conf_stack = np.stack(
        [np.array(prior["adjacency_confidence_matrix"], dtype=float) for prior in runs],
        axis=0,
    )

    for i, za in enumerate(zones):
        for j, zb in enumerate(zones):
            if j <= i:
                continue
            vals = conf_stack[:, i, j]
            rows.append(
                {
                    "zone_a": za,
                    "zone_b": zb,
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "range": float(vals.max() - vals.min()),
                    "edge_frequency": float((vals > 0).mean()),
                }
            )

    columns = [
        "zone_a",
        "zone_b",
        "mean",
        "std",
        "min",
        "max",
        "range",
        "edge_frequency",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    df_edges = pd.DataFrame(rows)
    df_edges = df_edges[df_edges["max"] > 0].copy()

    if df_edges.empty:
        return pd.DataFrame(columns=columns)

    return df_edges.sort_values(["range", "std"], ascending=False)


def _entropy_from_counts(counts: pd.Series) -> float:
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def build_semantic_agreement_summary(df_long: pd.DataFrame) -> pd.DataFrame:
    df = df_long.copy()
    df["label"] = df["value"].apply(value_to_label)

    rows = []
    for (zone, semantic), grp in df.groupby(["zone", "semantic"]):
        counts = grp["label"].value_counts()
        majority_label = counts.index[0]
        majority_ratio = float(counts.iloc[0] / counts.sum())
        rows.append(
            {
                "zone": zone,
                "semantic": semantic,
                "majority_label": majority_label,
                "majority_value": LABEL_TO_VALUE[majority_label],
                "majority_ratio": majority_ratio,
                "label_entropy": _entropy_from_counts(counts),
                "n_unique_labels": int(counts.shape[0]),
                "min_value": float(grp["value"].min()),
                "max_value": float(grp["value"].max()),
                "range": float(grp["value"].max() - grp["value"].min()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["majority_ratio", "range"],
        ascending=[True, False],
    )


def build_audit_report(
    df_long: pd.DataFrame,
    df_reasons: pd.DataFrame,
    adj_summary: pd.DataFrame,
) -> dict:
    agreement = build_semantic_agreement_summary(df_long)
    report = {
        "semantic_stability": {
            "mean_majority_ratio": float(agreement["majority_ratio"].mean()),
            "median_majority_ratio": float(agreement["majority_ratio"].median()),
            "mean_label_entropy": float(agreement["label_entropy"].mean()),
            "share_full_agreement": float((agreement["majority_ratio"] == 1.0).mean()),
            "share_within_one_label_step_range": float((agreement["range"] <= 0.2 + 1e-9).mean()),
            "share_large_flip_range_ge_0_4": float((agreement["range"] >= 0.4 - 1e-9).mean()),
        },
        "connection_confidence": {
            "n_edges": int(len(adj_summary)),
            "mean_confidence": float(adj_summary["mean"].mean()) if not adj_summary.empty else None,
            "median_confidence": float(adj_summary["mean"].median()) if not adj_summary.empty else None,
            "mean_confidence_range": float(adj_summary["range"].mean()) if not adj_summary.empty else None,
            "share_edges_varying_ge_0_4": float((adj_summary["range"] >= 0.4 - 1e-9).mean()) if not adj_summary.empty else None,
        },
    }

    if not df_reasons.empty:
        reason_col = df_reasons.get("reason")
        confidence_col = df_reasons.get("confidence")
        report["reason_trace"] = {
            "n_trace_rows": int(len(df_reasons)),
            "reason_coverage": float(reason_col.notna().mean()) if reason_col is not None else None,
            "basis_coverage": float(df_reasons["basis"].notna().mean()) if "basis" in df_reasons else None,
            "low_confidence_share": float((confidence_col == "low").mean()) if confidence_col is not None else None,
        }

    return report


def build_aggregated_prior(runs: List[dict], method: str = "median", quantize: bool = False) -> dict:
    if not runs:
        raise ValueError("Cannot aggregate empty prior runs.")
    if method not in {"median", "mean"}:
        raise ValueError('method must be "median" or "mean".')

    zones = runs[0]["zones"]
    sem_stack = np.stack(
        [np.array(prior["semantic_matrix"], dtype=float) for prior in runs],
        axis=0,
    )
    adj_stack = np.stack(
        [np.array(prior["adjacency_matrix"], dtype=float) for prior in runs],
        axis=0,
    )
    conf_stack = np.stack(
        [np.array(prior["adjacency_confidence_matrix"], dtype=float) for prior in runs],
        axis=0,
    )

    reducer = np.median if method == "median" else np.mean
    sem_reduced = reducer(sem_stack, axis=0)
    label_matrix = [[value_to_label(v) for v in row] for row in sem_reduced]

    if quantize:
        sem_out = np.array(
            [[LABEL_TO_VALUE[label] for label in row] for row in label_matrix],
            dtype=float,
        )
    else:
        sem_out = sem_reduced

    adj_reduced = (reducer(adj_stack, axis=0) >= 0.5).astype(float)
    conf_reduced = reducer(conf_stack, axis=0)

    out = {
        "zones": zones,
        "adjacency_matrix": adj_reduced.tolist(),
        "adjacency_confidence_matrix": conf_reduced.tolist(),
        "semantic_matrix": sem_out.tolist(),
        "semantic_labels": label_matrix,
        "semantic_names": SEMANTIC_NAMES,
        "aggregation": {
            "method": method,
            "n_runs": len(runs),
            "quantized_to_label_values": quantize,
            "label_to_value": LABEL_TO_VALUE,
        },
    }

    if "control_matrix" in runs[0]:
        control_stack = np.stack(
            [np.array(prior["control_matrix"], dtype=float) for prior in runs],
            axis=0,
        )
        control_reduced = reducer(control_stack, axis=0)
        out.update(
            {
                "control_matrix": control_reduced.tolist(),
                "control_names": runs[0].get("control_names", CONTROL_NAMES),
                "control_records": runs[0].get("control_records", {}),
                "control_summary": runs[0].get("control_summary", {}),
            }
        )

    if "llm_control_calibration_matrix" in runs[0]:
        llm_calib_stack = np.stack(
            [np.array(prior["llm_control_calibration_matrix"], dtype=float) for prior in runs],
            axis=0,
        )
        llm_calib_reduced = reducer(llm_calib_stack, axis=0)
        out.update(
            {
                "llm_control_calibration_matrix": llm_calib_reduced.tolist(),
                "llm_control_calibration_names": runs[0].get(
                    "llm_control_calibration_names",
                    LLM_CONTROL_CALIBRATION_NAMES,
                ),
                "llm_control_calibration_records": runs[0].get("llm_control_calibration_records", {}),
                "llm_control_calibration_summary": runs[0].get("llm_control_calibration_summary", {}),
            }
        )

    return out


def save_plots(df_long: pd.DataFrame, summary: pd.DataFrame, runs: List[dict], out_dir: str):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_long, x="semantic", y="value", color="#8ecae6")
    sns.stripplot(data=df_long, x="semantic", y="value", color="#1d3557", alpha=0.25, size=2)
    plt.xticks(rotation=25, ha="right")
    plt.ylim(-0.05, 1.05)
    plt.title("LLM Semantic Prior Distribution Across Runs")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "semantic_boxplot.png"), dpi=220)
    plt.close()

    top = summary.sort_values("range", ascending=False).head(30).copy()
    top["item"] = top["zone"] + " | " + top["semantic"]
    plt.figure(figsize=(10, max(6, 0.28 * len(top))))
    sns.barplot(data=top, y="item", x="range", color="#f4a261")
    plt.xlim(0, 1)
    plt.title("Top Semantic Prior Variation Ranges")
    plt.xlabel("max - min across runs")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "semantic_top_variation.png"), dpi=220)
    plt.close()

    pivot = (
        df_long.assign(item=lambda d: d["zone"] + " | " + d["semantic"])
        .pivot_table(index="item", columns="run", values="value", aggfunc="mean")
    )
    top_items = top["item"].tolist()
    pivot = pivot.loc[[x for x in top_items if x in pivot.index]]
    plt.figure(figsize=(10, max(6, 0.28 * len(pivot))))
    sns.heatmap(pivot, vmin=0, vmax=1, cmap="viridis", annot=True, fmt=".2f")
    plt.title("Semantic Prior Values by Run, Top Variable Items")
    plt.xlabel("run")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "semantic_run_heatmap.png"), dpi=220)
    plt.close()

    conf_stack = np.stack(
        [np.array(prior["adjacency_confidence_matrix"], dtype=float) for prior in runs],
        axis=0,
    )
    conf_std = conf_stack.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(conf_stack[0])
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_std, cmap="mako", xticklabels=False, yticklabels=False)
    plt.title("Adjacency Confidence Std Across Runs")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "adjacency_confidence_std_heatmap.png"), dpi=220)
    plt.close()


def load_existing_prior_runs(out_dir: str, target_runs: int) -> List[dict]:
    run_paths = sorted(glob.glob(os.path.join(out_dir, "prior_run_*.json")))
    run_paths = run_paths[:target_runs]

    runs = []
    for path in run_paths:
        with open(path, "r", encoding="utf-8") as f:
            runs.append(json.load(f))

    return runs


def run_stability_experiment(args):
    os.makedirs(args.out_dir, exist_ok=True)
    if not hasattr(args, "multi_agent"):
        args.multi_agent = not getattr(args, "single_agent", False)

    if Together is None:
        raise RuntimeError(
            "Missing Python package: together. Install it with `pip install together` "
            "in the environment where you run this script."
        )

    api_key = load_api_key()
    if not api_key:
        raise RuntimeError("Missing TOGETHER_API_KEY environment variable.")

    print(f"[Load] CSV: {args.csv}")
    print(f"[Load] IDF: {args.idf}")
    if getattr(args, "reference_idf", None):
        print(f"[Load] Reference IDF for control diff: {args.reference_idf}")
    df_raw = pd.read_csv(args.csv)
    _, zones = build_multizone_timeseries(df_raw)
    if args.max_zones:
        zones = zones[: args.max_zones]

    if not zones:
        raise ValueError("No zones found from CSV.")

    with open(args.idf, "r", encoding="utf-8") as f:
        idf_text = f.read()
    reference_idf_text = None
    if getattr(args, "reference_idf", None):
        with open(args.reference_idf, "r", encoding="utf-8") as f:
            reference_idf_text = f.read()

    client = Together(api_key=api_key)
    if getattr(args, "resume", False):
        runs = load_existing_prior_runs(args.out_dir, args.runs)
        print(f"[Resume] Loaded existing prior runs: {len(runs)}/{args.runs}")
    else:
        runs = []

    start_run_idx = len(runs) + 1
    if start_run_idx > args.runs:
        print(
            f"[Resume] Target already complete: {len(runs)}/{args.runs}. "
            "Re-aggregating only."
        )

    for run_idx in range(start_run_idx, args.runs + 1):
        run_seed = None if args.seed is None else args.seed + run_idx - 1
        print(
            f"\n[Run {run_idx}/{args.runs}] "
            f"model={args.model}, temperature={args.temperature}, seed={run_seed}"
        )
        prior = extract_idf_priors_once(
            idf_text=idf_text,
            zones=zones,
            client=client,
            model=args.model,
            temperature=args.temperature,
            seed=run_seed,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            sleep_s=args.sleep_s,
            include_connections=getattr(args, "include_connections", False),
            raw_output_dir=os.path.join(args.out_dir, "raw_llm_outputs"),
            run_idx=run_idx,
            thermal_context_chars=getattr(args, "thermal_context_chars", 20000),
            schedule_context_chars=getattr(args, "schedule_context_chars", 20000),
            use_label_output=not getattr(args, "numeric_output", False),
            multi_agent=getattr(args, "multi_agent", True),
            reference_idf_text=reference_idf_text,
        )
        runs.append(prior)

        with open(os.path.join(args.out_dir, f"prior_run_{run_idx:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(prior, f, indent=2)

    for prior in runs:
        control_summary = prior.get("control_summary", {})
        needs_control_refresh = "control_matrix" not in prior
        if reference_idf_text is not None and not control_summary.get("has_reference_idf", False):
            needs_control_refresh = True
        if needs_control_refresh:
            prior.update(
                extract_control_priors(
                    idf_text=idf_text,
                    zones=prior.get("zones", zones),
                    reference_idf_text=reference_idf_text,
                )
            )
        llm_records = prior.get("llm_control_calibration_records", {})
        has_operator_records = (
            isinstance(llm_records, dict)
            and any(
                isinstance(v, dict)
                and v.get("adaptation_operator")
                and v.get("dominant_regime")
                for v in llm_records.values()
            )
        )
        if reference_idf_text is not None and (
            "llm_control_calibration_matrix" not in prior or not has_operator_records
        ):
            sem_matrix_existing = np.array(prior.get("semantic_matrix", []), dtype=float)
            prior.update(
                extract_llm_control_calibration_priors(
                    control_prior=prior,
                    zones=prior.get("zones", zones),
                    semantic_matrix=sem_matrix_existing,
                    client=client,
                    model=args.model,
                    temperature=args.temperature,
                    seed=args.seed,
                    batch_size=args.batch_size,
                    max_tokens=args.max_tokens,
                    sleep_s=args.sleep_s,
                    raw_output_dir=os.path.join(args.out_dir, "raw_llm_outputs"),
                    run_idx=0,
                )
            )

    df_long = flatten_semantic_runs(runs)
    df_reasons = flatten_reason_traces(runs)
    summary = summarize_variability(df_long)
    adj_summary = summarize_adjacency_variability(runs)
    agreement_summary = build_semantic_agreement_summary(df_long)
    audit_report = build_audit_report(df_long, df_reasons, adj_summary)

    df_long.to_csv(os.path.join(args.out_dir, "semantic_values_long.csv"), index=False)
    if not df_reasons.empty:
        df_reasons.to_csv(os.path.join(args.out_dir, "agent_reason_traces.csv"), index=False)
        with open(os.path.join(args.out_dir, "agent_reason_traces.json"), "w", encoding="utf-8") as f:
            json.dump(df_reasons.to_dict(orient="records"), f, indent=2)
    summary.to_csv(os.path.join(args.out_dir, "semantic_variability_summary.csv"), index=False)
    agreement_summary.to_csv(os.path.join(args.out_dir, "semantic_agreement_summary.csv"), index=False)
    adj_summary.to_csv(os.path.join(args.out_dir, "adjacency_variability_summary.csv"), index=False)
    with open(os.path.join(args.out_dir, "llm_prior_audit_report.json"), "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    with open(os.path.join(args.out_dir, "all_prior_runs.json"), "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)

    save_plots(df_long, summary, runs, args.out_dir)

    median_prior = build_aggregated_prior(runs, method="median", quantize=False)
    median_quantized_prior = build_aggregated_prior(runs, method="median", quantize=True)
    mean_prior = build_aggregated_prior(runs, method="mean", quantize=False)
    mean_quantized_prior = build_aggregated_prior(runs, method="mean", quantize=True)

    with open(os.path.join(args.out_dir, "median_prior.json"), "w", encoding="utf-8") as f:
        json.dump(median_prior, f, indent=2)
    with open(os.path.join(args.out_dir, "median_quantized_prior.json"), "w", encoding="utf-8") as f:
        json.dump(median_quantized_prior, f, indent=2)
    with open(os.path.join(args.out_dir, "mean_prior.json"), "w", encoding="utf-8") as f:
        json.dump(mean_prior, f, indent=2)
    with open(os.path.join(args.out_dir, "mean_quantized_prior.json"), "w", encoding="utf-8") as f:
        json.dump(mean_quantized_prior, f, indent=2)

    print("\n[Done] Stability outputs saved:")
    print(f"  {os.path.join(args.out_dir, 'semantic_variability_summary.csv')}")
    print(f"  {os.path.join(args.out_dir, 'semantic_agreement_summary.csv')}")
    print(f"  {os.path.join(args.out_dir, 'llm_prior_audit_report.json')}")
    if not df_reasons.empty:
        print(f"  {os.path.join(args.out_dir, 'agent_reason_traces.csv')}")
    print(f"  {os.path.join(args.out_dir, 'mean_prior.json')}")
    print(f"  {os.path.join(args.out_dir, 'mean_quantized_prior.json')}")
    print(f"  {os.path.join(args.out_dir, 'median_prior.json')}")
    print(f"  {os.path.join(args.out_dir, 'semantic_boxplot.png')}")

    return {
        "runs": runs,
        "mean_prior": mean_prior,
        "mean_quantized_prior": mean_quantized_prior,
        "median_prior": median_prior,
        "median_quantized_prior": median_quantized_prior,
    }


def building_name_from_csv(csv_path: str) -> str:
    name = os.path.splitext(os.path.basename(csv_path))[0]
    return re.sub(r"_fewshot$|_few_shot$|_fulldata$|_full_data$", "", name)

def build_zone_cluster_evidence(idf_text: str, zones: List[str], prior: dict) -> List[dict]:
    sem = np.array(prior.get("semantic_matrix", []), dtype=float)
    if sem.ndim != 2 or sem.shape[0] != len(zones):
        sem = np.ones((len(zones), len(SEMANTIC_NAMES)), dtype=float) * 0.5
    if sem.shape[1] < len(SEMANTIC_NAMES):
        sem = np.hstack(
            [sem, np.ones((len(zones), len(SEMANTIC_NAMES) - sem.shape[1])) * 0.5]
        )
    sem = sem[:, : len(SEMANTIC_NAMES)]

    control_matrix = np.array(prior.get("control_matrix", []), dtype=float)
    has_control = control_matrix.ndim == 2 and control_matrix.shape[0] == len(zones)

    idf_blocks = [block for block in idf_text.split(";") if block.strip()]
    evidence = []
    for i, zone in enumerate(zones):
        z_clean = ultra_clean(zone)
        related = [block for block in idf_blocks if z_clean and z_clean in ultra_clean(block)]
        text = "\n".join(related).lower()
        type_counts = {
            "surface": sum("buildingsurface:detailed" in b.lower() for b in related),
            "fenestration": sum("fenestrationsurface:detailed" in b.lower() for b in related),
            "people": sum(b.lstrip().lower().startswith("people") for b in related),
            "lights": sum(b.lstrip().lower().startswith("lights") for b in related),
            "equipment": sum("equipment" in b.lower() for b in related),
            "thermostat": sum("thermostat" in b.lower() for b in related),
            "ideal_loads": sum("zonehvac:ideal" in b.lower() for b in related),
        }
        schedule_tokens = sorted(
            set(re.findall(r"([-A-Za-z0-9_ ]*(?:schedule|setpoint|occupancy|activity)[-A-Za-z0-9_ ]*)", text))
        )[:10]
        semantic = {
            name: round(float(sem[i, j]), 3)
            for j, name in enumerate(SEMANTIC_NAMES)
        }
        item = {
            "zone": zone,
            "zone_name_tokens": re.findall(r"[A-Za-z]+", zone.lower())[:12],
            "semantic_prior": semantic,
            "idf_object_counts": type_counts,
            "schedule_name_hints": schedule_tokens,
        }
        if has_control:
            item["control_prior"] = {
                name: round(float(control_matrix[i, j]), 3)
                for j, name in enumerate(CONTROL_NAMES[: control_matrix.shape[1]])
            }
        evidence.append(item)
    return evidence

def build_llm_zone_cluster_prompts(building_name: str, zones: List[str], evidence: List[dict]):
    system_prompt = """
You are a building thermal-dynamics modeling expert. Your task is to form
zone clusters for a multi-zone GRU surrogate model using only IDF-derived
evidence and the provided semantic priors. Do not use measured temperature
curves, model errors, or OOD test results.

Return strict JSON only. No markdown.
""".strip()

    schema = {
        "n_groups": 3,
        "groups": [
            {
                "group_id": 0,
                "label": "short readable mechanism label",
                "thermal_hypothesis": "why these zones should share dynamics",
                "mechanism_profile": {
                    "thermal_inertia": "low/medium/high",
                    "internal_gain_volatility": "low/medium/high",
                    "envelope_exposure": "low/medium/high",
                    "control_sensitivity": "low/medium/high",
                    "coupling_role": "core/perimeter/transition/service/high_gain/mixed",
                },
                "zones": ["exact zone names from input"],
                "confidence": 0.0,
            }
        ],
        "zone_records": {} if len(zones) > 60 else {
            "exact zone name": {
                "group_id": 0,
                "confidence": 0.0,
                "evidence": ["short evidence item 1", "short evidence item 2"],
            }
        },
    }
    zone_record_instruction = (
        "- For this large building, zone_records may be an empty object; the complete assignment must be in groups[*].zones."
        if len(zones) > 60
        else "- Fill zone_records for every zone, using the same group_id as groups[*].zones."
    )

    user_prompt = f"""
Building: {building_name}

Goal:
Cluster zones by expected thermal-response/error mode for a multi-zone thermal
rollout model. The clusters will become separate recurrent branches, so each
group should contain zones that plausibly need similar temporal dynamics.

Use these mechanisms, in priority order:
1. Internal-gain volatility and schedule/control volatility.
2. Envelope/weather exposure and solar/window indications.
3. HVAC/control setpoint evidence and likely response sensitivity.
4. Thermal inertia and physics-prior trust from the semantic prior.
5. Zone role inferred from name only when IDF evidence is ambiguous.

Important constraints:
- Choose n_groups adaptively from 2 to 5.
- Prefer 3 or 4 groups unless the evidence strongly suggests otherwise.
- Avoid singleton groups unless the zone is a clear outlier such as kitchen,
  cafeteria, gym, server/equipment, laboratory, healthcare treatment, lobby, or
  another high-gain/special-control space.
- Do not simply group by adjacency degree or by floor/number suffix.
- Use exact zone names from the input.
- Every zone must appear exactly once in exactly one groups[*].zones list.
{zone_record_instruction}
- This is a building-specific cluster prior, not a replacement for the existing
  semantic_matrix prior.

JSON schema example:
{json.dumps(schema, indent=2)}

Zone evidence:
{json.dumps(evidence, indent=2)}
""".strip()
    return system_prompt, user_prompt

def validate_llm_cluster_prior(raw: dict, zones: List[str]) -> dict:
    zone_to_group = {}
    zone_records = raw.get("zone_records", {})
    if isinstance(zone_records, dict):
        for z, rec in zone_records.items():
            matched = match_zone_name(z, zones)
            if matched and isinstance(rec, dict) and "group_id" in rec:
                zone_to_group[matched] = int(rec["group_id"])

    for group in raw.get("groups", []) if isinstance(raw.get("groups", []), list) else []:
        gid = int(group.get("group_id", group.get("id", 0)))
        for z in group.get("zones", []):
            matched = match_zone_name(z, zones)
            if matched:
                zone_to_group[matched] = gid

    if not zone_to_group:
        raise ValueError(
            "LLM cluster output contains no usable groups or zone_records assignments."
        )

    missing = [z for z in zones if z not in zone_to_group]
    if missing:
        fallback_gid = max(zone_to_group.values(), default=-1) + 1
        for z in missing:
            zone_to_group[z] = fallback_gid

    used = sorted(set(zone_to_group.values()))
    remap = {old: new for new, old in enumerate(used)}
    zone_cluster_ids = [remap[zone_to_group[z]] for z in zones]
    n_groups = int(max(zone_cluster_ids) + 1) if zone_cluster_ids else 0

    groups = []
    raw_groups = raw.get("groups", []) if isinstance(raw.get("groups", []), list) else []
    raw_by_old = {int(g.get("group_id", g.get("id", 0))): g for g in raw_groups if isinstance(g, dict)}
    for old_gid, new_gid in remap.items():
        raw_group = raw_by_old.get(old_gid, {})
        group_zones = [z for z, gid in zone_to_group.items() if remap[gid] == new_gid]
        groups.append({
            "group_id": int(new_gid),
            "label": raw_group.get("label", f"LLM cluster {new_gid}"),
            "thermal_hypothesis": raw_group.get("thermal_hypothesis", ""),
            "mechanism_profile": raw_group.get("mechanism_profile", {}),
            "zones": group_zones,
            "confidence": float(raw_group.get("confidence", 0.5)),
            "decision_basis": raw_group.get("decision_basis", raw_group.get("basis", "")),
            "reason": raw_group.get("reason", ""),
        })

    clean_records = {}
    for z in zones:
        raw_rec = {}
        for candidate, rec in zone_records.items() if isinstance(zone_records, dict) else []:
            if match_zone_name(candidate, [z]) == z and isinstance(rec, dict):
                raw_rec = rec
                break
        clean_records[z] = {
            "group_id": int(remap[zone_to_group[z]]),
            "confidence": float(raw_rec.get("confidence", 0.5)),
            "evidence": raw_rec.get("evidence", []),
            "decision": raw_rec.get("decision", ""),
        }

    return {
        "source": "llm_idf_semantic_cluster_prior",
        "n_groups": n_groups,
        "zones": zones,
        "zone_cluster_ids": zone_cluster_ids,
        "groups": groups,
        "zone_records": clean_records,
        "raw_llm_cluster_output": raw,
    }

def build_consensus_cluster_prior(
    cluster_runs: List[dict],
    zones: List[str],
    building_name: str,
    threshold: float = 0.6,
) -> dict:
    valid_runs = [
        run for run in cluster_runs
        if isinstance(run.get("zone_cluster_ids"), list)
        and len(run.get("zone_cluster_ids")) == len(zones)
    ]
    if not valid_runs:
        raise ValueError("No valid LLM cluster runs to aggregate.")

    n_zones = len(zones)
    coassign = np.zeros((n_zones, n_zones), dtype=float)
    for run in valid_runs:
        labels = [int(x) for x in run["zone_cluster_ids"]]
        for i in range(n_zones):
            for j in range(n_zones):
                if labels[i] == labels[j]:
                    coassign[i, j] += 1.0
    coassign /= float(len(valid_runs))

    adjacency = coassign >= float(threshold)
    np.fill_diagonal(adjacency, True)
    labels = np.full(n_zones, -1, dtype=int)
    current = 0
    for i in range(n_zones):
        if labels[i] >= 0:
            continue
        stack = [i]
        labels[i] = current
        while stack:
            u = stack.pop()
            for v in np.where(adjacency[u])[0]:
                if labels[v] < 0:
                    labels[v] = current
                    stack.append(v)
        current += 1

    individual_ks = [len(set(int(x) for x in run["zone_cluster_ids"])) for run in valid_runs]
    median_individual_k = int(np.median(individual_ks)) if individual_ks else 0
    if current <= 1 and median_individual_k > 1:
        best_idx = 0
        best_score = float("inf")
        for run_idx, run in enumerate(valid_runs):
            run_labels = np.asarray([int(x) for x in run["zone_cluster_ids"]], dtype=int)
            run_same = (run_labels[:, None] == run_labels[None, :]).astype(float)
            score = float(np.mean(np.abs(run_same - coassign)))
            if score < best_score:
                best_score = score
                best_idx = run_idx

        medoid = json.loads(json.dumps(valid_runs[best_idx]))
        medoid["source"] = "llm_idf_semantic_cluster_prior_consensus_medoid"
        medoid["building_name"] = building_name
        medoid["consensus"] = {
            "num_runs": len(valid_runs),
            "threshold": float(threshold),
            "method": "medoid_fallback_after_connected_component_collapse",
            "reason": "Thresholded co-assignment graph collapsed to one connected component; selected the run closest to the average co-assignment matrix to avoid chain-effect over-merging.",
            "connected_component_k": int(current),
            "individual_run_ks": [int(k) for k in individual_ks],
            "selected_run_index": int(best_idx + 1),
            "selected_run_score": float(best_score),
            "coassignment_matrix": coassign.round(4).tolist(),
        }
        print(
            f"[LLM Cluster Consensus] Connected-component consensus collapsed to K={current}; "
            f"using medoid run {best_idx + 1}/{len(valid_runs)} with "
            f"K={medoid.get('n_groups')} and score={best_score:.4f}",
            flush=True,
        )
        return medoid

    groups = []
    zone_records = {}
    for gid in range(current):
        idx = np.where(labels == gid)[0]
        group_zones = [zones[i] for i in idx]
        candidate_labels = []
        candidate_hypotheses = []
        zone_evidence = {z: [] for z in group_zones}
        for run in valid_runs:
            run_labels = [int(x) for x in run["zone_cluster_ids"]]
            run_groups = {
                int(g.get("group_id", -1)): g
                for g in run.get("groups", [])
                if isinstance(g, dict)
            }
            for i in idx:
                raw_group = run_groups.get(run_labels[i], {})
                if raw_group.get("label"):
                    candidate_labels.append(str(raw_group["label"]))
                if raw_group.get("thermal_hypothesis"):
                    candidate_hypotheses.append(str(raw_group["thermal_hypothesis"]))
                rec = run.get("zone_records", {}).get(zones[i], {})
                if isinstance(rec, dict):
                    zone_evidence[zones[i]].extend([str(x) for x in rec.get("evidence", [])[:2]])

        label = max(set(candidate_labels), key=candidate_labels.count) if candidate_labels else f"Consensus cluster {gid}"
        hypothesis = (
            max(set(candidate_hypotheses), key=candidate_hypotheses.count)
            if candidate_hypotheses
            else "Zones are grouped by repeated LLM agreement over IDF-derived thermal mechanisms."
        )
        groups.append({
            "group_id": int(gid),
            "label": label,
            "thermal_hypothesis": hypothesis,
            "mechanism_profile": {},
            "zones": group_zones,
            "confidence": float(np.mean(coassign[np.ix_(idx, idx)])) if len(idx) else 0.0,
        })
        for z in group_zones:
            uniq_evidence = list(dict.fromkeys(zone_evidence[z]))[:5]
            zone_records[z] = {
                "group_id": int(gid),
                "confidence": float(np.mean(coassign[zones.index(z), idx])) if len(idx) else 0.0,
                "evidence": uniq_evidence,
            }

    return {
        "source": "llm_idf_semantic_cluster_prior_consensus",
        "building_name": building_name,
        "n_groups": int(current),
        "zones": zones,
        "zone_cluster_ids": labels.astype(int).tolist(),
        "groups": groups,
        "zone_records": zone_records,
        "consensus": {
            "num_runs": len(valid_runs),
            "threshold": float(threshold),
            "coassignment_matrix": coassign.round(4).tolist(),
        },
        "cluster_runs": valid_runs,
    }

def generate_llm_cluster_prior(
    idf_text: str,
    zones: List[str],
    prior: dict,
    building_name: str,
    client: Together,
    model: str,
    temperature: float,
    seed: Optional[int],
    max_tokens: int,
    raw_output_dir: Optional[str],
    run_idx: int = 1,
    total_runs: int = 1,
) -> dict:
    print(
        f"[LLM Cluster {run_idx}/{total_runs}] building={building_name}, zones={len(zones)}, "
        f"model={model}, temperature={temperature}, seed={seed}",
        flush=True,
    )
    evidence = build_zone_cluster_evidence(idf_text, zones, prior)
    print(f"[LLM Cluster {run_idx}/{total_runs}] IDF/semantic evidence prepared.", flush=True)

    system_prompt, user_prompt = build_llm_zone_cluster_prompts(building_name, zones, evidence)
    print(
        f"[LLM Cluster {run_idx}/{total_runs}] Prompt prepared. Calling LLM "
        f"(max_tokens={max_tokens})...",
        flush=True,
    )
    cluster_prior = None
    retry_prompt = user_prompt
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            raw = call_llm_json(
                client=client,
                model=model,
                temperature=temperature,
                seed=None if seed is None else seed + attempt - 1,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                user_prompt=retry_prompt,
                raw_output_dir=raw_output_dir,
                run_idx=run_idx,
                batch_idx=attempt,
                agent_name="zone_cluster_prior",
                allow_empty_fallback=False,
            )
            print(f"[LLM Cluster {run_idx}/{total_runs}] LLM returned. Validating cluster JSON...", flush=True)
            cluster_prior = validate_llm_cluster_prior(raw, zones)
            if len(zones) >= 20 and int(cluster_prior.get("n_groups", 0)) <= 1:
                raise ValueError(
                    f"Degenerate LLM cluster prior K={cluster_prior.get('n_groups')} for {len(zones)} zones."
                )
            break
        except Exception as e:
            print(
                f"[LLM Cluster {run_idx}/{total_runs}] Invalid cluster output on attempt "
                f"{attempt}/{max_attempts}: {e}",
                flush=True,
            )
            if attempt >= max_attempts:
                raise
            retry_prompt = user_prompt + f"""

Previous attempt was invalid: {e}
Return one strict JSON object only. Do not omit groups. Every input zone must appear exactly once in groups[*].zones. For buildings with many zones, keep zone_records as {{}} and rely on groups[*].zones.
""".strip()

    if cluster_prior is None:
        raise RuntimeError("LLM cluster prior generation failed without an exception.")
    labels = cluster_prior.get("zone_cluster_ids", [])
    if labels:
        counts = {}
        for gid in labels:
            gid = int(gid)
            counts[gid] = counts.get(gid, 0) + 1
        print(
            f"[LLM Cluster {run_idx}/{total_runs}] Validated K={cluster_prior.get('n_groups')}, "
            f"counts={counts}",
            flush=True,
        )
        for group in cluster_prior.get("groups", []):
            print(
                f"  [Cluster {group.get('group_id')}] "
                f"{group.get('label', '')} | n={len(group.get('zones', []))}",
                flush=True,
            )
            hypothesis = group.get("thermal_hypothesis", "")
            if hypothesis:
                print(f"    hypothesis: {hypothesis}", flush=True)
    return cluster_prior

def generate_consensus_llm_cluster_prior(
    idf_text: str,
    zones: List[str],
    prior: dict,
    building_name: str,
    client: Together,
    model: str,
    temperature: float,
    seed: Optional[int],
    max_tokens: int,
    raw_output_dir: Optional[str],
    cluster_runs_dir: str,
    cluster_runs: int = 10,
    force_cluster_rerun: bool = False,
    consensus_threshold: float = 0.6,
    sleep_s: float = 0.5,
) -> dict:
    os.makedirs(cluster_runs_dir, exist_ok=True)
    if force_cluster_rerun:
        print(f"[LLM Cluster Consensus] force_cluster_rerun=True; ignoring existing cluster_run_*.json", flush=True)

    runs = []
    for run_idx in range(1, int(cluster_runs) + 1):
        run_path = os.path.join(cluster_runs_dir, f"cluster_run_{run_idx:03d}.json")
        if not force_cluster_rerun and os.path.exists(run_path):
            print(f"[LLM Cluster Consensus] Loaded existing run {run_idx}/{cluster_runs}: {run_path}", flush=True)
            with open(run_path, "r", encoding="utf-8") as f:
                cached_run = json.load(f)
            if len(zones) >= 20 and int(cached_run.get("n_groups", 0)) <= 1:
                print(
                    f"[LLM Cluster Consensus] Cached run is degenerate "
                    f"(K={cached_run.get('n_groups')}); regenerating {run_path}",
                    flush=True,
                )
            else:
                runs.append(cached_run)
                continue

        run_seed = None if seed is None else seed + run_idx - 1
        cluster_prior = generate_llm_cluster_prior(
            idf_text=idf_text,
            zones=zones,
            prior=prior,
            building_name=building_name,
            client=client,
            model=model,
            temperature=temperature,
            seed=run_seed,
            max_tokens=max_tokens,
            raw_output_dir=raw_output_dir,
            run_idx=run_idx,
            total_runs=cluster_runs,
        )
        with open(run_path, "w", encoding="utf-8") as f:
            json.dump(cluster_prior, f, indent=2)
        print(f"[LLM Cluster Consensus] Saved run {run_idx}/{cluster_runs}: {run_path}", flush=True)
        runs.append(cluster_prior)
        if sleep_s and run_idx < int(cluster_runs):
            time.sleep(float(sleep_s))

    print(
        f"[LLM Cluster Consensus] Aggregating {len(runs)} runs "
        f"with threshold={consensus_threshold}",
        flush=True,
    )
    consensus = build_consensus_cluster_prior(
        cluster_runs=runs,
        zones=zones,
        building_name=building_name,
        threshold=consensus_threshold,
    )
    counts = {}
    for gid in consensus["zone_cluster_ids"]:
        gid = int(gid)
        counts[gid] = counts.get(gid, 0) + 1
    print(
        f"[LLM Cluster Consensus] Final K={consensus['n_groups']}, counts={counts}",
        flush=True,
    )
    for group in consensus.get("groups", []):
        print(
            f"  [Consensus Cluster {group.get('group_id')}] "
            f"{group.get('label', '')} | n={len(group.get('zones', []))}",
            flush=True,
        )
    return consensus

# End of the public semantic and cluster-prior core.
