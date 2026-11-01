"""Original deterministic correction operator, plus standalone application below."""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import math
import re
import json
import time
import numpy as np
import pandas as pd
import config as cfg

def clean_token(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())

def infer_temperature_zones(frame: pd.DataFrame) -> list[str]:
    zones = []
    for column in frame.columns:
        name = str(column)
        for prefix in ("T_pred__", "T_in__", "T_true__"):
            if name.startswith(prefix):
                zones.append(name.replace(prefix, "", 1))
                break
    return list(dict.fromkeys(zones))


def find_zone_column(frame: pd.DataFrame, zone: str, prediction: bool) -> str | None:
    prefixes = ("T_pred__", "T_in__") if prediction else ("T_in__", "T_true__")
    for prefix in prefixes:
        candidate = f"{prefix}{zone}"
        if candidate in frame.columns:
            return candidate
    key = clean_token(zone)
    for column in frame.columns:
        name = str(column)
        if key in clean_token(name) and "temperature" in name.lower():
            return column
    return None


def get_zone_record(records: dict, zone: str) -> dict:
    if zone in records and isinstance(records[zone], dict):
        return records[zone]
    key = clean_token(zone)
    for name, record in records.items():
        if clean_token(name) == key and isinstance(record, dict):
            return record
    return {}


def aligned_daily_profile(profile: object, length: int) -> np.ndarray | None:
    if profile is None:
        return None
    values = np.asarray(profile, dtype=float)
    if values.size == 96 and length % 95 == 0:
        return np.tile(values[1:], length // 95)
    return np.tile(values, int(math.ceil(length / len(values))))[:length]


def deterministic_operator(control_record: dict, rule_record: dict) -> str:
    operator = str(rule_record.get("adaptation_operator") or "")
    if operator:
        return operator
    heating = abs(float(control_record.get("heating_shift_mean_abs_c") or 0.0))
    cooling = abs(float(control_record.get("cooling_shift_mean_abs_c") or 0.0))
    if heating > 1.5 * cooling:
        return "heating_boundary_push"
    if cooling > 1.5 * heating:
        return "cooling_boundary_push"
    return "center_translation"


def nominal_correction(
    prediction: np.ndarray,
    control_record: dict,
    rule_record: dict,
) -> tuple[np.ndarray, np.ndarray, str]:
    length = len(prediction)
    old_h = aligned_daily_profile(control_record.get("reference_heating_profile_24h"), length)
    old_c = aligned_daily_profile(control_record.get("reference_cooling_profile_24h"), length)
    new_h = aligned_daily_profile(control_record.get("heating_profile_24h"), length)
    new_c = aligned_daily_profile(control_record.get("cooling_profile_24h"), length)
    if any(value is None for value in (old_h, old_c, new_h, new_c)):
        return np.zeros(length), np.zeros(length), "missing_profiles"

    operator = deterministic_operator(control_record, rule_record)
    old_center = 0.5 * (old_h + old_c)
    new_center = 0.5 * (new_h + new_c)
    old_band = np.where(np.abs(old_c - old_h) < 0.25, 0.25, old_c - old_h)
    new_band = np.where(np.abs(new_c - new_h) < 0.25, 0.25, new_c - new_h)
    center = new_center - old_center
    band = new_center + ((prediction - old_center) / old_band) * new_band - prediction
    heating = np.maximum(new_h - prediction, 0.0)
    cooling = -np.maximum(prediction - new_c, 0.0)

    if operator == "center_translation":
        candidate = center
    elif operator == "deadband_coordinate_mapping":
        candidate = band
    elif operator == "heating_boundary_push":
        candidate = heating
    elif operator == "cooling_boundary_push":
        candidate = cooling
    elif operator == "mixed_deadband_mapping":
        candidate = 0.5 * (center + band) + heating + cooling
    else:
        candidate = np.zeros(length)

    changed = np.logical_or(np.abs(new_h - old_h) > 1e-8, np.abs(new_c - old_c) > 1e-8)
    limit = np.maximum(np.abs(new_h - old_h), np.abs(new_c - old_c))
    return changed.astype(float) * candidate, limit, operator


def apply_strengths(
    prediction_frame: pd.DataFrame,
    control_prior: dict,
    strengths: dict[str, float],
    prepared_candidates: dict[str, dict] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    corrected = prediction_frame.copy()
    candidates = (
        prepare_correction_candidates(prediction_frame, control_prior)
        if prepared_candidates is None else prepared_candidates
    )
    report = []
    for zone, prepared in candidates.items():
        prediction_column = prepared["prediction_column"]
        prediction = prepared["prediction"]
        candidate = prepared["candidate"]
        limit = prepared["limit"]
        operator = prepared["operator"]
        strength = float(strengths.get(zone, strengths.get(clean_token(zone), 0.0)))
        correction = np.clip(strength * candidate, -limit, limit)
        corrected[prediction_column] = prediction + correction
        report.append({
            "Zone": zone,
            "Strength": strength,
            "Operator": operator,
            "Correction_MAE_C": float(np.mean(np.abs(correction))),
            "Correction_RMS_C": float(np.sqrt(np.mean(correction ** 2))),
            "Modified_Points": int(np.sum(np.abs(correction) > 1e-12)),
        })
    return corrected, report


def episode_gate_profile(
    prior_record: dict,
    control_record: dict,
    length: int,
) -> np.ndarray:
    """Expand episode-specific gates from one 24-hour schedule to all rows."""
    episode_records = prior_record.get("episode_gate_records", [])
    if not isinstance(episode_records, list) or not episode_records:
        # Backward-compatible fallback for older scalar priors.
        return np.full(length, float(prior_record.get("correction_gate", 0.0)))

    schedule_profile = control_record.get("reference_heating_profile_24h")
    if schedule_profile is None:
        schedule_profile = control_record.get("heating_profile_24h")
    try:
        points_per_day = len(schedule_profile)
    except TypeError:
        points_per_day = 96
    if points_per_day <= 0:
        points_per_day = 96

    daily_gate = np.zeros(points_per_day, dtype=float)
    for episode in episode_records:
        if not isinstance(episode, dict):
            continue
        start_hour = float(episode.get("start_hour", 0.0))
        end_hour = float(episode.get("end_hour", 0.0))
        start = int(round(start_hour / 24.0 * points_per_day))
        end = int(round(end_hour / 24.0 * points_per_day))
        start = int(np.clip(start, 0, points_per_day))
        end = int(np.clip(end, 0, points_per_day))
        if end <= start:
            continue
        daily_gate[start:end] = float(episode.get("correction_gate", 0.0))

    aligned = aligned_daily_profile(daily_gate, length)
    return np.zeros(length, dtype=float) if aligned is None else aligned


def apply_episode_gates(
    prediction_frame: pd.DataFrame,
    control_prior: dict,
    prior: dict,
    prepared_candidates: dict[str, dict] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply each LLM gate only within its corresponding schedule episode."""
    corrected = prediction_frame.copy()
    candidates = (
        prepare_correction_candidates(prediction_frame, control_prior)
        if prepared_candidates is None else prepared_candidates
    )
    control_records = control_prior.get("control_records", {})
    prior_records = prior.get("zone_gate_records", {})
    if isinstance(prior_records, list):
        prior_records = {
            item["zone"]: item
            for item in prior_records
            if isinstance(item, dict) and item.get("zone")
        }

    report = []
    for zone, prepared in candidates.items():
        prediction_column = prepared["prediction_column"]
        prediction = prepared["prediction"]
        candidate = prepared["candidate"]
        limit = prepared["limit"]
        operator = prepared["operator"]
        control_record = get_zone_record(control_records, zone)
        prior_record = get_zone_record(prior_records, zone)
        gates = episode_gate_profile(prior_record, control_record, len(prediction))
        correction = np.clip(gates * candidate, -limit, limit)
        corrected[prediction_column] = prediction + correction

        episode_records = prior_record.get("episode_gate_records", [])
        episode_values = [
            float(item.get("correction_gate", 0.0))
            for item in episode_records
            if isinstance(item, dict)
        ]
        active = np.abs(candidate) > 1e-12
        applied_mean = float(np.mean(gates[active])) if active.any() else 0.0
        report.append({
            "Zone": zone,
            "Strength": float(prior_record.get("mean_episode_gate", applied_mean)),
            "Applied_Mean_Gate": applied_mean,
            "Min_Episode_Gate": min(episode_values, default=0.0),
            "Max_Episode_Gate": max(episode_values, default=0.0),
            "Episode_Count": len(episode_values),
            "Operator": operator,
            "Correction_MAE_C": float(np.mean(np.abs(correction))),
            "Correction_RMS_C": float(np.sqrt(np.mean(correction ** 2))),
            "Modified_Points": int(np.sum(np.abs(correction) > 1e-12)),
        })
    return corrected, report


def prepare_correction_candidates(
    prediction_frame: pd.DataFrame,
    control_prior: dict,
) -> dict[str, dict]:
    """Build the Rule-1 correction once so fixed and LLM gates can share it."""
    control_records = control_prior.get("control_records", {})
    rule_records = control_prior.get("rule_control_calibration_records", {})
    prepared = {}
    for zone in infer_temperature_zones(prediction_frame):
        prediction_column = find_zone_column(prediction_frame, zone, prediction=True)
        control = get_zone_record(control_records, zone)
        if prediction_column is None or not control or not control.get("control_changed_flag"):
            continue
        prediction = prediction_frame[prediction_column].to_numpy(dtype=float)
        candidate, limit, operator = nominal_correction(
            prediction, control, get_zone_record(rule_records, zone)
        )
        prepared[zone] = {
            "prediction_column": prediction_column,
            "prediction": prediction,
            "candidate": candidate,
            "limit": limit,
            "operator": operator,
        }
    return prepared


def parse_time_axis(frame: pd.DataFrame, year: int = 2023) -> pd.Series | None:
    if "datetime" in frame.columns:
        values = pd.to_datetime(frame["datetime"], errors="coerce")
        if not values.isna().all():
            return values
    column = next((value for value in frame.columns if "date/time" in str(value).lower()), None)
    if column is None:
        return None
    text = frame[column].astype(str).str.strip()
    parts = text.str.extract(r"^(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2}):(\d{2})")
    numeric = parts.apply(pd.to_numeric, errors="coerce")
    dates = pd.to_datetime(pd.DataFrame({
        "year": np.full(len(text), year), "month": numeric[0], "day": numeric[1],
    }), errors="coerce")
    # EnergyPlus 24:00 is the following midnight, as in the original model loader.
    seconds = numeric[2] * 3600 + numeric[3] * 60 + numeric[4]
    values = dates + pd.to_timedelta(seconds, unit="s")
    unmatched = parts[0].isna()
    if unmatched.any():
        values.loc[unmatched] = pd.to_datetime(text.loc[unmatched], errors="coerce")
    return values


def make_target_eval_frame(frame: pd.DataFrame, predict_period: int = 96) -> pd.DataFrame:
    axis = parse_time_axis(frame)
    if axis is not None:
        frame = frame.copy()
        frame["datetime"] = axis
    if axis is not None and not axis.isna().all() and bool((axis.dt.month == 8).any()):
        frame = frame.loc[axis.dt.month == 8].reset_index(drop=True)
    days = len(frame) // predict_period
    pieces = [
        frame.iloc[day * predict_period + 1:(day + 1) * predict_period]
        for day in range(days)
    ]
    return pd.concat(pieces, ignore_index=True) if pieces else frame.reset_index(drop=True)


def build_alignment(
    prediction_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return row positions shared by every zone and post-processing method."""
    prediction_time = parse_time_axis(prediction_frame)
    target_time = parse_time_axis(target_frame)
    if prediction_time is not None and target_time is not None:
        left = pd.DataFrame({
            "time": prediction_time.dt.strftime("%m-%d %H:%M:%S"),
            "prediction_position": np.arange(len(prediction_frame)),
        }).dropna(subset=["time"])
        right = pd.DataFrame({
            "time": target_time.dt.strftime("%m-%d %H:%M:%S"),
            "target_position": np.arange(len(target_frame)),
        }).dropna(subset=["time"])
        left["occurrence"] = left.groupby("time").cumcount()
        right["occurrence"] = right.groupby("time").cumcount()
        merged = left.merge(right, on=["time", "occurrence"], how="inner")
        if not merged.empty:
            return (
                merged["prediction_position"].to_numpy(dtype=int),
                merged["target_position"].to_numpy(dtype=int),
            )
    length = min(len(prediction_frame), len(target_frame))
    positions = np.arange(length, dtype=int)
    return positions, positions


def aligned_values(
    prediction_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    prediction_column: str,
    target_column: str,
    alignment: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    prediction_positions, target_positions = alignment or build_alignment(
        prediction_frame, target_frame
    )
    return (
        prediction_frame[prediction_column].to_numpy(dtype=float)[prediction_positions],
        target_frame[target_column].to_numpy(dtype=float)[target_positions],
    )


def compute_metrics(
    prediction_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    alignment: tuple[np.ndarray, np.ndarray] | None = None,
    zones: Sequence[str] | None = None,
) -> tuple[dict, list[dict]]:
    if alignment is None:
        alignment = build_alignment(prediction_frame, target_frame)
    if zones is None:
        zones = infer_temperature_zones(prediction_frame)
    zone_rows = []
    errors = []
    for zone in zones:
        prediction_column = find_zone_column(prediction_frame, zone, prediction=True)
        target_column = find_zone_column(target_frame, zone, prediction=False)
        if prediction_column is None or target_column is None:
            continue
        prediction, target = aligned_values(
            prediction_frame, target_frame, prediction_column, target_column, alignment
        )
        error = prediction - target
        finite = np.isfinite(error)
        if not finite.any():
            continue
        error = error[finite]
        errors.append(error)
        zone_rows.append({
            "Zone": zone,
            "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "MAE": float(np.mean(np.abs(error))),
            "Bias": float(np.mean(error)),
            "Points": int(len(error)),
        })
    if not errors:
        raise RuntimeError("No aligned prediction and target temperature columns")
    all_error = np.concatenate(errors)
    macro = {
        "RMSE": float(np.sqrt(np.mean(all_error ** 2))),
        "MAE": float(np.mean(np.abs(all_error))),
        "Bias": float(np.mean(all_error)),
        "Points": int(len(all_error)),
        "Zones": len(zone_rows),
    }
    return macro, zone_rows


# Preserve the original numerical functions above. Exact-air evaluation is opt-in
# and stored separately, rather than silently changing historical measurements.
_legacy_find_zone_column = find_zone_column
COLUMN_MODE = 'legacy'


def find_zone_column(frame, zone, prediction):
    if COLUMN_MODE == 'legacy':
        return _legacy_find_zone_column(frame,zone,prediction)
    for prefix in (('T_pred__','T_in__') if prediction else ('T_in__','T_true__')):
        if prefix+zone in frame.columns:return prefix+zone
    matches=[]
    for column in frame.columns:
        text=str(column)
        if ':' not in text:continue
        if clean_token(text.split(':',1)[0])!=clean_token(zone):continue
        label=text.lower()
        if 'air temperature' not in label:continue
        matches.append((0 if 'zone mean air temperature' in label else 1,column))
    if not matches:return None
    rank=min(x[0] for x in matches);best=[x[1] for x in matches if x[0]==rank]
    if len(best)!=1:raise ValueError('Ambiguous air temperature target for '+zone)
    return best[0]


def validate_prior(prior,control,prepared,expected_runs=None,expected_style=None):
    metadata=prior.get('metadata',{})
    if expected_runs is not None and int(metadata.get('n_runs',-1)) != int(expected_runs):
        raise ValueError('Prior sample count does not match this run')
    if expected_style is not None and metadata.get('prompt_style') != expected_style:
        raise ValueError('Prior prompt style does not match this run')
    records=prior.get('zone_gate_records',{})
    if isinstance(records,list):
        names=[x['zone'] for x in records]
        if len(names)!=len(set(names)):raise ValueError('Duplicate prior zones')
        records={x['zone']:x for x in records}
    for zone,item in prepared.items():
        record=get_zone_record(records,zone)
        if not record or 'episode_gate_records' not in record:
            raise ValueError('Prior does not cover active zone '+zone)
        occupied=np.zeros(96,bool)
        for ep in record['episode_gate_records']:
            start,end,g=float(ep['start_hour']),float(ep['end_hour']),float(ep['correction_gate'])
            if not all(math.isfinite(v) for v in (start,end,g)) or not 0<=start<end<=24 or not 0<=g<=1:
                raise ValueError('Invalid phase interval/gate: '+zone)
            n=len(get_zone_record(control['control_records'],zone)['reference_heating_profile_24h'])
            if len(occupied)!=n:occupied=np.zeros(n,bool)
            lo,hi=int(round(start/24*n)),int(round(end/24*n))
            if hi<=lo or occupied[lo:hi].any():raise ValueError('Overlapping/empty intervals: '+zone)
            occupied[lo:hi]=True
        # Empty phases are valid only for an explicitly zero nominal candidate.
        if not record['episode_gate_records'] and np.any(np.abs(item['candidate'])>1e-12):
            raise ValueError('No phases for nonzero correction candidate: '+zone)


def score_common(frame,target,alignment,zones,masks):
    rows=[];errors=[]
    for zone in zones:
        pc=find_zone_column(frame,zone,True);tc=find_zone_column(target,zone,False)
        a,b=aligned_values(frame,target,pc,tc,alignment)
        mask=masks[zone]
        if not np.isfinite(a[mask]).all():raise ValueError('Strategy changed valid-point coverage: '+zone)
        error=a[mask]-b[mask]
        if not len(error):continue
        errors.append(error)
        rows.append({'Zone':zone,'RMSE':float(np.sqrt(np.mean(error**2))),
            'MAE':float(np.mean(np.abs(error))),'Bias':float(np.mean(error)),'Points':len(error)})
    if not errors:raise ValueError('No common finite evaluation points')
    e=np.concatenate(errors)
    return {'RMSE':float(np.sqrt(np.mean(e**2))),'MAE':float(np.mean(np.abs(e))),
        'Bias':float(np.mean(e)),'Points':len(e),'Zones':len(rows)},rows


def process_case(row,s):
    identity={key:row[key] for key in cfg.KEYS}
    raw=pd.read_csv(row['Raw_CSV']);target=make_target_eval_frame(pd.read_csv(row['Target_CSV']))
    control=cfg.read_json(row['Control_Prior_JSON'])
    path=cfg.prior_path(s,row['Building'],row['OOD_Label'])
    if not path.is_file():raise FileNotFoundError('Prior not ready: '+str(path))
    prior=cfg.read_json(path)
    prepared=prepare_correction_candidates(raw,control)
    if s['prior_source']=='generated':
        validate_prior(prior,control,prepared,s['samples'],s['prompt_style'])
    else:
        validate_prior(prior,control,prepared)
    alignment=build_alignment(raw,target)
    zones=[z for z in infer_temperature_zones(raw) if find_zone_column(target,z,False) is not None]
    masks={}
    for zone in zones:
        a,b=aligned_values(raw,target,find_zone_column(raw,zone,True),find_zone_column(target,zone,False),alignment)
        masks[zone]=np.isfinite(a)&np.isfinite(b)
    fresh,report=apply_episode_gates(raw,control,prior,prepared)
    def frames():
        yield 'Raw',raw
        for alpha in s['fixed_strengths']:
            frame,_=apply_strengths(raw,control,{z:float(alpha) for z in prepared},prepared)
            yield 'Rule-%.2f'%alpha,frame
        yield cfg.OURS,fresh
    case_rows=[];zone_rows=[]
    for method,frame in frames():
        metrics,local=score_common(frame,target,alignment,zones,masks)
        case_rows.append(dict(identity,Postprocess_Method=method,**metrics))
        if s['save_zone_metrics']:
            zone_rows.extend(dict(identity,Postprocess_Method=method,**r) for r in local)
    if s['save_corrected']:
        base=s['run_root'] if s['prior_source']=='existing' else cfg.generation_root(s)
        dest=base/'corrected'/('existing' if s['prior_source']=='existing' else 'samples_%d'%s['samples'])
        dest=dest/cfg.safe_name(row['Architecture'])/cfg.safe_name(row['Method'])/cfg.safe_name(row['Model_Tag'])/('seed_'+str(row['Seed']))/row['Building']/row['Train_Days']/('variant_'+str(row['External_Variant']))
        dest.mkdir(parents=True,exist_ok=True)
        fresh.to_csv(dest/(row['OOD_Label']+'.csv'),index=False)
    columns=[{'Zone':z,'Prediction_Column':str(find_zone_column(raw,z,True)),
              'Target_Column':str(find_zone_column(target,z,False))} for z in zones]
    return {'case_rows':case_rows,'zone_rows':zone_rows,
        'corrections':[dict(identity,**r) for r in report],
        'target_columns':columns,'prior_metadata':prior.get('metadata',{})}


def load_cases(s):
    rows=cfg.read_index(s['case_index']);selected=[];seen=set()
    for position,row in enumerate(rows):
        if not cfg.selected(row,s):continue
        missing=set(cfg.KEYS+['Raw_CSV','Target_CSV','Control_Prior_JSON'])-set(row)
        if missing:raise ValueError('Case index lacks '+str(sorted(missing)))
        key=tuple(row[k] for k in cfg.KEYS)
        if key in seen:raise ValueError('Duplicate case identity '+str(key))
        seen.add(key)
        row=dict(row)
        for column in ('Raw_CSV','Target_CSV','Control_Prior_JSON'):
            row[column]=str(cfg.resolve_input(row[column],s['case_index'],s['final_root']))
        row['_position']=position
        selected.append(row)
    if not selected:raise ValueError('No selected cases')
    return selected


def atomic_csv(path,rows,columns):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    pd.DataFrame(rows,columns=columns).to_csv(temporary,index=False)
    temporary.replace(path)


def run(s,force=False):
    global COLUMN_MODE
    COLUMN_MODE=s['target_matching']
    plan=load_cases(s);out=cfg.metric_root(s)
    settings={'prior_source':s['prior_source'],'prior_root':str(s['existing_priors']) if s['prior_source']=='existing' else str(cfg.generation_root(s)/'priors'/('samples_%d'%s['samples'])),
        'fixed_strengths':s['fixed_strengths'],'target_matching':COLUMN_MODE,
        'save_zone_metrics':s['save_zone_metrics'],
        'operator':'original_core_95_first','alignment':'original_core_timestamp_occurrence_or_position'}
    setting_path=out/'settings.json'
    if setting_path.is_file() and cfg.read_json(setting_path)!=settings:
        raise ValueError('Evaluation settings changed. Choose another run_root; saved results are retained.')
    cfg.write_json(setting_path,settings)
    cases=[];zones=[];reports=[];failed=[]
    # Keep successfully scored unselected cases in the same evaluation ledger.
    identities={tuple(str(r[k]) for k in cfg.KEYS) for r in plan}
    def retained(name):
        path=out/name
        if not path.is_file() or not path.stat().st_size:return []
        try:
            frame=pd.read_csv(path,keep_default_na=False,dtype={k:str for k in cfg.KEYS})
        except pd.errors.EmptyDataError:return []
        if frame.empty:return []
        mask=frame[cfg.KEYS].apply(lambda r:tuple(str(x) for x in r) not in identities,axis=1)
        return frame.loc[mask].to_dict('records')
    other_cases=retained('case_metrics.csv');other_zones=retained('zone_metrics.csv');other_reports=retained('correction_reports.csv')
    metric_cols=cfg.KEYS+['Postprocess_Method','RMSE','MAE','Bias','Points','Zones']
    zone_cols=cfg.KEYS+['Postprocess_Method','Zone','RMSE','MAE','Bias','Points']
    report_cols=cfg.KEYS+['Zone','Strength','Applied_Mean_Gate','Min_Episode_Gate','Max_Episode_Gate','Episode_Count','Operator','Correction_MAE_C','Correction_RMS_C','Modified_Points']
    failure_cols=cfg.KEYS+['Error_type','Error']
    def save():
        atomic_csv(out/'case_metrics.csv',other_cases+cases,metric_cols)
        atomic_csv(out/'zone_metrics.csv',other_zones+zones,zone_cols)
        atomic_csv(out/'correction_reports.csv',other_reports+reports,report_cols)
        atomic_csv(out/'failures.csv',failed,failure_cols)
        done=len({tuple(r[k] for k in cfg.KEYS) for r in cases})
        cfg.write_json(out/'status.json',{'planned_selected_cases':len(plan),'completed_selected_cases':done,
            'failed_selected_cases':len(failed),'is_complete':done==len(plan), 'target_matching':COLUMN_MODE})
    print('Offline evaluation:',len(plan),'cases; prior source:',s['prior_source'],'; target matching:',COLUMN_MODE)
    print('Selection: prompt=%s samples=%s OOD=%s train_days=%s' %
          (s['prompt_style'],s['samples'],','.join(s['oods']),','.join(s['train_days'])),flush=True)
    for i,row in enumerate(plan,1):
        identity={k:row[k] for k in cfg.KEYS}
        try:
            paths=[Path(row[k]) for k in ('Raw_CSV','Target_CSV','Control_Prior_JSON')]+[cfg.prior_path(s,row['Building'],row['OOD_Label'])]
            signature={'identity':identity,'files':[[str(p),p.stat().st_size,p.stat().st_mtime_ns] for p in paths], 'evaluation':settings}
            cache=out/'case_cache'/('%06d.json'%row['_position'])
            saved=None
            if cache.is_file() and not force:
                try:
                    saved=cfg.read_json(cache)
                except (json.JSONDecodeError,UnicodeDecodeError):
                    # Only derived offline case caches can be rebuilt here.
                    # Invalid priors, control files and LLM responses still fail visibly.
                    backup=cache.with_name(cache.name+'.invalid_'+str(time.time_ns()))
                    cache.replace(backup)
                    print('REBUILD OFFLINE CACHE',cache,'; original retained at',backup,flush=True)
            if saved and saved.get('inputs')==signature:
                result=saved['result']
            else:
                result=process_case(row,s)
                cfg.write_json(cache,{'inputs':signature,'result':result})
            cases.extend(result['case_rows']);zones.extend(result['zone_rows']);reports.extend(result['corrections'])
        except Exception as exc:
            failed.append(dict(identity,Error_type=type(exc).__name__,Error=str(exc)))
            print('CASE NOT READY',row['Building'],row['OOD_Label'],type(exc).__name__,str(exc),flush=True)
        if i%s['checkpoint_every']==0 or i==len(plan):
            save();print('EVALUATE',i,'/',len(plan),'failures',len(failed),flush=True)
    print('Saved:',out)
    return out


if __name__=='__main__':
    settings,_,force=cfg.cli()
    run(settings,force)
