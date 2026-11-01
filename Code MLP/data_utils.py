import re
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List

import config

def build_multizone_timeseries(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    def norm_c(c: str): return re.sub(r"\s+", " ", str(c).strip().lower())
    def clean_z(s: str): return re.sub(r"\s+", " ", str(s).strip())
    def extr(c: str, kw: str):
        r = re.sub(r"\[.*?\]", "", c.strip()).strip()
        if ":" in r and norm_c(kw) in norm_c(r.split(":", 1)[1]): return clean_z(r.split(":", 1)[0])
        return clean_z(r)

    c_norm = {c: norm_c(c) for c in df_raw.columns}
    time_c = next((c for c in df_raw.columns if "date/time" in c_norm[c]), None)
    out_c = next((c for c in df_raw.columns if "site outdoor air drybulb temperature" in c_norm[c]), None)
    if out_c is None: raise ValueError("Cannot find outdoor temperature column.")
        
    z_map = {}
    for c in df_raw.columns:
        nc = c_norm[c]
        if "zone air temperature [c]" in nc: z_map.setdefault(extr(c, "temperature"), {})["temp"] = c
        elif "sensible heating" in nc: z_map.setdefault(extr(c, "heating"), {})["heat"] = c
        elif "sensible cooling" in nc: z_map.setdefault(extr(c, "cooling"), {})["cool"] = c
        elif "internal total heating" in nc: z_map.setdefault(extr(c, "internal"), {})["internal"] = c
        elif "zone windows total heat gain rate" in nc: z_map.setdefault(extr(c, "windows total heat gain"), {})["rad"] = c
            
    rows = {}
    if time_c is not None:
        out = []
        for s in df_raw[time_c].astype(str):
            m = re.match(r"^\s*(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2}):(\d{2})", s.strip())
            if m:
                mo, d, h, mi, sec = map(int, m.groups())
                out.append(datetime(2023, mo, d, 0, mi, sec) + timedelta(days=1) if h == 24 else datetime(2023, mo, d, h, mi, sec))
            else:
                try: out.append(pd.to_datetime(s.strip().split(" 24:")[0] + " 00:" + s.strip().split(" 24:")[1]) + pd.Timedelta(days=1) if " 24:" in s else pd.to_datetime(s.strip()))
                except Exception: out.append(pd.NaT)
        rows["datetime"] = pd.to_datetime(out)
        
    rows["T_out"] = pd.to_numeric(df_raw[out_c], errors="coerce").astype(float)
    zones = []
    
    for z, info in z_map.items():
        if "temp" not in info: continue
        T = pd.to_numeric(df_raw[info["temp"]], errors="coerce").astype(float)
        rows[f"T_in__{z}"] = T.values
        rows[f"Q_hvac__{z}"] = (pd.to_numeric(df_raw[info["cool"]], errors="coerce").astype(float).values if "cool" in info else np.zeros(len(T))) / config.HEAT_SCALE
        rows[f"Q_occ__{z}"] = (pd.to_numeric(df_raw[info["internal"]], errors="coerce").astype(float).values if "internal" in info else np.zeros(len(T))) / config.HEAT_SCALE
        rows[f"Q_rad__{z}"] = (pd.to_numeric(df_raw[info["rad"]], errors="coerce").astype(float).values if "rad" in info else np.zeros(len(T))) / config.HEAT_SCALE
        zones.append(z)
        
    return pd.DataFrame(rows).fillna(0).reset_index(drop=True), zones

def pack_dataset_author_style(df: pd.DataFrame, zones: List[str], predict_period: int, device: str = "cpu", fit_scaler=False, scaler_X=None, scaler_Y=None):
    T_in = np.column_stack([df[f"T_in__{z}"].values for z in zones])
    P = np.column_stack([df[f"Q_hvac__{z}"].values for z in zones])
    T_out = df["T_out"].values.reshape(-1, 1)
    T_rad = np.column_stack([df[f"Q_rad__{z}"].values for z in zones])
    T_occ = np.column_stack([df[f"Q_occ__{z}"].values for z in zones])
    X = np.hstack((T_in, P, T_out, T_rad, T_occ))
    n_zones = len(zones)
    
    if fit_scaler:
        scaler_X = MinMaxScaler().fit(X)
        scaler_Y = MinMaxScaler()
        y_min = scaler_X.data_min_[:n_zones].astype(float).copy()
        y_max = scaler_X.data_max_[:n_zones].astype(float).copy()
        if config.USE_PADDED_TEMP_SCALER and config.TEMP_SCALER_PAD_C > 0:
            y_min -= float(config.TEMP_SCALER_PAD_C)
            y_max += float(config.TEMP_SCALER_PAD_C)

        if config.TEMP_SCALER_MODE == "global":
            y_min_fit = np.full(n_zones, float(np.nanmin(y_min)), dtype=float)
            y_max_fit = np.full(n_zones, float(np.nanmax(y_max)), dtype=float)
        else:
            y_min_fit, y_max_fit = y_min, y_max

        scaler_Y.fit(np.vstack([y_min_fit, y_max_fit]))

    X_scaled = scaler_X.transform(X)
    Y_scaled = scaler_Y.transform(T_in)
    X_scaled[:, :n_zones] = Y_scaled

    S_Y = X_scaled[:, :n_zones]
    S_P = X_scaled[:, n_zones:2 * n_zones]
    S_X_env = X_scaled[:, 2 * n_zones:]

    N_sample = int(X_scaled.shape[0] / predict_period)
    data_out = np.zeros((predict_period, N_sample, X_scaled.shape[1]))
    label_out = np.zeros((predict_period - 1, N_sample, n_zones))

    for i in range(N_sample):
        i_start = i * predict_period
        i_end = i_start + predict_period
        data_out[:, i, :n_zones] = S_Y[i_start:i_end, :]
        data_out[:, i, n_zones:2 * n_zones] = S_P[i_start:i_end, :]
        data_out[:, i, 2 * n_zones:] = S_X_env[i_start:i_end, :]
        if config.PREDICT_RESIDUAL_DELTA:
            label_out[:, i, :] = S_Y[i_start + 1:i_end, :] - S_Y[i_start:i_end - 1, :]
        else:
            label_out[:, i, :] = S_Y[i_start + 1:i_end, :]

    return (
        torch.tensor(data_out, dtype=torch.float64, device=device),
        torch.tensor(label_out, dtype=torch.float64, device=device),
        scaler_X, scaler_Y,
    )

def pack_history_context_author_style(
    df_full: pd.DataFrame,
    target_indices,
    zones: List[str],
    predict_period: int,
    history_steps: int,
    scaler_X,
    scaler_Y,
    device: str = "cpu",
):
    """Pack real samples immediately preceding each daily target segment.

    The returned tensor has shape [history_steps, n_daily_samples, features].
    Context rows initialize a History MLP and are never prediction targets.
    """
    history_steps = max(1, int(history_steps))
    target_indices = np.asarray(target_indices, dtype=int)
    n_samples = int(len(target_indices) / predict_period)
    n_zones = len(zones)

    T_in = np.column_stack([df_full[f"T_in__{z}"].values for z in zones])
    P = np.column_stack([df_full[f"Q_hvac__{z}"].values for z in zones])
    T_out = df_full["T_out"].values.reshape(-1, 1)
    T_rad = np.column_stack([df_full[f"Q_rad__{z}"].values for z in zones])
    T_occ = np.column_stack([df_full[f"Q_occ__{z}"].values for z in zones])
    X = np.hstack((T_in, P, T_out, T_rad, T_occ))

    X_scaled = scaler_X.transform(X)
    X_scaled[:, :n_zones] = scaler_Y.transform(T_in)
    context = np.zeros((history_steps, n_samples, X_scaled.shape[1]))

    for sample_index in range(n_samples):
        target_start = int(target_indices[sample_index * predict_period])
        context_indices = np.arange(
            target_start - history_steps,
            target_start,
            dtype=int,
        )
        # Only the beginning of the entire source file needs padding.
        context_indices = np.clip(context_indices, 0, len(df_full) - 1)
        context[:, sample_index, :] = X_scaled[context_indices, :]

    return torch.tensor(context, dtype=torch.float64, device=device)

def build_next_step_eval_frame(df: pd.DataFrame) -> pd.DataFrame:
    predict_period = int(round(24 / config.DT_HOURS))
    n_sample = int(len(df) / predict_period)
    frames = []
    for i in range(n_sample):
        i_start = i * predict_period
        i_end = i_start + predict_period
        frames.append(df.iloc[i_start + 1:i_end])
    if not frames: return df.iloc[0:0].copy()
    return pd.concat(frames, axis=0).reset_index(drop=True)

def persistence_predict_from_frame(df: pd.DataFrame, zones: List[str]) -> np.ndarray:
    predict_period = int(round(24 / config.DT_HOURS))
    n_sample = int(len(df) / predict_period)
    anchor_interval = max(1, int(config.ROLLOUT_EVAL_ANCHOR_INTERVAL)) if config.ROLLOUT_EVAL_CLOSED_LOOP_RESIDUAL else 1
    day_preds = []

    for i in range(n_sample):
        i_start = i * predict_period
        i_end = i_start + predict_period
        day_df = df.iloc[i_start:i_end]
        zone_values = np.column_stack([day_df[f"T_in__{z}"].values.astype(float) for z in zones])
        step_preds = []
        total_steps = max(0, len(zone_values) - 1)
        for start in range(0, total_steps, anchor_interval):
            end = min(start + anchor_interval, total_steps)
            anchor_temp = zone_values[start:start + 1, :]
            step_preds.append(np.repeat(anchor_temp, end - start, axis=0))
        if step_preds: day_preds.append(np.vstack(step_preds))

    if not day_preds: return np.empty((0, len(zones)), dtype=float)
    return np.vstack(day_preds)
