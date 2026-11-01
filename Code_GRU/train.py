import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple

import config

def get_building_learning_rate(dataset_file: str = None, day_tag: int = None) -> float:
    if not dataset_file: return float(config.NN_LR)
    base_name = os.path.splitext(os.path.basename(str(dataset_file)))[0]
    # 如果要动态拦截学习率，这里就可以生效了（兼容之前的 TRAINING 调优）
    for (building_key, train_days), lr in config.BUILDING_DAY_LR_OVERRIDES.items():
        if str(building_key) in base_name and day_tag == int(train_days):
            return float(lr)
    for key, lr in config.BUILDING_LR_OVERRIDES.items():
        if str(key) in base_name: return float(lr)
    return float(config.NN_LR)

def safe_filename_token(s: str):
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")

def target_mode_model_name(model_name: str) -> str:
    if config.PREDICT_RESIDUAL_DELTA and "+ DeltaTarget" not in model_name:
        return f"{model_name} + DeltaTarget"
    return model_name

def get_model_checkpoint_path(result_dir: str, dataset_file: str, zones_count: int, model_name: str, day_tag: int, seed: int):
    base = safe_filename_token(os.path.splitext(os.path.basename(dataset_file))[0])
    run_tag = f"{day_tag}d" if day_tag > 0 else "full"
    model_tag = safe_filename_token(target_mode_model_name(model_name))
    ckpt_dir = os.path.join(result_dir, config.MODEL_CHECKPOINT_DIR_NAME, config.MODEL_CHECKPOINT_VERSION, base, run_tag)
    return os.path.join(ckpt_dir, f"{model_tag}_zones{zones_count}_seed{seed}.pt")

def save_model_checkpoint(model: nn.Module, ckpt_path: str, model_name: str, dataset_file: str, zones_count: int, seed: int, day_tag: int):
    if not config.SAVE_MODEL_CHECKPOINTS: return
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    model_name = target_mode_model_name(model_name)
    payload = {
        "model_name": model_name, "dataset_file": os.path.basename(dataset_file), "zones_count": zones_count,
        "seed": seed, "experiment_day_tag": day_tag,
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    torch.save(payload, ckpt_path)
    print(f"  [Checkpoint] Saved: {ckpt_path}")

def load_model_checkpoint_if_available(model: nn.Module, ckpt_path: str, device):
    if not config.LOAD_MODEL_CHECKPOINTS or not os.path.exists(ckpt_path): return False
    payload = torch.load(ckpt_path, map_location=device)
    state_dict = payload.get("state_dict", payload)
    
    current_state = model.state_dict()
    compatible_state = {}
    for k, v in state_dict.items():
        if k in current_state and tuple(v.shape) == tuple(current_state[k].shape):
            compatible_state[k] = v
    current_state.update(compatible_state)
    model.load_state_dict(current_state, strict=False)
    print(f"  [Checkpoint] Loaded: {ckpt_path}")
    return True

def _majority_parent_by_child(child_labels, parent_labels):
    child = np.asarray(child_labels, dtype=int).reshape(-1)
    parent = np.asarray(parent_labels, dtype=int).reshape(-1)
    if child.shape[0] != parent.shape[0]:
        raise ValueError(
            f"child_labels length {child.shape[0]} does not match parent_labels length {parent.shape[0]}"
        )
    mapping = {}
    for child_id in sorted(set(child.tolist())):
        parent_ids = parent[child == child_id]
        if parent_ids.size == 0:
            mapping[int(child_id)] = 0
            continue
        values, counts = np.unique(parent_ids, return_counts=True)
        mapping[int(child_id)] = int(values[np.argmax(counts)])
    return mapping

def semantic_initialize_cluster_model(
    model: nn.Module,
    result_dir: str,
    dataset_file: str,
    zones_count: int,
    seed: int,
    day_tag: int,
    child_labels,
    parent_labels,
    parent_model_name: str,
    device,
):
    """Initialize ROF child cluster parameters from their majority LLM parent cluster."""
    parent_ckpt_path = get_model_checkpoint_path(
        result_dir, dataset_file, zones_count, parent_model_name, day_tag, seed
    )
    if not os.path.exists(parent_ckpt_path):
        print(f"  [SemanticInit Skip] Missing parent checkpoint: {parent_ckpt_path}")
        return False

    payload = torch.load(parent_ckpt_path, map_location=device)
    parent_state = payload.get("state_dict", payload)
    current_state = model.state_dict()
    child_to_parent = _majority_parent_by_child(child_labels, parent_labels)

    copied = 0
    prefixes = []
    if hasattr(model, "cluster_gru_cells"):
        prefixes.append(("cluster_gru_cells", "cluster_gru_cells"))
    if hasattr(model, "cluster_lstm_cells"):
        prefixes.append(("cluster_lstm_cells", "cluster_lstm_cells"))
    if hasattr(model, "cluster_decoders"):
        prefixes.append(("cluster_decoders", "cluster_decoders"))

    for child_id, parent_id in child_to_parent.items():
        for target_prefix, parent_prefix in prefixes:
            src_prefix = f"{parent_prefix}.{parent_id}."
            dst_prefix = f"{target_prefix}.{child_id}."
            for key in list(current_state.keys()):
                if not key.startswith(dst_prefix):
                    continue
                suffix = key[len(dst_prefix):]
                parent_key = src_prefix + suffix
                if parent_key not in parent_state:
                    continue
                src = parent_state[parent_key]
                dst = current_state[key]
                if tuple(src.shape) != tuple(dst.shape):
                    continue
                current_state[key] = src.detach().to(device=dst.device, dtype=dst.dtype).clone()
                copied += 1

    if copied == 0:
        print(f"  [SemanticInit Warning] Parent checkpoint found but no compatible cluster tensors copied: {parent_ckpt_path}")
        return False

    model.load_state_dict(current_state, strict=False)
    print(
        f"  [SemanticInit] Initialized {copied} tensors from parent '{target_mode_model_name(parent_model_name)}'. "
        f"child_to_parent={child_to_parent}"
    )
    return True

def align_pred_and_label(pred: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if pred.shape[0] == label.shape[0] + 1: pred = pred[1:]
    elif label.shape[0] == pred.shape[0] + 1: label = label[1:]
    elif pred.shape[0] != label.shape[0]:
        n = min(pred.shape[0], label.shape[0])
        pred, label = pred[:n], label[:n]
    return pred, label

def residual_rollout_training_losses(pred_delta, true_delta, train_data, lossfn, rollout_initial_state=None, true_temperature_data=None):
    pred_delta, true_delta = align_pred_and_label(pred_delta, true_delta)
    target_data = train_data if true_temperature_data is None else true_temperature_data
    n = min(pred_delta.shape[0], true_delta.shape[0], target_data.shape[0] - 1)
    pred_delta, true_delta = pred_delta[:n], true_delta[:n]

    y0 = train_data[0:1, :, :pred_delta.shape[-1]] if rollout_initial_state is None else rollout_initial_state
    true_temp = target_data[1:n + 1, :, :pred_delta.shape[-1]]
    pred_temp = y0 + torch.cumsum(pred_delta, dim=0)
    
    rollout_loss = lossfn(pred_temp, true_temp)
    bias_loss = torch.mean(torch.mean(pred_delta - true_delta, dim=0) ** 2)
    return rollout_loss, bias_loss

def make_gaussian_noisy_training_input(model: nn.Module, clean_input: torch.Tensor) -> torch.Tensor:
    std = float(getattr(config, "GAUSSIAN_INPUT_NOISE_STD", 0.0))
    if std <= 0:
        return clean_input

    noisy_input = clean_input.detach().clone()
    y_dim = getattr(model, "y_dim", None)
    if getattr(config, "GAUSSIAN_INPUT_NOISE_TEMP_ONLY", True) and y_dim is not None:
        noise_slice = noisy_input[:, :, :y_dim]
        noisy_input[:, :, :y_dim] = torch.clamp(noise_slice + torch.randn_like(noise_slice) * std, 0.0, 1.0)
    else:
        noisy_input = torch.clamp(noisy_input + torch.randn_like(noisy_input) * std, 0.0, 1.0)
    return noisy_input

def _loss_value(x):
    return None if x is None else float(x.detach().item())

def _format_loss_term(name: str, value, weight: float) -> str:
    if value is None:
        return ""
    raw = _loss_value(value)
    return f" | {name}: {raw:.6f} * {weight:.2e}={raw * weight:.6f}"

def _standardize_zone_vector(v: torch.Tensor):
    v = v.reshape(-1)
    if v.numel() <= 1: return None
    std = torch.std(v)
    if not torch.isfinite(std) or float(std.detach().cpu()) < 1e-8: return None
    return (v - torch.mean(v)) / (std + 1e-8)

def compute_prior_sensitivity_alignment_loss(model: nn.Module, clean_input: torch.Tensor) -> torch.Tensor:
    y_dim = getattr(model, "y_dim", None)
    p_dim = getattr(model, "p_dim", 0)
    if y_dim is None or config.PRIOR_SENS_ALIGN_EPS <= 0: return torch.tensor(0.0, dtype=torch.float64, device=clean_input.device)

    base_input = clean_input.detach()
    losses = []
    
    # Check Occupancy Sensitivity
    if p_dim == y_dim and hasattr(model, "occ_prior"):
        p_start, p_end = y_dim, y_dim + p_dim
        p_base = base_input[:, :, p_start:p_end]
        minus_input, plus_input = base_input.clone(), base_input.clone()
        minus_input[:, :, p_start:p_end] = torch.clamp(p_base - config.PRIOR_SENS_ALIGN_EPS, 0.0, 1.0)
        plus_input[:, :, p_start:p_end] = torch.clamp(p_base + config.PRIOR_SENS_ALIGN_EPS, 0.0, 1.0)

        pred_minus = model(minus_input)[0] if isinstance(model(minus_input), tuple) else model(minus_input)
        pred_plus = model(plus_input)[0] if isinstance(model(plus_input), tuple) else model(plus_input)
        
        slope = (pred_plus[1:] - pred_minus[1:]) / (2.0 * config.PRIOR_SENS_ALIGN_EPS)
        response_strength = torch.mean(torch.abs(slope), dim=(0, 1))
        rza, pza = _standardize_zone_vector(response_strength), _standardize_zone_vector(model.occ_prior.to(response_strength.device))
        if rza is not None and pza is not None: losses.append(torch.mean((rza - pza)**2))
        
    if not losses: return torch.tensor(0.0, dtype=torch.float64, device=clean_input.device)
    return torch.stack(losses).mean()

# ————————————————————————————————————
# 主训练循环 (完美包含所有约束条件触发)
# ————————————————————————————————————
def run_unified_training(
    model_name: str,
    model: nn.Module,
    m_train_data,
    m_train_label,
    ss_Y,
    seed: int,
    dataset_file: str,
    day_tag: int,
    result_dir: str,
    train_history_context=None,
):
    model_name = target_mode_model_name(model_name)
    ckpt_path = get_model_checkpoint_path(result_dir, dataset_file, m_train_label.shape[-1], model_name, day_tag, seed)
    
    loaded = False
    if not config.FORCE_RETRAIN_EXISTING_CHECKPOINTS:
        loaded = load_model_checkpoint_if_available(model, ckpt_path, m_train_data.device)
        
    if config.ROLLOUT_EVAL_ONLY and not loaded:
        print(f"  [Rollout Eval Skip] {model_name} | checkpoint not found.")
        return None

    if loaded: return model
    if config.NN_EPOCHS == 0: return model
    
    torch.manual_seed(seed)
    current_lr = get_building_learning_rate(dataset_file, day_tag)
    optimizer = optim.Adam(model.parameters(), lr=current_lr)
    lossfn = nn.MSELoss()

    best_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    print(f"\n--- Training {model_name} | Seed: {seed} ---")
    print(
        f"[{model_name}] Config:"
        f" lr={current_lr:.2e}, residual={config.PREDICT_RESIDUAL_DELTA},"
        f" delta_w={config.RESIDUAL_TRAIN_DELTA_LAMBDA:.2e},"
        f" rollout_w={config.RESIDUAL_TRAIN_ROLLOUT_LAMBDA:.2e},"
        f" bias_w={config.RESIDUAL_TRAIN_BIAS_LAMBDA:.2e},"
        f" gaussian_aug={getattr(model, 'enable_gaussian_input_noise_training', False)},"
        f" gaussian_std={getattr(config, 'GAUSSIAN_INPUT_NOISE_STD', 0.0):.4f}"
    )
    for epoch in range(1, config.NN_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        train_input = m_train_data[:-1, :, :]
        train_label = m_train_label
        rollout_initial_state = None
        rollout_target_data = m_train_data
        gaussian_noise_abs_mean, gaussian_noise_abs_max = None, None
        if getattr(model, "enable_gaussian_input_noise_training", False):
            clean_for_noise_stats = train_input
            train_input = make_gaussian_noisy_training_input(model, train_input)
            y_dim = getattr(model, "y_dim", train_input.shape[-1])
            gaussian_diff = train_input[:, :, :y_dim] - clean_for_noise_stats[:, :, :y_dim]
            gaussian_noise_abs_mean = float(gaussian_diff.abs().mean().detach().item())
            gaussian_noise_abs_max = float(gaussian_diff.abs().max().detach().item())

        out = (
            model(train_input, history_context=train_history_context)
            if train_history_context is not None and hasattr(model, "history_window")
            else model(train_input)
        )
        pred = out[0] if isinstance(out, tuple) else out
        pred_aligned, label_aligned = align_pred_and_label(pred, train_label)
        loss_mse = lossfn(pred_aligned, label_aligned)

        # 辅助 Regularizations/Losses
        loss_reg, loss_sens_align, loss_decoder_shrink = (None,) * 3
        alpha_reg = 0.0

        if (hasattr(model, "W_graph_delta") and hasattr(model, "off_diag_mask") and epoch > config.TOPOLOGY_DELTA_REG_START_EPOCH):
            loss_reg = torch.mean((model.W_graph_delta * model.off_diag_mask) ** 2)
            alpha_reg = config.TOPOLOGY_DELTA_REG_ALPHA
        if getattr(model, "enable_prior_sensitivity_alignment_regularization", False) and epoch >= config.PRIOR_SENS_ALIGN_START_EPOCH:
            loss_sens_align = compute_prior_sensitivity_alignment_loss(model, m_train_data[:-1, :, :])
        if (
            getattr(model, "enable_cluster_decoder_shrink_regularization", False)
            and epoch >= getattr(config, "CLUSTER_DECODER_SHRINK_START_EPOCH", 200)
            and hasattr(model, "decoder_shrink_regularization_loss")
        ):
            loss_decoder_shrink = model.decoder_shrink_regularization_loss()

        loss = loss_mse
        loss_rollout, loss_bias = None, None
        if config.PREDICT_RESIDUAL_DELTA and config.RESIDUAL_TRAIN_USE_ROLLOUT_LOSS:
            loss_rollout, loss_bias = residual_rollout_training_losses(
                pred, train_label, train_input, lossfn,
                rollout_initial_state=rollout_initial_state,
                true_temperature_data=rollout_target_data,
            )
            loss = (config.RESIDUAL_TRAIN_DELTA_LAMBDA * loss_mse + config.RESIDUAL_TRAIN_ROLLOUT_LAMBDA * loss_rollout + config.RESIDUAL_TRAIN_BIAS_LAMBDA * loss_bias)
            
        supervised_loss = loss
        stop_metric = float(supervised_loss.detach().item()) if config.PREDICT_RESIDUAL_DELTA else float(loss_mse.detach().item())

        # 叠加所有处罚项以完成导数计算
        if loss_reg is not None: loss += alpha_reg * loss_reg
        if loss_sens_align is not None: loss += config.PRIOR_SENS_ALIGN_LAMBDA * loss_sens_align
        if loss_decoder_shrink is not None: loss += config.CLUSTER_DECODER_SHRINK_LAMBDA * loss_decoder_shrink

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if stop_metric < best_loss - config.NN_MIN_DELTA:
            best_loss = stop_metric
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epoch % config.NN_VERBOSE_EVERY == 0 or epoch == 1:
            msg = (
                f"[{model_name}] Epoch: {epoch:5d}"
                f" | StopLoss: {stop_metric:.6f}"
                f" | TotalLoss: {float(loss.detach().item()):.6f}"
            )
            msg += _format_loss_term("Delta", loss_mse, config.RESIDUAL_TRAIN_DELTA_LAMBDA)
            msg += _format_loss_term("Rollout", loss_rollout, config.RESIDUAL_TRAIN_ROLLOUT_LAMBDA)
            msg += _format_loss_term("Bias", loss_bias, config.RESIDUAL_TRAIN_BIAS_LAMBDA)
            msg += _format_loss_term("Reg", loss_reg, alpha_reg)
            msg += _format_loss_term("SensAlign", loss_sens_align, config.PRIOR_SENS_ALIGN_LAMBDA)
            msg += _format_loss_term("DecShrink", loss_decoder_shrink, config.CLUSTER_DECODER_SHRINK_LAMBDA)
            if gaussian_noise_abs_mean is not None:
                msg += f" | GNoise(abs_mean={gaussian_noise_abs_mean:.6f}, abs_max={gaussian_noise_abs_max:.6f})"
            msg += f" | wait: {epochs_no_improve}"
            print(msg)
        
        if epochs_no_improve >= config.NN_PATIENCE:
            print(f"[{model_name}] Early Stop at epoch {epoch}. Best Score: {best_loss:.6f}")
            break

    if best_state: model.load_state_dict(best_state)
    save_model_checkpoint(model, ckpt_path, model_name, dataset_file, m_train_label.shape[-1], seed, day_tag)
    return model

def anchored_residual_rollout(delta_scaled: np.ndarray, input_scaled: np.ndarray, anchor_interval: int) -> np.ndarray:
    if anchor_interval <= 1: return input_scaled + delta_scaled
    rolled = np.empty_like(delta_scaled)
    total_steps = delta_scaled.shape[0]
    for start in range(0, total_steps, anchor_interval):
        end = min(start + anchor_interval, total_steps)
        anchor_state = input_scaled[start:start+1, :, :]
        rolled[start:end] = anchor_state + np.cumsum(delta_scaled[start:end], axis=0)
    return rolled

def predict_and_unscale_unified(
    model, test_data, ss_Y, zones, history_context=None,
):
    model.eval()
    with torch.no_grad():
        out = (
            model(test_data[:-1, :, :], history_context=history_context)
            if history_context is not None and hasattr(model, "history_window")
            else model(test_data[:-1, :, :])
        )
    out_np = out[0].cpu().numpy() if isinstance(out, tuple) else out.cpu().numpy()
    input_scaled = test_data[:-1, :, :len(zones)].cpu().numpy()

    if out_np.shape[0] == input_scaled.shape[0] + 1: out_np = out_np[1:]
    elif out_np.shape[0] != input_scaled.shape[0]:
        n = min(out_np.shape[0], input_scaled.shape[0])
        out_np, input_scaled = out_np[:n], input_scaled[:n]

    if config.PREDICT_RESIDUAL_DELTA:
        if config.ROLLOUT_EVAL_CLOSED_LOOP_RESIDUAL:
            out_np = anchored_residual_rollout(out_np, input_scaled, config.ROLLOUT_EVAL_ANCHOR_INTERVAL)
        else:
            out_np = input_scaled + out_np
            
    return ss_Y.inverse_transform(out_np.reshape((-1, len(zones)), order="F"))

def print_first_step_residual_diagnostics(
    model,
    test_data,
    ss_Y,
    zones,
    model_name: str,
    split_tag: str,
    history_context=None,
):
    if not getattr(config, "EVAL_FIRST_STEP_DIAGNOSTICS", False):
        return
    if test_data.shape[0] < 2:
        return

    model.eval()
    y_dim = len(zones)
    with torch.no_grad():
        out = (
            model(test_data[:-1, :, :], history_context=history_context)
            if history_context is not None and hasattr(model, "history_window")
            else model(test_data[:-1, :, :])
        )
    pred = out[0] if isinstance(out, tuple) else out
    if pred.shape[0] == test_data[:-1].shape[0] + 1:
        pred = pred[1:]
    elif pred.shape[0] != test_data[:-1].shape[0]:
        n = min(pred.shape[0], test_data[:-1].shape[0])
        pred = pred[:n]

    y0_scaled = test_data[0, :, :y_dim].detach().cpu().numpy()
    y1_true_scaled = test_data[1, :, :y_dim].detach().cpu().numpy()
    first_pred = pred[0, :, :].detach().cpu().numpy()
    if config.PREDICT_RESIDUAL_DELTA:
        y1_pred_scaled = y0_scaled + first_pred
    else:
        y1_pred_scaled = first_pred

    y0 = ss_Y.inverse_transform(y0_scaled)
    y1_true = ss_Y.inverse_transform(y1_true_scaled)
    y1_pred = ss_Y.inverse_transform(y1_pred_scaled)

    true_step = y1_true - y0
    pred_step = y1_pred - y0
    first_err = y1_pred - y1_true

    true_abs = float(np.mean(np.abs(true_step)))
    pred_abs = float(np.mean(np.abs(pred_step)))
    err_abs = float(np.mean(np.abs(first_err)))
    err_rmse = float(np.sqrt(np.mean(first_err ** 2)))
    ratio = pred_abs / max(true_abs, 1e-12)

    tag = f"{model_name} | {split_tag}"
    print(
        f"  [FirstStepDiag | {tag}] "
        f"true_delta_abs_mean={true_abs:.4f}C, "
        f"pred_delta_abs_mean={pred_abs:.4f}C, "
        f"ratio={ratio:.2f}, "
        f"first_step_MAE={err_abs:.4f}C, "
        f"first_step_RMSE={err_rmse:.4f}C"
    )

    top_k = int(getattr(config, "EVAL_DIAGNOSTIC_TOP_ZONES", 3))
    if top_k > 0 and len(zones) > 0:
        zone_err = np.mean(np.abs(first_err), axis=0)
        order = np.argsort(zone_err)[::-1][:top_k]
        detail = ", ".join(
            f"{zones[i]}:err={zone_err[i]:.3f}C,pred_d={np.mean(pred_step[:, i]):.3f}C,true_d={np.mean(true_step[:, i]):.3f}C"
            for i in order
        )
        print(f"  [FirstStepDiag | {tag}] worst_zones: {detail}")

def print_raw_rollout_zone_diagnostics(
    model,
    eval_data,
    ss_Y,
    zones,
    model_name: str,
    split_tag: str,
    dataset_file: str = "",
    history_context=None,
):
    if not getattr(config, "RAW_ROLLOUT_DEBUG", False):
        return

    building_key = str(getattr(config, "RAW_ROLLOUT_DEBUG_BUILDING", "") or "").strip()
    if building_key and building_key not in os.path.basename(str(dataset_file)):
        return

    model_keyword = str(getattr(config, "RAW_ROLLOUT_DEBUG_MODEL_KEYWORD", "") or "").strip()
    if model_keyword and model_keyword not in str(model_name):
        return

    allowed_splits = getattr(config, "RAW_ROLLOUT_DEBUG_SPLITS", ("ID", "OOD"))
    if allowed_splits and split_tag not in set(allowed_splits):
        return

    zone_name = str(getattr(config, "RAW_ROLLOUT_DEBUG_ZONE", "") or "").strip()
    if not zone_name:
        return

    zone_lookup = {str(z).lower(): i for i, z in enumerate(zones)}
    zone_idx = zone_lookup.get(zone_name.lower())
    if zone_idx is None:
        clean_target = re.sub(r"[^a-z0-9]+", "", zone_name.lower())
        for i, z in enumerate(zones):
            if clean_target == re.sub(r"[^a-z0-9]+", "", str(z).lower()):
                zone_idx = i
                break
    if zone_idx is None:
        print(f"  [RawRolloutDiag] zone not found: {zone_name}; available examples={zones[:8]}")
        return

    if eval_data.shape[0] < 2:
        return

    batch_index = int(getattr(config, "RAW_ROLLOUT_DEBUG_BATCH_INDEX", 0))
    batch_index = max(0, min(batch_index, eval_data.shape[1] - 1))
    steps = int(getattr(config, "RAW_ROLLOUT_DEBUG_STEPS", 24))
    steps = max(1, min(steps, eval_data.shape[0] - 1))
    batch_summary = int(getattr(config, "RAW_ROLLOUT_DEBUG_BATCH_SUMMARY", 8))

    model.eval()
    y_dim = len(zones)
    with torch.no_grad():
        out = (
            model(eval_data[:-1, :, :], history_context=history_context)
            if history_context is not None and hasattr(model, "history_window")
            else model(eval_data[:-1, :, :])
        )
    pred = out[0] if isinstance(out, tuple) else out
    pred_had_initial_frame = pred.shape[0] == eval_data[:-1].shape[0] + 1
    if pred_had_initial_frame:
        pred = pred[1:]
    elif pred.shape[0] != eval_data[:-1].shape[0]:
        n = min(pred.shape[0], eval_data[:-1].shape[0])
        pred = pred[:n]

    input_scaled = eval_data[:-1, :, :y_dim].detach().cpu().numpy()
    true_next_scaled = eval_data[1:, :, :y_dim].detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    pred_delta_scaled = pred_np if config.PREDICT_RESIDUAL_DELTA else pred_np - input_scaled[:pred_np.shape[0]]
    true_delta_scaled = true_next_scaled[:pred_np.shape[0]] - input_scaled[:pred_np.shape[0]]

    if config.PREDICT_RESIDUAL_DELTA:
        if config.ROLLOUT_EVAL_CLOSED_LOOP_RESIDUAL:
            pred_temp_scaled = anchored_residual_rollout(
                pred_np,
                input_scaled[:pred_np.shape[0]],
                int(config.ROLLOUT_EVAL_ANCHOR_INTERVAL),
            )
        else:
            pred_temp_scaled = input_scaled[:pred_np.shape[0]] + pred_np
    else:
        pred_temp_scaled = pred_np

    def inv_zone(scaled_values):
        arr = np.zeros((len(scaled_values), y_dim), dtype=float)
        arr[:, zone_idx] = np.asarray(scaled_values, dtype=float)
        inv = ss_Y.inverse_transform(arr)
        return inv[:, zone_idx]

    b = batch_index
    y_t_s = input_scaled[:steps, b, zone_idx]
    y_true_s = true_next_scaled[:steps, b, zone_idx]
    y_pred_s = pred_temp_scaled[:steps, b, zone_idx]
    pred_d_s = pred_delta_scaled[:steps, b, zone_idx]
    true_d_s = true_delta_scaled[:steps, b, zone_idx]

    y_t_c = inv_zone(y_t_s)
    y_true_c = inv_zone(y_true_s)
    y_pred_c = inv_zone(y_pred_s)
    pred_d_c = y_pred_c - y_t_c
    true_d_c = y_true_c - y_t_c

    tag = f"{model_name} | {split_tag} | {os.path.basename(str(dataset_file))}"
    print("\n" + "  " + "=" * 112)
    print(f"  [RawRolloutDiag] {tag}")
    print(
        f"  zone={zones[zone_idx]} (idx={zone_idx}), batch={b}, steps={steps}, "
        f"pred_had_initial_frame={pred_had_initial_frame}, residual={config.PREDICT_RESIDUAL_DELTA}, "
        f"closed_loop={config.ROLLOUT_EVAL_CLOSED_LOOP_RESIDUAL}, anchor_interval={config.ROLLOUT_EVAL_ANCHOR_INTERVAL}"
    )
    print(
        f"  scaler zone min/max: {float(ss_Y.data_min_[zone_idx]):.4f}C / "
        f"{float(ss_Y.data_max_[zone_idx]):.4f}C"
    )
    print(
        "  step | y_t(C) | true_next(C) | pred_next(C) | true_d(C) | pred_d(C) "
        "| true_d_scaled | pred_d_scaled | err(C)"
    )
    for t in range(steps):
        print(
            f"  {t:4d} | {y_t_c[t]:7.3f} | {y_true_c[t]:12.3f} | {y_pred_c[t]:12.3f} "
            f"| {true_d_c[t]:9.3f} | {pred_d_c[t]:9.3f} | {true_d_s[t]:13.5f} "
            f"| {pred_d_s[t]:13.5f} | {y_pred_c[t] - y_true_c[t]:7.3f}"
        )

    if batch_summary > 0:
        n_batches = min(batch_summary, pred_np.shape[1])
        print("  first-step by batch/day:")
        for bb in range(n_batches):
            y0 = inv_zone([input_scaled[0, bb, zone_idx]])[0]
            yt = inv_zone([true_next_scaled[0, bb, zone_idx]])[0]
            yp = inv_zone([pred_temp_scaled[0, bb, zone_idx]])[0]
            print(
                f"    batch={bb:02d}: y0={y0:.3f}C, true1={yt:.3f}C, "
                f"pred1={yp:.3f}C, true_d={yt - y0:+.3f}C, pred_d={yp - y0:+.3f}C"
            )
    print("  " + "=" * 112 + "\n")

def print_and_log_metrics(y_pred, df, zones, model_name, dataset_file, day_tag, result_dir, is_external=False):
    from eval_utils import log_result_to_json, save_prediction_csv
    from data_utils import build_next_step_eval_frame
    import os
    
    target_df = build_next_step_eval_frame(df)
    y_true = np.column_stack([target_df[f"T_in__{z}"].values.astype(float) for z in zones])
    min_len = min(len(y_pred), len(y_true))
    y_pred, y_true = y_pred[:min_len], y_true[:min_len]

    m_mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if m_mask.sum() == 0: return

    p2_g, t2_g = y_pred[m_mask], y_true[m_mask]
    g_rmse = float(np.sqrt(np.mean((p2_g - t2_g) ** 2)))
    g_mae = float(np.mean(np.abs(p2_g - t2_g)))
    from sklearn.metrics import r2_score
    g_r2 = float(r2_score(t2_g, p2_g))
    
    tag = "OOD Test" if is_external else "ID Global"
    print(f"  [{model_name} | {tag}] RMSE: {g_rmse:.4f} \tMAE: {g_mae:.4f} \tR²: {g_r2:.4f}")

    exclude_first_steps = int(getattr(config, "EVAL_EXCLUDE_FIRST_STEPS", 0))
    if exclude_first_steps > 0:
        predict_period = int(round(24 / config.DT_HOURS))
        eval_period = max(1, predict_period - 1)
        step_idx = np.arange(len(y_pred)) % eval_period
        keep = step_idx >= min(exclude_first_steps, eval_period)
        keep_mask = keep[:, None] & m_mask
        if keep_mask.sum() > 0:
            p_keep, t_keep = y_pred[keep_mask], y_true[keep_mask]
            rmse_keep = float(np.sqrt(np.mean((p_keep - t_keep) ** 2)))
            mae_keep = float(np.mean(np.abs(p_keep - t_keep)))
            print(f"  [{model_name} | {tag} | excl_first_{exclude_first_steps}] RMSE: {rmse_keep:.4f} \tMAE: {mae_keep:.4f}")

    zone_metrics = {}
    for z_idx, z_name in enumerate(zones):
        p_z, t_z = y_pred[:, z_idx], y_true[:, z_idx]
        z_mask = np.isfinite(p_z) & np.isfinite(t_z)
        if z_mask.sum() > 0:
            zone_metrics[z_name] = {
                "RMSE": round(float(np.sqrt(np.mean((p_z[z_mask] - t_z[z_mask]) ** 2))), 4),
                "MAE":  round(float(np.mean(np.abs(p_z[z_mask] - t_z[z_mask]))), 4),
            }

    save_prediction_csv(y_pred, df, zones, model_name, dataset_file, day_tag, result_dir)
    log_json_path = os.path.join(result_dir, f"{os.path.basename(dataset_file).replace('.csv', '')}_results_{day_tag}d.json" if day_tag > 0 else f"{os.path.basename(dataset_file).replace('.csv', '')}_results_full.json")
    log_result_to_json(log_json_path, dataset_file, len(zones), model_name, day_tag, g_rmse, g_mae, g_r2, zone_metrics)
