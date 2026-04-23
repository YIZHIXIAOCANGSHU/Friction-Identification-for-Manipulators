from __future__ import annotations

from dataclasses import asdict

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from friction_identification_core.config import IdentificationConfig
from friction_identification_core.models import (
    MotorDynamicIdentificationResult,
    MotorIdentificationResult,
    RoundCapture,
)


TRACKING_ERROR_ABSOLUTE_LIMIT = 0.03
TRACKING_ERROR_RATIO_LIMIT = 0.12
SATURATION_COMMAND_RATIO = 0.98
MAX_RECOMMENDED_QUALITY_RATIO = 0.20
CYCLE_HOLDOUT_STRIDE = 3


def _smooth_velocity(velocity: np.ndarray, config: IdentificationConfig) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
    if velocity.size < 3:
        return velocity.copy()

    window = int(config.savgol_window)
    polyorder = int(config.savgol_polyorder)
    if window % 2 == 0:
        window += 1
    max_window = velocity.size if velocity.size % 2 == 1 else velocity.size - 1
    window = min(window, max_window)
    minimum_window = polyorder + 2
    if minimum_window % 2 == 0:
        minimum_window += 1
    if window >= minimum_window and window > polyorder:
        return savgol_filter(velocity, window_length=window, polyorder=polyorder, mode="interp")

    kernel_size = min(5, int(velocity.size))
    if kernel_size <= 1:
        return velocity.copy()
    kernel = np.ones(kernel_size, dtype=np.float64) / float(kernel_size)
    return np.convolve(velocity, kernel, mode="same")


def _build_design_matrix(velocity: np.ndarray, velocity_scale: float) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
    scale = max(float(velocity_scale), 1.0e-6)
    return np.column_stack(
        [
            np.tanh(velocity / scale),
            velocity,
            np.ones_like(velocity),
        ]
    )


def _label_balance_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    weights = np.zeros(labels.size, dtype=np.float64)
    unique_labels = tuple(dict.fromkeys(labels.tolist()))
    if not unique_labels:
        return np.ones(labels.size, dtype=np.float64)

    per_label_weight = 1.0 / float(len(unique_labels))
    for label in unique_labels:
        label_mask = labels == label
        label_count = int(np.count_nonzero(label_mask))
        if label_count <= 0:
            continue
        weights[label_mask] = per_label_weight / float(label_count)

    mean_weight = max(float(np.mean(weights[weights > 0.0])), 1.0e-8)
    return np.where(weights > 0.0, weights / mean_weight, 0.0)


def _huber_weights(residual: np.ndarray, delta: float) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    mad = np.median(np.abs(residual - np.median(residual)))
    scale = mad / 0.6745 if mad > 1.0e-8 else max(float(np.std(residual)), 1.0e-3)
    normalized = np.abs(residual) / (scale * max(float(delta), 1.0e-6))
    weights = np.ones_like(normalized)
    mask = normalized > 1.0
    weights[mask] = 1.0 / normalized[mask]
    return weights


def _solve_weighted_ridge(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    regularization: float,
) -> np.ndarray:
    clipped_weights = np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 1.0e-8, None)
    sqrt_w = np.sqrt(clipped_weights)[:, None]
    design_w = design * sqrt_w
    target_w = target * sqrt_w[:, 0]
    lhs = design_w.T @ design_w + float(regularization) * np.eye(design.shape[1], dtype=np.float64)
    rhs = design_w.T @ target_w
    return np.linalg.solve(lhs, rhs)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if total <= 1.0e-12:
        return 1.0 if residual <= 1.0e-12 else 0.0
    return 1.0 - residual / total


def _candidate_velocity_scales(config: IdentificationConfig, velocity: np.ndarray) -> tuple[float, ...]:
    candidates = {float(value) for value in config.velocity_scale_candidates if float(value) > 0.0}
    speed = np.abs(np.asarray(velocity, dtype=np.float64).reshape(-1))
    speed = speed[np.isfinite(speed) & (speed > max(float(config.zero_velocity_threshold), 1.0e-6))]
    if speed.size:
        percentiles = np.percentile(speed, [10.0, 50.0, 90.0])
        dynamic = [
            0.5 * percentiles[0],
            1.0 * percentiles[0],
            0.25 * percentiles[1],
            0.5 * percentiles[1],
            0.25 * percentiles[2],
        ]
        for candidate in dynamic:
            if np.isfinite(candidate) and candidate > 0.0:
                candidates.add(float(np.clip(candidate, 1.0e-4, 1.0)))
    if not candidates:
        candidates.add(0.02)
    return tuple(sorted(candidates))


def _resolve_capture_limit(capture: RoundCapture, *, key: str, override: float | None) -> float:
    if override is not None and np.isfinite(float(override)) and float(override) > 0.0:
        return float(override)
    value = capture.metadata.get(key)
    if value is None:
        return float("nan")
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0.0:
        return float("nan")
    return resolved


def _ratio_of_failures(base_mask: np.ndarray, ok_mask: np.ndarray) -> float:
    base_mask = np.asarray(base_mask, dtype=bool)
    ok_mask = np.asarray(ok_mask, dtype=bool)
    total = int(np.count_nonzero(base_mask))
    if total <= 0:
        return 0.0
    failed = int(np.count_nonzero(base_mask & (~ok_mask)))
    return float(failed / total)


def _capture_quality_metadata(config: IdentificationConfig, capture: RoundCapture) -> tuple[str | None, dict[str, object]]:
    synced_before_capture = bool(capture.metadata.get("synced_before_capture", True))
    sequence_error_count = int(capture.metadata.get("sequence_error_count", 0))
    sequence_error_ratio = float(capture.metadata.get("sequence_error_ratio", 0.0))
    target_frame_count = int(capture.metadata.get("target_frame_count", capture.sample_count))
    target_frame_ratio = float(capture.metadata.get("target_frame_ratio", 1.0 if capture.sample_count else 0.0))

    metadata = {
        "synced_before_capture": synced_before_capture,
        "sequence_error_count": sequence_error_count,
        "sequence_error_ratio": sequence_error_ratio,
        "target_frame_count": target_frame_count,
        "target_frame_ratio": target_frame_ratio,
    }
    if not synced_before_capture:
        return "sync_not_acquired", metadata
    if target_frame_ratio < float(config.min_target_frame_ratio):
        return "insufficient_target_frames", metadata
    if sequence_error_ratio > float(config.max_sequence_error_ratio):
        return "excessive_sequence_error_ratio", metadata
    return None, metadata


def _cycle_numbers(phase_name: np.ndarray) -> np.ndarray:
    values = np.zeros(np.asarray(phase_name).size, dtype=np.int64)
    for index, phase in enumerate(np.asarray(phase_name).astype(str)):
        if not phase.startswith("excitation_cycle_"):
            continue
        token = phase.rsplit("_", 1)[-1]
        if token.isdigit():
            values[index] = int(token)
    return values


def _velocity_band_assignment(
    velocity_cmd: np.ndarray,
    *,
    max_velocity: float,
    band_edges: tuple[float, ...],
) -> np.ndarray:
    normalized_speed = np.abs(np.asarray(velocity_cmd, dtype=np.float64))
    normalized_speed /= max(float(max_velocity), 1.0e-6)
    bands = np.zeros(normalized_speed.size, dtype=np.int64)
    for band_index, (lower, upper) in enumerate(zip(band_edges[:-1], band_edges[1:]), start=1):
        if band_index == len(band_edges) - 1:
            band_mask = (normalized_speed >= float(lower)) & (normalized_speed <= float(upper) + 1.0e-12)
        else:
            band_mask = (normalized_speed >= float(lower)) & (normalized_speed < float(upper))
        bands[band_mask] = int(band_index)
    return bands


def _band_descriptor(band_index: int, band_edges: tuple[float, ...]) -> str:
    lower = float(band_edges[band_index - 1])
    upper = float(band_edges[band_index])
    return f"band_{band_index:02d}@[{lower:.2f},{upper:.2f}]"


def _conclusion_fields(
    *,
    identified: bool,
    status: str,
    validation_mode: str,
    valid_rmse: float,
    saturation_ratio: float,
    tracking_error_ratio: float,
    parameters: np.ndarray,
) -> tuple[str, str]:
    if not identified:
        return "reject", f"辨识失败: {status}"
    if not np.all(np.isfinite(parameters)):
        return "reject", "辨识参数存在非有限值"
    if saturation_ratio > MAX_RECOMMENDED_QUALITY_RATIO:
        return "reject", f"样本饱和比例过高 ({saturation_ratio:.1%})"
    if tracking_error_ratio > MAX_RECOMMENDED_QUALITY_RATIO:
        return "reject", f"跟踪误差比例过高 ({tracking_error_ratio:.1%})"
    if validation_mode != "velocity_band_holdout":
        return "caution", "仅完成训练集拟合，未形成速度带留出验证"
    return "recommended", f"速度带留出验证通过，valid RMSE={valid_rmse:.6f}"


def _empty_result(
    capture: RoundCapture,
    *,
    status: str,
    valid_sample_ratio: float,
    sample_mask: np.ndarray,
    identification_window_mask: np.ndarray | None = None,
    tracking_ok_mask: np.ndarray | None = None,
    saturation_ok_mask: np.ndarray | None = None,
    metadata: dict[str, object] | None = None,
) -> MotorIdentificationResult:
    result_metadata = {
        "status": status,
        "recommended_for_runtime": False,
        "conclusion_level": "reject",
        "conclusion_text": f"辨识失败: {status}",
        "identification_sample_count": int(np.count_nonzero(sample_mask)),
        "validation_mode": "train_only",
        "validation_reason": status,
        "train_velocity_bands": [],
        "valid_velocity_bands": [],
        "saturation_ratio": 0.0,
        "tracking_error_ratio": 0.0,
    }
    if metadata:
        result_metadata.update(metadata)
    sample_mask = np.asarray(sample_mask, dtype=bool)
    identification_window_mask = (
        sample_mask.copy()
        if identification_window_mask is None
        else np.asarray(identification_window_mask, dtype=bool)
    )
    tracking_ok_mask = (
        np.ones(capture.sample_count, dtype=bool)
        if tracking_ok_mask is None
        else np.asarray(tracking_ok_mask, dtype=bool)
    )
    saturation_ok_mask = (
        np.ones(capture.sample_count, dtype=bool)
        if saturation_ok_mask is None
        else np.asarray(saturation_ok_mask, dtype=bool)
    )
    return MotorIdentificationResult(
        motor_id=int(capture.target_motor_id),
        motor_name=str(capture.motor_name),
        identified=False,
        coulomb=float("nan"),
        viscous=float("nan"),
        offset=float("nan"),
        velocity_scale=float("nan"),
        torque_pred=np.full(capture.sample_count, np.nan, dtype=np.float64),
        torque_target=np.asarray(capture.torque_feedback, dtype=np.float64),
        sample_mask=sample_mask,
        identification_window_mask=identification_window_mask,
        tracking_ok_mask=tracking_ok_mask,
        saturation_ok_mask=saturation_ok_mask,
        train_mask=np.zeros(capture.sample_count, dtype=bool),
        valid_mask=np.zeros(capture.sample_count, dtype=bool),
        train_rmse=float("nan"),
        valid_rmse=float("nan"),
        train_r2=float("nan"),
        valid_r2=float("nan"),
        valid_sample_ratio=float(valid_sample_ratio),
        sample_count=int(np.count_nonzero(sample_mask)),
        metadata=result_metadata,
    )


def identify_motor_friction(
    config: IdentificationConfig,
    capture: RoundCapture,
    *,
    max_torque: float | None = None,
    max_velocity: float | None = None,
) -> MotorIdentificationResult:
    quality_status, quality_metadata = _capture_quality_metadata(config, capture)
    if quality_status is not None:
        return _empty_result(
            capture,
            status=quality_status,
            valid_sample_ratio=0.0,
            sample_mask=np.zeros(capture.sample_count, dtype=bool),
            metadata=quality_metadata,
        )

    position = np.asarray(capture.position, dtype=np.float64)
    velocity_raw = np.asarray(capture.velocity, dtype=np.float64)
    velocity_cmd = np.asarray(capture.velocity_cmd, dtype=np.float64)
    torque_target = np.asarray(capture.torque_feedback, dtype=np.float64)
    command_raw = np.asarray(capture.command_raw, dtype=np.float64)
    command = np.asarray(capture.command, dtype=np.float64)
    motor_id = np.asarray(capture.motor_id, dtype=np.int64)
    id_match_ok = np.asarray(capture.id_match_ok, dtype=bool)
    phase_name = np.asarray(capture.phase_name).astype(str)
    resolved_max_torque = _resolve_capture_limit(capture, key="target_max_torque", override=max_torque)
    resolved_max_velocity = _resolve_capture_limit(capture, key="target_max_velocity", override=max_velocity)
    if not np.isfinite(resolved_max_velocity):
        resolved_max_velocity = max(float(np.nanmax(np.abs(velocity_cmd))), 1.0e-6)

    velocity = _smooth_velocity(velocity_raw, config)
    identification_window_mask = np.isfinite(position) & np.isfinite(velocity) & np.isfinite(torque_target) & np.isfinite(velocity_cmd)
    identification_window_mask &= np.isfinite(command_raw) & np.isfinite(command)
    identification_window_mask &= motor_id == int(capture.target_motor_id)
    identification_window_mask &= id_match_ok
    identification_window_mask &= np.char.startswith(phase_name, "excitation_cycle_")

    velocity_error = velocity - velocity_cmd
    tracking_limit = np.maximum(
        TRACKING_ERROR_ABSOLUTE_LIMIT,
        TRACKING_ERROR_RATIO_LIMIT * np.abs(velocity_cmd),
    )
    tracking_ok_mask = np.abs(velocity_error) <= tracking_limit
    if np.isfinite(resolved_max_torque):
        saturation_ok_mask = (
            np.abs(command_raw) < SATURATION_COMMAND_RATIO * resolved_max_torque
        ) & (
            np.abs(command) < SATURATION_COMMAND_RATIO * resolved_max_torque
        )
    else:
        saturation_ok_mask = np.ones(capture.sample_count, dtype=bool)

    saturation_ratio = _ratio_of_failures(identification_window_mask, saturation_ok_mask)
    tracking_error_ratio = _ratio_of_failures(identification_window_mask, tracking_ok_mask)

    sample_mask = identification_window_mask & tracking_ok_mask & saturation_ok_mask
    valid_count = int(np.count_nonzero(sample_mask))
    valid_ratio = float(valid_count / capture.sample_count) if capture.sample_count else 0.0
    if valid_count < int(config.min_samples):
        return _empty_result(
            capture,
            status="insufficient_identification_window_samples",
            valid_sample_ratio=valid_ratio,
            sample_mask=sample_mask,
            identification_window_mask=identification_window_mask,
            tracking_ok_mask=tracking_ok_mask,
            saturation_ok_mask=saturation_ok_mask,
            metadata={
                **quality_metadata,
                "saturation_ratio": saturation_ratio,
                "tracking_error_ratio": tracking_error_ratio,
            },
        )

    position_span = float(np.ptp(position[sample_mask])) if valid_count else 0.0
    if position_span < float(config.min_motion_span):
        return _empty_result(
            capture,
            status="insufficient_motion_span",
            valid_sample_ratio=valid_ratio,
            sample_mask=sample_mask,
            identification_window_mask=identification_window_mask,
            tracking_ok_mask=tracking_ok_mask,
            saturation_ok_mask=saturation_ok_mask,
            metadata={
                **quality_metadata,
                "position_span": position_span,
                "saturation_ratio": saturation_ratio,
                "tracking_error_ratio": tracking_error_ratio,
            },
        )

    positive_count = int(np.count_nonzero(velocity_cmd[sample_mask] > float(config.zero_velocity_threshold)))
    negative_count = int(np.count_nonzero(velocity_cmd[sample_mask] < -float(config.zero_velocity_threshold)))
    if positive_count < int(config.min_direction_samples) or negative_count < int(config.min_direction_samples):
        return _empty_result(
            capture,
            status="insufficient_bidirectional_window",
            valid_sample_ratio=valid_ratio,
            sample_mask=sample_mask,
            identification_window_mask=identification_window_mask,
            tracking_ok_mask=tracking_ok_mask,
            saturation_ok_mask=saturation_ok_mask,
            metadata={
                **quality_metadata,
                "position_span": position_span,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "saturation_ratio": saturation_ratio,
                "tracking_error_ratio": tracking_error_ratio,
            },
        )

    band_ids = _velocity_band_assignment(
        velocity_cmd,
        max_velocity=float(resolved_max_velocity),
        band_edges=config.validation_velocity_band_edges_ratio,
    )
    present_band_ids = sorted(int(band_id) for band_id in np.unique(band_ids[sample_mask]) if int(band_id) > 0)
    valid_band_ids = tuple(band_id for band_id in present_band_ids if band_id % 2 == 0)
    validation_mode = "train_only"
    validation_reason = "no_even_non_empty_velocity_bands"
    valid_mask = np.zeros(capture.sample_count, dtype=bool)
    if valid_band_ids:
        valid_mask = sample_mask & np.isin(band_ids, np.asarray(valid_band_ids, dtype=np.int64))
        validation_mode = "velocity_band_holdout"
        validation_reason = ""
    train_mask = sample_mask & (~valid_mask)
    if validation_mode == "velocity_band_holdout" and (
        np.count_nonzero(train_mask) < 3 or np.count_nonzero(valid_mask) < 3
    ):
        validation_mode = "train_only"
        validation_reason = "insufficient_validation_band_samples"
        valid_mask[:] = False
        train_mask = sample_mask.copy()

    train_band_ids = sorted(int(band_id) for band_id in np.unique(band_ids[train_mask]) if int(band_id) > 0)
    valid_band_ids = sorted(int(band_id) for band_id in np.unique(band_ids[valid_mask]) if int(band_id) > 0)
    train_band_labels = [_band_descriptor(band_id, config.validation_velocity_band_edges_ratio) for band_id in train_band_ids]
    valid_band_labels = [_band_descriptor(band_id, config.validation_velocity_band_edges_ratio) for band_id in valid_band_ids]

    best_result: dict[str, np.ndarray | float] | None = None
    train_labels = np.asarray([f"band_{int(band_id):02d}" for band_id in band_ids[train_mask]], dtype=object)
    for velocity_scale in _candidate_velocity_scales(config, velocity[train_mask]):
        train_velocity = velocity[train_mask]
        train_torque = torque_target[train_mask]
        design_train = _build_design_matrix(train_velocity, velocity_scale)
        weights = _label_balance_weights(train_labels)
        coeffs = _solve_weighted_ridge(design_train, train_torque, weights, config.regularization)

        for _ in range(int(config.max_iterations)):
            residual = train_torque - design_train @ coeffs
            robust_weights = weights * _huber_weights(residual, config.huber_delta)
            updated = _solve_weighted_ridge(design_train, train_torque, robust_weights, config.regularization)
            if np.linalg.norm(updated - coeffs) <= 1.0e-8 * max(1.0, np.linalg.norm(coeffs)):
                coeffs = updated
                break
            coeffs = updated

        prediction = _build_design_matrix(velocity, velocity_scale) @ coeffs
        train_rmse = _rmse(torque_target[train_mask], prediction[train_mask])
        train_r2 = _r2(torque_target[train_mask], prediction[train_mask])
        valid_rmse = _rmse(torque_target[valid_mask], prediction[valid_mask])
        valid_r2 = _r2(torque_target[valid_mask], prediction[valid_mask])
        score = valid_rmse if validation_mode == "velocity_band_holdout" and np.isfinite(valid_rmse) else train_rmse

        if best_result is None or float(score) < float(best_result["score"]):
            best_result = {
                "coeffs": coeffs,
                "prediction": prediction,
                "velocity_scale": float(velocity_scale),
                "train_rmse": float(train_rmse),
                "valid_rmse": float(valid_rmse),
                "train_r2": float(train_r2),
                "valid_r2": float(valid_r2),
                "score": float(score),
            }

    if best_result is None:
        return _empty_result(
            capture,
            status="fit_failed",
            valid_sample_ratio=valid_ratio,
            sample_mask=sample_mask,
            identification_window_mask=identification_window_mask,
            tracking_ok_mask=tracking_ok_mask,
            saturation_ok_mask=saturation_ok_mask,
            metadata={
                **quality_metadata,
                "position_span": position_span,
                "saturation_ratio": saturation_ratio,
                "tracking_error_ratio": tracking_error_ratio,
            },
        )

    coeffs = np.asarray(best_result["coeffs"], dtype=np.float64).reshape(-1)
    conclusion_level, conclusion_text = _conclusion_fields(
        identified=True,
        status="identified",
        validation_mode=validation_mode,
        valid_rmse=float(best_result["valid_rmse"]),
        saturation_ratio=saturation_ratio,
        tracking_error_ratio=tracking_error_ratio,
        parameters=coeffs,
    )
    metadata = {
        **quality_metadata,
        "status": "identified",
        "recommended_for_runtime": conclusion_level == "recommended",
        "conclusion_level": conclusion_level,
        "conclusion_text": conclusion_text,
        "position_span": position_span,
        "identification_sample_count": valid_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "train_velocity_bands": train_band_labels,
        "valid_velocity_bands": valid_band_labels,
        "validation_mode": validation_mode,
        "validation_reason": validation_reason,
        "band_weight_strategy": "balanced_by_velocity_band",
        "saturation_ratio": saturation_ratio,
        "tracking_error_ratio": tracking_error_ratio,
        "velocity_band_edges_ratio": list(config.validation_velocity_band_edges_ratio),
        "identification_config": asdict(config),
    }
    return MotorIdentificationResult(
        motor_id=int(capture.target_motor_id),
        motor_name=str(capture.motor_name),
        identified=True,
        coulomb=float(coeffs[0]),
        viscous=float(coeffs[1]),
        offset=float(coeffs[2]),
        velocity_scale=float(best_result["velocity_scale"]),
        torque_pred=np.asarray(best_result["prediction"], dtype=np.float64),
        torque_target=torque_target,
        sample_mask=sample_mask,
        identification_window_mask=identification_window_mask,
        tracking_ok_mask=tracking_ok_mask,
        saturation_ok_mask=saturation_ok_mask,
        train_mask=train_mask,
        valid_mask=valid_mask,
        train_rmse=float(best_result["train_rmse"]),
        valid_rmse=float(best_result["valid_rmse"]),
        train_r2=float(best_result["train_r2"]),
        valid_r2=float(best_result["valid_r2"]),
        valid_sample_ratio=valid_ratio,
        sample_count=int(valid_count),
        metadata=metadata,
    )


def _segment_index_groups(mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    if indices.size == 0:
        return []
    split_points = np.where(np.diff(indices) > 1)[0] + 1
    return [segment for segment in np.split(indices, split_points) if segment.size > 0]


def _decode_lugre_parameters(raw_parameters: np.ndarray) -> dict[str, float]:
    raw_parameters = np.asarray(raw_parameters, dtype=np.float64).reshape(-1)
    fc = float(np.exp(raw_parameters[0]))
    fs = float(fc + np.exp(raw_parameters[1]))
    return {
        "fc": fc,
        "fs": fs,
        "vs": float(np.exp(raw_parameters[2])),
        "sigma0": float(np.exp(raw_parameters[3])),
        "sigma1": float(np.exp(raw_parameters[4])),
        "sigma2": float(np.exp(raw_parameters[5])),
        "offset": float(raw_parameters[6]),
    }


def _lugre_g(velocity: np.ndarray, *, fc: float, fs: float, vs: float) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=np.float64)
    scale = max(float(vs), 1.0e-8)
    return float(fc) + (float(fs) - float(fc)) * np.exp(-((velocity / scale) ** 2))


def _simulate_lugre_segment(
    velocity: np.ndarray,
    dt: np.ndarray,
    *,
    fc: float,
    fs: float,
    vs: float,
    sigma0: float,
    sigma1: float,
    sigma2: float,
    offset: float,
    z0: float = 0.0,
) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    prediction = np.zeros(velocity.size, dtype=np.float64)
    z = float(z0)
    g = _lugre_g(velocity, fc=fc, fs=fs, vs=vs)
    for index, (v_i, dt_i, g_i) in enumerate(zip(velocity, dt, g)):
        dt_step = max(float(dt_i), 1.0e-6)
        z_dot = float(v_i) - abs(float(v_i)) * z / max(float(g_i), 1.0e-8)
        prediction[index] = float(sigma0) * z + float(sigma1) * z_dot + float(sigma2) * float(v_i) + float(offset)
        z = z + dt_step * z_dot
    return prediction


def _simulate_lugre_on_mask(
    velocity: np.ndarray,
    time: np.ndarray,
    mask: np.ndarray,
    parameters: dict[str, float],
) -> np.ndarray:
    prediction = np.full(np.asarray(mask, dtype=bool).size, np.nan, dtype=np.float64)
    for segment in _segment_index_groups(mask):
        segment_velocity = np.asarray(velocity[segment], dtype=np.float64)
        segment_time = np.asarray(time[segment], dtype=np.float64)
        if segment_time.size <= 1:
            dt = np.full(segment_time.size, 1.0e-3, dtype=np.float64)
        else:
            dt = np.diff(np.concatenate(([segment_time[0]], segment_time)))
            dt[0] = dt[1] if dt.size > 1 else max(float(np.median(np.diff(segment_time))), 1.0e-3)
        prediction[segment] = _simulate_lugre_segment(segment_velocity, dt, **parameters)
    return prediction


def _empty_dynamic_result(
    capture: RoundCapture,
    *,
    status: str,
    sample_mask: np.ndarray,
    valid_sample_ratio: float,
    metadata: dict[str, object] | None = None,
) -> MotorDynamicIdentificationResult:
    result_metadata = {
        "status": status,
        "validation_mode": "train_only",
        "validation_reason": status,
        "train_cycles": [],
        "valid_cycles": [],
    }
    if metadata:
        result_metadata.update(metadata)
    empty_mask = np.zeros(capture.sample_count, dtype=bool)
    return MotorDynamicIdentificationResult(
        motor_id=int(capture.target_motor_id),
        motor_name=str(capture.motor_name),
        identified=False,
        fc=float("nan"),
        fs=float("nan"),
        vs=float("nan"),
        sigma0=float("nan"),
        sigma1=float("nan"),
        sigma2=float("nan"),
        offset=float("nan"),
        torque_pred=np.full(capture.sample_count, np.nan, dtype=np.float64),
        torque_target=np.asarray(capture.torque_feedback, dtype=np.float64),
        sample_mask=np.asarray(sample_mask, dtype=bool),
        train_mask=empty_mask.copy(),
        valid_mask=empty_mask.copy(),
        validation_warmup_mask=empty_mask.copy(),
        train_rmse=float("nan"),
        valid_rmse=float("nan"),
        train_r2=float("nan"),
        valid_r2=float("nan"),
        valid_sample_ratio=float(valid_sample_ratio),
        sample_count=int(np.count_nonzero(sample_mask)),
        metadata=result_metadata,
    )


def identify_motor_friction_lugre(
    config: IdentificationConfig,
    capture: RoundCapture,
    static_result: MotorIdentificationResult,
) -> MotorDynamicIdentificationResult:
    sample_mask = np.asarray(static_result.sample_mask, dtype=bool)
    valid_ratio = float(np.count_nonzero(sample_mask) / capture.sample_count) if capture.sample_count else 0.0
    if not bool(static_result.identified):
        return _empty_dynamic_result(
            capture,
            status="static_identification_unavailable",
            sample_mask=sample_mask,
            valid_sample_ratio=valid_ratio,
            metadata={"static_validation_rmse": float(static_result.valid_rmse)},
        )

    cycle_numbers = _cycle_numbers(capture.phase_name)
    present_cycles = sorted(int(cycle) for cycle in np.unique(cycle_numbers[sample_mask]) if int(cycle) > 0)
    valid_cycles = [cycle for cycle in present_cycles if cycle % CYCLE_HOLDOUT_STRIDE == 0]
    validation_mode = "cycle_holdout"
    validation_reason = ""
    valid_mask = sample_mask & np.isin(cycle_numbers, np.asarray(valid_cycles, dtype=np.int64))
    train_mask = sample_mask & (~valid_mask)
    if not valid_cycles or np.count_nonzero(train_mask) < 3 or np.count_nonzero(valid_mask) < 3:
        validation_mode = "train_only"
        validation_reason = "insufficient_holdout_cycles"
        valid_cycles = []
        valid_mask[:] = False
        train_mask = sample_mask.copy()

    warmup_mask = np.zeros(capture.sample_count, dtype=bool)
    warmup_samples = int(config.validation_warmup_samples)
    if valid_cycles and warmup_samples > 0:
        for cycle in valid_cycles:
            cycle_indices = np.flatnonzero(valid_mask & (cycle_numbers == int(cycle)))
            warmup_mask[cycle_indices[:warmup_samples]] = True
    score_valid_mask = valid_mask & (~warmup_mask)
    if validation_mode == "cycle_holdout" and np.count_nonzero(score_valid_mask) < 3:
        validation_mode = "train_only"
        validation_reason = "insufficient_scored_validation_samples"
        valid_cycles = []
        valid_mask[:] = False
        score_valid_mask[:] = False
        warmup_mask[:] = False
        train_mask = sample_mask.copy()

    velocity = _smooth_velocity(np.asarray(capture.velocity, dtype=np.float64), config)
    time = np.asarray(capture.time, dtype=np.float64)
    torque_target = np.asarray(capture.torque_feedback, dtype=np.float64)
    initial_raw = np.asarray(
        [
            np.log(max(abs(float(static_result.coulomb)) * 0.85, 1.0e-4)),
            np.log(max(abs(float(static_result.coulomb)) * 0.30, 1.0e-4)),
            np.log(max(float(static_result.velocity_scale), 1.0e-4)),
            np.log(5.0),
            np.log(0.05),
            np.log(max(float(static_result.viscous), 1.0e-4)),
            float(static_result.offset),
        ],
        dtype=np.float64,
    )

    train_velocity = velocity[train_mask]
    train_time = time[train_mask]
    train_target = torque_target[train_mask]

    def residual(raw_parameters: np.ndarray) -> np.ndarray:
        parameters = _decode_lugre_parameters(raw_parameters)
        prediction = _simulate_lugre_on_mask(
            train_velocity,
            train_time,
            np.ones(train_velocity.size, dtype=bool),
            parameters,
        )
        return np.asarray(prediction - train_target, dtype=np.float64)

    optimized = least_squares(residual, initial_raw, method="trf")
    parameters = _decode_lugre_parameters(optimized.x)
    prediction = _simulate_lugre_on_mask(velocity, time, sample_mask, parameters)
    train_rmse = _rmse(torque_target[train_mask], prediction[train_mask])
    train_r2 = _r2(torque_target[train_mask], prediction[train_mask])
    valid_rmse = _rmse(torque_target[score_valid_mask], prediction[score_valid_mask])
    valid_r2 = _r2(torque_target[score_valid_mask], prediction[score_valid_mask])

    return MotorDynamicIdentificationResult(
        motor_id=int(capture.target_motor_id),
        motor_name=str(capture.motor_name),
        identified=True,
        fc=float(parameters["fc"]),
        fs=float(parameters["fs"]),
        vs=float(parameters["vs"]),
        sigma0=float(parameters["sigma0"]),
        sigma1=float(parameters["sigma1"]),
        sigma2=float(parameters["sigma2"]),
        offset=float(parameters["offset"]),
        torque_pred=prediction,
        torque_target=torque_target,
        sample_mask=sample_mask,
        train_mask=train_mask,
        valid_mask=score_valid_mask,
        validation_warmup_mask=warmup_mask,
        train_rmse=float(train_rmse),
        valid_rmse=float(valid_rmse),
        train_r2=float(train_r2),
        valid_r2=float(valid_r2),
        valid_sample_ratio=valid_ratio,
        sample_count=int(np.count_nonzero(sample_mask)),
        metadata={
            "status": "identified",
            "validation_mode": validation_mode,
            "validation_reason": validation_reason,
            "train_cycles": [int(cycle) for cycle in sorted(int(cycle) for cycle in np.unique(cycle_numbers[train_mask]) if int(cycle) > 0)],
            "valid_cycles": [int(cycle) for cycle in valid_cycles],
            "validation_warmup_samples": warmup_samples,
            "static_validation_rmse": float(static_result.valid_rmse),
            "static_train_rmse": float(static_result.train_rmse),
            "optimizer_status": int(optimized.status),
            "optimizer_message": str(optimized.message),
            "nfev": int(optimized.nfev),
        },
    )
