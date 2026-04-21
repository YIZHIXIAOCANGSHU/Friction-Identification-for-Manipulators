from __future__ import annotations

from dataclasses import asdict

import numpy as np
from scipy.signal import savgol_filter

from friction_identification_core.config import IdentificationConfig
from friction_identification_core.models import MotorIdentificationResult, RoundCapture


TRACKING_ERROR_ABSOLUTE_LIMIT = 0.03
TRACKING_ERROR_RATIO_LIMIT = 0.12
SATURATION_COMMAND_RATIO = 0.98
HIGH_SPEED_RATIO_THRESHOLD = 0.90
MAX_RECOMMENDED_QUALITY_RATIO = 0.20


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


def _platform_balance_weights(phase_name: np.ndarray) -> np.ndarray:
    phases = np.asarray(phase_name).astype(str).reshape(-1)
    weights = np.zeros(phases.size, dtype=np.float64)
    unique_platforms = tuple(dict.fromkeys(phases.tolist()))
    if not unique_platforms:
        return np.ones(phases.size, dtype=np.float64)

    per_platform_weight = 1.0 / float(len(unique_platforms))
    for platform_name in unique_platforms:
        platform_mask = phases == str(platform_name)
        platform_count = int(np.count_nonzero(platform_mask))
        if platform_count <= 0:
            continue
        weights[platform_mask] = per_platform_weight / float(platform_count)

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


def _ordered_platform_names(phase_name: np.ndarray, sample_mask: np.ndarray) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for phase in np.asarray(phase_name)[np.asarray(sample_mask, dtype=bool)]:
        label = str(phase)
        if label in seen:
            continue
        seen.add(label)
        ordered.append(label)
    return tuple(ordered)


def _platform_descriptors(platform_names: tuple[str, ...], phase_name: np.ndarray, velocity_cmd: np.ndarray) -> tuple[str, ...]:
    descriptors: list[str] = []
    phase_name = np.asarray(phase_name).astype(str)
    velocity_cmd = np.asarray(velocity_cmd, dtype=np.float64)
    for platform_name in platform_names:
        platform_mask = phase_name == str(platform_name)
        platform_velocity = float(np.median(velocity_cmd[platform_mask])) if np.any(platform_mask) else float("nan")
        descriptors.append(f"{platform_name}@{platform_velocity:+.4f}")
    return tuple(descriptors)


def _phase_level(phase_name: str) -> int | None:
    tokens = str(phase_name).rsplit("_", 1)
    if len(tokens) != 2 or not tokens[1].isdigit():
        return None
    return int(tokens[1])


def _configured_platform_count(phase_name: np.ndarray) -> int:
    levels = [_phase_level(name) for name in np.asarray(phase_name).astype(str)]
    valid_levels = [level for level in levels if level is not None]
    return max(valid_levels) if valid_levels else 0


def _holdout_platform_names(platform_count: int) -> tuple[str, ...]:
    if platform_count >= 5:
        levels = (3, 5)
    elif platform_count == 4:
        levels = (2, 4)
    else:
        return ()

    holdout: list[str] = []
    for level in levels:
        holdout.append(f"steady_forward_{level:02d}")
        holdout.append(f"steady_reverse_{level:02d}")
    return tuple(holdout)


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


def _conclusion_fields(
    *,
    identified: bool,
    status: str,
    validation_mode: str,
    high_speed_platform_count: int,
    high_speed_valid_rmse: float,
    saturation_ratio: float,
    tracking_error_ratio: float,
    parameters: np.ndarray,
) -> tuple[str, str]:
    if not identified:
        return "reject", f"辨识失败: {status}"
    if not np.all(np.isfinite(parameters)):
        return "reject", "辨识参数存在非有限值"
    if saturation_ratio > MAX_RECOMMENDED_QUALITY_RATIO:
        return "reject", f"稳态样本饱和比例过高 ({saturation_ratio:.1%})"
    if tracking_error_ratio > MAX_RECOMMENDED_QUALITY_RATIO:
        return "reject", f"稳态样本跟踪误差比例过高 ({tracking_error_ratio:.1%})"
    if validation_mode != "platform_holdout":
        return "caution", "仅完成训练集拟合，未形成平台留出验证"
    if high_speed_platform_count < 2 or not np.isfinite(high_speed_valid_rmse):
        return "caution", "高速平台覆盖不足，缺少可用的高速验证结果"
    return "recommended", f"平台留出验证通过，高速段有效 RMSE={high_speed_valid_rmse:.6f}"


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
    if sequence_error_ratio > float(config.max_sequence_error_ratio):
        return "excessive_sequence_errors", metadata
    if target_frame_ratio < float(config.min_target_frame_ratio):
        return "insufficient_target_frames", metadata
    return None, metadata


def _empty_result(
    capture: RoundCapture,
    *,
    status: str,
    valid_sample_ratio: float,
    sample_mask: np.ndarray,
    steady_state_mask: np.ndarray | None = None,
    tracking_ok_mask: np.ndarray | None = None,
    saturation_ok_mask: np.ndarray | None = None,
    metadata: dict[str, object] | None = None,
) -> MotorIdentificationResult:
    result_metadata = {
        "status": status,
        "recommended_for_runtime": False,
        "conclusion_level": "reject",
        "conclusion_text": f"辨识失败: {status}",
        "dropped_platforms": [],
        "steady_sample_count": int(np.count_nonzero(sample_mask)),
        "high_speed_platform_count": 0,
        "high_speed_valid_rmse": float("nan"),
        "saturation_ratio": 0.0,
        "tracking_error_ratio": 0.0,
        "validation_mode": "train_only",
        "validation_reason": status,
        "train_platforms": [],
        "valid_platforms": [],
    }
    if metadata:
        result_metadata.update(metadata)
    sample_mask = np.asarray(sample_mask, dtype=bool)
    steady_state_mask = sample_mask.copy() if steady_state_mask is None else np.asarray(steady_state_mask, dtype=bool)
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
        steady_state_mask=steady_state_mask,
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

    velocity = _smooth_velocity(velocity_raw, config)
    steady_state_mask = np.isfinite(position) & np.isfinite(velocity) & np.isfinite(torque_target) & np.isfinite(velocity_cmd)
    steady_state_mask &= np.isfinite(command_raw) & np.isfinite(command)
    steady_state_mask &= motor_id == int(capture.target_motor_id)
    steady_state_mask &= id_match_ok
    steady_state_mask &= np.char.startswith(phase_name, "steady_")
    steady_state_mask &= np.abs(velocity_cmd) >= float(config.zero_velocity_threshold)
    steady_state_mask &= np.abs(velocity) >= float(config.zero_velocity_threshold)
    steady_state_mask &= np.sign(velocity) == np.sign(velocity_cmd)

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

    saturation_ratio = _ratio_of_failures(steady_state_mask, saturation_ok_mask)
    tracking_error_ratio = _ratio_of_failures(steady_state_mask, tracking_ok_mask)

    candidate_mask = steady_state_mask & tracking_ok_mask & saturation_ok_mask
    platform_names_before_drop = _ordered_platform_names(phase_name, steady_state_mask)
    retained_platforms: list[str] = []
    dropped_platforms: list[str] = []
    for platform_name in platform_names_before_drop:
        platform_mask = candidate_mask & (phase_name == str(platform_name))
        platform_count = int(np.count_nonzero(platform_mask))
        if platform_count < int(config.min_samples_per_platform):
            dropped_platforms.append(f"{platform_name} ({platform_count} samples)")
            continue
        retained_platforms.append(str(platform_name))

    sample_mask = candidate_mask & np.isin(phase_name, np.asarray(retained_platforms))
    valid_count = int(np.count_nonzero(sample_mask))
    valid_ratio = float(valid_count / capture.sample_count) if capture.sample_count else 0.0
    if valid_count < int(config.min_samples):
        return _empty_result(
            capture,
            status="insufficient_steady_state_samples",
            valid_sample_ratio=valid_ratio,
            sample_mask=sample_mask,
            steady_state_mask=steady_state_mask,
            tracking_ok_mask=tracking_ok_mask,
            saturation_ok_mask=saturation_ok_mask,
            metadata={
                **quality_metadata,
                "dropped_platforms": dropped_platforms,
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
            steady_state_mask=steady_state_mask,
            tracking_ok_mask=tracking_ok_mask,
            saturation_ok_mask=saturation_ok_mask,
            metadata={
                **quality_metadata,
                "position_span": position_span,
                "dropped_platforms": dropped_platforms,
                "saturation_ratio": saturation_ratio,
                "tracking_error_ratio": tracking_error_ratio,
            },
        )

    positive_count = int(np.count_nonzero(velocity_cmd[sample_mask] > float(config.zero_velocity_threshold)))
    negative_count = int(np.count_nonzero(velocity_cmd[sample_mask] < -float(config.zero_velocity_threshold)))
    platform_names = _ordered_platform_names(phase_name, sample_mask)
    if positive_count < int(config.min_direction_samples) or negative_count < int(config.min_direction_samples):
        return _empty_result(
            capture,
            status="insufficient_bidirectional_steady_state",
            valid_sample_ratio=valid_ratio,
            sample_mask=sample_mask,
            steady_state_mask=steady_state_mask,
            tracking_ok_mask=tracking_ok_mask,
            saturation_ok_mask=saturation_ok_mask,
            metadata={
                **quality_metadata,
                "dropped_platforms": dropped_platforms,
                "saturation_ratio": saturation_ratio,
                "tracking_error_ratio": tracking_error_ratio,
                "steady_platforms": list(_platform_descriptors(platform_names, phase_name, velocity_cmd)),
                "positive_count": positive_count,
                "negative_count": negative_count,
            },
        )

    train_platform_names = platform_names
    valid_platform_names: tuple[str, ...] = ()
    validation_mode = "train_only"
    validation_reason = "insufficient_platform_count"
    configured_platform_count = _configured_platform_count(phase_name)
    requested_holdout_platforms = _holdout_platform_names(configured_platform_count)
    if requested_holdout_platforms:
        platform_name_set = set(platform_names)
        if set(requested_holdout_platforms).issubset(platform_name_set):
            valid_platform_names = requested_holdout_platforms
            train_platform_names = tuple(name for name in platform_names if name not in valid_platform_names)
            validation_mode = "platform_holdout"
            validation_reason = ""
        else:
            validation_reason = "holdout_platform_dropped"

    valid_mask = np.zeros(capture.sample_count, dtype=bool)
    if valid_platform_names:
        valid_mask = np.isin(phase_name, np.asarray(valid_platform_names)) & sample_mask
    train_mask = sample_mask & (~valid_mask)
    if validation_mode == "platform_holdout" and (
        np.count_nonzero(train_mask) < 3 or np.count_nonzero(valid_mask) < 3
    ):
        valid_mask[:] = False
        train_mask = sample_mask.copy()
        valid_platform_names = ()
        train_platform_names = platform_names
        validation_mode = "train_only"
        validation_reason = "insufficient_platform_samples"
    elif validation_mode != "platform_holdout":
        train_mask = sample_mask.copy()

    high_speed_mask = sample_mask.copy()
    if np.isfinite(resolved_max_velocity):
        high_speed_mask &= np.abs(velocity_cmd) >= HIGH_SPEED_RATIO_THRESHOLD * resolved_max_velocity
    else:
        high_speed_mask &= False
    high_speed_platform_names = _ordered_platform_names(phase_name, high_speed_mask)

    common_metadata = {
        **quality_metadata,
        "position_span": position_span,
        "steady_sample_count": valid_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "steady_platforms": list(_platform_descriptors(platform_names, phase_name, velocity_cmd)),
        "train_platforms": list(_platform_descriptors(train_platform_names, phase_name, velocity_cmd)),
        "valid_platforms": list(_platform_descriptors(valid_platform_names, phase_name, velocity_cmd)),
        "dropped_platforms": dropped_platforms,
        "validation_mode": validation_mode,
        "validation_reason": validation_reason,
        "platform_weight_strategy": "balanced_by_platform",
        "saturation_ratio": saturation_ratio,
        "tracking_error_ratio": tracking_error_ratio,
        "high_speed_platform_count": int(len(high_speed_platform_names)),
        "high_speed_platforms": list(_platform_descriptors(high_speed_platform_names, phase_name, velocity_cmd)),
    }

    best_result: dict[str, np.ndarray | float] | None = None
    for velocity_scale in _candidate_velocity_scales(config, velocity[train_mask]):
        train_velocity = velocity[train_mask]
        train_torque = torque_target[train_mask]
        design_train = _build_design_matrix(train_velocity, velocity_scale)
        weights = _platform_balance_weights(phase_name[train_mask])
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
        high_speed_valid_mask = valid_mask & high_speed_mask
        high_speed_valid_rmse = _rmse(torque_target[high_speed_valid_mask], prediction[high_speed_valid_mask])
        score = valid_rmse if validation_mode == "platform_holdout" and np.isfinite(valid_rmse) else train_rmse

        if best_result is None or float(score) < float(best_result["score"]):
            best_result = {
                "coeffs": coeffs,
                "prediction": prediction,
                "velocity_scale": float(velocity_scale),
                "train_rmse": float(train_rmse),
                "valid_rmse": float(valid_rmse),
                "train_r2": float(train_r2),
                "valid_r2": float(valid_r2),
                "high_speed_valid_rmse": float(high_speed_valid_rmse),
                "score": float(score),
            }

    if best_result is None:
        return _empty_result(
            capture,
            status="fit_failed",
            valid_sample_ratio=valid_ratio,
            sample_mask=sample_mask,
            metadata=common_metadata,
        )

    coeffs = np.asarray(best_result["coeffs"], dtype=np.float64).reshape(-1)
    conclusion_level, conclusion_text = _conclusion_fields(
        identified=True,
        status="identified",
        validation_mode=validation_mode,
        high_speed_platform_count=int(len(high_speed_platform_names)),
        high_speed_valid_rmse=float(best_result["high_speed_valid_rmse"]),
        saturation_ratio=saturation_ratio,
        tracking_error_ratio=tracking_error_ratio,
        parameters=coeffs,
    )
    metadata = {
        **common_metadata,
        "status": "identified",
        "recommended_for_runtime": conclusion_level == "recommended",
        "conclusion_level": conclusion_level,
        "conclusion_text": conclusion_text,
        "high_speed_valid_rmse": float(best_result["high_speed_valid_rmse"]),
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
        steady_state_mask=steady_state_mask,
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
