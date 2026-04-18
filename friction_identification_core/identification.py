from __future__ import annotations

from dataclasses import asdict

import numpy as np
from scipy.signal import savgol_filter

from friction_identification_core.config import IdentificationConfig
from friction_identification_core.models import MotorIdentificationResult, RoundCapture


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


def _balance_weights(velocity: np.ndarray, zero_threshold: float) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
    weights = np.ones_like(velocity)
    positive = velocity > float(zero_threshold)
    negative = velocity < -float(zero_threshold)
    stationary = ~(positive | negative)

    positive_count = max(int(np.count_nonzero(positive)), 1)
    negative_count = max(int(np.count_nonzero(negative)), 1)
    stationary_count = max(int(np.count_nonzero(stationary)), 1)

    if np.any(positive):
        weights[positive] *= 0.5 / positive_count
    if np.any(negative):
        weights[negative] *= 0.5 / negative_count
    if np.any(stationary):
        weights[stationary] *= 0.1 / stationary_count

    mean_weight = max(float(np.mean(weights)), 1.0e-8)
    return weights / mean_weight


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


def _empty_result(capture: RoundCapture, *, status: str, valid_sample_ratio: float, sample_mask: np.ndarray) -> MotorIdentificationResult:
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
        train_mask=np.zeros(capture.sample_count, dtype=bool),
        valid_mask=np.zeros(capture.sample_count, dtype=bool),
        train_rmse=float("nan"),
        valid_rmse=float("nan"),
        train_r2=float("nan"),
        valid_r2=float("nan"),
        valid_sample_ratio=float(valid_sample_ratio),
        sample_count=int(capture.sample_count),
        metadata={"status": status},
    )


def identify_motor_friction(config: IdentificationConfig, capture: RoundCapture) -> MotorIdentificationResult:
    position = np.asarray(capture.position, dtype=np.float64)
    velocity_raw = np.asarray(capture.velocity, dtype=np.float64)
    torque_target = np.asarray(capture.torque_feedback, dtype=np.float64)
    motor_id = np.asarray(capture.motor_id, dtype=np.int64)

    velocity = _smooth_velocity(velocity_raw, config)
    sample_mask = np.isfinite(position) & np.isfinite(velocity) & np.isfinite(torque_target)
    sample_mask &= motor_id == int(capture.target_motor_id)
    sample_mask &= np.abs(velocity) >= float(config.zero_velocity_threshold)

    valid_count = int(np.count_nonzero(sample_mask))
    valid_ratio = float(valid_count / capture.sample_count) if capture.sample_count else 0.0
    if valid_count < int(config.min_samples):
        return _empty_result(
            capture,
            status="insufficient_samples",
            valid_sample_ratio=valid_ratio,
            sample_mask=sample_mask,
        )

    position_span = float(np.ptp(position[sample_mask])) if valid_count else 0.0
    if position_span < float(config.min_motion_span):
        return _empty_result(
            capture,
            status="insufficient_motion_span",
            valid_sample_ratio=valid_ratio,
            sample_mask=sample_mask,
        )

    positive_count = int(np.count_nonzero(velocity[sample_mask] > float(config.zero_velocity_threshold)))
    negative_count = int(np.count_nonzero(velocity[sample_mask] < -float(config.zero_velocity_threshold)))
    if positive_count == 0 or negative_count == 0:
        return _empty_result(
            capture,
            status="insufficient_bidirectional_motion",
            valid_sample_ratio=valid_ratio,
            sample_mask=sample_mask,
        )

    valid_indices = np.flatnonzero(sample_mask)
    valid_mask = np.zeros(capture.sample_count, dtype=bool)
    if valid_indices.size:
        trimmed = valid_indices[min(int(config.validation_warmup_samples), valid_indices.size) :]
        valid_mask[trimmed[:: int(config.validation_stride)]] = True
    train_mask = sample_mask & (~valid_mask)
    if np.count_nonzero(train_mask) < 3:
        valid_mask[:] = False
        train_mask = sample_mask.copy()

    best_result: dict[str, np.ndarray | float] | None = None
    for velocity_scale in _candidate_velocity_scales(config, velocity[sample_mask]):
        train_velocity = velocity[train_mask]
        train_torque = torque_target[train_mask]
        design_train = _build_design_matrix(train_velocity, velocity_scale)
        weights = _balance_weights(train_velocity, config.zero_velocity_threshold)
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
        score = valid_rmse if np.isfinite(valid_rmse) else train_rmse

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
        )

    coeffs = np.asarray(best_result["coeffs"], dtype=np.float64).reshape(-1)
    metadata = {
        "status": "identified",
        "positive_count": positive_count,
        "negative_count": negative_count,
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
        train_mask=train_mask,
        valid_mask=valid_mask,
        train_rmse=float(best_result["train_rmse"]),
        valid_rmse=float(best_result["valid_rmse"]),
        train_r2=float(best_result["train_r2"]),
        valid_r2=float(best_result["valid_r2"]),
        valid_sample_ratio=valid_ratio,
        sample_count=int(capture.sample_count),
        metadata=metadata,
    )
