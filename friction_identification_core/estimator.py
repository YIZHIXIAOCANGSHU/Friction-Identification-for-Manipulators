from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np
from scipy.optimize import lsq_linear

from friction_identification_core.models import FrictionIdentificationResult, JointFrictionParameters


def smooth_sign(velocity: np.ndarray, velocity_scale: float) -> np.ndarray:
    """Smooth sign approximation used for the Coulomb friction term."""

    scale = max(float(velocity_scale), 1e-6)
    return np.tanh(np.asarray(velocity, dtype=np.float64) / scale)


def build_friction_regression_matrix(
    velocity: np.ndarray,
    velocity_scale: float = 0.02,
    include_offset: bool = True,
) -> np.ndarray:
    """Build [sign(v), v, 1] regression features for one joint."""

    velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
    columns = [smooth_sign(velocity, velocity_scale), velocity]
    if include_offset:
        columns.append(np.ones_like(velocity))
    return np.column_stack(columns)


def predict_friction_torque(
    velocity: np.ndarray,
    parameters: JointFrictionParameters,
) -> np.ndarray:
    """Predict one joint's friction torque from velocity and fitted parameters."""

    regressor = build_friction_regression_matrix(
        velocity=velocity,
        velocity_scale=parameters.velocity_scale,
        include_offset=True,
    )
    coeffs = np.array(
        [parameters.coulomb, parameters.viscous, parameters.offset],
        dtype=np.float64,
    )
    return regressor @ coeffs


def _solve_weighted_regularized_ls(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    regularization: float,
    bounds: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    weights = np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 1e-8, None)
    sqrt_w = np.sqrt(weights)[:, None]
    design_w = design * sqrt_w
    target_w = target * sqrt_w[:, 0]

    if bounds is None:
        lhs = design_w.T @ design_w + regularization * np.eye(design.shape[1], dtype=np.float64)
        rhs = design_w.T @ target_w
        return np.linalg.solve(lhs, rhs)

    eye = np.sqrt(max(float(regularization), 0.0)) * np.eye(design.shape[1], dtype=np.float64)
    augmented_design = np.vstack([design_w, eye])
    augmented_target = np.concatenate([target_w, np.zeros(design.shape[1], dtype=np.float64)])
    lower, upper = bounds
    result = lsq_linear(
        augmented_design,
        augmented_target,
        bounds=(np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)),
        lsmr_tol="auto",
    )
    return result.x


def _build_huber_weights(residual: np.ndarray, huber_delta: float) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64)
    median = np.median(residual)
    mad = np.median(np.abs(residual - median))
    scale = mad / 0.6745 if mad > 1e-8 else max(np.std(residual), 1e-3)
    normalized = np.abs(residual) / (scale * huber_delta + 1e-12)
    return np.where(normalized <= 1.0, 1.0, 1.0 / normalized)


def _build_balance_weights(velocity: np.ndarray) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
    weights = np.ones_like(velocity)
    positive = velocity > 0.0
    negative = velocity < 0.0

    positive_count = max(int(np.count_nonzero(positive)), 1)
    negative_count = max(int(np.count_nonzero(negative)), 1)
    if np.any(positive):
        weights[positive] *= 0.5 / positive_count
    if np.any(negative):
        weights[negative] *= 0.5 / negative_count

    stationary = ~(positive | negative)
    if np.any(stationary):
        weights[stationary] *= 1.0 / max(int(np.count_nonzero(stationary)), 1)

    speed = np.abs(velocity)
    speed_scale = np.percentile(speed, 75) if speed.size > 0 else 0.0
    if speed_scale > 1e-6:
        weights *= 0.7 + 0.3 * np.clip(speed / speed_scale, 0.0, 2.0)

    return weights / max(np.mean(weights), 1e-8)


def _compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    residual = np.sum((y_true - y_pred) ** 2)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 1e-12:
        return 1.0 if residual <= 1e-12 else 0.0
    return float(1.0 - residual / denom)


def _build_periodic_validation_mask(num_samples: int) -> np.ndarray:
    mask = np.zeros(max(int(num_samples), 0), dtype=bool)
    if mask.size == 0:
        return mask
    mask[::5] = True
    mask[: min(20, mask.size)] = False
    if np.all(mask):
        mask[-1] = False
    return mask


def _build_candidate_velocity_scales_for_joint(
    joint_velocity: np.ndarray,
    *,
    default_velocity_scale: float,
    fallback: tuple[float, ...],
) -> tuple[float, ...]:
    joint_velocity = np.asarray(joint_velocity, dtype=np.float64).reshape(-1)
    speed = np.abs(joint_velocity)
    speed_nonzero = speed[np.isfinite(speed) & (speed > 1e-6)]
    if speed_nonzero.size <= 10:
        return fallback

    v_10, v_50, v_90 = np.percentile(speed_nonzero, [10, 50, 90])
    dynamic_scales = (
        v_10 * 0.3,
        v_10 * 0.5,
        v_10 * 0.8,
        v_50 * 0.2,
        v_50 * 0.4,
        v_90 * 0.1,
        v_90 * 0.2,
    )
    candidates = {
        float(default_velocity_scale),
        0.005,
        0.01,
        0.02,
        0.03,
        0.05,
        0.08,
        0.12,
    }
    for candidate in dynamic_scales:
        if np.isfinite(candidate) and candidate > 0.0:
            candidates.add(float(np.clip(candidate, 0.003, 0.3)))
    return tuple(sorted(candidates))


def fit_joint_friction(
    velocity: np.ndarray,
    torque: np.ndarray,
    *,
    velocity_scale: float = 0.02,
    regularization: float = 1e-8,
    max_iterations: int = 12,
    huber_delta: float = 1.35,
    min_velocity_threshold: float = 0.0,
    include_offset: bool = True,
    nonnegative: bool = True,
) -> JointFrictionParameters:
    velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
    torque = np.asarray(torque, dtype=np.float64).reshape(-1)
    valid = np.isfinite(velocity) & np.isfinite(torque)
    if min_velocity_threshold > 0.0:
        valid &= np.abs(velocity) >= min_velocity_threshold

    velocity_valid = velocity[valid]
    torque_valid = torque[valid]
    if velocity_valid.size < 8:
        raise ValueError("有效摩擦样本过少，无法稳定拟合。")

    design = build_friction_regression_matrix(
        velocity_valid,
        velocity_scale=velocity_scale,
        include_offset=include_offset,
    )
    weights = _build_balance_weights(velocity_valid)
    bounds = None
    if nonnegative:
        lower = np.zeros(design.shape[1], dtype=np.float64)
        upper = np.full(design.shape[1], np.inf, dtype=np.float64)
        if include_offset:
            lower[-1] = -0.05
            upper[-1] = 0.05
        bounds = (lower, upper)

    coeffs = _solve_weighted_regularized_ls(
        design,
        torque_valid,
        weights,
        regularization,
        bounds=bounds,
    )

    for _ in range(max_iterations):
        residual = torque_valid - design @ coeffs
        robust_weights = _build_huber_weights(residual, huber_delta=huber_delta)
        combined_weights = weights * robust_weights
        new_coeffs = _solve_weighted_regularized_ls(
            design,
            torque_valid,
            combined_weights,
            regularization,
            bounds=bounds,
        )
        if np.linalg.norm(new_coeffs - coeffs) <= 1e-8 * max(1.0, np.linalg.norm(coeffs)):
            coeffs = new_coeffs
            break
        coeffs = new_coeffs

    if not include_offset:
        coeffs = np.append(coeffs, 0.0)

    return JointFrictionParameters(
        coulomb=float(coeffs[0]),
        viscous=float(coeffs[1]),
        offset=float(coeffs[2]),
        velocity_scale=float(velocity_scale),
    )


def fit_multijoint_friction(
    velocity: np.ndarray,
    torque: np.ndarray,
    *,
    joint_names: Optional[Iterable[str]] = None,
    validation_mask: Optional[np.ndarray] = None,
    sample_mask: Optional[np.ndarray] = None,
    velocity_scale: float = 0.02,
    regularization: float = 1e-8,
    max_iterations: int = 12,
    huber_delta: float = 1.35,
    min_velocity_threshold: float = 0.0,
    true_coulomb: Optional[np.ndarray] = None,
    true_viscous: Optional[np.ndarray] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> FrictionIdentificationResult:
    velocity = np.asarray(velocity, dtype=np.float64)
    torque = np.asarray(torque, dtype=np.float64)
    if velocity.shape != torque.shape or velocity.ndim != 2:
        raise ValueError("velocity 和 torque 必须是同形状的二维数组 [N, J]。")

    num_samples, num_joints = velocity.shape
    if joint_names is None:
        joint_names = [f"joint_{idx + 1}" for idx in range(num_joints)]
    else:
        joint_names = list(joint_names)

    if sample_mask is None:
        joint_sample_mask = np.isfinite(velocity) & np.isfinite(torque)
    else:
        joint_sample_mask = np.asarray(sample_mask, dtype=bool)
        if joint_sample_mask.shape != velocity.shape:
            raise ValueError("sample_mask 必须与 velocity/torque 形状一致。")
        joint_sample_mask &= np.isfinite(velocity) & np.isfinite(torque)

    if validation_mask is None:
        validation_mask_matrix = np.zeros_like(joint_sample_mask, dtype=bool)
        for joint_idx in range(num_joints):
            joint_indices = np.flatnonzero(joint_sample_mask[:, joint_idx])
            if joint_indices.size == 0:
                continue
            joint_validation = _build_periodic_validation_mask(joint_indices.size)
            validation_mask_matrix[joint_indices[joint_validation], joint_idx] = True
    else:
        raw_validation_mask = np.asarray(validation_mask, dtype=bool)
        if raw_validation_mask.ndim == 1:
            if raw_validation_mask.size != num_samples:
                raise ValueError("validation_mask 长度必须与样本数一致。")
            validation_mask_matrix = joint_sample_mask & raw_validation_mask[:, None]
        elif raw_validation_mask.shape == velocity.shape:
            validation_mask_matrix = joint_sample_mask & raw_validation_mask
        else:
            raise ValueError("validation_mask 必须是长度为 N 的向量或形状为 [N, J] 的布尔数组。")

    train_mask = joint_sample_mask & (~validation_mask_matrix)

    parameters: list[JointFrictionParameters] = []
    predicted_torque = np.full_like(torque, np.nan, dtype=np.float64)
    train_rmse = np.full(num_joints, np.nan, dtype=np.float64)
    validation_rmse = np.full(num_joints, np.nan, dtype=np.float64)
    train_r2 = np.full(num_joints, np.nan, dtype=np.float64)
    validation_r2 = np.full(num_joints, np.nan, dtype=np.float64)
    candidate_velocity_scales = tuple(
        sorted(
            {
                float(velocity_scale),
                0.005,
                0.01,
                0.015,
                0.02,
                0.03,
                0.05,
                0.08,
                0.12,
                0.18,
                0.25,
            }
        )
    )

    for joint_idx in range(num_joints):
        if progress_callback is not None:
            progress_callback(joint_idx + 1, num_joints, joint_names[joint_idx])
        joint_train_mask = train_mask[:, joint_idx]
        joint_validation_mask = validation_mask_matrix[:, joint_idx]
        if np.count_nonzero(joint_train_mask) < 8:
            parameters.append(
                JointFrictionParameters(
                    coulomb=float("nan"),
                    viscous=float("nan"),
                    offset=float("nan"),
                    velocity_scale=float(velocity_scale),
                )
            )
            continue

        candidate_velocity_scales_joint = _build_candidate_velocity_scales_for_joint(
            velocity[joint_train_mask, joint_idx],
            default_velocity_scale=velocity_scale,
            fallback=candidate_velocity_scales,
        )
        best_score = None
        params = None
        for candidate_scale in candidate_velocity_scales_joint:
            for include_offset in (False, True):
                try:
                    candidate_params = fit_joint_friction(
                        velocity[joint_train_mask, joint_idx],
                        torque[joint_train_mask, joint_idx],
                        velocity_scale=candidate_scale,
                        regularization=regularization,
                        max_iterations=max_iterations,
                        huber_delta=huber_delta,
                        min_velocity_threshold=min_velocity_threshold,
                        include_offset=include_offset,
                        nonnegative=True,
                    )
                except ValueError:
                    continue
                candidate_pred = predict_friction_torque(velocity[:, joint_idx], candidate_params)
                selection_mask = joint_validation_mask if np.any(joint_validation_mask) else joint_train_mask
                selection_rmse = _compute_rmse(
                    torque[selection_mask, joint_idx],
                    candidate_pred[selection_mask],
                )
                complexity_penalty = 5e-4 * abs(candidate_params.offset)
                complexity_penalty += 2e-4 if include_offset else 0.0
                score = selection_rmse + complexity_penalty
                if best_score is None or score < best_score:
                    best_score = score
                    params = candidate_params

        if params is None:
            parameters.append(
                JointFrictionParameters(
                    coulomb=float("nan"),
                    viscous=float("nan"),
                    offset=float("nan"),
                    velocity_scale=float(velocity_scale),
                )
            )
            continue
        parameters.append(params)

        predicted_torque[:, joint_idx] = predict_friction_torque(velocity[:, joint_idx], params)
        train_rmse[joint_idx] = _compute_rmse(
            torque[joint_train_mask, joint_idx],
            predicted_torque[joint_train_mask, joint_idx],
        )
        validation_rmse[joint_idx] = _compute_rmse(
            torque[joint_validation_mask, joint_idx],
            predicted_torque[joint_validation_mask, joint_idx],
        )
        train_r2[joint_idx] = _compute_r2(
            torque[joint_train_mask, joint_idx],
            predicted_torque[joint_train_mask, joint_idx],
        )
        validation_r2[joint_idx] = _compute_r2(
            torque[joint_validation_mask, joint_idx],
            predicted_torque[joint_validation_mask, joint_idx],
        )

    return FrictionIdentificationResult(
        joint_names=joint_names,
        parameters=parameters,
        predicted_torque=predicted_torque,
        measured_torque=torque,
        train_mask=train_mask,
        validation_mask=validation_mask_matrix,
        train_rmse=train_rmse,
        validation_rmse=validation_rmse,
        train_r2=train_r2,
        validation_r2=validation_r2,
        true_coulomb=None if true_coulomb is None else np.asarray(true_coulomb, dtype=np.float64),
        true_viscous=None if true_viscous is None else np.asarray(true_viscous, dtype=np.float64),
    )
