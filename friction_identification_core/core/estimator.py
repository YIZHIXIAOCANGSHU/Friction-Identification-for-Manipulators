from __future__ import annotations

"""Robust least-squares estimators for joint friction identification."""

from typing import Callable, Iterable, Optional

import numpy as np
from scipy.optimize import lsq_linear

from friction_identification_core.core.models import FrictionIdentificationResult, JointFrictionParameters


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

    if validation_mask is None:
        validation_mask = np.zeros(num_samples, dtype=bool)
        validation_mask[::5] = True
        if np.all(validation_mask):
            validation_mask[-1] = False
    else:
        validation_mask = np.asarray(validation_mask, dtype=bool).reshape(-1)
        if validation_mask.size != num_samples:
            raise ValueError("validation_mask 长度必须与样本数一致。")

    train_mask = ~validation_mask
    if np.count_nonzero(train_mask) < max(10, 2 * num_joints):
        raise ValueError("训练样本不足，无法执行多关节摩擦辨识。")

    parameters: list[JointFrictionParameters] = []
    predicted_torque = np.zeros_like(torque)
    train_rmse = np.zeros(num_joints)
    validation_rmse = np.zeros(num_joints)
    train_r2 = np.zeros(num_joints)
    validation_r2 = np.zeros(num_joints)
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
        best_score = None
        params = None
        for candidate_scale in candidate_velocity_scales:
            for include_offset in (False, True):
                candidate_params = fit_joint_friction(
                    velocity[train_mask, joint_idx],
                    torque[train_mask, joint_idx],
                    velocity_scale=candidate_scale,
                    regularization=regularization,
                    max_iterations=max_iterations,
                    huber_delta=huber_delta,
                    min_velocity_threshold=min_velocity_threshold,
                    include_offset=include_offset,
                    nonnegative=True,
                )
                candidate_pred = predict_friction_torque(velocity[:, joint_idx], candidate_params)
                selection_mask = validation_mask if np.any(validation_mask) else train_mask
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

        assert params is not None
        parameters.append(params)

        predicted_torque[:, joint_idx] = predict_friction_torque(velocity[:, joint_idx], params)
        train_rmse[joint_idx] = _compute_rmse(
            torque[train_mask, joint_idx],
            predicted_torque[train_mask, joint_idx],
        )
        validation_rmse[joint_idx] = _compute_rmse(
            torque[validation_mask, joint_idx],
            predicted_torque[validation_mask, joint_idx],
        )
        train_r2[joint_idx] = _compute_r2(
            torque[train_mask, joint_idx],
            predicted_torque[train_mask, joint_idx],
        )
        validation_r2[joint_idx] = _compute_r2(
            torque[validation_mask, joint_idx],
            predicted_torque[validation_mask, joint_idx],
        )

    return FrictionIdentificationResult(
        joint_names=joint_names,
        parameters=parameters,
        predicted_torque=predicted_torque,
        measured_torque=torque,
        train_mask=train_mask,
        validation_mask=validation_mask,
        train_rmse=train_rmse,
        validation_rmse=validation_rmse,
        train_r2=train_r2,
        validation_r2=validation_r2,
        true_coulomb=None if true_coulomb is None else np.asarray(true_coulomb, dtype=np.float64),
        true_viscous=None if true_viscous is None else np.asarray(true_viscous, dtype=np.float64),
    )
