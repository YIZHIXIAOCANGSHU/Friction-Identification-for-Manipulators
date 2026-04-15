from __future__ import annotations

"""Tracking-analysis helpers shared by simulation workflows."""

import numpy as np

from .models import FrictionSampleBatch, TrackingEvaluationResult


def build_tracking_evaluation(
    *,
    label: str,
    batch: FrictionSampleBatch,
    controller_coulomb: np.ndarray,
    controller_viscous: np.ndarray,
) -> TrackingEvaluationResult:
    """Summarize trajectory-tracking quality for one controller setup."""

    joint_error = batch.q - batch.q_cmd
    ee_error = batch.ee_pos - batch.ee_pos_cmd
    joint_rmse = np.sqrt(np.mean(joint_error**2, axis=0))
    joint_max_abs_error = np.max(np.abs(joint_error), axis=0)
    ee_rmse_xyz = np.sqrt(np.mean(ee_error**2, axis=0))
    ee_error_norm = np.linalg.norm(ee_error, axis=1)
    return TrackingEvaluationResult(
        label=label,
        batch=batch,
        controller_coulomb=np.asarray(controller_coulomb, dtype=np.float64).copy(),
        controller_viscous=np.asarray(controller_viscous, dtype=np.float64).copy(),
        joint_rmse=joint_rmse,
        joint_max_abs_error=joint_max_abs_error,
        ee_rmse_xyz=ee_rmse_xyz,
        mean_joint_rmse=float(np.mean(joint_rmse)),
        ee_position_rmse=float(np.sqrt(np.mean(ee_error_norm**2))),
        ee_max_error=float(np.max(ee_error_norm)),
    )


def serialize_tracking_evaluation(result: TrackingEvaluationResult) -> dict[str, object]:
    """Convert a tracking summary into JSON-friendly primitives."""

    return {
        "label": result.label,
        "controller_coulomb": result.controller_coulomb.tolist(),
        "controller_viscous": result.controller_viscous.tolist(),
        "mean_joint_rmse_rad": result.mean_joint_rmse,
        "joint_rmse_rad": result.joint_rmse.tolist(),
        "joint_max_abs_error_rad": result.joint_max_abs_error.tolist(),
        "ee_rmse_xyz_m": result.ee_rmse_xyz.tolist(),
        "ee_position_rmse_m": result.ee_position_rmse,
        "ee_max_error_m": result.ee_max_error,
    }


def choose_tracking_winner(
    true_result: TrackingEvaluationResult,
    identified_result: TrackingEvaluationResult,
) -> str:
    """Pick the better tracking run using EE RMSE then joint RMSE as tiebreakers."""

    tolerance = 1e-9
    if identified_result.ee_position_rmse + tolerance < true_result.ee_position_rmse:
        return identified_result.label
    if true_result.ee_position_rmse + tolerance < identified_result.ee_position_rmse:
        return true_result.label

    if identified_result.mean_joint_rmse + tolerance < true_result.mean_joint_rmse:
        return identified_result.label
    if true_result.mean_joint_rmse + tolerance < identified_result.mean_joint_rmse:
        return true_result.label
    return "tie"
