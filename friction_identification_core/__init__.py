"""Portable friction-identification core for MuJoCo-driven robots."""

from .estimator import (
    build_friction_regression_matrix,
    fit_joint_friction,
    fit_multijoint_friction,
    predict_friction_torque,
)
from .models import FrictionSampleBatch, FrictionIdentificationResult, JointFrictionParameters

__all__ = [
    "FrictionSampleBatch",
    "FrictionIdentificationResult",
    "JointFrictionParameters",
    "build_friction_regression_matrix",
    "fit_joint_friction",
    "fit_multijoint_friction",
    "predict_friction_torque",
]
