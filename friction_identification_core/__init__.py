"""Portable friction-identification core for MuJoCo-driven robots."""

from .config import (
    DEFAULT_FRICTION_CONFIG,
    CollectionConfig,
    FitConfig,
    FrictionIdentificationConfig,
    RobotModelConfig,
    SampleFilterConfig,
)
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
    "CollectionConfig",
    "DEFAULT_FRICTION_CONFIG",
    "FitConfig",
    "FrictionIdentificationConfig",
    "RobotModelConfig",
    "SampleFilterConfig",
    "build_friction_regression_matrix",
    "fit_joint_friction",
    "fit_multijoint_friction",
    "predict_friction_torque",
]
