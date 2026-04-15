"""Portable friction-identification core for MuJoCo-driven robots."""

from .config import (
    DEFAULT_FRICTION_CONFIG,
    CollectionConfig,
    FitConfig,
    FrictionIdentificationConfig,
    RealUartConfig,
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
from .shared_logic import (
    ReferenceTrajectory,
    generate_segmented_excitation_trajectory,
    predict_friction_compensation_torque,
    sample_reference_trajectory,
)

# Re-export the most common config, model, and fitting entry points.
__all__ = [
    "FrictionSampleBatch",
    "FrictionIdentificationResult",
    "JointFrictionParameters",
    "ReferenceTrajectory",
    "CollectionConfig",
    "DEFAULT_FRICTION_CONFIG",
    "FitConfig",
    "FrictionIdentificationConfig",
    "RealUartConfig",
    "RobotModelConfig",
    "SampleFilterConfig",
    "build_friction_regression_matrix",
    "fit_joint_friction",
    "fit_multijoint_friction",
    "generate_segmented_excitation_trajectory",
    "predict_friction_compensation_torque",
    "predict_friction_torque",
    "sample_reference_trajectory",
]
