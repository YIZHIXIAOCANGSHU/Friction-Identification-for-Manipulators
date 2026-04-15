"""Portable friction-identification core for MuJoCo-driven robots."""

from friction_identification_core.config import DEFAULT_CONFIG_PATH, Config, load_config
from friction_identification_core.core.controller import FrictionIdentificationController
from friction_identification_core.core.estimator import (
    build_friction_regression_matrix,
    fit_joint_friction,
    fit_multijoint_friction,
    predict_friction_torque,
)
from friction_identification_core.core.models import (
    FrictionIdentificationResult,
    FrictionSampleBatch,
    JointFrictionParameters,
)
from friction_identification_core.core.safety import SafetyGuard
from friction_identification_core.core.trajectory import ReferenceTrajectory, sample_reference_trajectory
from friction_identification_core.hardware.runner import run_hardware
from friction_identification_core.simulation.runner import run_simulation

__all__ = [
    "Config",
    "DEFAULT_CONFIG_PATH",
    "FrictionIdentificationController",
    "SafetyGuard",
    "FrictionSampleBatch",
    "FrictionIdentificationResult",
    "JointFrictionParameters",
    "ReferenceTrajectory",
    "build_friction_regression_matrix",
    "fit_joint_friction",
    "fit_multijoint_friction",
    "load_config",
    "predict_friction_torque",
    "run_hardware",
    "run_simulation",
    "sample_reference_trajectory",
]
