"""Portable friction-identification core for MuJoCo-driven robots."""

from friction_identification_core.config import DEFAULT_CONFIG_PATH, Config, load_config
from friction_identification_core.controller import (
    FrictionIdentificationController,
    SafetyGuard,
    load_compensation_parameters,
    load_summary_vectors,
    predict_compensation_torque,
)
from friction_identification_core.estimator import (
    build_friction_regression_matrix,
    fit_joint_friction,
    fit_multijoint_friction,
    predict_friction_torque,
)
from friction_identification_core.models import (
    CollectedData,
    DataSource,
    FrictionIdentificationResult,
    FrictionSampleBatch,
    IdentificationInputs,
    JointFrictionParameters,
)
from friction_identification_core.pipeline import (
    IdentificationPipeline,
    PipelineRunResult,
    run_hardware,
    run_simulation,
)
from friction_identification_core.results import (
    IdentificationResults,
    JointResult,
    ResultPaths,
    ResultsManager,
    ResultStore,
)
from friction_identification_core.sources import HardwareSource, SimulationSource, build_source
from friction_identification_core.trajectory import ReferenceTrajectory, sample_reference_trajectory

__all__ = [
    "Config",
    "DEFAULT_CONFIG_PATH",
    "CollectedData",
    "DataSource",
    "FrictionIdentificationController",
    "SafetyGuard",
    "FrictionSampleBatch",
    "FrictionIdentificationResult",
    "IdentificationResults",
    "IdentificationInputs",
    "IdentificationPipeline",
    "JointResult",
    "JointFrictionParameters",
    "PipelineRunResult",
    "ReferenceTrajectory",
    "ResultPaths",
    "ResultsManager",
    "ResultStore",
    "HardwareSource",
    "SimulationSource",
    "build_friction_regression_matrix",
    "build_source",
    "fit_joint_friction",
    "fit_multijoint_friction",
    "load_compensation_parameters",
    "load_config",
    "load_summary_vectors",
    "predict_compensation_torque",
    "predict_friction_torque",
    "run_hardware",
    "run_simulation",
    "sample_reference_trajectory",
]
