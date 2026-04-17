from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("default.yaml")


def _as_float_array(values: Any, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {array.shape}.")
    return array


def _as_string_tuple(values: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in values)


@dataclass(frozen=True)
class RobotConfig:
    urdf_path: Path
    joint_names: tuple[str, ...]
    joint_limits: np.ndarray
    torque_limits: np.ndarray
    home_qpos: np.ndarray
    tcp_offset: np.ndarray
    end_effector_body: str


@dataclass(frozen=True)
class FrictionConfig:
    coulomb: np.ndarray
    viscous: np.ndarray


@dataclass(frozen=True)
class ExcitationConfig:
    duration: float
    base_frequency: float
    amplitude_scale: float


@dataclass(frozen=True)
class TransitionConfig:
    max_ee_speed: float
    min_duration: float
    settle_duration: float


@dataclass(frozen=True)
class IdentificationConfig:
    target_joint: int
    excitation: ExcitationConfig
    transition: TransitionConfig


@dataclass(frozen=True)
class ControllerConfig:
    kp: np.ndarray
    kd: np.ndarray
    feedback_scale: float


@dataclass(frozen=True)
class SafetyConfig:
    joint_limit_margin: float
    enable_torque_clamp: bool
    soft_limit_zone: float


@dataclass(frozen=True)
class SamplingConfig:
    rate: float
    timestep: float
    hardware_reference_step_factor: float


@dataclass(frozen=True)
class FittingConfig:
    velocity_scale: float
    regularization: float
    max_iterations: int
    huber_delta: float
    min_velocity_threshold: float


@dataclass(frozen=True)
class SerialConfig:
    port: str
    baudrate: int


@dataclass(frozen=True)
class VisualizationConfig:
    render: bool
    spawn_rerun: bool
    viewer_fps: float = 30.0
    rerun_log_stride: int = 1
    uart_text_log_interval: int = 100


@dataclass(frozen=True)
class OutputConfig:
    results_dir: Path
    simulation_results_filename: str = "simulation_results.npz"
    hardware_results_filename: str = "hardware_results.npz"
    simulation_prefix: str = "friction_identification"
    hardware_capture_prefix: str = "real_uart_capture"
    hardware_ident_prefix: str = "real_friction_identification"
    legacy_summary_filename: str = "real_friction_identification_summary.json"


@dataclass(frozen=True)
class Config:
    robot: RobotConfig
    simulation_friction: FrictionConfig
    identification: IdentificationConfig
    controller: ControllerConfig
    safety: SafetyConfig
    sampling: SamplingConfig
    fitting: FittingConfig
    serial: SerialConfig
    visualization: VisualizationConfig
    output: OutputConfig
    config_path: Path
    project_root: Path = PROJECT_ROOT

    @property
    def joint_count(self) -> int:
        return len(self.robot.joint_names)

    @property
    def target_joint(self) -> int:
        return int(self.identification.target_joint)

    @property
    def target_joint_name(self) -> str:
        return self.robot.joint_names[self.target_joint]

    @property
    def target_joint_mask(self) -> np.ndarray:
        mask = np.zeros(self.joint_count, dtype=bool)
        mask[self.target_joint] = True
        return mask

    @property
    def results_dir(self) -> Path:
        return self.output.results_dir

    @property
    def simulation_results_path(self) -> Path:
        return self.results_dir / self.output.simulation_results_filename

    @property
    def hardware_results_path(self) -> Path:
        return self.results_dir / self.output.hardware_results_filename

    @property
    def summary_path(self) -> Path:
        return self.hardware_results_path

    @property
    def legacy_summary_path(self) -> Path:
        return self.results_dir / self.output.legacy_summary_filename

    def resolve_project_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return self.project_root / candidate


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Config YAML root must be a mapping.")
    return payload


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    raw = _load_yaml(config_path)

    robot_raw = raw["robot"]
    joint_names = _as_string_tuple(robot_raw["joint_names"])
    joint_count = len(joint_names)

    robot = RobotConfig(
        urdf_path=(PROJECT_ROOT / robot_raw["urdf_path"]).resolve(),
        joint_names=joint_names,
        joint_limits=_as_float_array(robot_raw["joint_limits"], shape=(joint_count, 2)),
        torque_limits=_as_float_array(robot_raw["torque_limits"], shape=(joint_count,)),
        home_qpos=_as_float_array(robot_raw["home_qpos"], shape=(joint_count,)),
        tcp_offset=_as_float_array(robot_raw["tcp_offset"], shape=(3,)),
        end_effector_body=str(robot_raw["end_effector_body"]),
    )

    simulation_friction_raw = raw["simulation_friction"]
    simulation_friction = FrictionConfig(
        coulomb=_as_float_array(simulation_friction_raw["coulomb"], shape=(joint_count,)),
        viscous=_as_float_array(simulation_friction_raw["viscous"], shape=(joint_count,)),
    )

    identification_raw = raw["identification"]
    target_joint = int(identification_raw["target_joint"])
    if not 0 <= target_joint < joint_count:
        raise ValueError(f"identification.target_joint must be in [0, {joint_count - 1}].")
    identification = IdentificationConfig(
        target_joint=target_joint,
        excitation=ExcitationConfig(**identification_raw["excitation"]),
        transition=TransitionConfig(**identification_raw["transition"]),
    )

    controller_raw = raw["controller"]
    controller = ControllerConfig(
        kp=_as_float_array(controller_raw["kp"], shape=(joint_count,)),
        kd=_as_float_array(controller_raw["kd"], shape=(joint_count,)),
        feedback_scale=float(controller_raw["feedback_scale"]),
    )

    safety_raw = raw["safety"]
    safety = SafetyConfig(
        joint_limit_margin=float(safety_raw["joint_limit_margin"]),
        enable_torque_clamp=bool(safety_raw["enable_torque_clamp"]),
        soft_limit_zone=float(safety_raw.get("soft_limit_zone", 0.12)),
    )

    sampling = SamplingConfig(
        rate=float(raw["sampling"]["rate"]),
        timestep=float(raw["sampling"]["timestep"]),
        hardware_reference_step_factor=float(raw["sampling"].get("hardware_reference_step_factor", 4.0)),
    )

    fitting_raw = raw["fitting"]
    fitting = FittingConfig(
        velocity_scale=float(fitting_raw["velocity_scale"]),
        regularization=float(fitting_raw["regularization"]),
        max_iterations=int(fitting_raw["max_iterations"]),
        huber_delta=float(fitting_raw["huber_delta"]),
        min_velocity_threshold=float(fitting_raw["min_velocity_threshold"]),
    )

    serial_raw = raw["serial"]
    serial = SerialConfig(
        port=str(serial_raw["port"]),
        baudrate=int(serial_raw["baudrate"]),
    )

    visualization_raw = raw["visualization"]
    visualization = VisualizationConfig(
        render=bool(visualization_raw["render"]),
        spawn_rerun=bool(visualization_raw["spawn_rerun"]),
        viewer_fps=float(visualization_raw.get("viewer_fps", 30.0)),
        rerun_log_stride=int(visualization_raw.get("rerun_log_stride", 1)),
        uart_text_log_interval=int(visualization_raw.get("uart_text_log_interval", 100)),
    )

    output_raw = raw["output"]
    output = OutputConfig(
        results_dir=(PROJECT_ROOT / output_raw["results_dir"]).resolve(),
        simulation_results_filename=str(
            output_raw.get("simulation_results_filename", "simulation_results.npz")
        ),
        hardware_results_filename=str(
            output_raw.get("hardware_results_filename", "hardware_results.npz")
        ),
        simulation_prefix=str(output_raw.get("simulation_prefix", "friction_identification")),
        hardware_capture_prefix=str(output_raw.get("hardware_capture_prefix", "real_uart_capture")),
        hardware_ident_prefix=str(output_raw.get("hardware_ident_prefix", "real_friction_identification")),
        legacy_summary_filename=str(
            output_raw.get("legacy_summary_filename", output_raw.get("summary_filename", "real_friction_identification_summary.json"))
        ),
    )

    return Config(
        robot=robot,
        simulation_friction=simulation_friction,
        identification=identification,
        controller=controller,
        safety=safety,
        sampling=sampling,
        fitting=fitting,
        serial=serial,
        visualization=visualization,
        output=output,
        config_path=config_path.resolve(),
    )

__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_CONFIG_PATH",
    "Config",
    "RobotConfig",
    "FrictionConfig",
    "ExcitationConfig",
    "TransitionConfig",
    "IdentificationConfig",
    "ControllerConfig",
    "SafetyConfig",
    "SamplingConfig",
    "FittingConfig",
    "SerialConfig",
    "VisualizationConfig",
    "OutputConfig",
    "load_config",
]
