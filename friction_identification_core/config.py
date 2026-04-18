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


def _as_int_tuple(values: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in values)


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
class ExcitationConfig:
    profile: str
    window_mode: str
    duration: float
    sweep_cycles: int
    reversal_pause_s: float
    zero_crossing_dither_s: float
    harmonic_weights: np.ndarray
    speed_schedule: np.ndarray
    phase_offsets: np.ndarray


@dataclass(frozen=True)
class TransitionConfig:
    max_ee_speed: float
    min_duration: float
    settle_duration: float


@dataclass(frozen=True)
class SequentialConfig:
    joint_duration: float
    zero_position_duration: float
    inter_joint_delay: float
    num_groups: int
    inter_group_delay: float


@dataclass(frozen=True)
class IdentificationConfig:
    mode: str
    active_joints: tuple[int, ...]
    sequential: SequentialConfig
    excitation: ExcitationConfig
    transition: TransitionConfig


@dataclass(frozen=True)
class BatchCollectionConfig:
    num_batches: int
    inter_batch_delay: float


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
    feedback_stale_timeout_factor: float


@dataclass(frozen=True)
class VisualizationConfig:
    render: bool
    spawn_rerun: bool
    rerun_mode: str
    viewer_fps: float = 30.0
    rerun_log_stride: int = 5
    uart_text_log_interval: int = 100


@dataclass(frozen=True)
class StatusConfig:
    velocity_eps: float


@dataclass(frozen=True)
class OutputConfig:
    results_dir: Path
    hardware_capture_prefix: str = "hardware_capture"
    hardware_ident_prefix: str = "hardware_identification"
    hardware_summary_filename: str = "hardware_identification_summary.npz"
    hardware_report_filename: str = "hardware_identification_report.md"
    hardware_compensation_filename: str = "hardware_compensation_validation.npz"
    legacy_summary_filename: str = "real_friction_identification_summary.json"


@dataclass(frozen=True)
class Config:
    robot: RobotConfig
    identification: IdentificationConfig
    batch_collection: BatchCollectionConfig
    controller: ControllerConfig
    safety: SafetyConfig
    sampling: SamplingConfig
    fitting: FittingConfig
    serial: SerialConfig
    visualization: VisualizationConfig
    status: StatusConfig
    output: OutputConfig
    config_path: Path
    project_root: Path = PROJECT_ROOT

    @property
    def joint_count(self) -> int:
        return len(self.robot.joint_names)

    @property
    def active_joint_indices(self) -> np.ndarray:
        return np.asarray(self.identification.active_joints, dtype=np.int64)

    @property
    def active_joint_mask(self) -> np.ndarray:
        mask = np.zeros(self.joint_count, dtype=bool)
        mask[self.active_joint_indices] = True
        return mask

    @property
    def active_joint_names(self) -> list[str]:
        return [self.robot.joint_names[idx] for idx in self.identification.active_joints]

    @property
    def results_dir(self) -> Path:
        return self.output.results_dir

    @property
    def summary_path(self) -> Path:
        return self.results_dir / self.output.hardware_summary_filename

    @property
    def report_path(self) -> Path:
        return self.results_dir / self.output.hardware_report_filename

    @property
    def compensation_validation_path(self) -> Path:
        return self.results_dir / self.output.hardware_compensation_filename

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


def _parse_identification_config(raw: dict[str, Any], joint_count: int) -> IdentificationConfig:
    mode = str(raw.get("mode", "parallel")).strip().lower()
    if mode not in {"parallel", "sequential"}:
        raise ValueError("identification.mode must be 'parallel' or 'sequential'.")

    active_joints = _as_int_tuple(raw["active_joints"])
    if not active_joints:
        raise ValueError("identification.active_joints must not be empty.")
    if len(set(active_joints)) != len(active_joints):
        raise ValueError("identification.active_joints must not contain duplicates.")
    for joint_idx in active_joints:
        if not 0 <= joint_idx < joint_count:
            raise ValueError(f"active joint index {joint_idx} is outside [0, {joint_count - 1}].")

    excitation_raw = raw["excitation"]
    excitation = ExcitationConfig(
        profile=str(excitation_raw["profile"]),
        window_mode=str(excitation_raw.get("window_mode", "safe")).strip().lower(),
        duration=float(excitation_raw["duration"]),
        sweep_cycles=int(excitation_raw["sweep_cycles"]),
        reversal_pause_s=float(excitation_raw["reversal_pause_s"]),
        zero_crossing_dither_s=float(excitation_raw["zero_crossing_dither_s"]),
        harmonic_weights=_as_float_array(excitation_raw["harmonic_weights"]),
        speed_schedule=_as_float_array(excitation_raw["speed_schedule"]),
        phase_offsets=_as_float_array(excitation_raw["phase_offsets"], shape=(joint_count,)),
    )
    if excitation.window_mode not in {"safe", "hard", "unbounded"}:
        raise ValueError("identification.excitation.window_mode must be 'safe', 'hard', or 'unbounded'.")
    if excitation.harmonic_weights.size == 0:
        raise ValueError("identification.excitation.harmonic_weights must not be empty.")
    if excitation.speed_schedule.size == 0:
        raise ValueError("identification.excitation.speed_schedule must not be empty.")

    sequential_raw = raw.get("sequential", {})
    sequential = SequentialConfig(
        joint_duration=float(sequential_raw.get("joint_duration", excitation.duration)),
        zero_position_duration=float(sequential_raw.get("zero_position_duration", 3.0)),
        inter_joint_delay=float(sequential_raw.get("inter_joint_delay", 2.0)),
        num_groups=int(sequential_raw.get("num_groups", 1)),
        inter_group_delay=float(sequential_raw.get("inter_group_delay", 0.0)),
    )
    if sequential.joint_duration <= 0.0:
        raise ValueError("identification.sequential.joint_duration must be > 0.")
    if sequential.zero_position_duration < 0.0:
        raise ValueError("identification.sequential.zero_position_duration must be >= 0.")
    if sequential.inter_joint_delay < 0.0:
        raise ValueError("identification.sequential.inter_joint_delay must be >= 0.")
    if sequential.num_groups <= 0:
        raise ValueError("identification.sequential.num_groups must be >= 1.")
    if sequential.inter_group_delay < 0.0:
        raise ValueError("identification.sequential.inter_group_delay must be >= 0.")

    transition_raw = raw["transition"]
    transition = TransitionConfig(
        max_ee_speed=float(transition_raw["max_ee_speed"]),
        min_duration=float(transition_raw["min_duration"]),
        settle_duration=float(transition_raw["settle_duration"]),
    )
    return IdentificationConfig(
        mode=mode,
        active_joints=active_joints,
        sequential=sequential,
        excitation=excitation,
        transition=transition,
    )


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

    identification = _parse_identification_config(raw["identification"], joint_count)

    batch_collection_raw = raw.get("batch_collection", {})
    batch_collection = BatchCollectionConfig(
        num_batches=int(batch_collection_raw.get("num_batches", 1)),
        inter_batch_delay=float(batch_collection_raw.get("inter_batch_delay", 0.0)),
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

    sampling_raw = raw["sampling"]
    sampling = SamplingConfig(
        rate=float(sampling_raw["rate"]),
        timestep=float(sampling_raw["timestep"]),
        hardware_reference_step_factor=float(sampling_raw.get("hardware_reference_step_factor", 4.0)),
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
    feedback_stale_timeout_factor = float(serial_raw.get("feedback_stale_timeout_factor", 8.0))
    if feedback_stale_timeout_factor <= 0.0:
        raise ValueError("serial.feedback_stale_timeout_factor must be > 0.")
    serial = SerialConfig(
        port=str(serial_raw["port"]),
        baudrate=int(serial_raw["baudrate"]),
        feedback_stale_timeout_factor=feedback_stale_timeout_factor,
    )

    visualization_raw = raw["visualization"]
    rerun_mode = str(
        visualization_raw.get(
            "rerun_mode",
            "focused" if identification.mode == "sequential" else "full",
        )
    ).strip().lower()
    if rerun_mode not in {"full", "focused"}:
        raise ValueError("visualization.rerun_mode must be 'full' or 'focused'.")
    visualization = VisualizationConfig(
        render=bool(visualization_raw["render"]),
        spawn_rerun=bool(visualization_raw["spawn_rerun"]),
        rerun_mode=rerun_mode,
        viewer_fps=float(visualization_raw.get("viewer_fps", 30.0)),
        rerun_log_stride=int(visualization_raw.get("rerun_log_stride", 5)),
        uart_text_log_interval=int(visualization_raw.get("uart_text_log_interval", 100)),
    )

    status_raw = raw.get("status", {})
    status = StatusConfig(
        velocity_eps=float(status_raw.get("velocity_eps", 0.02)),
    )

    output_raw = raw["output"]
    output = OutputConfig(
        results_dir=(PROJECT_ROOT / output_raw["results_dir"]).resolve(),
        hardware_capture_prefix=str(output_raw.get("hardware_capture_prefix", "hardware_capture")),
        hardware_ident_prefix=str(output_raw.get("hardware_ident_prefix", "hardware_identification")),
        hardware_summary_filename=str(
            output_raw.get("hardware_summary_filename", "hardware_identification_summary.npz")
        ),
        hardware_report_filename=str(
            output_raw.get("hardware_report_filename", "hardware_identification_report.md")
        ),
        hardware_compensation_filename=str(
            output_raw.get("hardware_compensation_filename", "hardware_compensation_validation.npz")
        ),
        legacy_summary_filename=str(
            output_raw.get("legacy_summary_filename", "real_friction_identification_summary.json")
        ),
    )

    return Config(
        robot=robot,
        identification=identification,
        batch_collection=batch_collection,
        controller=controller,
        safety=safety,
        sampling=sampling,
        fitting=fitting,
        serial=serial,
        visualization=visualization,
        status=status,
        output=output,
        config_path=config_path.resolve(),
    )


__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_CONFIG_PATH",
    "BatchCollectionConfig",
    "Config",
    "ControllerConfig",
    "ExcitationConfig",
    "IdentificationConfig",
    "OutputConfig",
    "RobotConfig",
    "SafetyConfig",
    "SequentialConfig",
    "SamplingConfig",
    "SerialConfig",
    "StatusConfig",
    "TransitionConfig",
    "VisualizationConfig",
    "load_config",
]
