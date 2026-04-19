from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("default.yaml")
DEFAULT_TORQUE_LIMITS = np.array([40.0, 40.0, 27.0, 27.0, 7.0, 7.0, 9.0], dtype=np.float64)


def _as_int_tuple(values: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in values)


def _as_float_tuple(values: Any) -> tuple[float, ...]:
    return tuple(float(item) for item in values)


def _expand_float_vector(values: Any, size: int, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 1:
        return np.full(size, float(array[0]), dtype=np.float64)
    if array.size != size:
        raise ValueError(f"{name} must contain either 1 or {size} values.")
    return array.astype(np.float64, copy=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Config YAML root must be a mapping.")
    return payload


@dataclass(frozen=True)
class MotorsConfig:
    ids: tuple[int, ...]
    names: tuple[str, ...]
    enabled_ids: tuple[int, ...]

    def name_for(self, motor_id: int) -> str:
        try:
            index = self.ids.index(int(motor_id))
        except ValueError as exc:
            raise KeyError(f"Unknown motor_id: {motor_id}") from exc
        return self.names[index]


@dataclass(frozen=True)
class SerialConfig:
    port: str
    baudrate: int
    read_timeout: float
    write_timeout: float
    read_chunk_size: int
    flush_input_before_round: bool


@dataclass(frozen=True)
class ExcitationConfig:
    sample_rate: float
    duration: float
    hold_start: float
    hold_end: float


@dataclass(frozen=True)
class ControlConfig:
    max_velocity: np.ndarray
    max_torque: np.ndarray


@dataclass(frozen=True)
class IdentificationConfig:
    group_count: int
    regularization: float
    huber_delta: float
    max_iterations: int
    min_samples: int
    zero_velocity_threshold: float
    min_motion_span: float
    validation_stride: int
    validation_warmup_samples: int
    savgol_window: int
    savgol_polyorder: int
    velocity_scale_candidates: tuple[float, ...]


@dataclass(frozen=True)
class OutputConfig:
    results_dir: Path
    summary_filename: str
    summary_csv_filename: str
    summary_report_filename: str


@dataclass(frozen=True)
class Config:
    motors: MotorsConfig
    serial: SerialConfig
    excitation: ExcitationConfig
    control: ControlConfig
    identification: IdentificationConfig
    output: OutputConfig
    config_path: Path
    project_root: Path = PROJECT_ROOT

    @property
    def motor_ids(self) -> tuple[int, ...]:
        return self.motors.ids

    @property
    def enabled_motor_ids(self) -> tuple[int, ...]:
        return self.motors.enabled_ids

    @property
    def motor_count(self) -> int:
        return len(self.motors.ids)

    @property
    def group_count(self) -> int:
        return int(self.identification.group_count)

    @property
    def results_dir(self) -> Path:
        return self.output.results_dir

    def motor_index(self, motor_id: int) -> int:
        try:
            return self.motors.ids.index(int(motor_id))
        except ValueError as exc:
            raise KeyError(f"Unknown motor_id: {motor_id}") from exc

    def resolve_project_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return (self.project_root / candidate).resolve()


def _parse_motors(raw: dict[str, Any]) -> MotorsConfig:
    motor_ids = tuple(sorted(set(_as_int_tuple(raw.get("ids", range(1, 8))))))
    if not motor_ids:
        raise ValueError("motors.ids must not be empty.")
    if motor_ids != tuple(range(min(motor_ids), max(motor_ids) + 1)):
        raise ValueError("motors.ids must be a contiguous ascending sequence.")

    names_raw = raw.get("names")
    if names_raw is None:
        names = tuple(f"motor_{motor_id:02d}" for motor_id in motor_ids)
    else:
        names = tuple(str(item) for item in names_raw)
        if len(names) != len(motor_ids):
            raise ValueError("motors.names must match motors.ids length.")

    enabled_ids = tuple(sorted(set(_as_int_tuple(raw.get("enabled", motor_ids)))))
    if not enabled_ids:
        raise ValueError("motors.enabled must not be empty.")
    for motor_id in enabled_ids:
        if motor_id not in motor_ids:
            raise ValueError(f"Enabled motor_id {motor_id} is not present in motors.ids.")

    return MotorsConfig(ids=motor_ids, names=names, enabled_ids=enabled_ids)


def _parse_serial(raw: dict[str, Any]) -> SerialConfig:
    return SerialConfig(
        port=str(raw.get("port", "/dev/ttyUSB0")),
        baudrate=int(raw.get("baudrate", 115200)),
        read_timeout=float(raw.get("read_timeout", 0.02)),
        write_timeout=float(raw.get("write_timeout", 0.02)),
        read_chunk_size=max(int(raw.get("read_chunk_size", 256)), 19),
        flush_input_before_round=bool(raw.get("flush_input_before_round", True)),
    )


def _parse_excitation(raw: dict[str, Any]) -> ExcitationConfig:
    return ExcitationConfig(
        sample_rate=float(raw.get("sample_rate", 200.0)),
        duration=float(raw.get("duration", 18.0)),
        hold_start=float(raw.get("hold_start", 1.5)),
        hold_end=float(raw.get("hold_end", 1.5)),
    )


def _parse_control(raw: dict[str, Any], motor_count: int) -> ControlConfig:
    max_velocity = _expand_float_vector(raw.get("max_velocity", 0.5), motor_count, name="control.max_velocity")
    if np.any(max_velocity <= 0.0):
        raise ValueError("control.max_velocity must all be > 0.")
    max_torque = _expand_float_vector(
        raw.get("max_torque", raw.get("torque_limits", DEFAULT_TORQUE_LIMITS)),
        motor_count,
        name="control.max_torque",
    )
    if np.any(max_torque <= 0.0):
        raise ValueError("control.max_torque must all be > 0.")
    return ControlConfig(
        max_velocity=max_velocity,
        max_torque=max_torque,
    )


def _parse_identification(raw: dict[str, Any]) -> IdentificationConfig:
    velocity_scale_candidates = tuple(float(item) for item in raw.get("velocity_scale_candidates", (0.01, 0.02, 0.05)))
    if not velocity_scale_candidates:
        raise ValueError("identification.velocity_scale_candidates must not be empty.")
    return IdentificationConfig(
        group_count=max(int(raw.get("group_count", 1)), 1),
        regularization=float(raw.get("regularization", 1.0e-6)),
        huber_delta=float(raw.get("huber_delta", 1.5)),
        max_iterations=max(int(raw.get("max_iterations", 12)), 1),
        min_samples=max(int(raw.get("min_samples", 200)), 1),
        zero_velocity_threshold=max(float(raw.get("zero_velocity_threshold", 0.01)), 0.0),
        min_motion_span=max(float(raw.get("min_motion_span", 0.05)), 0.0),
        validation_stride=max(int(raw.get("validation_stride", 5)), 1),
        validation_warmup_samples=max(int(raw.get("validation_warmup_samples", 20)), 0),
        savgol_window=max(int(raw.get("savgol_window", 31)), 3),
        savgol_polyorder=max(int(raw.get("savgol_polyorder", 3)), 1),
        velocity_scale_candidates=velocity_scale_candidates,
    )


def _parse_output(raw: dict[str, Any], *, project_root: Path) -> OutputConfig:
    results_dir = Path(raw.get("results_dir", "results"))
    if not results_dir.is_absolute():
        results_dir = (project_root / results_dir).resolve()
    return OutputConfig(
        results_dir=results_dir,
        summary_filename=str(raw.get("summary_filename", "hardware_identification_summary.npz")),
        summary_csv_filename=str(raw.get("summary_csv_filename", "hardware_identification_summary.csv")),
        summary_report_filename=str(raw.get("summary_report_filename", "hardware_identification_report.md")),
    )


def load_config(path: str | Path) -> Config:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    payload = _load_yaml(candidate)

    motors = _parse_motors(payload.get("motors", {}))
    config = Config(
        motors=motors,
        serial=_parse_serial(payload.get("serial", {})),
        excitation=_parse_excitation(payload.get("excitation", {})),
        control=_parse_control(payload.get("control", {}), len(motors.ids)),
        identification=_parse_identification(payload.get("identification", {})),
        output=_parse_output(payload.get("output", {}), project_root=PROJECT_ROOT),
        config_path=candidate,
    )

    if config.excitation.sample_rate <= 0.0:
        raise ValueError("excitation.sample_rate must be > 0.")
    if config.excitation.duration <= 0.0:
        raise ValueError("excitation.duration must be > 0.")
    return config


def _parse_motor_override(raw: str | None, available_ids: tuple[int, ...]) -> tuple[int, ...] | None:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    if text == "all":
        return available_ids

    parsed: list[int] = []
    seen: set[int] = set()
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        motor_id = int(token)
        if motor_id not in available_ids:
            raise ValueError(f"motor_id {motor_id} is not present in config motors.ids.")
        if motor_id in seen:
            continue
        parsed.append(motor_id)
        seen.add(motor_id)
    if not parsed:
        raise ValueError("--motors did not resolve to any valid motor_id.")
    return tuple(sorted(parsed))


def apply_overrides(
    config: Config,
    *,
    output: str | None = None,
    motors: str | None = None,
    groups: int | None = None,
) -> Config:
    updated = config
    if output:
        updated = replace(
            updated,
            output=replace(updated.output, results_dir=updated.resolve_project_path(output)),
        )

    overridden_motor_ids = _parse_motor_override(motors, updated.motor_ids)
    if overridden_motor_ids is not None:
        updated = replace(updated, motors=replace(updated.motors, enabled_ids=overridden_motor_ids))

    if groups is not None:
        if int(groups) <= 0:
            raise ValueError("--groups must be a positive integer.")
        updated = replace(updated, identification=replace(updated.identification, group_count=int(groups)))
    return updated
