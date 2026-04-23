from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("default.yaml")
DEFAULT_TORQUE_LIMITS = np.array([40.0, 40.0, 27.0, 27.0, 7.0, 7.0, 9.0], dtype=np.float64)

_LEGACY_CONTROL_KEYS = {"velocity_p_gain", "torque_limits"}
_LEGACY_EXCITATION_KEYS = {"platforms", "transition_duration"}


def _as_int_tuple(values: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in values)


def _expand_float_vector(values: Any, size: int, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 1:
        return np.full(size, float(array[0]), dtype=np.float64)
    if array.size != size:
        raise ValueError(f"{name} must contain either 1 or {size} values.")
    return array.astype(np.float64, copy=True)


def _required_float(raw: dict[str, Any], key: str, *, name: str) -> float:
    if key not in raw:
        raise ValueError(f"{name} is required.")
    try:
        return float(raw[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc


def _required_int(raw: dict[str, Any], key: str, *, name: str) -> int:
    if key not in raw:
        raise ValueError(f"{name} is required.")
    try:
        return int(raw[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Config YAML root must be a mapping.")
    return payload


def _reject_legacy_keys(raw: dict[str, Any], *, section: str, keys: set[str]) -> None:
    for key in sorted(keys):
        if key in raw:
            raise ValueError(f"{section}.{key} is no longer supported.")


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
    sync_timeout: float
    sync_cycles_required: int


@dataclass(frozen=True)
class ExcitationConfig:
    sample_rate: float
    curve_type: str
    hold_start: float
    hold_end: float
    position_limit: float
    velocity_utilization: float
    base_frequency: float
    steady_cycles: int
    fade_in_cycles: int
    fade_out_cycles: int
    harmonic_multipliers: tuple[int, ...]
    harmonic_weights: tuple[float, ...]


@dataclass(frozen=True)
class ControlConfig:
    max_velocity: np.ndarray
    max_torque: np.ndarray
    position_gain: np.ndarray
    velocity_gain: np.ndarray
    zeroing_position_gain: np.ndarray
    zeroing_velocity_gain: np.ndarray
    zeroing_hard_velocity_limit: np.ndarray
    zeroing_velocity_limit: np.ndarray
    zeroing_position_tolerance: np.ndarray
    zeroing_velocity_tolerance: np.ndarray
    zeroing_required_frames: int
    zeroing_timeout: float
    speed_abort_ratio: np.ndarray
    zero_target_velocity_threshold: np.ndarray
    low_speed_abort_limit: np.ndarray


@dataclass(frozen=True)
class IdentificationConfig:
    group_count: int
    regularization: float
    huber_delta: float
    max_iterations: int
    min_samples: int
    min_direction_samples: int
    zero_velocity_threshold: float
    min_motion_span: float
    min_target_frame_ratio: float
    max_sequence_error_ratio: float
    validation_velocity_band_edges_ratio: tuple[float, ...]
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
        sync_timeout=max(float(raw.get("sync_timeout", 2.0)), 0.1),
        sync_cycles_required=max(int(raw.get("sync_cycles_required", 3)), 1),
    )


def _parse_excitation(raw: dict[str, Any]) -> ExcitationConfig:
    _reject_legacy_keys(raw, section="excitation", keys=_LEGACY_EXCITATION_KEYS)

    harmonic_multipliers = tuple(int(item) for item in raw.get("harmonic_multipliers", (1, 2, 3, 4, 5, 6)))
    harmonic_weights = tuple(float(item) for item in raw.get("harmonic_weights", (1.0, 0.75, 0.55, 0.40, 0.30, 0.25)))
    if not harmonic_multipliers:
        raise ValueError("excitation.harmonic_multipliers must not be empty.")
    if len(harmonic_multipliers) != len(harmonic_weights):
        raise ValueError("excitation.harmonic_multipliers and excitation.harmonic_weights must have the same length.")

    return ExcitationConfig(
        sample_rate=float(raw.get("sample_rate", 200.0)),
        curve_type=str(raw.get("curve_type", "multisine")),
        hold_start=float(raw.get("hold_start", 1.0)),
        hold_end=float(raw.get("hold_end", 1.0)),
        position_limit=float(raw.get("position_limit", 2.5)),
        velocity_utilization=float(raw.get("velocity_utilization", 0.85)),
        base_frequency=float(raw.get("base_frequency", 0.25)),
        steady_cycles=int(raw.get("steady_cycles", 6)),
        fade_in_cycles=int(raw.get("fade_in_cycles", 1)),
        fade_out_cycles=int(raw.get("fade_out_cycles", 1)),
        harmonic_multipliers=harmonic_multipliers,
        harmonic_weights=harmonic_weights,
    )


def _parse_control(raw: dict[str, Any], motor_count: int) -> ControlConfig:
    _reject_legacy_keys(raw, section="control", keys=_LEGACY_CONTROL_KEYS)

    max_velocity = _expand_float_vector(raw.get("max_velocity", 1.0), motor_count, name="control.max_velocity")
    max_torque = _expand_float_vector(raw.get("max_torque", DEFAULT_TORQUE_LIMITS), motor_count, name="control.max_torque")
    position_gain = _expand_float_vector(raw.get("position_gain", 0.08), motor_count, name="control.position_gain")
    velocity_gain = _expand_float_vector(raw.get("velocity_gain", 0.03), motor_count, name="control.velocity_gain")
    zeroing_position_gain = _expand_float_vector(
        raw.get("zeroing_position_gain", raw.get("position_gain", 0.8)),
        motor_count,
        name="control.zeroing_position_gain",
    )
    zeroing_velocity_gain = _expand_float_vector(
        raw.get("zeroing_velocity_gain", raw.get("velocity_gain", 0.18)),
        motor_count,
        name="control.zeroing_velocity_gain",
    )
    zeroing_hard_velocity_limit = _expand_float_vector(
        raw.get("zeroing_hard_velocity_limit", 2.0),
        motor_count,
        name="control.zeroing_hard_velocity_limit",
    )
    zeroing_velocity_limit = _expand_float_vector(
        raw.get("zeroing_velocity_limit", 0.40),
        motor_count,
        name="control.zeroing_velocity_limit",
    )
    zeroing_position_tolerance = _expand_float_vector(
        raw.get("zeroing_position_tolerance", 0.02),
        motor_count,
        name="control.zeroing_position_tolerance",
    )
    zeroing_velocity_tolerance = _expand_float_vector(
        raw.get("zeroing_velocity_tolerance", 0.02),
        motor_count,
        name="control.zeroing_velocity_tolerance",
    )
    speed_abort_ratio = _expand_float_vector(raw.get("speed_abort_ratio", 1.20), motor_count, name="control.speed_abort_ratio")
    zero_target_velocity_threshold = _expand_float_vector(
        raw.get("zero_target_velocity_threshold", 0.02),
        motor_count,
        name="control.zero_target_velocity_threshold",
    )
    low_speed_abort_limit = _expand_float_vector(
        raw.get("low_speed_abort_limit", 0.08),
        motor_count,
        name="control.low_speed_abort_limit",
    )

    zeroing_required_frames = int(raw.get("zeroing_required_frames", 8))
    zeroing_timeout = float(raw.get("zeroing_timeout", 8.0))

    if np.any(max_velocity <= 0.0):
        raise ValueError("control.max_velocity must all be > 0.")
    if np.any(max_torque <= 0.0):
        raise ValueError("control.max_torque must all be > 0.")
    if np.any(position_gain < 0.0):
        raise ValueError("control.position_gain must all be >= 0.")
    if np.any(velocity_gain < 0.0):
        raise ValueError("control.velocity_gain must all be >= 0.")
    if np.any(zeroing_position_gain < 0.0):
        raise ValueError("control.zeroing_position_gain must all be >= 0.")
    if np.any(zeroing_velocity_gain < 0.0):
        raise ValueError("control.zeroing_velocity_gain must all be >= 0.")
    if np.any(zeroing_hard_velocity_limit <= 0.0):
        raise ValueError("control.zeroing_hard_velocity_limit must all be > 0.")
    if np.any(zeroing_velocity_limit <= 0.0):
        raise ValueError("control.zeroing_velocity_limit must all be > 0.")
    if np.any(zeroing_position_tolerance < 0.0):
        raise ValueError("control.zeroing_position_tolerance must all be >= 0.")
    if np.any(zeroing_velocity_tolerance <= 0.0):
        raise ValueError("control.zeroing_velocity_tolerance must all be > 0.")
    if zeroing_required_frames <= 0:
        raise ValueError("control.zeroing_required_frames must be > 0.")
    if zeroing_timeout <= 0.0:
        raise ValueError("control.zeroing_timeout must be > 0.")
    if np.any(speed_abort_ratio < 1.0):
        raise ValueError("control.speed_abort_ratio must all be >= 1.0.")
    if np.any(zero_target_velocity_threshold < 0.0):
        raise ValueError("control.zero_target_velocity_threshold must all be >= 0.")
    if np.any(low_speed_abort_limit <= 0.0):
        raise ValueError("control.low_speed_abort_limit must all be > 0.")

    return ControlConfig(
        max_velocity=max_velocity,
        max_torque=max_torque,
        position_gain=position_gain,
        velocity_gain=velocity_gain,
        zeroing_position_gain=zeroing_position_gain,
        zeroing_velocity_gain=zeroing_velocity_gain,
        zeroing_hard_velocity_limit=zeroing_hard_velocity_limit,
        zeroing_velocity_limit=zeroing_velocity_limit,
        zeroing_position_tolerance=zeroing_position_tolerance,
        zeroing_velocity_tolerance=zeroing_velocity_tolerance,
        zeroing_required_frames=zeroing_required_frames,
        zeroing_timeout=zeroing_timeout,
        speed_abort_ratio=speed_abort_ratio,
        zero_target_velocity_threshold=zero_target_velocity_threshold,
        low_speed_abort_limit=low_speed_abort_limit,
    )


def _parse_identification(raw: dict[str, Any]) -> IdentificationConfig:
    velocity_scale_candidates = tuple(float(item) for item in raw.get("velocity_scale_candidates", (0.01, 0.02, 0.05)))
    band_edges = tuple(float(item) for item in raw.get("validation_velocity_band_edges_ratio", (0.05, 0.12, 0.25, 0.40, 0.60, 0.85)))
    if not velocity_scale_candidates:
        raise ValueError("identification.velocity_scale_candidates must not be empty.")
    if len(band_edges) < 2:
        raise ValueError("identification.validation_velocity_band_edges_ratio must contain at least two edges.")
    return IdentificationConfig(
        group_count=max(int(raw.get("group_count", 1)), 1),
        regularization=float(raw.get("regularization", 1.0e-6)),
        huber_delta=float(raw.get("huber_delta", 1.5)),
        max_iterations=max(int(raw.get("max_iterations", 12)), 1),
        min_samples=max(int(raw.get("min_samples", 200)), 1),
        min_direction_samples=max(int(raw.get("min_direction_samples", 300)), 1),
        zero_velocity_threshold=max(float(raw.get("zero_velocity_threshold", 0.015)), 0.0),
        min_motion_span=max(float(raw.get("min_motion_span", 0.05)), 0.0),
        min_target_frame_ratio=float(np.clip(raw.get("min_target_frame_ratio", 0.7), 0.0, 1.0)),
        max_sequence_error_ratio=max(float(raw.get("max_sequence_error_ratio", 0.2)), 0.0),
        validation_velocity_band_edges_ratio=band_edges,
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
        summary_report_filename=str(raw.get("summary_report_filename", "hardware_identification_summary.md")),
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
    if config.excitation.curve_type != "multisine":
        raise ValueError("excitation.curve_type must be 'multisine'.")
    if config.excitation.hold_start < 0.0:
        raise ValueError("excitation.hold_start must be >= 0.")
    if config.excitation.hold_end < 0.0:
        raise ValueError("excitation.hold_end must be >= 0.")
    if config.excitation.position_limit <= 0.0:
        raise ValueError("excitation.position_limit must be > 0.")
    if not (0.0 < config.excitation.velocity_utilization <= 1.0):
        raise ValueError("excitation.velocity_utilization must be within (0, 1].")
    if config.excitation.base_frequency <= 0.0:
        raise ValueError("excitation.base_frequency must be > 0.")
    if config.excitation.steady_cycles <= 0:
        raise ValueError("excitation.steady_cycles must be > 0.")
    if config.excitation.fade_in_cycles < 0:
        raise ValueError("excitation.fade_in_cycles must be >= 0.")
    if config.excitation.fade_out_cycles < 0:
        raise ValueError("excitation.fade_out_cycles must be >= 0.")
    if any(multiplier <= 0 for multiplier in config.excitation.harmonic_multipliers):
        raise ValueError("excitation.harmonic_multipliers must all be > 0.")
    if len(set(config.excitation.harmonic_multipliers)) != len(config.excitation.harmonic_multipliers):
        raise ValueError("excitation.harmonic_multipliers must be unique.")
    if any(weight <= 0.0 for weight in config.excitation.harmonic_weights):
        raise ValueError("excitation.harmonic_weights must all be > 0.")
    edges = np.asarray(config.identification.validation_velocity_band_edges_ratio, dtype=np.float64)
    if np.any(~np.isfinite(edges)) or np.any(edges <= 0.0) or np.any(edges > 1.0):
        raise ValueError("identification.validation_velocity_band_edges_ratio must contain values within (0, 1].")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("identification.validation_velocity_band_edges_ratio must be strictly increasing.")
    return config


def _parse_motor_override(
    raw: str | None,
    available_ids: tuple[int, ...],
    *,
    source_name: str = "config motors.ids",
) -> tuple[int, ...] | None:
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
            raise ValueError(f"motor_id {motor_id} is not present in {source_name}.")
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

    overridden_motor_ids = _parse_motor_override(
        motors,
        updated.enabled_motor_ids,
        source_name="config motors.enabled",
    )
    if overridden_motor_ids is not None:
        updated = replace(updated, motors=replace(updated.motors, enabled_ids=overridden_motor_ids))

    if groups is not None:
        if int(groups) <= 0:
            raise ValueError("--groups must be a positive integer.")
        updated = replace(updated, identification=replace(updated.identification, group_count=int(groups)))
    return updated
