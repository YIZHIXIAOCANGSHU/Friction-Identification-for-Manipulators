from __future__ import annotations

import numpy as np

from friction_identification_core.config import ExcitationConfig
from friction_identification_core.models import ReferenceTrajectory


_VELOCITY_LEVELS = (0.25, 0.5, 0.75, 1.0)
_RAMP_WEIGHT = 1.0
_HOLD_WEIGHT = 2.0


def _blend_segment(
    *,
    time: np.ndarray,
    velocity_cmd: np.ndarray,
    phase_name: np.ndarray,
    start_s: float,
    duration_s: float,
    start_velocity: float,
    end_velocity: float,
    phase: str,
) -> None:
    if duration_s <= 0.0:
        return
    end_s = start_s + duration_s
    mask = (time >= start_s) & (time < end_s)
    if not np.any(mask):
        return
    u = np.clip((time[mask] - start_s) / duration_s, 0.0, 1.0)
    blend = 0.5 - 0.5 * np.cos(np.pi * u)
    velocity_cmd[mask] = start_velocity + (end_velocity - start_velocity) * blend
    phase_name[mask] = phase


def _hold_segment(
    *,
    time: np.ndarray,
    velocity_cmd: np.ndarray,
    phase_name: np.ndarray,
    start_s: float,
    duration_s: float,
    velocity: float,
    phase: str,
) -> None:
    if duration_s <= 0.0:
        return
    end_s = start_s + duration_s
    mask = (time >= start_s) & (time < end_s)
    if not np.any(mask):
        return
    velocity_cmd[mask] = velocity
    phase_name[mask] = phase


def build_reference_trajectory(config: ExcitationConfig, *, max_velocity: float) -> ReferenceTrajectory:
    sample_rate = max(float(config.sample_rate), 1.0)
    dt = 1.0 / sample_rate
    motion_duration = max(float(config.duration), dt)
    total_duration = float(config.hold_start) + motion_duration + float(config.hold_end)
    sample_count = max(int(np.ceil(total_duration * sample_rate)), 2)

    time = np.arange(sample_count, dtype=np.float64) * dt
    velocity_cmd = np.zeros(sample_count, dtype=np.float64)
    phase_name = np.full(sample_count, "hold_start", dtype="<U32")

    level_velocities = tuple(float(level) * float(max_velocity) for level in _VELOCITY_LEVELS)
    segment_specs: list[tuple[str, float, float]] = []
    for level_index, level_velocity in enumerate(level_velocities, start=1):
        segment_specs.extend(
            (
                (f"ramp_to_forward_{level_index:02d}", level_velocity, _RAMP_WEIGHT),
                (f"hold_forward_{level_index:02d}", level_velocity, _HOLD_WEIGHT),
                (f"return_from_forward_{level_index:02d}", 0.0, _RAMP_WEIGHT),
                (f"ramp_to_reverse_{level_index:02d}", -level_velocity, _RAMP_WEIGHT),
                (f"hold_reverse_{level_index:02d}", -level_velocity, _HOLD_WEIGHT),
                (f"return_from_reverse_{level_index:02d}", 0.0, _RAMP_WEIGHT),
            )
        )
    weights = np.asarray([weight for _, _, weight in segment_specs], dtype=np.float64)
    durations = motion_duration * weights / np.sum(weights)

    current_time = float(config.hold_start)
    current_velocity = 0.0
    for (segment_name, target_velocity, weight), segment_duration in zip(segment_specs, durations):
        if weight == _HOLD_WEIGHT:
            _hold_segment(
                time=time,
                velocity_cmd=velocity_cmd,
                phase_name=phase_name,
                start_s=current_time,
                duration_s=float(segment_duration),
                velocity=float(target_velocity),
                phase=str(segment_name),
            )
        else:
            _blend_segment(
                time=time,
                velocity_cmd=velocity_cmd,
                phase_name=phase_name,
                start_s=current_time,
                duration_s=float(segment_duration),
                start_velocity=float(current_velocity),
                end_velocity=float(target_velocity),
                phase=str(segment_name),
            )
        current_time += float(segment_duration)
        current_velocity = float(target_velocity)

    hold_end_mask = time >= (float(config.hold_start) + motion_duration)
    phase_name[hold_end_mask] = "hold_end"
    velocity_cmd[hold_end_mask] = 0.0
    acceleration_cmd = np.gradient(velocity_cmd, dt)
    position_cmd = np.zeros(sample_count, dtype=np.float64)
    position_cmd[1:] = np.cumsum((velocity_cmd[:-1] + velocity_cmd[1:]) * 0.5 * dt)

    return ReferenceTrajectory(
        time=time,
        position_cmd=position_cmd,
        velocity_cmd=velocity_cmd,
        acceleration_cmd=acceleration_cmd,
        phase_name=phase_name,
        duration_s=total_duration,
    )
