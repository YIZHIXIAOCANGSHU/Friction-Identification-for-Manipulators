from __future__ import annotations

import numpy as np

from friction_identification_core.config import ExcitationConfig
from friction_identification_core.models import ReferenceTrajectory


def _segment_slice(time: np.ndarray, start_s: float, duration_s: float) -> slice:
    end_s = start_s + duration_s
    epsilon = 1.0e-12
    start_index = int(np.searchsorted(time, start_s - epsilon, side="left"))
    end_index = int(np.searchsorted(time, end_s - epsilon, side="left"))
    return slice(start_index, end_index)


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
    segment = _segment_slice(time, start_s, duration_s)
    if segment.start >= segment.stop:
        return
    segment_time = time[segment]
    u = np.clip((segment_time - start_s) / duration_s, 0.0, 1.0)
    blend = 0.5 - 0.5 * np.cos(np.pi * u)
    velocity_cmd[segment] = start_velocity + (end_velocity - start_velocity) * blend
    phase_name[segment] = phase


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
    segment = _segment_slice(time, start_s, duration_s)
    if segment.start >= segment.stop:
        return
    velocity_cmd[segment] = velocity
    phase_name[segment] = phase


def build_reference_trajectory(config: ExcitationConfig, *, max_velocity: float) -> ReferenceTrajectory:
    sample_rate = max(float(config.sample_rate), 1.0)
    dt = 1.0 / sample_rate
    segment_specs: list[tuple[str, float, float, str]] = [
        ("hold_start", 0.0, float(config.hold_start), "hold"),
    ]
    for level_index, platform in enumerate(config.platforms, start=1):
        platform_speed = float(platform.resolve_speed(max_velocity=float(max_velocity)))
        segment_specs.extend(
            (
                (
                    f"settle_forward_{level_index:02d}",
                    platform_speed,
                    float(platform.settle_duration),
                    "blend",
                ),
                (
                    f"steady_forward_{level_index:02d}",
                    platform_speed,
                    float(platform.steady_duration),
                    "hold",
                ),
            )
        )
    segment_specs.append(("transition_mid_zero", 0.0, float(config.transition_duration), "blend"))
    for level_index, platform in enumerate(config.platforms, start=1):
        platform_speed = float(platform.resolve_speed(max_velocity=float(max_velocity)))
        segment_specs.extend(
            (
                (
                    f"settle_reverse_{level_index:02d}",
                    -platform_speed,
                    float(platform.settle_duration),
                    "blend",
                ),
                (
                    f"steady_reverse_{level_index:02d}",
                    -platform_speed,
                    float(platform.steady_duration),
                    "hold",
                ),
            )
        )
    segment_specs.append(("transition_end_zero", 0.0, float(config.transition_duration), "blend"))
    segment_specs.append(("hold_end", 0.0, float(config.hold_end), "hold"))

    total_duration = max(sum(duration for _, _, duration, _ in segment_specs), dt)
    sample_count = max(int(np.ceil(total_duration * sample_rate - 1.0e-9)), 2)

    time = np.arange(sample_count, dtype=np.float64) * dt
    velocity_cmd = np.zeros(sample_count, dtype=np.float64)
    phase_name = np.full(sample_count, "hold_end", dtype="<U32")

    segment_durations = np.asarray([duration for _, _, duration, _ in segment_specs], dtype=np.float64)
    segment_start_times = np.concatenate(([0.0], np.cumsum(segment_durations[:-1], dtype=np.float64)))

    current_velocity = 0.0
    for (segment_name, target_velocity, segment_duration, segment_mode), start_time in zip(
        segment_specs,
        segment_start_times,
    ):
        if segment_mode == "hold":
            _hold_segment(
                time=time,
                velocity_cmd=velocity_cmd,
                phase_name=phase_name,
                start_s=float(start_time),
                duration_s=segment_duration,
                velocity=float(target_velocity),
                phase=str(segment_name),
            )
        else:
            _blend_segment(
                time=time,
                velocity_cmd=velocity_cmd,
                phase_name=phase_name,
                start_s=float(start_time),
                duration_s=segment_duration,
                start_velocity=float(current_velocity),
                end_velocity=float(target_velocity),
                phase=str(segment_name),
            )
        current_velocity = float(target_velocity)

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
