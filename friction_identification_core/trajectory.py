from __future__ import annotations

import numpy as np

from friction_identification_core.config import ExcitationConfig
from friction_identification_core.models import ReferenceTrajectory


def build_reference_trajectory(config: ExcitationConfig) -> ReferenceTrajectory:
    sample_rate = max(float(config.sample_rate), 1.0)
    dt = 1.0 / sample_rate
    motion_duration = max(float(config.duration), dt)
    total_duration = float(config.hold_start) + motion_duration + float(config.hold_end)
    sample_count = max(int(np.ceil(total_duration * sample_rate)), 2)

    time = np.arange(sample_count, dtype=np.float64) * dt
    position_cmd = np.zeros(sample_count, dtype=np.float64)
    velocity_cmd = np.zeros(sample_count, dtype=np.float64)
    acceleration_cmd = np.zeros(sample_count, dtype=np.float64)
    phase_name = np.full(sample_count, "hold_start", dtype="<U32")

    frequencies = tuple(float(freq) for freq in config.frequency_bands)
    amplitude = float(config.amplitude)
    segment_specs: list[tuple[str, float, float]] = []
    for band_index, frequency in enumerate(frequencies, start=1):
        segment_specs.append((f"forward_band_{band_index:02d}", amplitude, 1.0 / max(frequency, 1.0e-6)))
        segment_specs.append((f"reverse_band_{band_index:02d}", -amplitude, 1.0 / max(frequency, 1.0e-6)))
    segment_specs.append(("settle_to_zero", 0.0, 1.0 / max(frequencies[-1], 1.0e-6)))

    weights = np.asarray([weight for _, _, weight in segment_specs], dtype=np.float64)
    durations = motion_duration * weights / np.sum(weights)

    current_time = float(config.hold_start)
    current_position = 0.0
    for (segment_name, target_position, _), segment_duration in zip(segment_specs, durations):
        segment_start = current_time
        segment_end = current_time + float(segment_duration)
        if segment_end <= segment_start:
            continue
        mask = (time >= segment_start) & (time < segment_end)
        local_t = time[mask] - segment_start
        u = np.clip(local_t / (segment_end - segment_start), 0.0, 1.0)
        blend = 0.5 - 0.5 * np.cos(np.pi * u)
        delta = float(target_position) - float(current_position)
        position_cmd[mask] = current_position + delta * blend
        velocity_cmd[mask] = delta * 0.5 * np.pi * np.sin(np.pi * u) / (segment_end - segment_start)
        acceleration_cmd[mask] = (
            delta
            * 0.5
            * (np.pi / (segment_end - segment_start)) ** 2
            * np.cos(np.pi * u)
        )
        phase_name[mask] = segment_name
        current_time = segment_end
        current_position = float(target_position)

    hold_end_mask = time >= (float(config.hold_start) + motion_duration)
    phase_name[hold_end_mask] = "hold_end"
    position_cmd[hold_end_mask] = 0.0
    velocity_cmd[hold_end_mask] = 0.0
    acceleration_cmd[hold_end_mask] = 0.0

    return ReferenceTrajectory(
        time=time,
        position_cmd=position_cmd,
        velocity_cmd=velocity_cmd,
        acceleration_cmd=acceleration_cmd,
        phase_name=phase_name,
        duration_s=total_duration,
    )
