from __future__ import annotations

import numpy as np

from friction_identification_core.config import ExcitationConfig
from friction_identification_core.models import ReferenceTrajectory


def _schroeder_phases(harmonic_count: int) -> np.ndarray:
    indices = np.arange(harmonic_count, dtype=np.float64)
    return -np.pi * indices * (indices - 1.0) / max(float(harmonic_count), 1.0)


def _excitation_envelope(
    time: np.ndarray,
    *,
    fade_in_duration: float,
    steady_duration: float,
    fade_out_duration: float,
) -> np.ndarray:
    envelope = np.ones_like(time, dtype=np.float64)
    if fade_in_duration > 0.0:
        fade_in_mask = time < fade_in_duration
        u = np.clip(time[fade_in_mask] / fade_in_duration, 0.0, 1.0)
        envelope[fade_in_mask] = 0.5 - 0.5 * np.cos(np.pi * u)
    if fade_out_duration > 0.0:
        fade_out_start = fade_in_duration + steady_duration
        fade_out_mask = time >= fade_out_start
        u = np.clip((time[fade_out_mask] - fade_out_start) / fade_out_duration, 0.0, 1.0)
        envelope[fade_out_mask] = 0.5 + 0.5 * np.cos(np.pi * u)
    return np.clip(envelope, 0.0, 1.0)


def build_reference_trajectory(config: ExcitationConfig, *, max_velocity: float) -> ReferenceTrajectory:
    sample_rate = max(float(config.sample_rate), 1.0)
    dt = 1.0 / sample_rate
    cycle_duration = 1.0 / float(config.base_frequency)
    fade_in_duration = float(config.fade_in_cycles) * cycle_duration
    steady_duration = float(config.steady_cycles) * cycle_duration
    fade_out_duration = float(config.fade_out_cycles) * cycle_duration
    excitation_duration = fade_in_duration + steady_duration + fade_out_duration
    total_duration = float(config.hold_start) + excitation_duration + float(config.hold_end)
    sample_count = max(int(np.ceil(total_duration * sample_rate - 1.0e-9)), 2)

    time = np.arange(sample_count, dtype=np.float64) * dt
    position_cmd = np.zeros(sample_count, dtype=np.float64)
    velocity_cmd = np.zeros(sample_count, dtype=np.float64)
    acceleration_cmd = np.zeros(sample_count, dtype=np.float64)
    phase_name = np.full(sample_count, "hold_end", dtype="<U32")

    hold_start_end = float(config.hold_start)
    excitation_end = hold_start_end + excitation_duration
    hold_start_mask = time < hold_start_end
    hold_end_mask = time >= excitation_end
    excitation_mask = (~hold_start_mask) & (~hold_end_mask)

    phase_name[hold_start_mask] = "hold_start"
    phase_name[hold_end_mask] = "hold_end"

    if np.any(excitation_mask):
        excitation_time = time[excitation_mask] - hold_start_end
        envelope = _excitation_envelope(
            excitation_time,
            fade_in_duration=fade_in_duration,
            steady_duration=steady_duration,
            fade_out_duration=fade_out_duration,
        )
        phases = _schroeder_phases(len(config.harmonic_multipliers))
        q_raw = np.zeros(excitation_time.size, dtype=np.float64)
        for multiplier, weight, phase in zip(config.harmonic_multipliers, config.harmonic_weights, phases):
            omega = 2.0 * np.pi * float(multiplier) * float(config.base_frequency)
            q_raw += float(weight) * np.sin(omega * excitation_time + float(phase))
        q_unit = envelope * q_raw
        if np.any(envelope > 0.0):
            q_unit -= float(np.mean(q_unit[envelope > 0.0])) * envelope
        v_unit = np.gradient(q_unit, dt)
        a_unit = np.gradient(v_unit, dt)

        max_abs_position = max(float(np.max(np.abs(q_unit))), 1.0e-9)
        max_abs_velocity = max(float(np.max(np.abs(v_unit))), 1.0e-9)
        scale = min(
            float(config.position_limit) / max_abs_position,
            float(config.velocity_utilization) * float(max_velocity) / max_abs_velocity,
        )

        position_cmd[excitation_mask] = scale * q_unit
        velocity_cmd[excitation_mask] = scale * v_unit
        acceleration_cmd[excitation_mask] = scale * a_unit

        fade_in_end = fade_in_duration
        steady_end = fade_in_duration + steady_duration
        fade_out_end = fade_in_duration + steady_duration + fade_out_duration
        for local_index, t_exc in zip(np.flatnonzero(excitation_mask), excitation_time):
            if t_exc < fade_in_end:
                phase_name[local_index] = "fade_in"
                continue
            if t_exc < steady_end:
                cycle_index = int(np.floor((t_exc - fade_in_duration) / cycle_duration)) + 1
                cycle_index = min(max(cycle_index, 1), int(config.steady_cycles))
                phase_name[local_index] = f"excitation_cycle_{cycle_index:02d}"
                continue
            if t_exc < fade_out_end:
                phase_name[local_index] = "fade_out"

    position_max = float(np.max(np.abs(position_cmd)))
    velocity_max = float(np.max(np.abs(velocity_cmd)))
    velocity_limit = float(config.velocity_utilization) * float(max_velocity)
    if position_max > float(config.position_limit) + 1.0e-9:
        raise ValueError("Reference trajectory exceeds excitation.position_limit.")
    if velocity_max > velocity_limit + 1.0e-9:
        raise ValueError("Reference trajectory exceeds excitation.velocity_utilization * control.max_velocity.")

    return ReferenceTrajectory(
        time=time,
        position_cmd=position_cmd,
        velocity_cmd=velocity_cmd,
        acceleration_cmd=acceleration_cmd,
        phase_name=phase_name,
        duration_s=float(total_duration),
    )
