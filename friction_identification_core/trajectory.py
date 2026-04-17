from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from friction_identification_core.config import Config


@dataclass(frozen=True)
class ReferenceTrajectory:
    """Joint-space reference trajectory for the full 7-axis run."""

    time: np.ndarray
    q_cmd: np.ndarray
    qd_cmd: np.ndarray
    qdd_cmd: np.ndarray
    phase_name: np.ndarray | None = None


@dataclass(frozen=True)
class ReferenceSample:
    q_cmd: np.ndarray
    qd_cmd: np.ndarray
    qdd_cmd: np.ndarray
    phase_name: str


def resolve_active_joint_mask(
    num_joints: int,
    active_joints: np.ndarray | Sequence[bool] | None,
    *,
    require_any: bool = False,
) -> np.ndarray:
    if active_joints is None:
        mask = np.ones(int(num_joints), dtype=bool)
    else:
        array = np.asarray(active_joints)
        if array.dtype == bool:
            mask = array.reshape(-1)
        else:
            mask = np.zeros(int(num_joints), dtype=bool)
            indices = np.asarray(array, dtype=np.int64).reshape(-1)
            mask[indices] = True
    if mask.size != int(num_joints):
        raise ValueError("active_joints size must match the number of joints.")
    if require_any and not np.any(mask):
        raise ValueError("At least one joint must be active.")
    return mask.copy()


def resolve_joint_limit_arrays(
    joint_limits: np.ndarray,
    *,
    limited: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_limits = np.asarray(joint_limits, dtype=np.float64)
    if joint_limits.ndim != 2 or joint_limits.shape[1] != 2:
        raise ValueError("joint_limits must have shape [num_joints, 2].")

    lower = np.minimum(joint_limits[:, 0], joint_limits[:, 1]).astype(np.float64, copy=True)
    upper = np.maximum(joint_limits[:, 0], joint_limits[:, 1]).astype(np.float64, copy=True)
    if limited is None:
        limited = np.all(np.isfinite(joint_limits), axis=1)
    else:
        limited = np.asarray(limited, dtype=bool).reshape(-1)
    if limited.size != joint_limits.shape[0]:
        raise ValueError("limited mask size must match the number of joints.")

    lower[~limited] = -np.inf
    upper[~limited] = np.inf
    return lower, upper, limited


def _phase_durations(
    duration: float,
    *,
    reversal_pause_s: float,
    zero_crossing_dither_s: float,
    speed_segment_count: int,
) -> dict[str, float]:
    desired = np.asarray([0.06, 0.54, 0.18, 0.16, 0.06], dtype=np.float64) * float(duration)
    minimum = np.asarray(
        [
            0.6,
            max(4.0, 0.28 * float(duration)),
            max(1.2, 4.0 * float(reversal_pause_s), 3.0 * float(zero_crossing_dither_s)),
            max(1.5, 1.5 * max(int(speed_segment_count), 1)),
            0.6,
        ],
        dtype=np.float64,
    )
    durations = np.maximum(desired, minimum)
    durations *= float(duration) / max(float(np.sum(durations)), 1e-9)
    names = ("center_hold", "full_range_sweep", "reversal_dither", "speed_sweep", "hold")
    return {name: float(value) for name, value in zip(names, durations)}


def _triangle_wave(phase: np.ndarray) -> np.ndarray:
    frac = np.mod(np.asarray(phase, dtype=np.float64), 1.0)
    return 1.0 - np.abs(2.0 * frac - 1.0)


def _normalize_harmonic_weights(harmonic_weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(harmonic_weights, dtype=np.float64).reshape(-1)
    if weights.size == 0:
        return np.ones(1, dtype=np.float64)
    peak = np.sum(np.abs(weights))
    if peak <= 1e-9:
        return np.ones_like(weights)
    return weights / peak


def _harmonic_dither(local_t: np.ndarray, base_frequency: float, harmonic_weights: np.ndarray) -> np.ndarray:
    weights = _normalize_harmonic_weights(harmonic_weights)
    dither = np.zeros_like(local_t, dtype=np.float64)
    for harmonic_idx, weight in enumerate(weights, start=1):
        dither += weight * np.sin(2.0 * np.pi * base_frequency * harmonic_idx * local_t)
    peak = np.max(np.abs(dither)) if dither.size else 0.0
    if peak > 1e-9:
        dither /= peak
    return dither


def _safe_joint_window(
    joint_limits: np.ndarray,
    *,
    safety_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower, upper, limited = resolve_joint_limit_arrays(joint_limits)
    lower_safe = lower.copy()
    upper_safe = upper.copy()
    if np.any(limited):
        lower_safe[limited] = lower[limited] + float(safety_margin)
        upper_safe[limited] = upper[limited] - float(safety_margin)
    invalid = limited & (lower_safe >= upper_safe)
    if np.any(invalid):
        raise ValueError("Safety margin leaves no valid joint window for one or more joints.")
    return lower_safe, upper_safe, limited


def build_quintic_point_to_point_trajectory(
    *,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    duration: float,
    sample_rate: float,
    settle_duration: float = 0.0,
) -> ReferenceTrajectory:
    start_q = np.asarray(start_q, dtype=np.float64).reshape(-1)
    goal_q = np.asarray(goal_q, dtype=np.float64).reshape(-1)
    if start_q.shape != goal_q.shape:
        raise ValueError("start_q and goal_q must share the same shape.")

    duration = max(float(duration), 1e-6)
    sample_rate = max(float(sample_rate), 1e-6)
    settle_duration = max(float(settle_duration), 0.0)
    total_duration = duration + settle_duration
    num_samples = max(int(round(total_duration * sample_rate)), 2)
    t = np.linspace(0.0, total_duration, num_samples, endpoint=False)

    delta = goal_q - start_q
    q_cmd = np.empty((num_samples, start_q.size), dtype=np.float64)
    qd_cmd = np.zeros_like(q_cmd)
    qdd_cmd = np.zeros_like(q_cmd)

    move_mask = t < duration
    if np.any(move_mask):
        tau = np.clip(t[move_mask] / duration, 0.0, 1.0)
        tau2 = tau * tau
        tau3 = tau2 * tau
        tau4 = tau3 * tau
        tau5 = tau4 * tau

        blend = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
        blend_d = (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) / duration
        blend_dd = (60.0 * tau - 180.0 * tau2 + 120.0 * tau3) / (duration * duration)

        q_cmd[move_mask] = start_q + blend[:, None] * delta
        qd_cmd[move_mask] = blend_d[:, None] * delta
        qdd_cmd[move_mask] = blend_dd[:, None] * delta

    hold_mask = ~move_mask
    if np.any(hold_mask):
        q_cmd[hold_mask] = goal_q

    q_cmd[-1] = goal_q
    qd_cmd[-1] = 0.0
    qdd_cmd[-1] = 0.0
    phase_name = np.full(num_samples, "transition", dtype="<U32")
    if settle_duration > 0.0:
        phase_name[move_mask] = "startup_move"
        phase_name[hold_mask] = "startup_settle"
    return ReferenceTrajectory(time=t, q_cmd=q_cmd, qd_cmd=qd_cmd, qdd_cmd=qdd_cmd, phase_name=phase_name)


def build_excitation_start_pose(
    *,
    home_qpos: np.ndarray,
    excitation_centers: np.ndarray,
    active_joints: np.ndarray | Sequence[bool] | None = None,
) -> np.ndarray:
    home_qpos = np.asarray(home_qpos, dtype=np.float64).reshape(-1)
    excitation_centers = np.asarray(excitation_centers, dtype=np.float64).reshape(-1)
    if home_qpos.shape != excitation_centers.shape:
        raise ValueError("home_qpos and excitation_centers must share the same shape.")

    active_joint_mask = resolve_active_joint_mask(home_qpos.size, active_joints)
    start_q = home_qpos.copy()
    start_q[active_joint_mask] = excitation_centers[active_joint_mask]
    return start_q


def generate_parallel_full_range_excitation(
    *,
    home_qpos: np.ndarray,
    joint_limits: np.ndarray,
    safety_margin: float,
    duration: float,
    sample_rate: float,
    sweep_cycles: int,
    speed_schedule: np.ndarray,
    phase_offsets: np.ndarray,
    harmonic_weights: np.ndarray,
    reversal_pause_s: float,
    zero_crossing_dither_s: float,
    active_joints: np.ndarray | Sequence[bool] | None = None,
) -> ReferenceTrajectory:
    duration = max(float(duration), 1e-3)
    sample_rate = max(float(sample_rate), 1e-3)
    num_samples = max(int(round(duration * sample_rate)), 2)
    t = np.linspace(0.0, duration, num_samples, endpoint=False)

    home_qpos = np.asarray(home_qpos, dtype=np.float64).reshape(-1)
    phase_offsets = np.asarray(phase_offsets, dtype=np.float64).reshape(-1)
    if phase_offsets.size != home_qpos.size:
        raise ValueError("phase_offsets size must match the number of joints.")

    active_joint_mask = resolve_active_joint_mask(home_qpos.size, active_joints, require_any=True)
    lower_safe, upper_safe, limited = _safe_joint_window(joint_limits, safety_margin=safety_margin)
    centers = 0.5 * (lower_safe + upper_safe)
    span = np.maximum(upper_safe - lower_safe, 1e-6)
    start_q = build_excitation_start_pose(
        home_qpos=home_qpos,
        excitation_centers=centers,
        active_joints=active_joint_mask,
    )
    q_cmd = np.broadcast_to(start_q, (num_samples, home_qpos.size)).copy()
    phase_name = np.full(num_samples, "hold", dtype="<U32")

    durations = _phase_durations(
        duration,
        reversal_pause_s=reversal_pause_s,
        zero_crossing_dither_s=zero_crossing_dither_s,
        speed_segment_count=int(np.asarray(speed_schedule).size),
    )
    phase_edges = np.cumsum([0.0, *durations.values()])
    phase_edges[-1] = duration

    align_mask = (t >= phase_edges[0]) & (t < phase_edges[1])
    sweep_mask = (t >= phase_edges[1]) & (t < phase_edges[2])
    reversal_mask = (t >= phase_edges[2]) & (t < phase_edges[3])
    speed_mask = (t >= phase_edges[3]) & (t < phase_edges[4])
    hold_mask = t >= phase_edges[4]
    phase_name[align_mask] = "center_hold"
    phase_name[sweep_mask] = "full_range_sweep"
    phase_name[reversal_mask] = "reversal_dither"
    phase_name[speed_mask] = "speed_sweep"
    phase_name[hold_mask] = "hold"

    speed_schedule = np.asarray(speed_schedule, dtype=np.float64).reshape(-1)
    harmonic_weights = np.asarray(harmonic_weights, dtype=np.float64).reshape(-1)
    align_ratio = np.full(np.count_nonzero(align_mask), 0.5, dtype=np.float64)

    sweep_local_t = t[sweep_mask] - phase_edges[1]
    sweep_duration = max(durations["full_range_sweep"], 1e-6)
    sweep_u = sweep_local_t / sweep_duration

    reversal_local_t = t[reversal_mask] - phase_edges[2]
    reversal_duration = max(durations["reversal_dither"], 1e-6)
    reversal_u = reversal_local_t / reversal_duration
    reversal_cycles = max(
        int(round(reversal_duration / max(float(zero_crossing_dither_s), 1e-3))),
        max(int(sweep_cycles), 2),
    )

    speed_local_t = t[speed_mask] - phase_edges[3]
    speed_duration = max(durations["speed_sweep"], 1e-6)

    for joint_idx in np.flatnonzero(active_joint_mask):
        joint_phase = float(np.mod(phase_offsets[joint_idx], 1.0))
        joint_center = centers[joint_idx]
        joint_span = span[joint_idx]
        joint_lower = lower_safe[joint_idx]
        joint_upper = upper_safe[joint_idx]

        if align_ratio.size > 0:
            q_cmd[align_mask, joint_idx] = joint_center

        if sweep_u.size > 0:
            sweep_ratio = _triangle_wave(sweep_cycles * sweep_u + joint_phase)
            sweep_dither = 0.04 * _harmonic_dither(
                sweep_local_t,
                base_frequency=max(float(sweep_cycles), 1.0) / sweep_duration,
                harmonic_weights=harmonic_weights,
            )
            q_cmd[sweep_mask, joint_idx] = joint_lower + joint_span * np.clip(
                sweep_ratio + sweep_dither,
                0.0,
                1.0,
            )

        if reversal_u.size > 0:
            endpoint = (_triangle_wave(0.5 * reversal_cycles * reversal_u + joint_phase) >= 0.5).astype(np.float64)
            reversal_wave = np.sin(2.0 * np.pi * reversal_cycles * reversal_u + 2.0 * np.pi * joint_phase)
            reversal_ratio = np.clip(endpoint + 0.08 * reversal_wave, 0.0, 1.0)
            q_cmd[reversal_mask, joint_idx] = joint_lower + joint_span * reversal_ratio

        if speed_local_t.size > 0:
            start_idx = 0
            segment_edges = np.linspace(0.0, speed_duration, speed_schedule.size + 1)
            for segment_idx, speed_scale in enumerate(speed_schedule):
                end_idx = int(
                    np.searchsorted(speed_local_t, segment_edges[segment_idx + 1], side="left")
                    if segment_idx + 1 < segment_edges.size - 1
                    else speed_local_t.size
                )
                if end_idx <= start_idx:
                    continue
                segment_t = speed_local_t[start_idx:end_idx] - segment_edges[segment_idx]
                segment_duration = max(segment_edges[segment_idx + 1] - segment_edges[segment_idx], 1e-6)
                cycles = max(0.35, 0.55 + 1.8 * float(speed_scale))
                segment_ratio = 0.5 + 0.44 * np.sin(
                    2.0 * np.pi * cycles * segment_t / segment_duration + 2.0 * np.pi * joint_phase
                )
                segment_ratio += 0.05 * _harmonic_dither(
                    segment_t,
                    base_frequency=max(cycles, 0.2) / segment_duration,
                    harmonic_weights=harmonic_weights,
                )
                q_cmd[np.flatnonzero(speed_mask)[start_idx:end_idx], joint_idx] = joint_lower + joint_span * np.clip(
                    segment_ratio,
                    0.0,
                    1.0,
                )
                start_idx = end_idx

        if np.any(hold_mask):
            q_cmd[hold_mask, joint_idx] = joint_center

        if limited[joint_idx]:
            np.clip(q_cmd[:, joint_idx], joint_lower, joint_upper, out=q_cmd[:, joint_idx])

    gradient_order = 2 if num_samples >= 3 else 1
    dt = 1.0 / sample_rate
    qd_cmd = np.gradient(q_cmd, dt, axis=0, edge_order=gradient_order)
    qdd_cmd = np.gradient(qd_cmd, dt, axis=0, edge_order=gradient_order)
    return ReferenceTrajectory(
        time=t,
        q_cmd=q_cmd,
        qd_cmd=qd_cmd,
        qdd_cmd=qdd_cmd,
        phase_name=phase_name,
    )


def sample_reference_trajectory(
    reference: ReferenceTrajectory,
    elapsed_s: float,
    *,
    wrap: bool,
) -> ReferenceSample:
    if reference.time.size == 0:
        raise ValueError("reference trajectory must contain at least one sample.")

    if reference.time.size >= 2:
        dt = float(reference.time[1] - reference.time[0])
    else:
        dt = 0.0
    reference_duration = float(reference.time[-1] + max(dt, 0.0))

    reference_time = float(elapsed_s)
    if wrap and reference_duration > 1e-12:
        reference_time = reference_time % reference_duration
    elif reference_duration > 1e-12:
        reference_time = min(reference_time, np.nextafter(reference_duration, 0.0))
    else:
        reference_time = 0.0

    sample_idx = int(np.searchsorted(reference.time, reference_time, side="right") - 1)
    sample_idx = int(np.clip(sample_idx, 0, reference.time.shape[0] - 1))
    phase_name = ""
    if reference.phase_name is not None and sample_idx < reference.phase_name.shape[0]:
        phase_name = str(reference.phase_name[sample_idx])
    return ReferenceSample(
        q_cmd=reference.q_cmd[sample_idx].copy(),
        qd_cmd=reference.qd_cmd[sample_idx].copy(),
        qdd_cmd=reference.qdd_cmd[sample_idx].copy(),
        phase_name=phase_name,
    )


def build_excitation_trajectory(config: Config) -> ReferenceTrajectory:
    return generate_parallel_full_range_excitation(
        home_qpos=config.robot.home_qpos,
        joint_limits=config.robot.joint_limits,
        safety_margin=config.safety.joint_limit_margin,
        duration=config.identification.excitation.duration,
        sample_rate=config.sampling.rate,
        sweep_cycles=config.identification.excitation.sweep_cycles,
        speed_schedule=config.identification.excitation.speed_schedule,
        phase_offsets=config.identification.excitation.phase_offsets,
        harmonic_weights=config.identification.excitation.harmonic_weights,
        reversal_pause_s=config.identification.excitation.reversal_pause_s,
        zero_crossing_dither_s=config.identification.excitation.zero_crossing_dither_s,
        active_joints=config.active_joint_mask,
    )


def build_startup_pose(config: Config, excitation_reference: ReferenceTrajectory) -> np.ndarray:
    return build_excitation_start_pose(
        home_qpos=config.robot.home_qpos,
        excitation_centers=excitation_reference.q_cmd[0],
        active_joints=config.active_joint_mask,
    )


__all__ = [
    "ReferenceSample",
    "ReferenceTrajectory",
    "build_excitation_start_pose",
    "build_excitation_trajectory",
    "build_quintic_point_to_point_trajectory",
    "build_startup_pose",
    "generate_parallel_full_range_excitation",
    "resolve_active_joint_mask",
    "resolve_joint_limit_arrays",
    "sample_reference_trajectory",
]
