from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from friction_identification_core.config import Config


@dataclass(frozen=True)
class ReferenceTrajectory:
    """Unified description of a joint-space reference trajectory."""

    time: np.ndarray
    q_cmd: np.ndarray
    qd_cmd: np.ndarray
    qdd_cmd: np.ndarray


@dataclass(frozen=True)
class JointExcitationPlan:
    centers: np.ndarray
    amplitudes: np.ndarray
    safe_lower: np.ndarray
    safe_upper: np.ndarray
    limited: np.ndarray


def resolve_active_joint_mask(
    num_joints: int,
    active_joints: np.ndarray | Sequence[bool] | None,
    *,
    require_any: bool = False,
) -> np.ndarray:
    if active_joints is None:
        mask = np.ones(int(num_joints), dtype=bool)
    else:
        mask = np.asarray(active_joints, dtype=bool).reshape(-1)
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
    return ReferenceTrajectory(time=t, q_cmd=q_cmd, qd_cmd=qd_cmd, qdd_cmd=qdd_cmd)


def build_joint_excitation_plan(
    *,
    home_qpos: np.ndarray,
    joint_limits: np.ndarray,
    limited: np.ndarray | None,
    amplitude_scale: float,
) -> JointExcitationPlan:
    home_qpos = np.asarray(home_qpos, dtype=np.float64).reshape(-1)
    lower, upper, limited = resolve_joint_limit_arrays(joint_limits, limited=limited)
    if home_qpos.size != lower.size:
        raise ValueError("home_qpos size must match the number of joint limits.")

    centers = home_qpos.copy()
    amplitudes = np.full(home_qpos.size, 0.08, dtype=np.float64)
    safe_lower = lower.copy()
    safe_upper = upper.copy()

    for joint_idx in range(home_qpos.size):
        center = home_qpos[joint_idx]
        if limited[joint_idx]:
            span = upper[joint_idx] - lower[joint_idx]
            margin = float(np.clip(0.10 * span, 0.04, 0.12))
            safe_lo = lower[joint_idx] + margin
            safe_hi = upper[joint_idx] - margin
            if safe_lo >= safe_hi:
                safe_lo = lower[joint_idx] + 0.20 * span
                safe_hi = upper[joint_idx] - 0.20 * span

            center = float(np.clip(center, safe_lo, safe_hi))
            max_excursion = max(0.0, min(center - safe_lo, safe_hi - center))
            desired_span = max(0.06, min(0.18, 0.22 * span))
            if max_excursion < desired_span:
                center = 0.5 * (safe_lo + safe_hi)
                max_excursion = max(0.0, min(center - safe_lo, safe_hi - center))

            max_excursion = max(max_excursion, 0.02)
            amplitude = min(amplitude_scale * span, 0.78 * max_excursion, 0.32 * span)
            amplitude = max(amplitude, min(0.03, 0.40 * max_excursion))
            safe_lower[joint_idx] = safe_lo
            safe_upper[joint_idx] = safe_hi
        else:
            max_excursion = 0.18
            amplitude = min(amplitude_scale * 0.5, max_excursion)
            amplitude = max(amplitude, 0.03)

        centers[joint_idx] = center
        amplitudes[joint_idx] = amplitude

    return JointExcitationPlan(
        centers=centers,
        amplitudes=amplitudes,
        safe_lower=safe_lower,
        safe_upper=safe_upper,
        limited=limited,
    )


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


def generate_segmented_excitation_trajectory(
    *,
    home_qpos: np.ndarray,
    joint_limits: np.ndarray,
    limited: np.ndarray | None,
    duration: float,
    sample_rate: float,
    base_frequency: float,
    amplitude_scale: float,
    active_joints: np.ndarray | Sequence[bool] | None = None,
) -> ReferenceTrajectory:
    num_samples = max(int(round(float(duration) * float(sample_rate))), 2)
    t = np.linspace(0.0, float(duration), num_samples, endpoint=False)
    home_qpos = np.asarray(home_qpos, dtype=np.float64).reshape(-1)
    lower, upper, limited = resolve_joint_limit_arrays(joint_limits, limited=limited)
    plan = build_joint_excitation_plan(
        home_qpos=home_qpos,
        joint_limits=np.column_stack((lower, upper)),
        limited=limited,
        amplitude_scale=amplitude_scale,
    )

    num_joints = home_qpos.size
    active_joint_mask = resolve_active_joint_mask(num_joints, active_joints, require_any=True)
    active_joint_indices = np.flatnonzero(active_joint_mask)
    baseline_q = build_excitation_start_pose(
        home_qpos=home_qpos,
        excitation_centers=plan.centers,
        active_joints=active_joint_mask,
    )
    q_cmd = np.broadcast_to(baseline_q, (num_samples, num_joints)).copy()
    qd_cmd = np.zeros_like(q_cmd)
    qdd_cmd = np.zeros_like(q_cmd)

    active_duration = max(float(duration), 1.0 / float(sample_rate))
    segment_edges = np.linspace(0.0, float(duration), active_joint_indices.size + 1, dtype=np.float64)
    base_cycles = max(3.0, float(base_frequency) * active_duration)

    for active_order, joint_idx in enumerate(active_joint_indices):
        seg_start = segment_edges[active_order]
        seg_end = segment_edges[active_order + 1]
        segment_mask = (
            (t >= seg_start)
            & (t < seg_end if active_order < active_joint_indices.size - 1 else t <= seg_end)
        )
        if not np.any(segment_mask):
            continue

        local_t = t[segment_mask] - seg_start
        segment_duration = max(seg_end - seg_start, 1e-6)
        normalized_t = local_t / segment_duration
        envelope = np.sin(np.pi * normalized_t) ** 2

        cycles = base_cycles * (1.0 + 0.05 * active_order)
        omega = 2.0 * np.pi * cycles / segment_duration
        # Stack harmonics and a chirp so each segment spans a wider speed range.
        base_pattern = (
            np.sin(omega * local_t)
            + 0.30 * np.sin(2.1 * omega * local_t + 0.35)
            + 0.18 * np.sin(3.5 * omega * local_t + 0.7)
            + 0.10 * np.sin(5.2 * omega * local_t + 1.1)
        )
        chirp_freq_start = 0.5 * omega / (2.0 * np.pi)
        chirp_freq_end = 2.0 * omega / (2.0 * np.pi)
        chirp_phase = 2.0 * np.pi * (
            chirp_freq_start * local_t
            + 0.5 * (chirp_freq_end - chirp_freq_start) * (local_t**2) / segment_duration
        )
        chirp_pattern = 0.15 * np.sin(chirp_phase)
        pattern = envelope * (base_pattern + chirp_pattern)

        amplitude = plan.amplitudes[joint_idx]
        if plan.limited[joint_idx]:
            max_excursion = min(
                plan.centers[joint_idx] - plan.safe_lower[joint_idx],
                plan.safe_upper[joint_idx] - plan.centers[joint_idx],
            )
            peak = float(np.max(np.abs(pattern)))
            if peak > 1e-9:
                amplitude = min(amplitude, 0.98 * max_excursion / peak)

        q_cmd[segment_mask, joint_idx] = plan.centers[joint_idx] + amplitude * pattern

    q_cmd[0] = baseline_q
    if np.any(plan.limited):
        for joint_idx in range(num_joints):
            if plan.limited[joint_idx]:
                np.clip(q_cmd[:, joint_idx], lower[joint_idx], upper[joint_idx], out=q_cmd[:, joint_idx])

    gradient_order = 2 if num_samples >= 3 else 1
    dt = 1.0 / float(sample_rate)
    qd_cmd[:] = np.gradient(q_cmd, dt, axis=0, edge_order=gradient_order)
    qdd_cmd[:] = np.gradient(qd_cmd, dt, axis=0, edge_order=gradient_order)
    return ReferenceTrajectory(time=t, q_cmd=q_cmd, qd_cmd=qd_cmd, qdd_cmd=qdd_cmd)


def sample_reference_trajectory(
    reference: ReferenceTrajectory,
    elapsed_s: float,
    *,
    wrap: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return (
        reference.q_cmd[sample_idx].copy(),
        reference.qd_cmd[sample_idx].copy(),
        reference.qdd_cmd[sample_idx].copy(),
    )


def build_target_joint_mask(config: Config) -> np.ndarray:
    return config.target_joint_mask


def build_excitation_trajectory(config: Config) -> ReferenceTrajectory:
    return generate_segmented_excitation_trajectory(
        home_qpos=config.robot.home_qpos,
        joint_limits=config.robot.joint_limits,
        limited=None,
        duration=config.identification.excitation.duration,
        sample_rate=config.sampling.rate,
        base_frequency=config.identification.excitation.base_frequency,
        amplitude_scale=config.identification.excitation.amplitude_scale,
        active_joints=config.target_joint_mask,
    )


def build_startup_pose(config: Config, excitation_reference: ReferenceTrajectory) -> np.ndarray:
    return build_excitation_start_pose(
        home_qpos=config.robot.home_qpos,
        excitation_centers=excitation_reference.q_cmd[0],
        active_joints=config.target_joint_mask,
    )


__all__ = [
    "ReferenceTrajectory",
    "build_excitation_start_pose",
    "build_quintic_point_to_point_trajectory",
    "build_target_joint_mask",
    "build_excitation_trajectory",
    "build_startup_pose",
    "generate_segmented_excitation_trajectory",
    "resolve_active_joint_mask",
    "resolve_joint_limit_arrays",
    "sample_reference_trajectory",
]
