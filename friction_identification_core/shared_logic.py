from __future__ import annotations

"""仿真与真机共用的激励、限位和摩擦计算核心逻辑。"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .models import JointFrictionParameters


@dataclass(frozen=True)
class ReferenceTrajectory:
    """统一描述一条可被仿真或真机复用的关节参考轨迹。"""

    time: np.ndarray
    q_cmd: np.ndarray
    qd_cmd: np.ndarray
    qdd_cmd: np.ndarray


@dataclass(frozen=True)
class JointExcitationPlan:
    """记录逐关节激励时使用的中心位与安全幅值。"""

    centers: np.ndarray
    amplitudes: np.ndarray
    safe_lower: np.ndarray
    safe_upper: np.ndarray
    limited: np.ndarray


def resolve_joint_limit_arrays(
    joint_limits: np.ndarray,
    *,
    limited: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """标准化关节上下限数组，并保留哪些关节真的启用了限位。"""

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


def shrink_joint_limit_window(
    lower: np.ndarray,
    upper: np.ndarray,
    limited: np.ndarray,
    *,
    margin: float,
    keep_midpoint_inside: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """在真实限位内再缩出一层工作窗口，用于保守激励或筛样。"""

    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)
    limited = np.asarray(limited, dtype=bool).reshape(-1)
    if lower.size != upper.size or lower.size != limited.size:
        raise ValueError("lower, upper, and limited must share the same length.")

    shrunk_lower = lower.copy()
    shrunk_upper = upper.copy()
    if not np.any(limited):
        return shrunk_lower, shrunk_upper

    margin = max(float(margin), 0.0)
    shrunk_lower[limited] = lower[limited] + margin
    shrunk_upper[limited] = upper[limited] - margin

    if keep_midpoint_inside:
        midpoint = 0.5 * (lower + upper)
        shrunk_lower[limited] = np.minimum(shrunk_lower[limited], midpoint[limited])
        shrunk_upper[limited] = np.maximum(shrunk_upper[limited], midpoint[limited])

    return shrunk_lower, shrunk_upper


def build_joint_excitation_plan(
    *,
    home_qpos: np.ndarray,
    joint_limits: np.ndarray,
    limited: np.ndarray | None,
    amplitude_scale: float,
) -> JointExcitationPlan:
    """根据 home 位和限位信息，为每个关节生成统一的激励计划。"""

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

            # 当 home 位太靠边时，把激励中心移回中间，保证正反方向都有有效激励。
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


def generate_segmented_excitation_trajectory(
    *,
    home_qpos: np.ndarray,
    joint_limits: np.ndarray,
    limited: np.ndarray | None,
    duration: float,
    sample_rate: float,
    base_frequency: float,
    amplitude_scale: float,
) -> ReferenceTrajectory:
    """生成逐关节分段激励轨迹，供仿真与真机共享。"""

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
    initial_q = np.zeros(num_joints, dtype=np.float64)
    q_cmd = np.broadcast_to(plan.centers, (num_samples, num_joints)).copy()
    qd_cmd = np.zeros_like(q_cmd)
    qdd_cmd = np.zeros_like(q_cmd)

    transition_duration = min(1.0, max(0.0, 0.08 * float(duration)))
    active_duration = max(float(duration) - transition_duration, 1.0 / float(sample_rate))
    segment_edges = np.linspace(transition_duration, float(duration), num_joints + 1, dtype=np.float64)
    base_cycles = max(3.0, float(base_frequency) * active_duration)

    for joint_idx in range(num_joints):
        seg_start = segment_edges[joint_idx]
        seg_end = segment_edges[joint_idx + 1]
        segment_mask = (t >= seg_start) & (t < seg_end if joint_idx < num_joints - 1 else t <= seg_end)
        if not np.any(segment_mask):
            continue

        local_t = t[segment_mask] - seg_start
        segment_duration = max(seg_end - seg_start, 1e-6)
        normalized_t = local_t / segment_duration
        envelope = np.sin(np.pi * normalized_t) ** 2

        cycles = base_cycles * (1.0 + 0.05 * joint_idx)
        omega = 2.0 * np.pi * cycles / segment_duration
        harmonic_ratio = 2.1
        phase_shift = 0.35 * joint_idx
        pattern = envelope * (
            np.sin(omega * local_t)
            + 0.28 * np.sin(harmonic_ratio * omega * local_t + phase_shift)
        )

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

    transition_mask = t < transition_duration
    if np.any(transition_mask):
        normalized = np.clip(t[transition_mask] / max(transition_duration, 1e-6), 0.0, 1.0)
        blend = normalized * normalized * (3.0 - 2.0 * normalized)
        q_cmd[transition_mask] = initial_q + blend[:, None] * (plan.centers - initial_q)
    q_cmd[0] = initial_q

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
    """按当前时间从共享参考轨迹中取一个离散采样点。"""

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


def build_position_window_mask(
    q: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """逐样本判断关节位置是否全部落在给定工作窗口内。"""

    q = np.asarray(q, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64).reshape(1, -1)
    upper = np.asarray(upper, dtype=np.float64).reshape(1, -1)
    return np.all((q >= lower) & (q <= upper), axis=1)


def shape_limit_aware_torque_command(
    *,
    q: np.ndarray,
    torque_command: np.ndarray,
    torque_limits: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    soft_margin: float,
    recenter_torque_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """靠近真实限位时衰减朝外力矩，超窗时改写为回中力矩。"""

    q = np.asarray(q, dtype=np.float64).reshape(-1)
    torque_command = np.asarray(torque_command, dtype=np.float64).reshape(-1)
    torque_limits = np.asarray(torque_limits, dtype=np.float64).reshape(-1)
    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)
    if not (q.size == torque_command.size == torque_limits.size == lower.size == upper.size):
        raise ValueError("All torque-shaping vectors must share the same length.")

    filtered = torque_command.copy()
    scale = np.ones_like(filtered)
    soft_margin = max(float(soft_margin), 1e-9)
    center = 0.5 * (lower + upper)

    positive_mask = filtered > 0.0
    negative_mask = filtered < 0.0
    if np.any(positive_mask):
        available_upper = upper[positive_mask] - q[positive_mask]
        scale[positive_mask] = np.clip(available_upper / soft_margin, 0.0, 1.0)
    if np.any(negative_mask):
        available_lower = q[negative_mask] - lower[negative_mask]
        scale[negative_mask] = np.clip(available_lower / soft_margin, 0.0, 1.0)
    filtered *= scale

    outside_window = (q < lower) | (q > upper)
    if np.any(outside_window):
        recenter = np.clip((center - q) / soft_margin, -1.0, 1.0)
        recenter *= torque_limits * float(recenter_torque_ratio)
        filtered[outside_window] = recenter[outside_window]

    blocked_mask = np.abs(filtered - torque_command) > 1e-12
    return filtered, blocked_mask, scale


def find_joint_limit_violation(
    *,
    q: np.ndarray,
    joint_names: Sequence[str],
    joint_limits: np.ndarray,
    margin: float,
) -> str | None:
    """检查是否越过硬安全限位，若越界则返回可直接打印的错误信息。"""

    q = np.asarray(q, dtype=np.float64).reshape(-1)
    lower, upper, limited = resolve_joint_limit_arrays(joint_limits)
    if q.size != lower.size or len(joint_names) != lower.size:
        raise ValueError("q, joint_names, and joint_limits must describe the same joints.")

    margin = max(float(margin), 0.0)
    lower = lower - margin
    upper = upper + margin
    violation_mask = limited & ((q < lower) | (q > upper))
    if not np.any(violation_mask):
        return None

    joint_idx = int(np.flatnonzero(violation_mask)[0])
    return (
        f"检测到关节超限，已触发安全停机: {joint_names[joint_idx]}="
        f"{q[joint_idx]:.6f} rad, 安全限位=[{lower[joint_idx]:.6f}, {upper[joint_idx]:.6f}]"
    )


def build_simulation_clean_sample_mask(
    *,
    q: np.ndarray,
    qd: np.ndarray,
    tau_constraint: np.ndarray,
    tau_friction: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    limited: np.ndarray,
    limit_margin: float,
    constraint_tolerance: float,
    min_motion_speed: float = 0.02,
) -> np.ndarray:
    """用统一规则筛掉仿真中的限位、约束和静止污染样本。"""

    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qd, dtype=np.float64)
    tau_constraint = np.asarray(tau_constraint, dtype=np.float64)
    tau_friction = np.asarray(tau_friction, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)
    limited = np.asarray(limited, dtype=bool).reshape(-1)

    if np.any(limited):
        margin_to_limits = np.minimum(q - lower[None, :], upper[None, :] - q)
        margin_to_limits[:, ~limited] = np.inf
        away_from_limits = np.all(margin_to_limits > float(limit_margin), axis=1)
    else:
        away_from_limits = np.ones(q.shape[0], dtype=bool)

    constraint_is_clean = np.all(np.abs(tau_constraint) < float(constraint_tolerance), axis=1)
    finite = (
        np.all(np.isfinite(q), axis=1)
        & np.all(np.isfinite(qd), axis=1)
        & np.all(np.isfinite(tau_friction), axis=1)
    )
    moving = np.any(np.abs(qd) > float(min_motion_speed), axis=1)
    return away_from_limits & constraint_is_clean & finite & moving


def build_residual_clean_sample_mask(
    *,
    q: np.ndarray,
    tau_residual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    torque_limits: np.ndarray,
    torque_limit_scale: float,
) -> np.ndarray:
    """筛出真机残差力矩中有限、在工作窗内且幅值合理的样本。"""

    q = np.asarray(q, dtype=np.float64)
    tau_residual = np.asarray(tau_residual, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)
    torque_limits = np.asarray(torque_limits, dtype=np.float64).reshape(-1)

    within_window = build_position_window_mask(q, lower=lower, upper=upper)
    finite = np.all(np.isfinite(q), axis=1) & np.all(np.isfinite(tau_residual), axis=1)
    residual_reasonable = np.all(np.abs(tau_residual) <= (torque_limits * float(torque_limit_scale)), axis=1)
    return finite & within_window & residual_reasonable


def predict_friction_compensation_torque(
    velocity: np.ndarray,
    parameters: Sequence[JointFrictionParameters],
    *,
    torque_limits: np.ndarray,
) -> np.ndarray:
    """根据辨识结果计算摩擦补偿力矩，并统一进行安全裁剪。"""

    velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
    torque_limits = np.asarray(torque_limits, dtype=np.float64).reshape(-1)
    if velocity.size != len(parameters) or torque_limits.size != len(parameters):
        raise ValueError("velocity, torque_limits, and parameters must describe the same joints.")

    torque = np.zeros_like(velocity)
    for idx, param in enumerate(parameters):
        scale = max(float(param.velocity_scale), 1e-6)
        torque[idx] = param.coulomb * np.tanh(velocity[idx] / scale) + param.viscous * velocity[idx] + param.offset
    return np.clip(torque, -torque_limits, torque_limits)
