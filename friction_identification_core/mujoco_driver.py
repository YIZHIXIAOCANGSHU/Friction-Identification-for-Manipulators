from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np

from .am_d02_scene import build_am_d02_model
from .models import FrictionSampleBatch


@dataclass
class MujocoFrictionCollectionConfig:
    duration: float = 30.0
    sample_rate: float = 400.0
    timestep: float = 0.0005
    base_frequency: float = 0.12
    amplitude_scale: float = 0.18
    render: bool = False
    realtime: bool = False
    feedback_scale: float = 0.12


class MujocoFrictionCollector:
    def __init__(
        self,
        model_path: str,
        joint_names: list[str],
        actuator_names: Optional[list[str]] = None,
        *,
        timestep: float = 0.0005,
        render: bool = False,
        home_qpos: Optional[np.ndarray] = None,
        end_effector_body: Optional[str] = None,
        tcp_offset: Optional[np.ndarray] = None,
        joint_limit_overrides: Optional[np.ndarray] = None,
        friction_loss: Optional[np.ndarray] = None,
        damping: Optional[np.ndarray] = None,
        inverse_friction_loss: Optional[np.ndarray] = None,
        inverse_damping: Optional[np.ndarray] = None,
    ):
        self.model = self._load_model(model_path=model_path, tcp_offset=tcp_offset)
        self.inverse_model = self._load_model(model_path=model_path, tcp_offset=tcp_offset)
        self.data = mujoco.MjData(self.model)
        self.inverse_data = mujoco.MjData(self.inverse_model)
        self.model.opt.timestep = timestep
        self.inverse_model.opt.timestep = timestep
        self.render = render
        self.viewer = None
        self.joint_names = list(joint_names)
        self.actuator_names = list(actuator_names or [])
        self.home_qpos = None if home_qpos is None else np.asarray(home_qpos, dtype=np.float64).copy()
        self.joint_ids = []
        self.dof_addrs = []
        self.qpos_addrs = []
        self.actuator_ids = []

        for name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"找不到关节: {name}")
            self.joint_ids.append(joint_id)
            self.dof_addrs.append(self.model.jnt_dofadr[joint_id])
            self.qpos_addrs.append(self.model.jnt_qposadr[joint_id])

        for name in self.actuator_names:
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                raise ValueError(f"找不到执行器: {name}")
            self.actuator_ids.append(actuator_id)

        self.joint_ids = np.asarray(self.joint_ids, dtype=np.int32)
        self.dof_addrs = np.asarray(self.dof_addrs, dtype=np.int32)
        self.qpos_addrs = np.asarray(self.qpos_addrs, dtype=np.int32)
        self.actuator_ids = np.asarray(self.actuator_ids, dtype=np.int32)
        self.ee_body_id = -1
        if end_effector_body is not None:
            self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, end_effector_body)
            if self.ee_body_id < 0:
                raise ValueError(f"找不到末端 body: {end_effector_body}")

        if joint_limit_overrides is not None:
            self._apply_joint_limit_overrides(joint_limit_overrides)
        self._apply_friction_overrides(friction_loss=friction_loss, damping=damping)
        self._apply_inverse_friction_overrides(
            friction_loss=friction_loss if inverse_friction_loss is None else inverse_friction_loss,
            damping=damping if inverse_damping is None else inverse_damping,
        )

        # Friction identification only needs dynamics from the articulated model.
        # Disable contact collisions to avoid large artificial constraint torques.
        self.model.geom_contype[:] = 0
        self.model.geom_conaffinity[:] = 0
        self.inverse_model.geom_contype[:] = 0
        self.inverse_model.geom_conaffinity[:] = 0

        self.kp = np.array([115.0, 100.0, 35.0, 40.0, 22.0, 20.0, 20.0], dtype=np.float64)
        self.kd = np.array([5.0, 5.0, 2.2, 2.0, 1.0, 1.0, 1.0], dtype=np.float64)

    @staticmethod
    def _load_model(model_path: str, tcp_offset: Optional[np.ndarray]) -> mujoco.MjModel:
        if model_path.lower().endswith(".urdf"):
            if tcp_offset is None:
                raise ValueError("tcp_offset is required when loading a URDF model.")
            return build_am_d02_model(model_path, np.asarray(tcp_offset, dtype=np.float64))
        return mujoco.MjModel.from_xml_path(model_path)

    @staticmethod
    def _print_status(message: str) -> None:
        print(f"[INFO] {message}", flush=True)

    @staticmethod
    def _maybe_report_progress(
        *,
        label: str,
        completed: int,
        total: int,
        last_reported_percent: int,
        percent_step: int = 10,
        extra: str = "",
    ) -> int:
        if total <= 0:
            return last_reported_percent

        if completed >= total:
            percent = 100
            if last_reported_percent >= 100:
                return last_reported_percent
        else:
            percent = int(100.0 * completed / total)
            next_threshold = max(percent_step, last_reported_percent + percent_step)
            if percent < next_threshold:
                return last_reported_percent

        suffix = f", {extra}" if extra else ""
        MujocoFrictionCollector._print_status(
            f"{label}: {percent:3d}% ({completed}/{total}{suffix})"
        )
        return percent

    def _apply_joint_limit_overrides(self, joint_limit_overrides: np.ndarray) -> None:
        joint_limit_overrides = np.asarray(joint_limit_overrides, dtype=np.float64)
        if joint_limit_overrides.shape != (len(self.joint_ids), 2):
            raise ValueError("joint_limit_overrides must have shape [num_joints, 2].")
        for joint_id, limits in zip(self.joint_ids, joint_limit_overrides):
            self.model.jnt_limited[joint_id] = 1
            self.model.jnt_range[joint_id] = limits
            self.inverse_model.jnt_limited[joint_id] = 1
            self.inverse_model.jnt_range[joint_id] = limits

    def _apply_friction_overrides(
        self,
        *,
        friction_loss: Optional[np.ndarray],
        damping: Optional[np.ndarray],
    ) -> None:
        self._apply_friction_overrides_to_model(
            self.model,
            friction_loss=friction_loss,
            damping=damping,
        )

    def _apply_inverse_friction_overrides(
        self,
        *,
        friction_loss: Optional[np.ndarray],
        damping: Optional[np.ndarray],
    ) -> None:
        self._apply_friction_overrides_to_model(
            self.inverse_model,
            friction_loss=friction_loss,
            damping=damping,
        )

    def _apply_friction_overrides_to_model(
        self,
        model: mujoco.MjModel,
        *,
        friction_loss: Optional[np.ndarray],
        damping: Optional[np.ndarray],
    ) -> None:
        if friction_loss is not None:
            friction_loss = np.asarray(friction_loss, dtype=np.float64).reshape(-1)
            if friction_loss.size != len(self.dof_addrs):
                raise ValueError("friction_loss size must match the number of joints.")
            model.dof_frictionloss[self.dof_addrs] = friction_loss
        if damping is not None:
            damping = np.asarray(damping, dtype=np.float64).reshape(-1)
            if damping.size != len(self.dof_addrs):
                raise ValueError("damping size must match the number of joints.")
            model.dof_damping[self.dof_addrs] = damping

    def _get_home_joint_positions(self) -> np.ndarray:
        if self.home_qpos is not None:
            return self.home_qpos.copy()

        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            return np.array([self.model.key_qpos[key_id, addr] for addr in self.qpos_addrs], dtype=np.float64)

        ranges = self.model.jnt_range[self.joint_ids]
        centers = np.mean(ranges, axis=1)
        return centers.astype(np.float64)

    def get_true_friction_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        coulomb = np.array([self.model.dof_frictionloss[addr] for addr in self.dof_addrs], dtype=np.float64)
        viscous = np.array([self.model.dof_damping[addr] for addr in self.dof_addrs], dtype=np.float64)
        return coulomb, viscous

    def get_controller_friction_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        coulomb = np.array(
            [self.inverse_model.dof_frictionloss[addr] for addr in self.dof_addrs],
            dtype=np.float64,
        )
        viscous = np.array(
            [self.inverse_model.dof_damping[addr] for addr in self.dof_addrs],
            dtype=np.float64,
        )
        return coulomb, viscous

    def set_inverse_friction_parameters(
        self,
        *,
        friction_loss: Optional[np.ndarray],
        damping: Optional[np.ndarray],
    ) -> None:
        self._apply_inverse_friction_overrides(
            friction_loss=friction_loss,
            damping=damping,
        )

    def _get_joint_limits(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lower = np.full(len(self.joint_ids), -np.inf, dtype=np.float64)
        upper = np.full(len(self.joint_ids), np.inf, dtype=np.float64)
        limited = np.zeros(len(self.joint_ids), dtype=bool)
        for idx, joint_id in enumerate(self.joint_ids):
            if not self.model.jnt_limited[joint_id]:
                continue
            lo, hi = self.model.jnt_range[joint_id]
            lower[idx] = min(lo, hi)
            upper[idx] = max(lo, hi)
            limited[idx] = True
        return lower, upper, limited

    def _build_joint_excitation_plan(
        self,
        *,
        amplitude_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lower, upper, limited = self._get_joint_limits()
        home = self._get_home_joint_positions()
        num_joints = len(self.joint_ids)

        centers = home.copy()
        amplitudes = np.full(num_joints, 0.08, dtype=np.float64)
        safe_lower = lower.copy()
        safe_upper = upper.copy()

        for joint_idx in range(num_joints):
            center = home[joint_idx]
            if limited[joint_idx]:
                span = upper[joint_idx] - lower[joint_idx]
                margin = float(np.clip(0.10 * span, 0.04, 0.12))
                safe_lo = lower[joint_idx] + margin
                safe_hi = upper[joint_idx] - margin
                if safe_lo >= safe_hi:
                    safe_lo = lower[joint_idx] + 0.2 * span
                    safe_hi = upper[joint_idx] - 0.2 * span

                center = float(np.clip(center, safe_lo, safe_hi))
                max_excursion = max(0.0, min(center - safe_lo, safe_hi - center))

                # If the home pose sits too close to one side, move to the
                # safe-range midpoint so the active segment excites both motion
                # directions and avoids grazing the joint limits.
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

        return centers, amplitudes, safe_lower, safe_upper, limited

    def generate_excitation_trajectory(
        self,
        *,
        duration: float,
        sample_rate: float,
        base_frequency: float,
        amplitude_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        num_samples = max(int(round(duration * sample_rate)), 2)
        t = np.linspace(0.0, duration, num_samples, endpoint=False)
        num_joints = len(self.joint_ids)
        centers, amplitudes, safe_lower, safe_upper, limited = self._build_joint_excitation_plan(
            amplitude_scale=amplitude_scale
        )
        q_cmd = np.broadcast_to(centers, (num_samples, num_joints)).copy()
        qd_cmd = np.zeros_like(q_cmd)
        qdd_cmd = np.zeros_like(q_cmd)

        segment_edges = np.linspace(0.0, duration, num_joints + 1, dtype=np.float64)
        base_cycles = max(3.0, base_frequency * duration)

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

            amplitude = amplitudes[joint_idx]
            if limited[joint_idx]:
                max_excursion = min(
                    centers[joint_idx] - safe_lower[joint_idx],
                    safe_upper[joint_idx] - centers[joint_idx],
                )
                peak = float(np.max(np.abs(pattern)))
                if peak > 1e-9:
                    amplitude = min(amplitude, 0.98 * max_excursion / peak)

            q_cmd[segment_mask, joint_idx] = centers[joint_idx] + amplitude * pattern

        lower, upper, _ = self._get_joint_limits()
        if np.any(limited):
            for joint_idx in range(num_joints):
                if limited[joint_idx]:
                    np.clip(q_cmd[:, joint_idx], lower[joint_idx], upper[joint_idx], out=q_cmd[:, joint_idx])

        gradient_order = 2 if num_samples >= 3 else 1
        qd_cmd[:] = np.gradient(q_cmd, 1.0 / sample_rate, axis=0, edge_order=gradient_order)
        qdd_cmd[:] = np.gradient(qd_cmd, 1.0 / sample_rate, axis=0, edge_order=gradient_order)

        return t, q_cmd, qd_cmd, qdd_cmd

    def reset(self, q_init: np.ndarray) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        for qpos_addr, value in zip(self.qpos_addrs, q_init):
            self.data.qpos[qpos_addr] = value
        mujoco.mj_forward(self.model, self.data)

    def _inverse_dynamics(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        qpos_backup = self.inverse_data.qpos.copy()
        qvel_backup = self.inverse_data.qvel.copy()
        qacc_backup = self.inverse_data.qacc.copy()

        for qpos_addr, dof_addr, q_i, qd_i, qdd_i in zip(self.qpos_addrs, self.dof_addrs, q, qd, qdd):
            self.inverse_data.qpos[qpos_addr] = q_i
            self.inverse_data.qvel[dof_addr] = qd_i
            self.inverse_data.qacc[dof_addr] = qdd_i

        mujoco.mj_inverse(self.inverse_model, self.inverse_data)
        tau = np.array(
            [self.inverse_data.qfrc_inverse[dof_addr] for dof_addr in self.dof_addrs],
            dtype=np.float64,
        )

        self.inverse_data.qpos[:] = qpos_backup
        self.inverse_data.qvel[:] = qvel_backup
        self.inverse_data.qacc[:] = qacc_backup
        mujoco.mj_forward(self.inverse_model, self.inverse_data)
        return tau

    def _get_joint_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = np.array([self.data.qpos[addr] for addr in self.qpos_addrs], dtype=np.float64)
        qd = np.array([self.data.qvel[addr] for addr in self.dof_addrs], dtype=np.float64)
        qdd = np.array([self.data.qacc[addr] for addr in self.dof_addrs], dtype=np.float64)
        return q, qd, qdd

    def _get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if self.ee_body_id < 0:
            return np.zeros(3, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return (
            self.data.xpos[self.ee_body_id].copy(),
            self.data.xquat[self.ee_body_id].copy(),
        )

    def _get_friction_components(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tau_passive = -np.array([self.data.qfrc_passive[addr] for addr in self.dof_addrs], dtype=np.float64)
        tau_constraint = -np.array([self.data.qfrc_constraint[addr] for addr in self.dof_addrs], dtype=np.float64)
        return tau_passive, tau_constraint, tau_passive + tau_constraint

    def _set_torque(self, tau: np.ndarray) -> None:
        if self.actuator_ids.size > 0:
            self.data.ctrl[:] = 0.0
            for actuator_id, torque in zip(self.actuator_ids, tau):
                self.data.ctrl[actuator_id] = float(torque)
            return

        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self.dof_addrs] = tau

    def evaluate_end_effector_trajectory(self, q_traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        num_samples = q_traj.shape[0]
        ee_pos = np.zeros((num_samples, 3), dtype=np.float64)
        ee_quat = np.zeros((num_samples, 4), dtype=np.float64)

        qpos_backup = self.data.qpos.copy()
        qvel_backup = self.data.qvel.copy()
        qacc_backup = self.data.qacc.copy()

        try:
            self._print_status(f"Evaluating end-effector reference trajectory for {num_samples} samples")
            last_reported_percent = 0
            self.data.qvel[:] = 0.0
            self.data.qacc[:] = 0.0
            for sample_idx, q_sample in enumerate(q_traj):
                for qpos_addr, q_i in zip(self.qpos_addrs, q_sample):
                    self.data.qpos[qpos_addr] = q_i
                mujoco.mj_forward(self.model, self.data)
                ee_pos[sample_idx], ee_quat[sample_idx] = self._get_ee_pose()
                last_reported_percent = self._maybe_report_progress(
                    label="EE reference evaluation",
                    completed=sample_idx + 1,
                    total=num_samples,
                    last_reported_percent=last_reported_percent,
                    percent_step=20,
                )
        finally:
            self.data.qpos[:] = qpos_backup
            self.data.qvel[:] = qvel_backup
            self.data.qacc[:] = qacc_backup
            mujoco.mj_forward(self.model, self.data)

        return ee_pos, ee_quat

    def collect(
        self,
        q_cmd: np.ndarray,
        qd_cmd: np.ndarray,
        qdd_cmd: np.ndarray,
        ee_pos_cmd: np.ndarray,
        ee_quat_cmd: np.ndarray,
        *,
        sample_rate: float,
        feedback_scale: float,
        realtime: bool,
    ) -> FrictionSampleBatch:
        num_samples = q_cmd.shape[0]
        self.reset(q_cmd[0])

        if self.render:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -20
            self.viewer.cam.distance = 1.8
            self.viewer.cam.lookat[:] = [0.0, 0.0, 1.1]

        time_log = np.zeros(num_samples, dtype=np.float64)
        q_log = np.zeros_like(q_cmd)
        qd_log = np.zeros_like(q_cmd)
        qdd_log = np.zeros_like(q_cmd)
        ee_pos_log = np.zeros_like(ee_pos_cmd)
        ee_quat_log = np.zeros_like(ee_quat_cmd)
        tau_ctrl_log = np.zeros_like(q_cmd)
        tau_passive_log = np.zeros_like(q_cmd)
        tau_constraint_log = np.zeros_like(q_cmd)
        tau_friction_log = np.zeros_like(q_cmd)

        sim_time = 0.0
        sample_dt = 1.0 / sample_rate
        wall_start = time.time()
        self._print_status(
            "Collecting MuJoCo samples "
            f"(duration={num_samples / sample_rate:.1f}s, sample_rate={sample_rate:.1f}Hz, "
            f"samples={num_samples})"
        )
        last_reported_percent = 0

        for sample_idx in range(num_samples):
            q_curr, qd_curr, _ = self._get_joint_state()
            tau_ff = self._inverse_dynamics(q_cmd[sample_idx], qd_cmd[sample_idx], qdd_cmd[sample_idx])
            tau_fb = self.kp * (q_cmd[sample_idx] - q_curr) + self.kd * (qd_cmd[sample_idx] - qd_curr)
            tau_ctrl = tau_ff + feedback_scale * tau_fb
            self._set_torque(tau_ctrl)

            next_sample_time = (sample_idx + 1) * sample_dt
            while sim_time + 1e-12 < next_sample_time:
                mujoco.mj_step(self.model, self.data)
                sim_time += self.model.opt.timestep
                if self.viewer is not None and self.viewer.is_running():
                    self.viewer.sync()
                if realtime:
                    elapsed = time.time() - wall_start
                    if sim_time > elapsed:
                        time.sleep(sim_time - elapsed)

            q_meas, qd_meas, qdd_meas = self._get_joint_state()
            ee_pos_meas, ee_quat_meas = self._get_ee_pose()
            tau_passive, tau_constraint, tau_friction = self._get_friction_components()

            time_log[sample_idx] = sim_time
            q_log[sample_idx] = q_meas
            qd_log[sample_idx] = qd_meas
            qdd_log[sample_idx] = qdd_meas
            ee_pos_log[sample_idx] = ee_pos_meas
            ee_quat_log[sample_idx] = ee_quat_meas
            tau_ctrl_log[sample_idx] = tau_ctrl
            tau_passive_log[sample_idx] = tau_passive
            tau_constraint_log[sample_idx] = tau_constraint
            tau_friction_log[sample_idx] = tau_friction
            last_reported_percent = self._maybe_report_progress(
                label="MuJoCo collection",
                completed=sample_idx + 1,
                total=num_samples,
                last_reported_percent=last_reported_percent,
                percent_step=5,
                extra=f"sim_time={sim_time:.1f}s/{num_samples / sample_rate:.1f}s",
            )

        wall_elapsed = time.time() - wall_start
        self._print_status(
            f"MuJoCo collection complete in {wall_elapsed:.1f}s wall time"
        )

        return FrictionSampleBatch(
            time=time_log,
            q=q_log,
            qd=qd_log,
            qdd=qdd_log,
            ee_pos=ee_pos_log,
            ee_quat=ee_quat_log,
            q_cmd=q_cmd,
            qd_cmd=qd_cmd,
            qdd_cmd=qdd_cmd,
            ee_pos_cmd=ee_pos_cmd,
            ee_quat_cmd=ee_quat_cmd,
            tau_ctrl=tau_ctrl_log,
            tau_passive=tau_passive_log,
            tau_constraint=tau_constraint_log,
            tau_friction=tau_friction_log,
        )

    def build_clean_sample_mask(
        self,
        batch: FrictionSampleBatch,
        *,
        limit_margin: float = 0.05,
        constraint_tolerance: float = 0.35,
    ) -> np.ndarray:
        lower, upper, limited = self._get_joint_limits()
        if np.any(limited):
            lower = lower[None, :]
            upper = upper[None, :]
            margin_to_limits = np.minimum(batch.q - lower, upper - batch.q)
            margin_to_limits[:, ~limited] = np.inf
            away_from_limits = np.all(margin_to_limits > limit_margin, axis=1)
        else:
            away_from_limits = np.ones(batch.time.shape[0], dtype=bool)

        constraint_is_clean = np.all(np.abs(batch.tau_constraint) < constraint_tolerance, axis=1)
        finite = (
            np.all(np.isfinite(batch.q), axis=1)
            & np.all(np.isfinite(batch.qd), axis=1)
            & np.all(np.isfinite(batch.tau_friction), axis=1)
        )
        moving = np.any(np.abs(batch.qd) > 0.02, axis=1)
        return away_from_limits & constraint_is_clean & finite & moving

    def run_openarm_collection(self, config: MujocoFrictionCollectionConfig) -> FrictionSampleBatch:
        self._print_status(
            "Generating segmented joint excitation trajectory "
            f"(duration={config.duration:.1f}s, base_frequency={config.base_frequency:.3f}Hz, "
            f"amplitude_scale={config.amplitude_scale:.3f})"
        )
        t, q_cmd, qd_cmd, qdd_cmd = self.generate_excitation_trajectory(
            duration=config.duration,
            sample_rate=config.sample_rate,
            base_frequency=config.base_frequency,
            amplitude_scale=config.amplitude_scale,
        )
        self._print_status(f"Excitation trajectory ready with {t.shape[0]} samples")
        return self.run_reference_trajectory(
            q_cmd=q_cmd,
            qd_cmd=qd_cmd,
            qdd_cmd=qdd_cmd,
            sample_rate=config.sample_rate,
            feedback_scale=config.feedback_scale,
            realtime=config.realtime,
            time_reference=t,
        )

    def run_reference_trajectory(
        self,
        *,
        q_cmd: np.ndarray,
        qd_cmd: np.ndarray,
        qdd_cmd: np.ndarray,
        sample_rate: float,
        feedback_scale: float,
        realtime: bool,
        time_reference: Optional[np.ndarray] = None,
        ee_pos_cmd: Optional[np.ndarray] = None,
        ee_quat_cmd: Optional[np.ndarray] = None,
    ) -> FrictionSampleBatch:
        if ee_pos_cmd is None or ee_quat_cmd is None:
            ee_pos_cmd, ee_quat_cmd = self.evaluate_end_effector_trajectory(q_cmd)

        batch = self.collect(
            q_cmd=q_cmd,
            qd_cmd=qd_cmd,
            qdd_cmd=qdd_cmd,
            ee_pos_cmd=ee_pos_cmd,
            ee_quat_cmd=ee_quat_cmd,
            sample_rate=sample_rate,
            feedback_scale=feedback_scale,
            realtime=realtime,
        )
        if time_reference is None:
            batch.time[:] = np.arange(batch.time.shape[0], dtype=np.float64) / float(sample_rate)
        else:
            time_reference = np.asarray(time_reference, dtype=np.float64).reshape(-1)
            if time_reference.shape[0] != batch.time.shape[0]:
                raise ValueError("time_reference length must match the number of trajectory samples.")
            batch.time[:] = time_reference
        return batch

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
