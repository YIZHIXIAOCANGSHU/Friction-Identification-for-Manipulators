from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from friction_identification_core.config import Config
from friction_identification_core.models import JointFrictionParameters


class InverseDynamicsBackend(Protocol):
    def inverse_dynamics(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        ...


class SafetyGuard:
    """Only keep joint-limit detection and torque clamping."""

    def __init__(self, config: Config, active_joint_mask: np.ndarray | None = None) -> None:
        self.joint_names = list(config.robot.joint_names)
        self.joint_limits = np.asarray(config.robot.joint_limits, dtype=np.float64)
        self.torque_limits = np.asarray(config.robot.torque_limits, dtype=np.float64)
        self.margin = float(config.safety.joint_limit_margin)
        self.soft_limit_zone = max(float(config.safety.soft_limit_zone), 0.0)
        self.enable_torque_clamp = bool(config.safety.enable_torque_clamp)
        if active_joint_mask is None:
            self.active_joint_mask = np.ones(len(self.joint_names), dtype=bool)
        else:
            self.active_joint_mask = np.asarray(active_joint_mask, dtype=bool).reshape(-1)

    def safe_joint_window(self) -> tuple[np.ndarray, np.ndarray]:
        lower = self.joint_limits[:, 0] + self.margin
        upper = self.joint_limits[:, 1] - self.margin
        return lower, upper

    def hard_joint_window(self) -> tuple[np.ndarray, np.ndarray]:
        return self.joint_limits[:, 0].copy(), self.joint_limits[:, 1].copy()

    def check_joint_limits(self, q: np.ndarray, *, use_safe_margin: bool = False) -> bool:
        lower, upper = self.safe_joint_window() if use_safe_margin else self.hard_joint_window()
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        within = (q >= lower) & (q <= upper)
        within[~self.active_joint_mask] = True
        return bool(np.all(within))

    def get_violation_message(self, q: np.ndarray, *, use_safe_margin: bool = False) -> str | None:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        lower, upper = self.safe_joint_window() if use_safe_margin else self.hard_joint_window()
        violation = np.flatnonzero((q < lower) | (q > upper))
        violation = violation[self.active_joint_mask[violation]]
        if violation.size == 0:
            return None
        joint_idx = int(violation[0])
        range_label = "安全关节范围" if use_safe_margin else "物理关节范围"
        return (
            f"{self.joint_names[joint_idx]} 超出{range_label}: "
            f"q={q[joint_idx]:.6f} rad, "
            f"range=[{lower[joint_idx]:.6f}, {upper[joint_idx]:.6f}]"
        )

    def assert_joint_limits(self, q: np.ndarray, *, use_safe_margin: bool = False) -> None:
        message = self.get_violation_message(q, use_safe_margin=use_safe_margin)
        if message is not None:
            raise RuntimeError(message)

    def clamp_torque(self, tau: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype=np.float64).reshape(-1)
        if not self.enable_torque_clamp:
            return tau.copy()
        return np.clip(tau, -self.torque_limits, self.torque_limits)

    def soften_torque_near_joint_limits(self, q: np.ndarray, tau: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        tau = np.asarray(tau, dtype=np.float64).reshape(-1).copy()
        if self.soft_limit_zone <= 1e-9:
            return tau

        lower, upper = self.safe_joint_window()
        for joint_idx, active in enumerate(self.active_joint_mask):
            if not active:
                continue

            joint_tau = tau[joint_idx]
            if joint_tau > 0.0:
                outward_margin = upper[joint_idx] - q[joint_idx]
            elif joint_tau < 0.0:
                outward_margin = q[joint_idx] - lower[joint_idx]
            else:
                continue

            if outward_margin <= 0.0:
                tau[joint_idx] = 0.0
                continue

            if outward_margin < self.soft_limit_zone:
                tau[joint_idx] *= outward_margin / self.soft_limit_zone
        return tau


class FrictionIdentificationController:
    """Unified feedforward plus PD controller for both simulation and hardware."""

    def __init__(
        self,
        config: Config,
        backend: InverseDynamicsBackend,
        safety: SafetyGuard | None = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.safety = safety
        self.kp = np.asarray(config.controller.kp, dtype=np.float64)
        self.kd = np.asarray(config.controller.kd, dtype=np.float64)
        self.feedback_scale = float(config.controller.feedback_scale)
        self.target_joint = int(config.identification.target_joint)

    def compute_torque(
        self,
        q_cmd: np.ndarray,
        qd_cmd: np.ndarray,
        qdd_cmd: np.ndarray,
        q_curr: np.ndarray,
        qd_curr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tau_ff = np.asarray(self.backend.inverse_dynamics(q_cmd, qd_cmd, qdd_cmd), dtype=np.float64)
        tau_fb = self.kp * (np.asarray(q_cmd) - np.asarray(q_curr)) + self.kd * (
            np.asarray(qd_cmd) - np.asarray(qd_curr)
        )
        tau = tau_ff + self.feedback_scale * tau_fb

        mask = np.zeros_like(tau, dtype=bool)
        mask[self.target_joint] = True
        tau_ff = tau_ff.copy()
        tau_fb = tau_fb.copy()
        tau = tau.copy()
        tau_ff[~mask] = 0.0
        tau_fb[~mask] = 0.0
        tau[~mask] = 0.0

        if self.safety is not None:
            tau = self.safety.clamp_torque(tau)
            tau = self.safety.soften_torque_near_joint_limits(q_curr, tau)
        return tau_ff, tau_fb, tau


def load_summary_vectors(summary_path: Path, joint_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load identified friction vectors from the new NPZ store or the legacy JSON summary."""

    zeros = np.zeros(joint_count, dtype=np.float64)
    for candidate in _summary_candidates(summary_path):
        if not candidate.exists():
            continue
        if candidate.suffix.lower() == ".npz":
            with np.load(candidate, allow_pickle=False) as payload:
                return (
                    _read_result_vector(payload, "coulomb", joint_count),
                    _read_result_vector(payload, "viscous", joint_count),
                    _read_result_vector(payload, "offset", joint_count),
                )

        with open(candidate, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        def _read_legacy_vector(key: str) -> np.ndarray:
            values = np.asarray(payload.get(key, [0.0] * joint_count), dtype=np.float64).reshape(-1)
            if values.size != joint_count:
                return np.zeros(joint_count, dtype=np.float64)
            return values

        return (
            _read_legacy_vector("estimated_coulomb"),
            _read_legacy_vector("estimated_viscous"),
            _read_legacy_vector("estimated_offset"),
        )

    return zeros.copy(), zeros.copy(), zeros.copy()


def has_compensation_results(summary_path: Path) -> bool:
    return any(candidate.exists() for candidate in _summary_candidates(summary_path))


def _summary_candidates(summary_path: Path) -> list[Path]:
    summary_path = Path(summary_path)
    candidates = [summary_path]
    if summary_path.suffix.lower() == ".npz":
        candidates.append(summary_path.with_name("real_friction_identification_summary.json"))
    elif summary_path.suffix.lower() == ".json":
        candidates.insert(0, summary_path.with_suffix(".npz"))
    return candidates


def _read_result_vector(payload: np.lib.npyio.NpzFile, key: str, joint_count: int) -> np.ndarray:
    raw = payload[key] if key in payload.files else np.zeros(joint_count, dtype=np.float64)
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if values.size != joint_count:
        return np.zeros(joint_count, dtype=np.float64)
    return np.nan_to_num(values, nan=0.0)


def load_compensation_parameters(
    summary_path: Path,
    joint_count: int,
    *,
    velocity_scale: float = 0.03,
) -> list[JointFrictionParameters]:
    coulomb, viscous, offset = load_summary_vectors(summary_path, joint_count)
    return [
        JointFrictionParameters(
            coulomb=float(coulomb[idx]),
            viscous=float(viscous[idx]),
            offset=float(offset[idx]),
            velocity_scale=float(velocity_scale),
        )
        for idx in range(joint_count)
    ]


def predict_compensation_torque(
    velocity: np.ndarray,
    parameters: list[JointFrictionParameters],
    torque_limits: np.ndarray,
) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
    torque_limits = np.asarray(torque_limits, dtype=np.float64).reshape(-1)
    torque = np.zeros_like(velocity)
    for idx, param in enumerate(parameters):
        scale = max(float(param.velocity_scale), 1e-6)
        torque[idx] = (
            param.coulomb * np.tanh(velocity[idx] / scale)
            + param.viscous * velocity[idx]
            + param.offset
        )
    return np.clip(torque, -torque_limits, torque_limits)


__all__ = [
    "FrictionIdentificationController",
    "InverseDynamicsBackend",
    "SafetyGuard",
    "has_compensation_results",
    "load_compensation_parameters",
    "load_summary_vectors",
    "predict_compensation_torque",
]
