from __future__ import annotations

import numpy as np

from friction_identification_core.config import Config


class SafetyGuard:
    """Only keep joint-limit detection and torque clamping."""

    def __init__(self, config: Config, active_joint_mask: np.ndarray | None = None) -> None:
        self.joint_names = list(config.robot.joint_names)
        self.joint_limits = np.asarray(config.robot.joint_limits, dtype=np.float64)
        self.torque_limits = np.asarray(config.robot.torque_limits, dtype=np.float64)
        self.margin = float(config.safety.joint_limit_margin)
        self.enable_torque_clamp = bool(config.safety.enable_torque_clamp)
        if active_joint_mask is None:
            self.active_joint_mask = np.ones(len(self.joint_names), dtype=bool)
        else:
            self.active_joint_mask = np.asarray(active_joint_mask, dtype=bool).reshape(-1)

    def safe_joint_window(self) -> tuple[np.ndarray, np.ndarray]:
        lower = self.joint_limits[:, 0] + self.margin
        upper = self.joint_limits[:, 1] - self.margin
        return lower, upper

    def check_joint_limits(self, q: np.ndarray) -> bool:
        lower, upper = self.safe_joint_window()
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        within = (q >= lower) & (q <= upper)
        within[~self.active_joint_mask] = True
        return bool(np.all(within))

    def get_violation_message(self, q: np.ndarray) -> str | None:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        lower, upper = self.safe_joint_window()
        violation = np.flatnonzero(
            (q < lower) | (q > upper)
        )
        violation = violation[self.active_joint_mask[violation]]
        if violation.size == 0:
            return None
        joint_idx = int(violation[0])
        return (
            f"{self.joint_names[joint_idx]} 超出安全关节范围: "
            f"q={q[joint_idx]:.6f} rad, "
            f"safe_range=[{lower[joint_idx]:.6f}, {upper[joint_idx]:.6f}]"
        )

    def assert_joint_limits(self, q: np.ndarray) -> None:
        message = self.get_violation_message(q)
        if message is not None:
            raise RuntimeError(message)

    def clamp_torque(self, tau: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype=np.float64).reshape(-1)
        if not self.enable_torque_clamp:
            return tau.copy()
        return np.clip(tau, -self.torque_limits, self.torque_limits)
