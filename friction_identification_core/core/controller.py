from __future__ import annotations

from typing import Protocol

import numpy as np

from friction_identification_core.config import Config
from friction_identification_core.core.safety import SafetyGuard


class InverseDynamicsBackend(Protocol):
    def inverse_dynamics(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        ...


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
        tau_fb = self.kp * (np.asarray(q_cmd) - np.asarray(q_curr)) + self.kd * (np.asarray(qd_cmd) - np.asarray(qd_curr))
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
        return tau_ff, tau_fb, tau
