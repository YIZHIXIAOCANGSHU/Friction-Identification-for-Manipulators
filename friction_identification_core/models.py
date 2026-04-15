from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class JointFrictionParameters:
    coulomb: float
    viscous: float
    offset: float = 0.0
    velocity_scale: float = 0.02


@dataclass
class FrictionSampleBatch:
    time: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    q_cmd: np.ndarray
    qd_cmd: np.ndarray
    qdd_cmd: np.ndarray
    tau_ctrl: np.ndarray
    tau_passive: np.ndarray
    tau_constraint: np.ndarray
    tau_friction: np.ndarray

    def subset(self, mask: np.ndarray) -> "FrictionSampleBatch":
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        return FrictionSampleBatch(
            time=self.time[mask],
            q=self.q[mask],
            qd=self.qd[mask],
            qdd=self.qdd[mask],
            q_cmd=self.q_cmd[mask],
            qd_cmd=self.qd_cmd[mask],
            qdd_cmd=self.qdd_cmd[mask],
            tau_ctrl=self.tau_ctrl[mask],
            tau_passive=self.tau_passive[mask],
            tau_constraint=self.tau_constraint[mask],
            tau_friction=self.tau_friction[mask],
        )


@dataclass
class FrictionIdentificationResult:
    joint_names: list[str]
    parameters: list[JointFrictionParameters]
    predicted_torque: np.ndarray
    measured_torque: np.ndarray
    train_mask: np.ndarray
    validation_mask: np.ndarray
    train_rmse: np.ndarray
    validation_rmse: np.ndarray
    train_r2: np.ndarray
    validation_r2: np.ndarray
    true_coulomb: Optional[np.ndarray] = None
    true_viscous: Optional[np.ndarray] = None
