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
    ee_pos: np.ndarray
    ee_quat: np.ndarray
    q_cmd: np.ndarray
    qd_cmd: np.ndarray
    qdd_cmd: np.ndarray
    ee_pos_cmd: np.ndarray
    ee_quat_cmd: np.ndarray
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
            ee_pos=self.ee_pos[mask],
            ee_quat=self.ee_quat[mask],
            q_cmd=self.q_cmd[mask],
            qd_cmd=self.qd_cmd[mask],
            qdd_cmd=self.qdd_cmd[mask],
            ee_pos_cmd=self.ee_pos_cmd[mask],
            ee_quat_cmd=self.ee_quat_cmd[mask],
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


@dataclass
class TrackingEvaluationResult:
    label: str
    batch: FrictionSampleBatch
    controller_coulomb: np.ndarray
    controller_viscous: np.ndarray
    joint_rmse: np.ndarray
    joint_max_abs_error: np.ndarray
    ee_rmse_xyz: np.ndarray
    mean_joint_rmse: float
    ee_position_rmse: float
    ee_max_error: float
