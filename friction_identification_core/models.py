from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Protocol

import numpy as np

if TYPE_CHECKING:
    from friction_identification_core.controller import (
        FrictionIdentificationController,
        InverseDynamicsBackend,
        SafetyGuard,
    )
    from friction_identification_core.results import IdentificationResults
    from friction_identification_core.trajectory import ReferenceTrajectory


@dataclass
class JointFrictionParameters:
    """Identified friction coefficients for one joint."""

    coulomb: float
    viscous: float
    offset: float = 0.0
    velocity_scale: float = 0.02


@dataclass
class FrictionSampleBatch:
    """Time-aligned batch of joint, torque, and end-effector samples."""

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
    """Per-joint fitting outputs together with train/validation diagnostics."""

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
    """Summary of how one controller parameter set tracks a reference motion."""

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


@dataclass
class CollectedData:
    """Collected hardware payload for one batch or one compensation validation run."""

    source: str
    mode: str
    time: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    q_cmd: np.ndarray
    qd_cmd: np.ndarray
    tau_command: np.ndarray
    tau_measured: np.ndarray
    qdd: np.ndarray | None = None
    qdd_cmd: np.ndarray | None = None
    tau_track_ff: np.ndarray | None = None
    tau_track_fb: np.ndarray | None = None
    tau_friction_comp: np.ndarray | None = None
    tau_rigid: np.ndarray | None = None
    tau_residual: np.ndarray | None = None
    tau_passive: np.ndarray | None = None
    tau_constraint: np.ndarray | None = None
    tau_friction: np.ndarray | None = None
    clean_mask: np.ndarray | None = None
    joint_refresh_mask: np.ndarray | None = None
    joint_clean_mask: np.ndarray | None = None
    rotation_state: np.ndarray | None = None
    range_ratio: np.ndarray | None = None
    limit_margin_remaining: np.ndarray | None = None
    batch_index: np.ndarray | None = None
    phase_name: np.ndarray | None = None
    ee_pos: np.ndarray | None = None
    ee_quat: np.ndarray | None = None
    ee_pos_cmd: np.ndarray | None = None
    ee_quat_cmd: np.ndarray | None = None
    mos_temperature: np.ndarray | None = None
    coil_temperature: np.ndarray | None = None
    uart_cycle_hz: np.ndarray | None = None
    uart_latency_ms: np.ndarray | None = None
    uart_transfer_kbps: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        return int(self.time.shape[0])


@dataclass
class IdentificationInputs:
    """Inputs consumed by the shared friction estimator."""

    velocity: np.ndarray
    torque: np.ndarray
    joint_names: list[str]
    clean_mask: np.ndarray | None = None
    sample_mask: np.ndarray | None = None
    true_coulomb: np.ndarray | None = None
    true_viscous: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DataSource(Protocol):
    """Hardware source interface used by the parallel identification pipeline."""

    source_name: str
    inverse_dynamics_backend: "InverseDynamicsBackend"

    def build_reference(self, *, joint_index: int | None = None) -> "ReferenceTrajectory | None":
        ...

    def supports_identification(self, mode: str) -> bool:
        ...

    def collect(
        self,
        *,
        mode: str,
        reference: "ReferenceTrajectory | None",
        controller: "FrictionIdentificationController",
        safety: "SafetyGuard",
        batch_index: int = 1,
        total_batches: int = 1,
        target_joint_index: int | None = None,
        group_index: int = 1,
        total_groups: int = 1,
    ) -> CollectedData:
        ...

    def prepare_identification(self, data: CollectedData) -> IdentificationInputs | None:
        ...

    def publish_identification_result(
        self,
        data: CollectedData,
        result: FrictionIdentificationResult,
    ) -> None:
        ...

    def finalize(
        self,
        data: CollectedData | None,
        result: FrictionIdentificationResult | None,
    ) -> None:
        ...

    def publish_summary(self, summary: "IdentificationResults") -> None:
        ...
