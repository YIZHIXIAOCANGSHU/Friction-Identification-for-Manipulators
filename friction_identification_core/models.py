from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReferenceSample:
    position_cmd: float
    velocity_cmd: float
    acceleration_cmd: float
    phase_name: str


@dataclass(frozen=True)
class ReferenceTrajectory:
    time: np.ndarray
    position_cmd: np.ndarray
    velocity_cmd: np.ndarray
    acceleration_cmd: np.ndarray
    phase_name: np.ndarray
    duration_s: float

    def sample(self, elapsed_s: float) -> ReferenceSample:
        if self.time.size == 0:
            return ReferenceSample(0.0, 0.0, 0.0, "empty")
        index = int(np.searchsorted(self.time, float(elapsed_s), side="right") - 1)
        index = min(max(index, 0), int(self.time.size) - 1)
        return ReferenceSample(
            position_cmd=float(self.position_cmd[index]),
            velocity_cmd=float(self.velocity_cmd[index]),
            acceleration_cmd=float(self.acceleration_cmd[index]),
            phase_name=str(self.phase_name[index]),
        )


@dataclass(frozen=True)
class RoundCapture:
    group_index: int
    round_index: int
    target_motor_id: int
    motor_name: str
    time: np.ndarray
    motor_id: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    torque_feedback: np.ndarray
    command: np.ndarray
    position_cmd: np.ndarray
    velocity_cmd: np.ndarray
    acceleration_cmd: np.ndarray
    phase_name: np.ndarray
    state: np.ndarray
    mos_temperature: np.ndarray
    id_match_ok: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        return int(self.time.size)


@dataclass(frozen=True)
class MotorIdentificationResult:
    motor_id: int
    motor_name: str
    identified: bool
    coulomb: float
    viscous: float
    offset: float
    velocity_scale: float
    torque_pred: np.ndarray
    torque_target: np.ndarray
    sample_mask: np.ndarray
    train_mask: np.ndarray
    valid_mask: np.ndarray
    train_rmse: float
    valid_rmse: float
    train_r2: float
    valid_r2: float
    valid_sample_ratio: float
    sample_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
