from __future__ import annotations

import numpy as np

from friction_identification_core.config import Config
from friction_identification_core.models import ReferenceSample
from friction_identification_core.serial_protocol import FeedbackFrame


class SingleMotorController:
    def __init__(self, config: Config) -> None:
        self._config = config

    def update(self, motor_id: int, reference: ReferenceSample, feedback: FeedbackFrame) -> tuple[float, float]:
        index = self._config.motor_index(motor_id)
        max_velocity = float(self._config.control.max_velocity[index])
        max_torque = float(self._config.control.max_torque[index])
        velocity_error = float(reference.velocity_cmd) - float(feedback.velocity)
        raw_command = max_torque * velocity_error / max(max_velocity, 1.0e-9)
        limited_command = float(np.clip(raw_command, -max_torque, max_torque))
        return float(raw_command), limited_command
