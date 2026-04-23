from __future__ import annotations

import numpy as np

from friction_identification_core.config import Config
from friction_identification_core.models import MotorCompensationParameters, ReferenceSample
from friction_identification_core.serial_protocol import FeedbackFrame


class SingleMotorController:
    def __init__(self, config: Config) -> None:
        self._config = config

    def update(
        self,
        motor_id: int,
        reference: ReferenceSample,
        feedback: FeedbackFrame,
        *,
        compensation: MotorCompensationParameters | None = None,
        position_gain: float | None = None,
        velocity_gain: float | None = None,
    ) -> tuple[float, float]:
        index = self._config.motor_index(motor_id)
        max_torque = float(self._config.control.max_torque[index])
        if compensation is None:
            if position_gain is None:
                position_gain = float(self._config.control.position_gain[index])
            if velocity_gain is None:
                velocity_gain = float(self._config.control.velocity_gain[index])
            position_error = float(reference.position_cmd) - float(feedback.position)
            velocity_error = float(reference.velocity_cmd) - float(feedback.velocity)
            raw_command = float(position_gain) * position_error + float(velocity_gain) * velocity_error
        else:
            raw_command = compensation.feedforward_torque(float(feedback.velocity))
        limited_command = float(np.clip(raw_command, -max_torque, max_torque))
        return float(raw_command), limited_command
