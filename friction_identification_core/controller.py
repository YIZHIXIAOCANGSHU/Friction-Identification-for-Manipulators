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
    ) -> tuple[float, float]:
        index = self._config.motor_index(motor_id)
        max_torque = float(self._config.control.max_torque[index])
        if compensation is None:
            velocity_p_gain = float(self._config.control.velocity_p_gain[index])
            velocity_error = float(reference.velocity_cmd) - float(feedback.velocity)
            raw_command = velocity_p_gain * velocity_error
        else:
            raw_command = compensation.feedforward_torque(float(feedback.velocity))
        limited_command = float(np.clip(raw_command, -max_torque, max_torque))
        return float(raw_command), limited_command
