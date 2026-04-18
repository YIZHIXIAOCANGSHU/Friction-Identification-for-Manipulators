from __future__ import annotations

from friction_identification_core.config import Config
from friction_identification_core.models import ReferenceSample
from friction_identification_core.serial_protocol import FeedbackFrame


class SingleMotorController:
    def __init__(self, config: Config) -> None:
        self._config = config

    def update(self, motor_id: int, reference: ReferenceSample, feedback: FeedbackFrame) -> float:
        index = self._config.motor_index(motor_id)
        kp = float(self._config.control.kp[index])
        kd = float(self._config.control.kd[index])
        output_scale = float(self._config.control.output_scale[index])
        position_error = float(reference.position_cmd) - float(feedback.position)
        velocity_error = float(reference.velocity_cmd) - float(feedback.velocity)
        return output_scale * (kp * position_error + kd * velocity_error)
