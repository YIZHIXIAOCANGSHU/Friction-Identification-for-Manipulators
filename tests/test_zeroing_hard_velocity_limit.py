from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from friction_identification_core.config import load_config
from friction_identification_core.controller import SingleMotorController
from friction_identification_core.pipeline import _perform_zeroing, _runtime_abort_from_frame
from friction_identification_core.serial_protocol import (
    FeedbackFrame,
    RECV_FRAME_HEAD,
    RECV_FRAME_SIZE,
    RECV_FRAME_STRUCT,
    SerialFrameParser,
    SingleMotorCommandAdapter,
)


class _RecorderStub:
    def log_zeroing_event(self, **_kwargs) -> None:
        return None

    def log_live_command_packet(self, **_kwargs) -> None:
        return None

    def log_zeroing_sample(self, **_kwargs) -> None:
        return None


class _ConstantFrameTransport:
    def __init__(
        self,
        *,
        motor_id: int,
        position: float,
        velocity: float,
        torque: float,
        temperature: float = 31.0,
    ) -> None:
        self._frame = RECV_FRAME_STRUCT.pack(
            RECV_FRAME_HEAD,
            int(motor_id),
            1,
            float(position),
            float(velocity),
            float(torque),
            float(temperature),
        )
        self.writes: list[bytes] = []

    def read(self, size: int) -> bytes:
        frame_size = len(self._frame)
        repeat_count = max(1, int(size + frame_size - 1) // frame_size)
        return self._frame * repeat_count

    def write(self, payload: bytes) -> int:
        self.writes.append(bytes(payload))
        return len(payload)

    def reset_input_buffer(self) -> None:
        return None


class _VelocitySequenceTransport:
    def __init__(
        self,
        *,
        motor_id: int,
        position: float,
        velocities: tuple[float, ...],
        torque: float,
        temperature: float = 31.0,
    ) -> None:
        if not velocities:
            raise ValueError("velocities must not be empty.")
        self._motor_id = int(motor_id)
        self._position = float(position)
        self._velocities = tuple(float(velocity) for velocity in velocities)
        self._torque = float(torque)
        self._temperature = float(temperature)
        self._read_index = 0
        self.writes: list[bytes] = []

    def _next_frame(self) -> bytes:
        velocity_index = min(self._read_index, len(self._velocities) - 1)
        velocity = self._velocities[velocity_index]
        self._read_index += 1
        return RECV_FRAME_STRUCT.pack(
            RECV_FRAME_HEAD,
            self._motor_id,
            1,
            self._position,
            velocity,
            self._torque,
            self._temperature,
        )

    def read(self, size: int) -> bytes:
        return self._next_frame()

    def write(self, payload: bytes) -> int:
        self.writes.append(bytes(payload))
        return len(payload)

    def reset_input_buffer(self) -> None:
        return None


class ZeroingHardVelocityLimitTests(unittest.TestCase):
    def _write_config(self, root: Path, *, zeroing_hard_velocity_limit: float) -> Path:
        config_path = root / "config.yaml"
        config_path.write_text(
            "\n".join(
                (
                    "motors:",
                    "  ids: [1, 2, 3, 4, 5, 6, 7]",
                    "  enabled: [1]",
                    "control:",
                    "  zeroing_position_gain: 0.8",
                    "  zeroing_velocity_gain: 0.18",
                    f"  zeroing_hard_velocity_limit: {float(zeroing_hard_velocity_limit):.6f}",
                    "  zeroing_velocity_limit: 0.4",
                    "  zeroing_position_tolerance: 0.02",
                    "  zeroing_velocity_tolerance: 0.02",
                    "  zeroing_required_frames: 4",
                    "  zeroing_timeout: 0.2",
                    "output:",
                    f'  results_dir: "{(root / "results").as_posix()}"',
                    "",
                )
            ),
            encoding="utf-8",
        )
        return config_path

    def test_load_config_reads_zeroing_hard_velocity_limit_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_config(self._write_config(Path(tmpdir), zeroing_hard_velocity_limit=1.25))

        self.assertEqual(config.control.zeroing_hard_velocity_limit.tolist(), [1.25] * 7)
        self.assertEqual(config.control.zeroing_velocity_limit.tolist(), [0.4] * 7)

    def test_zeroing_uses_yaml_hard_velocity_limit_for_abort(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_config(self._write_config(Path(tmpdir), zeroing_hard_velocity_limit=0.5))

            transport = _ConstantFrameTransport(
                motor_id=1,
                position=0.05,
                velocity=0.6,
                torque=0.0,
            )
            parser = SerialFrameParser(max_motor_id=max(config.motor_ids))
            controller = SingleMotorController(config)
            command_adapter = SingleMotorCommandAdapter(
                motor_count=7,
                torque_limits=config.control.max_torque,
            )

            with self.assertRaisesRegex(
                ValueError,
                r"reason=velocity_limit_exceeded.*velocity_limit=0.500000",
            ):
                _perform_zeroing(
                    config=config,
                    transport=transport,
                    parser=parser,
                    command_adapter=command_adapter,
                    controller=controller,
                    rerun_recorder=_RecorderStub(),
                )

            self.assertEqual(
                transport.writes[-1],
                SingleMotorCommandAdapter(motor_count=7).pack(1, 0.0),
            )

    def test_zeroing_ignores_single_frame_velocity_spike_during_locking(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_config(self._write_config(Path(tmpdir), zeroing_hard_velocity_limit=2.0))
            config = replace(
                config,
                serial=replace(
                    config.serial,
                    read_chunk_size=RECV_FRAME_SIZE,
                    read_timeout=0.001,
                ),
            )

            transport = _VelocitySequenceTransport(
                motor_id=1,
                position=0.0,
                velocities=(2.2, 0.0, 0.0, 0.0, 0.0),
                torque=0.0,
            )
            parser = SerialFrameParser(max_motor_id=max(config.motor_ids))
            controller = SingleMotorController(config)
            command_adapter = SingleMotorCommandAdapter(
                motor_count=7,
                torque_limits=config.control.max_torque,
            )

            _perform_zeroing(
                config=config,
                transport=transport,
                parser=parser,
                command_adapter=command_adapter,
                controller=controller,
                rerun_recorder=_RecorderStub(),
            )

            self.assertGreaterEqual(len(transport.writes), 2)
            self.assertEqual(
                transport.writes[-1],
                SingleMotorCommandAdapter(motor_count=7).pack(1, 0.0),
            )

    def test_zeroing_does_not_use_excitation_position_limit_as_abort_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_config(self._write_config(Path(tmpdir), zeroing_hard_velocity_limit=2.0))
            config = replace(
                config,
                control=replace(
                    config.control,
                    zeroing_timeout=0.02,
                ),
            )

            transport = _ConstantFrameTransport(
                motor_id=1,
                position=3.0,
                velocity=0.0,
                torque=0.0,
            )
            parser = SerialFrameParser(max_motor_id=max(config.motor_ids))
            controller = SingleMotorController(config)
            command_adapter = SingleMotorCommandAdapter(
                motor_count=7,
                torque_limits=config.control.max_torque,
            )

            with self.assertRaisesRegex(ValueError, r"reason=zeroing_timeout"):
                _perform_zeroing(
                    config=config,
                    transport=transport,
                    parser=parser,
                    command_adapter=command_adapter,
                    controller=controller,
                    rerun_recorder=_RecorderStub(),
                )

    def test_runtime_position_limit_abort_remains_enabled_outside_zeroing(self) -> None:
        frame = FeedbackFrame(
            motor_id=1,
            state=1,
            position=3.0,
            velocity=0.0,
            torque=0.0,
            mos_temperature=31.0,
        )

        abort_event = _runtime_abort_from_frame(
            frame=frame,
            stage="identify",
            target_motor_id=1,
            group_index=1,
            round_index=1,
            phase_name="excitation_cycle_01",
            velocity_limit=1.0,
            torque_limit=1.0,
            position_limit=2.5,
        )

        self.assertIsNotNone(abort_event)
        assert abort_event is not None
        self.assertEqual(abort_event.reason, "position_limit_exceeded")
        self.assertEqual(abort_event.stage, "identify")


if __name__ == "__main__":
    unittest.main()
