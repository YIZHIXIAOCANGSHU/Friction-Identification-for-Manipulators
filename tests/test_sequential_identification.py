from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from friction_identification_core.config import DEFAULT_CONFIG_PATH, load_config
from friction_identification_core.controller import SingleMotorController
from friction_identification_core.identification import identify_motor_friction
from friction_identification_core.models import ReferenceSample, RoundCapture
from friction_identification_core.pipeline import run_sequential_identification
from friction_identification_core.serial_protocol import (
    COMMAND_PAYLOAD_STRUCT,
    FeedbackFrame,
    RECV_FRAME_HEAD,
    RECV_FRAME_STRUCT,
    SerialFrameParser,
    SingleMotorCommandAdapter,
    calculate_xor_checksum,
)
from friction_identification_core.trajectory import build_reference_trajectory


class FakeTransport:
    def __init__(self, motor_ids: tuple[int, ...]) -> None:
        self._motor_ids = motor_ids
        self._step = 0
        self.writes: list[bytes] = []
        self.closed = False

    def read(self, size: int) -> bytes:
        frames = bytearray()
        for motor_id in self._motor_ids:
            angle = 0.2 * self._step + 0.1 * motor_id
            position = 0.2 * np.sin(angle)
            velocity = 0.8 * np.cos(angle)
            torque = 0.9 * np.tanh(velocity / 0.05) + 0.08 * velocity + 0.01 * motor_id
            frames.extend(
                RECV_FRAME_STRUCT.pack(
                    RECV_FRAME_HEAD,
                    motor_id,
                    1,
                    float(position),
                    float(velocity),
                    float(torque),
                    float(30.0 + motor_id),
                )
            )
            self._step += 1
        return bytes(frames[:size])

    def write(self, payload: bytes) -> int:
        self.writes.append(payload)
        return len(payload)

    def reset_input_buffer(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class SequentialIdentificationTests(unittest.TestCase):
    def test_parser_and_command_adapter(self) -> None:
        parser = SerialFrameParser(max_motor_id=7)
        payload = RECV_FRAME_STRUCT.pack(RECV_FRAME_HEAD, 3, 2, 1.0, -2.0, 0.5, 40.0)
        parser.feed(b"\x00\x01" + payload)
        frame = parser.pop_frame()
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame.motor_id, 3)
        self.assertAlmostEqual(frame.velocity, -2.0, places=6)

        config = load_config(DEFAULT_CONFIG_PATH)
        adapter = SingleMotorCommandAdapter(motor_count=7, torque_limits=config.control.max_torque)
        command = adapter.pack(5, 99.0)
        payload_values = COMMAND_PAYLOAD_STRUCT.unpack(command[2:30])
        self.assertAlmostEqual(adapter.limit_command(5, 99.0), 7.0, places=6)
        self.assertAlmostEqual(payload_values[4], 7.0, places=6)
        self.assertTrue(all(abs(value) < 1.0e-9 for index, value in enumerate(payload_values) if index != 4))
        self.assertEqual(command[30], calculate_xor_checksum(command[:30]))

    def test_controller_clips_raw_output_to_motor_limit(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)
        controller = SingleMotorController(config)
        reference = ReferenceSample(position_cmd=0.0, velocity_cmd=2.0, acceleration_cmd=0.0, phase_name="test")
        feedback = FeedbackFrame(
            motor_id=5,
            state=1,
            position=0.0,
            velocity=0.0,
            torque=0.0,
            mos_temperature=30.0,
        )

        raw_command, limited_command = controller.update(5, reference, feedback)

        self.assertGreater(raw_command, 7.0)
        self.assertAlmostEqual(limited_command, 7.0, places=6)

    def test_reference_trajectory_covers_expected_phases(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)
        excitation = replace(
            config.excitation,
            duration=3.0,
            hold_start=0.2,
            hold_end=0.2,
        )
        trajectory = build_reference_trajectory(excitation, max_velocity=0.4)
        self.assertEqual(str(trajectory.phase_name[0]), "hold_start")
        self.assertIn("hold_forward_01", set(trajectory.phase_name.tolist()))
        self.assertIn("hold_reverse_01", set(trajectory.phase_name.tolist()))
        self.assertEqual(str(trajectory.phase_name[-1]), "hold_end")
        self.assertLessEqual(float(np.max(np.abs(trajectory.velocity_cmd))), 0.4 + 1.0e-6)
        self.assertAlmostEqual(float(trajectory.velocity_cmd[0]), 0.0, places=6)
        self.assertAlmostEqual(float(trajectory.velocity_cmd[-1]), 0.0, places=6)
        self.assertAlmostEqual(float(trajectory.position_cmd[-1]), 0.0, delta=0.02)

    def test_identification_recovers_synthetic_parameters(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)
        identification_config = replace(
            config.identification,
            min_samples=50,
            validation_stride=7,
            validation_warmup_samples=10,
            savgol_window=21,
            savgol_polyorder=3,
        )
        time_axis = np.linspace(0.0, 10.0, 800, endpoint=False)
        velocity = 0.7 * np.sin(2.0 * np.pi * 0.3 * time_axis)
        dt = time_axis[1] - time_axis[0]
        position = np.cumsum(velocity) * dt
        true_coulomb = 0.85
        true_viscous = 0.12
        true_offset = -0.03
        torque = true_coulomb * np.tanh(velocity / 0.05) + true_viscous * velocity + true_offset
        torque += 0.01 * np.sin(2.0 * np.pi * 0.9 * time_axis)

        capture = RoundCapture(
            group_index=1,
            round_index=1,
            target_motor_id=1,
            motor_name="motor_01",
            time=time_axis,
            motor_id=np.ones_like(time_axis, dtype=np.int64),
            position=position,
            velocity=velocity,
            torque_feedback=torque,
            command_raw=np.zeros_like(time_axis),
            command=np.zeros_like(time_axis),
            position_cmd=np.zeros_like(time_axis),
            velocity_cmd=np.zeros_like(time_axis),
            acceleration_cmd=np.zeros_like(time_axis),
            phase_name=np.full(time_axis.size, "synthetic"),
            state=np.zeros_like(time_axis, dtype=np.uint8),
            mos_temperature=np.full(time_axis.size, 35.0),
            id_match_ok=np.ones_like(time_axis, dtype=bool),
            metadata={},
        )

        result = identify_motor_friction(identification_config, capture)
        self.assertTrue(result.identified)
        self.assertAlmostEqual(result.coulomb, true_coulomb, delta=0.15)
        self.assertAlmostEqual(result.viscous, true_viscous, delta=0.08)
        self.assertAlmostEqual(result.offset, true_offset, delta=0.08)

    def test_end_to_end_pipeline_writes_summary(self) -> None:
        base_config = load_config(DEFAULT_CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1, 3)),
                excitation=replace(
                    base_config.excitation,
                    duration=0.02,
                    hold_start=0.0,
                    hold_end=0.0,
                    sample_rate=50.0,
                ),
                control=replace(
                    base_config.control,
                    max_velocity=np.full_like(base_config.control.max_velocity, 0.1),
                ),
                identification=replace(
                    base_config.identification,
                    group_count=1,
                    min_samples=3,
                    validation_stride=2,
                    validation_warmup_samples=0,
                    savgol_window=5,
                    savgol_polyorder=2,
                ),
                serial=replace(base_config.serial, read_chunk_size=512, read_timeout=0.001),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )

            result = run_sequential_identification(
                config,
                transport_factory=lambda: FakeTransport(config.motor_ids),
            )

            self.assertEqual(len(result.artifacts), 2)
            self.assertTrue(result.summary_paths.root_summary_path.exists())
            self.assertTrue(result.summary_paths.manifest_path.exists())
            self.assertTrue(result.summary_paths.rerun_recording_path.exists())

            manifest = json.loads(result.summary_paths.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["motor_order"], [1, 3])
            self.assertEqual(len(manifest["capture_files"]), 2)
            self.assertEqual(len(manifest["identification_files"]), 2)
            self.assertEqual(manifest["rerun_recording_path"], str(result.summary_paths.rerun_recording_path))

            with np.load(result.summary_paths.root_summary_path, allow_pickle=False) as summary:
                self.assertEqual(summary["motor_ids"].tolist(), [1, 2, 3, 4, 5, 6, 7])
                self.assertEqual(summary["identified_mask"].shape[0], 7)

            with np.load(Path(manifest["capture_files"][0]), allow_pickle=False) as capture:
                self.assertIn("command_raw", capture.files)


if __name__ == "__main__":
    unittest.main()
