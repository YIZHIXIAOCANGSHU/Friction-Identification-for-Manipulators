from __future__ import annotations

import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from friction_identification_core.__main__ import build_parser, main
from friction_identification_core.config import DEFAULT_CONFIG_PATH, ExcitationPlatformConfig, load_config
from friction_identification_core.controller import SingleMotorController
from friction_identification_core.identification import identify_motor_friction
from friction_identification_core.models import MotorCompensationParameters, ReferenceSample, RoundCapture
from friction_identification_core.pipeline import run_compensation_validation, run_sequential_identification
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
from friction_identification_core.visualization import RerunRecorder


README_PATH = DEFAULT_CONFIG_PATH.parents[1] / "README.md"


class SequencedFakeTransport:
    def __init__(self, motor_ids: tuple[int, ...], *, bad_cycles: int = 0, sleep_s: float = 0.001) -> None:
        self._motor_ids = tuple(int(motor_id) for motor_id in motor_ids)
        self._bad_cycles = int(bad_cycles)
        self._sleep_s = float(sleep_s)
        self._step = 0
        self._cycle_index = 0
        self._pending = bytearray()
        self.writes: list[bytes] = []
        self.closed = False

    def _cycle_motor_ids(self) -> tuple[int, ...]:
        if self._cycle_index < self._bad_cycles:
            first, *rest = self._motor_ids
            return (first, first, *rest)
        return self._motor_ids

    def _build_cycle_bytes(self) -> bytes:
        frames = bytearray()
        for motor_id in self._cycle_motor_ids():
            angle = 0.08 * self._step + 0.05 * motor_id
            position = 0.2 * np.sin(angle)
            velocity = 0.25 * np.sin(0.4 * angle)
            torque = 0.85 * np.tanh(velocity / 0.05) + 0.09 * velocity + 0.01 * motor_id
            frames.extend(
                RECV_FRAME_STRUCT.pack(
                    RECV_FRAME_HEAD,
                    int(motor_id),
                    1,
                    float(position),
                    float(velocity),
                    float(torque),
                    float(30.0 + motor_id),
                )
            )
            self._step += 1
        self._cycle_index += 1
        return bytes(frames)

    def read(self, size: int) -> bytes:
        if self._sleep_s > 0.0:
            time.sleep(self._sleep_s)
        while len(self._pending) < int(size):
            self._pending.extend(self._build_cycle_bytes())
        chunk = bytes(self._pending[:size])
        del self._pending[:size]
        return chunk

    def write(self, payload: bytes) -> int:
        self.writes.append(payload)
        return len(payload)

    def reset_input_buffer(self) -> None:
        self._pending.clear()

    def close(self) -> None:
        self.closed = True


class OutOfOrderFakeTransport(SequencedFakeTransport):
    def _cycle_motor_ids(self) -> tuple[int, ...]:
        return (1, 3, 2, 4, 5, 7, 6)


def _platform(
    speed: float | None,
    settle_duration: float,
    steady_duration: float,
    *,
    speed_ratio: float | None = None,
) -> ExcitationPlatformConfig:
    return ExcitationPlatformConfig(
        speed=None if speed is None else float(speed),
        speed_ratio=None if speed_ratio is None else float(speed_ratio),
        settle_duration=float(settle_duration),
        steady_duration=float(steady_duration),
    )


def _identification_config(**kwargs: object):
    config = load_config(DEFAULT_CONFIG_PATH).identification
    overrides = {
        "min_samples": 20,
        "min_samples_per_platform": 20,
        "min_direction_samples": 20,
        "min_motion_span": 0.01,
        "savgol_window": 11,
        "savgol_polyorder": 3,
    }
    overrides.update(kwargs)
    return replace(config, **overrides)


def _build_capture(
    phase_blocks: list[tuple[str, float, int]],
    *,
    motor_id: int = 1,
    dt: float = 0.01,
    max_torque: float = 5.0,
    max_velocity: float = 0.5,
    metadata: dict[str, object] | None = None,
) -> RoundCapture:
    phase_values: list[str] = []
    velocity_cmd_values: list[float] = []
    velocity_values: list[float] = []
    previous_velocity = 0.0
    for phase_name, target_velocity, sample_count in phase_blocks:
        if sample_count <= 0:
            continue
        if phase_name.startswith("settle_") or phase_name.startswith("transition_"):
            command_block = np.linspace(previous_velocity, target_velocity, sample_count, endpoint=False, dtype=np.float64)
        else:
            command_block = np.full(sample_count, target_velocity, dtype=np.float64)
        if phase_name.startswith("steady_"):
            phase_index = np.arange(sample_count, dtype=np.float64)
            velocity_block = command_block + 0.003 * np.sin(0.4 * phase_index)
        else:
            velocity_block = command_block.copy()
        phase_values.extend([phase_name] * sample_count)
        velocity_cmd_values.extend(command_block.tolist())
        velocity_values.extend(velocity_block.tolist())
        previous_velocity = float(target_velocity)

    time_axis = np.arange(len(phase_values), dtype=np.float64) * float(dt)
    velocity_cmd = np.asarray(velocity_cmd_values, dtype=np.float64)
    velocity = np.asarray(velocity_values, dtype=np.float64)
    position = np.zeros_like(time_axis)
    if time_axis.size > 1:
        position[1:] = np.cumsum((velocity[:-1] + velocity[1:]) * 0.5 * float(dt))
    torque = 0.82 * np.tanh(velocity / 0.05) + 0.11 * velocity - 0.02
    torque += 0.004 * np.cos(np.arange(time_axis.size, dtype=np.float64) * 0.3)

    base_metadata: dict[str, object] = {
        "synced_before_capture": True,
        "sequence_error_count": 0,
        "sequence_error_ratio": 0.0,
        "target_frame_count": int(time_axis.size),
        "target_frame_ratio": 1.0,
        "target_max_torque": float(max_torque),
        "target_max_velocity": float(max_velocity),
    }
    if metadata:
        base_metadata.update(metadata)

    return RoundCapture(
        group_index=1,
        round_index=1,
        target_motor_id=int(motor_id),
        motor_name=f"motor_{int(motor_id):02d}",
        time=time_axis,
        motor_id=np.full(time_axis.size, int(motor_id), dtype=np.int64),
        position=position,
        velocity=velocity,
        torque_feedback=torque,
        command_raw=np.zeros_like(time_axis),
        command=np.zeros_like(time_axis),
        position_cmd=np.zeros_like(time_axis),
        velocity_cmd=velocity_cmd,
        acceleration_cmd=np.gradient(velocity_cmd, float(dt)) if time_axis.size else np.zeros(0, dtype=np.float64),
        phase_name=np.asarray(phase_values),
        state=np.zeros(time_axis.size, dtype=np.uint8),
        mos_temperature=np.full(time_axis.size, 35.0, dtype=np.float64),
        id_match_ok=np.ones(time_axis.size, dtype=bool),
        metadata=base_metadata,
    )


def _write_runtime_summary(
    path: Path,
    *,
    motor_ids: tuple[int, ...],
    recommended_motor_ids: tuple[int, ...] | None = None,
) -> Path:
    recommended_set = set(motor_ids if recommended_motor_ids is None else recommended_motor_ids)
    motor_id_array = np.asarray(motor_ids, dtype=np.int64)
    motor_names = np.asarray([f"motor_{motor_id:02d}" for motor_id in motor_ids])
    recommended = np.asarray([motor_id in recommended_set for motor_id in motor_ids], dtype=bool)
    base = np.asarray([0.85 + 0.01 * index for index, _motor_id in enumerate(motor_ids)], dtype=np.float64)
    np.savez(
        path,
        motor_ids=motor_id_array,
        motor_names=motor_names,
        recommended_for_runtime=recommended,
        coulomb=base.copy(),
        viscous=0.1 * base,
        offset=np.linspace(-0.02, 0.02, num=len(motor_ids), dtype=np.float64),
        velocity_scale=np.full(len(motor_ids), 0.05, dtype=np.float64),
        validation_rmse=np.full(len(motor_ids), 0.01, dtype=np.float64),
        validation_r2=np.full(len(motor_ids), 0.95, dtype=np.float64),
        conclusion_level=np.asarray(["recommended" if flag else "caution" for flag in recommended]),
        conclusion_text=np.asarray(["ok" if flag else "not recommended" for flag in recommended]),
        high_speed_valid_rmse=np.full(len(motor_ids), 0.02, dtype=np.float64),
        high_speed_platform_count=np.full(len(motor_ids), 2, dtype=np.int64),
        saturation_ratio=np.zeros(len(motor_ids), dtype=np.float64),
        tracking_error_ratio=np.zeros(len(motor_ids), dtype=np.float64),
    )
    return path


class SequentialIdentificationTests(unittest.TestCase):
    def test_default_config_exposes_explicit_excitation_platforms(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)

        self.assertGreater(config.excitation.sample_rate, 0.0)
        self.assertGreaterEqual(config.excitation.hold_start, 0.0)
        self.assertGreaterEqual(config.excitation.hold_end, 0.0)
        self.assertGreaterEqual(config.excitation.transition_duration, 0.0)
        self.assertEqual(len(config.excitation.platforms), 5)
        self.assertEqual(
            tuple(platform.speed_ratio for platform in config.excitation.platforms),
            (0.12, 0.28, 0.50, 0.72, 0.90),
        )
        self.assertTrue(all(platform.speed is None for platform in config.excitation.platforms))
        self.assertTrue(all(platform.settle_duration > 0.0 for platform in config.excitation.platforms))
        self.assertTrue(all(platform.steady_duration > 0.0 for platform in config.excitation.platforms))

    def test_config_rejects_invalid_excitation_platforms(self) -> None:
        cases = {
            "missing_platforms": """
excitation:
  sample_rate: 200.0
  hold_start: 0.0
  hold_end: 0.0
  transition_duration: 0.1
""",
            "negative_hold_start": """
excitation:
  sample_rate: 200.0
  hold_start: -0.1
  hold_end: 0.0
  transition_duration: 0.1
  platforms:
    - speed: 0.08
      settle_duration: 0.2
      steady_duration: 0.3
""",
            "negative_transition_duration": """
excitation:
  sample_rate: 200.0
  hold_start: 0.0
  hold_end: 0.0
  transition_duration: -0.1
  platforms:
    - speed: 0.08
      settle_duration: 0.2
      steady_duration: 0.3
""",
            "zero_speed": """
excitation:
  sample_rate: 200.0
  hold_start: 0.0
  hold_end: 0.0
  transition_duration: 0.1
  platforms:
    - speed: 0.0
      settle_duration: 0.2
      steady_duration: 0.3
""",
            "missing_speed_and_ratio": """
excitation:
  sample_rate: 200.0
  hold_start: 0.0
  hold_end: 0.0
  transition_duration: 0.1
  platforms:
    - settle_duration: 0.2
      steady_duration: 0.3
""",
            "speed_and_ratio_together": """
excitation:
  sample_rate: 200.0
  hold_start: 0.0
  hold_end: 0.0
  transition_duration: 0.1
  platforms:
    - speed: 0.08
      speed_ratio: 0.3
      settle_duration: 0.2
      steady_duration: 0.3
""",
            "ratio_too_high": """
excitation:
  sample_rate: 200.0
  hold_start: 0.0
  hold_end: 0.0
  transition_duration: 0.1
  platforms:
    - speed_ratio: 0.95
      settle_duration: 0.2
      steady_duration: 0.3
""",
            "zero_steady_duration": """
excitation:
  sample_rate: 200.0
  hold_start: 0.0
  hold_end: 0.0
  transition_duration: 0.1
  platforms:
    - speed: 0.08
      settle_duration: 0.2
      steady_duration: 0.0
""",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            for case_name, payload in cases.items():
                with self.subTest(case=case_name):
                    config_path.write_text(payload, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_config(config_path)

    def test_readme_and_cli_match_runtime_modes(self) -> None:
        parser = build_parser()
        mode_action = next(action for action in parser._actions if action.dest == "mode")
        self.assertEqual(tuple(mode_action.choices), ("identify", "compensate", "default", "sequential"))

        readme = README_PATH.read_text(encoding="utf-8")
        self.assertIn("CLI 主模式是 `identify` 和 `compensate`", readme)
        self.assertIn("`default` 和 `sequential` 仅作为 `identify` 的兼容别名", readme)

    def test_main_enables_rerun_viewer_for_identify_mode(self) -> None:
        loaded_config = object()
        configured = object()

        with (
            mock.patch("friction_identification_core.__main__.load_config", return_value=loaded_config),
            mock.patch("friction_identification_core.__main__.apply_overrides", return_value=configured),
            mock.patch("friction_identification_core.__main__.run_sequential_identification") as run_mock,
            mock.patch("friction_identification_core.__main__.run_compensation_validation") as compensate_mock,
        ):
            main([])

        run_mock.assert_called_once_with(configured, show_rerun_viewer=True)
        compensate_mock.assert_not_called()

    def test_main_routes_compensate_mode_to_compensation_runner(self) -> None:
        loaded_config = object()
        configured = object()

        with (
            mock.patch("friction_identification_core.__main__.load_config", return_value=loaded_config),
            mock.patch("friction_identification_core.__main__.apply_overrides", return_value=configured),
            mock.patch("friction_identification_core.__main__.run_sequential_identification") as run_mock,
            mock.patch("friction_identification_core.__main__.run_compensation_validation") as compensate_mock,
        ):
            main(["--mode", "compensate"])

        run_mock.assert_not_called()
        compensate_mock.assert_called_once_with(configured, show_rerun_viewer=True)

    def test_main_maps_legacy_aliases_to_identify_mode(self) -> None:
        loaded_config = object()
        configured = object()

        with (
            mock.patch("friction_identification_core.__main__.load_config", return_value=loaded_config),
            mock.patch("friction_identification_core.__main__.apply_overrides", return_value=configured),
            mock.patch("friction_identification_core.__main__.run_sequential_identification") as run_mock,
            mock.patch("friction_identification_core.__main__.run_compensation_validation") as compensate_mock,
        ):
            main(["--mode", "default"])

        run_mock.assert_called_once_with(configured, show_rerun_viewer=True)
        compensate_mock.assert_not_called()

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

    def test_parser_reset_clears_partial_buffer(self) -> None:
        parser = SerialFrameParser(max_motor_id=7)
        payload = RECV_FRAME_STRUCT.pack(RECV_FRAME_HEAD, 4, 1, 0.1, 0.2, 0.3, 25.0)
        parser.feed(payload[:7])
        parser.reset()
        self.assertIsNone(parser.pop_frame())
        parser.feed(payload)
        frame = parser.pop_frame()
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame.motor_id, 4)

    def test_parser_accepts_valid_frame_without_following_frame_head(self) -> None:
        parser = SerialFrameParser(max_motor_id=7)
        payload = RECV_FRAME_STRUCT.pack(RECV_FRAME_HEAD, 2, 1, 0.1, 0.2, 0.3, 25.0)
        parser.feed(payload + b"\x01\x02\x03\x04")
        frame = parser.pop_frame()
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame.motor_id, 2)
        self.assertAlmostEqual(frame.velocity, 0.2, places=6)

    def test_controller_clips_raw_output_to_motor_limit(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)
        controller = SingleMotorController(config)
        motor_index = config.motor_index(5)
        reference = ReferenceSample(
            position_cmd=0.0,
            velocity_cmd=float(config.control.max_velocity[motor_index]) * 2.0,
            acceleration_cmd=0.0,
            phase_name="test",
        )
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

    def test_controller_adds_friction_feedforward_in_compensation_mode(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)
        controller = SingleMotorController(config)
        params = MotorCompensationParameters(
            motor_id=5,
            motor_name="motor_05",
            coulomb=0.8,
            viscous=0.2,
            offset=0.1,
            velocity_scale=0.05,
        )
        reference = ReferenceSample(
            position_cmd=0.0,
            velocity_cmd=1.0,
            acceleration_cmd=0.0,
            phase_name="steady_forward_01",
        )
        feedback = FeedbackFrame(
            motor_id=5,
            state=1,
            position=0.0,
            velocity=0.8,
            torque=0.0,
            mos_temperature=30.0,
        )

        raw_command, limited_command = controller.update(5, reference, feedback, compensation=params)

        self.assertGreater(raw_command, 0.0)
        self.assertAlmostEqual(raw_command, limited_command, places=6)
        self.assertGreater(raw_command, 0.25)

    def test_capture_waits_for_sync_before_sampling(self) -> None:
        base_config = load_config(DEFAULT_CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1,)),
                excitation=replace(
                    base_config.excitation,
                    hold_start=0.0,
                    hold_end=0.0,
                    sample_rate=40.0,
                    transition_duration=0.01,
                    platforms=(
                        _platform(0.08, 0.02, 0.02),
                    ),
                ),
                identification=replace(
                    base_config.identification,
                    min_samples=1,
                    min_motion_span=0.0,
                    min_target_frame_ratio=0.1,
                    max_sequence_error_ratio=0.6,
                ),
                serial=replace(
                    base_config.serial,
                    read_chunk_size=1024,
                    read_timeout=0.001,
                    sync_timeout=0.08,
                    sync_cycles_required=2,
                ),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )

            result = run_sequential_identification(
                config,
                transport_factory=lambda: SequencedFakeTransport(config.motor_ids, bad_cycles=3, sleep_s=0.001),
            )

            capture = result.artifacts[0].capture
            metadata = capture.metadata
            self.assertTrue(metadata["synced_before_capture"])
            self.assertGreater(float(metadata["sync_wait_duration_s"]), 0.0)
            self.assertEqual(int(metadata["sequence_error_count"]), 0)
            self.assertGreater(float(metadata["planned_duration_s"]), 0.0)
            self.assertGreater(float(metadata["actual_capture_duration_s"]), 0.0)
            self.assertGreater(float(metadata["round_total_duration_s"]), 0.0)
            self.assertLess(float(capture.time[0]), 0.02)

    def test_capture_starts_after_target_motor_feedback_even_when_bus_order_is_noisy(self) -> None:
        base_config = load_config(DEFAULT_CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1,)),
                excitation=replace(
                    base_config.excitation,
                    hold_start=0.0,
                    hold_end=0.0,
                    sample_rate=40.0,
                    transition_duration=0.01,
                    platforms=(
                        _platform(0.08, 0.02, 0.06),
                    ),
                ),
                identification=replace(
                    base_config.identification,
                    min_samples=1,
                    min_motion_span=0.0,
                    min_target_frame_ratio=0.1,
                    max_sequence_error_ratio=1.0,
                ),
                serial=replace(
                    base_config.serial,
                    read_chunk_size=1024,
                    read_timeout=0.001,
                    sync_timeout=0.08,
                    sync_cycles_required=3,
                ),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )

            result = run_sequential_identification(
                config,
                transport_factory=lambda: OutOfOrderFakeTransport(config.motor_ids, sleep_s=0.001),
            )

            capture = result.artifacts[0].capture
            metadata = capture.metadata
            self.assertTrue(metadata["synced_before_capture"])
            self.assertEqual(int(metadata["sync_required_target_frames"]), 3)
            self.assertGreaterEqual(int(metadata["target_sync_frame_count"]), 3)
            self.assertEqual(int(metadata["sequence_error_count"]), 0)
            self.assertEqual(float(metadata["sequence_error_ratio"]), 0.0)
            self.assertGreater(capture.sample_count, 0)
            self.assertGreater(float(np.max(np.abs(capture.command))), 0.0)

    def test_reference_trajectory_uses_platform_settle_and_steady_phases(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)
        excitation = replace(
            config.excitation,
            hold_start=0.2,
            hold_end=0.2,
            sample_rate=100.0,
            transition_duration=0.1,
            platforms=(
                _platform(None, 0.1, 0.2, speed_ratio=0.20),
                _platform(None, 0.1, 0.2, speed_ratio=0.30),
                _platform(None, 0.2, 0.3, speed_ratio=0.45),
                _platform(None, 0.2, 0.4, speed_ratio=0.90),
            ),
        )
        trajectory = build_reference_trajectory(excitation, max_velocity=0.4)
        phase_names = trajectory.phase_name.tolist()
        ordered_phases: list[str] = []
        for phase_name in phase_names:
            if ordered_phases and ordered_phases[-1] == str(phase_name):
                continue
            ordered_phases.append(str(phase_name))

        self.assertEqual(
            ordered_phases,
            [
                "hold_start",
                "settle_forward_01",
                "steady_forward_01",
                "settle_forward_02",
                "steady_forward_02",
                "settle_forward_03",
                "steady_forward_03",
                "settle_forward_04",
                "steady_forward_04",
                "transition_mid_zero",
                "settle_reverse_01",
                "steady_reverse_01",
                "settle_reverse_02",
                "steady_reverse_02",
                "settle_reverse_03",
                "steady_reverse_03",
                "settle_reverse_04",
                "steady_reverse_04",
                "transition_end_zero",
                "hold_end",
            ],
        )
        self.assertAlmostEqual(float(trajectory.duration_s), 4.0, places=6)
        self.assertEqual(int(np.count_nonzero(trajectory.phase_name == "hold_start")), 20)
        self.assertEqual(int(np.count_nonzero(trajectory.phase_name == "transition_mid_zero")), 10)
        self.assertEqual(int(np.count_nonzero(trajectory.phase_name == "hold_end")), 20)

        expected_forward_levels = [0.08, 0.12, 0.18, 0.36]
        for level_index, expected_speed in enumerate(expected_forward_levels, start=1):
            mask = trajectory.phase_name == f"steady_forward_{level_index:02d}"
            self.assertEqual(int(np.count_nonzero(mask)), int(excitation.platforms[level_index - 1].steady_duration * 100.0))
            self.assertAlmostEqual(float(np.median(trajectory.velocity_cmd[mask])), expected_speed, places=6)
            reverse_mask = trajectory.phase_name == f"steady_reverse_{level_index:02d}"
            self.assertAlmostEqual(float(np.median(trajectory.velocity_cmd[reverse_mask])), -expected_speed, places=6)
        self.assertAlmostEqual(float(trajectory.position_cmd[-1]), 0.0, delta=0.05)

    def test_reference_trajectory_rejects_platform_above_ninety_percent_max_velocity(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)
        excitation = replace(
            config.excitation,
            sample_rate=100.0,
            hold_start=0.0,
            hold_end=0.0,
            transition_duration=0.1,
            platforms=(
                _platform(None, 0.1, 0.2, speed_ratio=0.95),
            ),
        )

        with self.assertRaises(ValueError):
            build_reference_trajectory(excitation, max_velocity=0.4)

    def test_identification_uses_only_steady_state_samples(self) -> None:
        capture = _build_capture(
            [
                ("hold_start", 0.0, 10),
                ("settle_forward_01", 0.08, 12),
                ("steady_forward_01", 0.08, 30),
                ("transition_mid_zero", 0.0, 10),
                ("settle_reverse_01", -0.08, 12),
                ("steady_reverse_01", -0.08, 30),
                ("hold_end", 0.0, 10),
            ]
        )
        result = identify_motor_friction(_identification_config(min_samples=20), capture)

        self.assertTrue(result.identified)
        self.assertTrue(np.array_equal(result.sample_mask, result.steady_state_mask))
        used_phases = {str(phase) for phase in capture.phase_name[result.steady_state_mask]}
        self.assertEqual(used_phases, {"steady_forward_01", "steady_reverse_01"})
        self.assertFalse(np.any(result.steady_state_mask[capture.phase_name == "settle_forward_01"]))
        self.assertFalse(np.any(result.steady_state_mask[capture.phase_name == "transition_mid_zero"]))

    def test_platform_holdout_validation_tracks_platform_coverage(self) -> None:
        capture = _build_capture(
            [
                ("hold_start", 0.0, 10),
                ("settle_forward_01", 0.08, 10),
                ("steady_forward_01", 0.08, 140),
                ("settle_forward_02", 0.18, 10),
                ("steady_forward_02", 0.18, 140),
                ("settle_forward_03", 0.25, 10),
                ("steady_forward_03", 0.25, 140),
                ("settle_forward_04", 0.33, 10),
                ("steady_forward_04", 0.33, 140),
                ("settle_forward_05", 0.45, 10),
                ("steady_forward_05", 0.45, 140),
                ("transition_mid_zero", 0.0, 10),
                ("settle_reverse_01", -0.08, 10),
                ("steady_reverse_01", -0.08, 140),
                ("settle_reverse_02", -0.18, 10),
                ("steady_reverse_02", -0.18, 140),
                ("settle_reverse_03", -0.25, 10),
                ("steady_reverse_03", -0.25, 140),
                ("settle_reverse_04", -0.33, 10),
                ("steady_reverse_04", -0.33, 140),
                ("settle_reverse_05", -0.45, 10),
                ("steady_reverse_05", -0.45, 140),
                ("hold_end", 0.0, 10),
            ],
            max_velocity=0.5,
        )
        result = identify_motor_friction(
            _identification_config(min_samples=40, min_samples_per_platform=120, min_direction_samples=300),
            capture,
        )

        self.assertTrue(result.identified)
        self.assertEqual(result.metadata["validation_mode"], "platform_holdout")
        train_platforms = tuple(result.metadata["train_platforms"])
        valid_platforms = tuple(result.metadata["valid_platforms"])
        self.assertTrue(train_platforms)
        self.assertTrue(valid_platforms)
        self.assertTrue(set(train_platforms).isdisjoint(valid_platforms))

        valid_phase_names = {item.split("@", 1)[0] for item in valid_platforms}
        train_phase_names = {item.split("@", 1)[0] for item in train_platforms}
        self.assertEqual(valid_phase_names, {"steady_forward_03", "steady_forward_05", "steady_reverse_03", "steady_reverse_05"})
        self.assertEqual({str(phase) for phase in capture.phase_name[result.valid_mask]}, valid_phase_names)
        self.assertTrue({str(phase) for phase in capture.phase_name[result.train_mask]}.issubset(train_phase_names))

    def test_identification_drops_high_speed_platforms_when_tracking_or_saturation_is_bad(self) -> None:
        capture = _build_capture(
            [
                ("steady_forward_01", 0.08, 140),
                ("steady_forward_02", 0.18, 140),
                ("steady_forward_03", 0.25, 140),
                ("steady_forward_04", 0.36, 140),
                ("steady_forward_05", 0.45, 140),
                ("steady_reverse_01", -0.08, 140),
                ("steady_reverse_02", -0.18, 140),
                ("steady_reverse_03", -0.25, 140),
                ("steady_reverse_04", -0.36, 140),
                ("steady_reverse_05", -0.45, 140),
            ],
            max_torque=5.0,
            max_velocity=0.5,
        )
        bad_velocity = capture.velocity.copy()
        forward_high_mask = capture.phase_name == "steady_forward_05"
        bad_velocity[forward_high_mask] = capture.velocity_cmd[forward_high_mask] + 0.08
        bad_command_raw = capture.command_raw.copy()
        bad_command = capture.command.copy()
        reverse_high_mask = capture.phase_name == "steady_reverse_05"
        bad_command_raw[reverse_high_mask] = 4.95
        bad_command[reverse_high_mask] = 4.95
        degraded_capture = replace(
            capture,
            velocity=bad_velocity,
            command_raw=bad_command_raw,
            command=bad_command,
        )

        result = identify_motor_friction(
            _identification_config(min_samples=40, min_samples_per_platform=120, min_direction_samples=300),
            degraded_capture,
        )

        self.assertTrue(result.identified)
        dropped_platforms = tuple(result.metadata["dropped_platforms"])
        self.assertTrue(any("steady_forward_05" in item for item in dropped_platforms))
        self.assertTrue(any("steady_reverse_05" in item for item in dropped_platforms))
        self.assertLess(int(result.metadata["high_speed_platform_count"]), 2)
        self.assertEqual(result.metadata["conclusion_level"], "caution")
        self.assertFalse(bool(result.metadata["recommended_for_runtime"]))

    def test_train_only_identification_is_marked_as_caution(self) -> None:
        capture = _build_capture(
            [
                ("steady_forward_01", 0.08, 160),
                ("steady_forward_02", 0.18, 160),
                ("steady_forward_03", 0.25, 160),
                ("steady_reverse_01", -0.08, 160),
                ("steady_reverse_02", -0.18, 160),
                ("steady_reverse_03", -0.25, 160),
            ],
            max_velocity=0.5,
        )

        result = identify_motor_friction(
            _identification_config(min_samples=40, min_samples_per_platform=120, min_direction_samples=300),
            capture,
        )

        self.assertTrue(result.identified)
        self.assertEqual(result.metadata["validation_mode"], "train_only")
        self.assertEqual(result.metadata["conclusion_level"], "caution")
        self.assertFalse(bool(result.metadata["recommended_for_runtime"]))

    def test_identification_returns_clear_failure_statuses(self) -> None:
        cases = {
            "sync_not_acquired": _build_capture(
                [("steady_forward_01", 0.08, 20), ("steady_reverse_01", -0.08, 20)],
                metadata={"synced_before_capture": False},
            ),
            "insufficient_steady_state_samples": _build_capture(
                [("steady_forward_01", 0.08, 4), ("steady_reverse_01", -0.08, 4)]
            ),
            "insufficient_bidirectional_steady_state": _build_capture(
                [("steady_forward_01", 0.08, 30), ("steady_forward_02", 0.18, 30)]
            ),
        }

        for expected_status, capture in cases.items():
            with self.subTest(status=expected_status):
                result = identify_motor_friction(_identification_config(min_samples=10), capture)
                self.assertFalse(result.identified)
                self.assertEqual(result.metadata["status"], expected_status)

    def test_end_to_end_pipeline_writes_quality_metrics_and_summary(self) -> None:
        base_config = load_config(DEFAULT_CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1, 3)),
                excitation=replace(
                    base_config.excitation,
                    hold_start=0.0,
                    hold_end=0.0,
                    sample_rate=80.0,
                    transition_duration=0.01,
                    platforms=(
                        _platform(None, 0.02, 1.8, speed_ratio=0.12),
                        _platform(None, 0.02, 1.8, speed_ratio=0.28),
                        _platform(None, 0.02, 1.8, speed_ratio=0.50),
                        _platform(None, 0.02, 1.8, speed_ratio=0.72),
                        _platform(None, 0.02, 1.8, speed_ratio=0.90),
                    ),
                ),
                control=replace(
                    base_config.control,
                    max_velocity=np.full_like(base_config.control.max_velocity, 0.25),
                ),
                identification=replace(
                    base_config.identification,
                    min_samples=5,
                    min_motion_span=0.0,
                    min_target_frame_ratio=0.1,
                    max_sequence_error_ratio=0.6,
                    min_samples_per_platform=20,
                    min_direction_samples=40,
                ),
                serial=replace(
                    base_config.serial,
                    read_chunk_size=1024,
                    read_timeout=0.001,
                    sync_timeout=0.1,
                    sync_cycles_required=1,
                ),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )

            result = run_sequential_identification(
                config,
                transport_factory=lambda: SequencedFakeTransport(config.motor_ids, bad_cycles=1, sleep_s=0.001),
            )

            self.assertEqual(len(result.artifacts), 2)
            self.assertTrue(result.summary_paths.root_summary_path.exists())
            self.assertTrue(result.summary_paths.manifest_path.exists())

            manifest = json.loads(result.summary_paths.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["mode"], "identify")
            self.assertEqual(manifest["motor_order"], [1, 3])

            with np.load(result.summary_paths.root_summary_path, allow_pickle=False) as summary:
                self.assertIn("status", summary.files)
                self.assertIn("sequence_error_count", summary.files)
                self.assertIn("target_frame_ratio", summary.files)
                self.assertIn("planned_duration_s", summary.files)
                self.assertIn("actual_capture_duration_s", summary.files)
                self.assertIn("round_total_duration_s", summary.files)
                self.assertIn("synced_before_capture", summary.files)
                self.assertIn("train_platforms", summary.files)
                self.assertIn("valid_platforms", summary.files)
                self.assertIn("recommended_for_runtime", summary.files)
                self.assertIn("conclusion_level", summary.files)
                self.assertIn("conclusion_text", summary.files)
                self.assertIn("saturation_ratio", summary.files)
                self.assertIn("tracking_error_ratio", summary.files)
                self.assertIn("high_speed_platform_count", summary.files)
                self.assertIn("high_speed_valid_rmse", summary.files)
                self.assertEqual(summary["motor_ids"].tolist(), [1, 2, 3, 4, 5, 6, 7])
                self.assertTrue(bool(summary["synced_before_capture"][0]))
                self.assertEqual(int(summary["sequence_error_count"][0]), 0)
                self.assertEqual(summary["recommended_for_runtime"].dtype, np.bool_)

            with np.load(Path(manifest["capture_files"][0]), allow_pickle=False) as capture:
                metadata = json.loads(str(capture["metadata"]))
                self.assertIn("sequence_error_ratio", metadata)
                self.assertIn("target_frame_count", metadata)
                self.assertIn("synced_before_capture", metadata)
                self.assertIn("target_max_torque", metadata)
                self.assertIn("target_max_velocity", metadata)
                self.assertIn("planned_duration_s", metadata)
                self.assertIn("actual_capture_duration_s", metadata)
                self.assertIn("round_total_duration_s", metadata)
                self.assertEqual(int(metadata["sequence_error_count"]), 0)

            with np.load(Path(manifest["identification_files"][0]), allow_pickle=False) as identification:
                self.assertIn("steady_state_mask", identification.files)
                self.assertIn("train_mask", identification.files)
                self.assertIn("valid_mask", identification.files)
                self.assertIn("tracking_ok_mask", identification.files)
                self.assertIn("saturation_ok_mask", identification.files)

            csv_header = result.summary_paths.root_summary_csv_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("recommended_for_runtime", csv_header)
            self.assertIn("conclusion_level", csv_header)
            self.assertIn("high_speed_valid_rmse", csv_header)

            report_text = result.summary_paths.root_summary_report_path.read_text(encoding="utf-8")
            self.assertIn("## Runtime Conclusions", report_text)
            self.assertIn("recommended_for_runtime", report_text)

    def test_compensation_mode_requires_recommended_runtime_parameters(self) -> None:
        base_config = load_config(DEFAULT_CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            params_path = _write_runtime_summary(
                Path(tmpdir) / "summary.npz",
                motor_ids=(1, 3),
                recommended_motor_ids=(1,),
            )
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1, 3)),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )

            with self.assertRaises(ValueError):
                run_compensation_validation(
                    config,
                    transport_factory=lambda: SequencedFakeTransport(config.motor_ids, sleep_s=0.001),
                    parameters_path=params_path,
                )

    def test_compensation_mode_uses_summary_parameters_and_skips_identification_files(self) -> None:
        base_config = load_config(DEFAULT_CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            params_path = _write_runtime_summary(
                Path(tmpdir) / "summary.npz",
                motor_ids=(1, 3),
                recommended_motor_ids=(1, 3),
            )
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1, 3)),
                excitation=replace(
                    base_config.excitation,
                    hold_start=0.0,
                    hold_end=0.0,
                    sample_rate=60.0,
                    transition_duration=0.01,
                    platforms=(
                        _platform(None, 0.02, 0.06, speed_ratio=0.12),
                    ),
                ),
                identification=replace(
                    base_config.identification,
                    group_count=1,
                    min_samples=1,
                    min_motion_span=0.0,
                    min_target_frame_ratio=0.1,
                ),
                serial=replace(
                    base_config.serial,
                    read_chunk_size=1024,
                    read_timeout=0.001,
                    sync_timeout=0.08,
                    sync_cycles_required=1,
                ),
                control=replace(
                    base_config.control,
                    max_velocity=np.full_like(base_config.control.max_velocity, 0.25),
                ),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )

            result = run_compensation_validation(
                config,
                transport_factory=lambda: SequencedFakeTransport(config.motor_ids, sleep_s=0.001),
                parameters_path=params_path,
            )

            self.assertEqual(len(result.artifacts), 2)
            self.assertIsNone(result.summary_paths)

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["mode"], "compensate")
            self.assertEqual(manifest["motor_order"], [1, 3])
            self.assertEqual(manifest["identification_files"], [])
            self.assertEqual(len(manifest["capture_files"]), 2)
            self.assertIn("compensation_parameters_path", manifest)

            with np.load(Path(manifest["capture_files"][0]), allow_pickle=False) as capture:
                metadata = json.loads(str(capture["metadata"]))
                self.assertEqual(metadata["mode"], "compensate")
                self.assertIn("tracking_velocity_rmse", metadata)
                self.assertIn("planned_duration_s", metadata)
                self.assertIn("actual_capture_duration_s", metadata)
                self.assertGreater(float(metadata["tracking_velocity_rmse"]), 0.0)

    def test_rerun_recorder_logs_quality_and_conclusion_paths(self) -> None:
        logged_paths: list[str] = []
        spawn_calls: list[dict[str, object]] = []

        class _FakeRecordingStream:
            def __init__(self, _name: str) -> None:
                self.saved_path = None

            def save(self, path: Path) -> None:
                self.saved_path = Path(path)

            def spawn(self, **kwargs) -> None:
                spawn_calls.append(dict(kwargs))

            def send_blueprint(self, *_args, **_kwargs) -> None:
                return None

            def log(self, path: str, _payload, static: bool = False) -> None:
                _ = static
                logged_paths.append(path)

            def set_time(self, *_args, **_kwargs) -> None:
                return None

            def disconnect(self) -> None:
                return None

        fake_rr = SimpleNamespace(
            RecordingStream=_FakeRecordingStream,
            TextDocument=lambda text, media_type=None: {"text": text, "media_type": media_type},
            TextLog=lambda text: {"text": text},
            SeriesLines=lambda names=None: {"names": tuple(names or ())},
            Scalars=lambda values: {"values": tuple(values)},
            BarChart=lambda values, abscissa=None: {"values": tuple(values), "abscissa": tuple(abscissa or ())},
            blueprint=SimpleNamespace(
                Vertical=lambda *args, **kwargs: ("Vertical", args, kwargs),
                Horizontal=lambda *args, **kwargs: ("Horizontal", args, kwargs),
                Tabs=lambda *args, **kwargs: ("Tabs", args, kwargs),
                TextDocumentView=lambda **kwargs: ("TextDocumentView", kwargs),
                TimeSeriesView=lambda **kwargs: ("TimeSeriesView", kwargs),
                BarChartView=lambda **kwargs: ("BarChartView", kwargs),
                Blueprint=lambda *args, **kwargs: ("Blueprint", args, kwargs),
            ),
        )

        capture = _build_capture(
            [
                ("steady_forward_01", 0.08, 140),
                ("steady_forward_02", 0.18, 140),
                ("steady_forward_03", 0.25, 140),
                ("steady_forward_04", 0.36, 140),
                ("steady_forward_05", 0.45, 140),
                ("steady_reverse_01", -0.08, 140),
                ("steady_reverse_02", -0.18, 140),
                ("steady_reverse_03", -0.25, 140),
                ("steady_reverse_04", -0.36, 140),
                ("steady_reverse_05", -0.45, 140),
            ],
            max_velocity=0.5,
        )
        result = identify_motor_friction(
            _identification_config(min_samples=40, min_samples_per_platform=120, min_direction_samples=300),
            capture,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.npz"
            report_path = Path(tmpdir) / "report.md"
            np.savez(
                summary_path,
                motor_ids=np.asarray([1], dtype=np.int64),
                coulomb=np.asarray([result.coulomb], dtype=np.float64),
                viscous=np.asarray([result.viscous], dtype=np.float64),
                offset=np.asarray([result.offset], dtype=np.float64),
                velocity_scale=np.asarray([result.velocity_scale], dtype=np.float64),
                validation_rmse=np.asarray([result.valid_rmse], dtype=np.float64),
                validation_r2=np.asarray([result.valid_r2], dtype=np.float64),
                high_speed_valid_rmse=np.asarray([result.metadata["high_speed_valid_rmse"]], dtype=np.float64),
                high_speed_platform_count=np.asarray([result.metadata["high_speed_platform_count"]], dtype=np.int64),
                saturation_ratio=np.asarray([result.metadata["saturation_ratio"]], dtype=np.float64),
                tracking_error_ratio=np.asarray([result.metadata["tracking_error_ratio"]], dtype=np.float64),
                recommended_for_runtime=np.asarray([result.metadata["recommended_for_runtime"]], dtype=bool),
                conclusion_level=np.asarray([result.metadata["conclusion_level"]]),
                conclusion_text=np.asarray([result.metadata["conclusion_text"]]),
                motor_names=np.asarray([capture.motor_name]),
            )
            report_path.write_text("# report\n", encoding="utf-8")

            with mock.patch("friction_identification_core.visualization.rr", fake_rr):
                recorder = RerunRecorder(
                    summary_path.with_suffix(".rrd"),
                    motor_ids=(1,),
                    motor_names={1: capture.motor_name},
                    show_viewer=True,
                )
                recorder.log_identification(capture, result)
                recorder.log_summary(summary_path=summary_path, report_path=report_path)
                recorder.close()

        expected_paths = {
            "rounds/group_01/motor_01/round_01/signals/velocity_error",
            "rounds/group_01/motor_01/round_01/signals/saturation_flag",
            "rounds/group_01/motor_01/round_01/identification/residual",
            "rounds/group_01/motor_01/round_01/identification/sample_masks",
            "rounds/group_01/motor_01/round_01/quality/summary",
            "summary/high_speed_valid_rmse",
            "summary/saturation_ratio",
            "summary/tracking_error_ratio",
            "summary/recommended_for_runtime",
            "summary/conclusions",
        }
        self.assertEqual(spawn_calls, [{"connect": True, "detach_process": True}])
        self.assertTrue(expected_paths.issubset(set(logged_paths)))

    def test_rerun_recorder_uses_by_motor_blueprint_and_logs_sent_commands(self) -> None:
        logged_entries: list[tuple[str, object, bool]] = []
        blueprint_calls: list[tuple[object, dict[str, object]]] = []

        class _FakeRecordingStream:
            def __init__(self, _name: str) -> None:
                self.saved_path = None

            def save(self, path: Path) -> None:
                self.saved_path = Path(path)

            def send_blueprint(self, blueprint, **kwargs) -> None:
                blueprint_calls.append((blueprint, dict(kwargs)))

            def log(self, path: str, payload, static: bool = False) -> None:
                logged_entries.append((path, payload, static))

            def set_time(self, *_args, **_kwargs) -> None:
                return None

            def disconnect(self) -> None:
                return None

        fake_rr = SimpleNamespace(
            RecordingStream=_FakeRecordingStream,
            TextDocument=lambda text, media_type=None: {"text": text, "media_type": media_type},
            TextLog=lambda text: {"text": text},
            SeriesLines=lambda names=None: {"names": tuple(names or ())},
            Scalars=lambda values: {"values": tuple(values)},
            BarChart=lambda values, abscissa=None: {"values": tuple(values), "abscissa": tuple(abscissa or ())},
            blueprint=SimpleNamespace(
                Vertical=lambda *args, **kwargs: ("Vertical", args, kwargs),
                Horizontal=lambda *args, **kwargs: ("Horizontal", args, kwargs),
                Tabs=lambda *args, **kwargs: ("Tabs", args, kwargs),
                TextDocumentView=lambda **kwargs: ("TextDocumentView", kwargs),
                TimeSeriesView=lambda **kwargs: ("TimeSeriesView", kwargs),
                BarChartView=lambda **kwargs: ("BarChartView", kwargs),
                Blueprint=lambda *args, **kwargs: ("Blueprint", args, kwargs),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("friction_identification_core.visualization.rr", fake_rr):
                recorder = RerunRecorder(
                    Path(tmpdir) / "live.rrd",
                    motor_ids=(1, 2, 3),
                    motor_names={1: "motor_01", 2: "motor_02", 3: "motor_03"},
                )
                recorder.log_live_command_packet(
                    group_index=1,
                    round_index=2,
                    active_motor_id=2,
                    sent_commands=np.asarray([0.0, 1.25, 0.0], dtype=np.float64),
                    expected_velocities=np.asarray([0.0, 0.3, 0.0], dtype=np.float64),
                )
                recorder.log_live_motor_sample(
                    group_index=1,
                    round_index=2,
                    active_motor_id=2,
                    motor_id=2,
                    position=0.1,
                    velocity=0.2,
                    expected_velocity=0.3,
                    target_torque=1.25,
                    feedback_torque=0.75,
                    state=1,
                    mos_temperature=31.0,
                    phase_name="steady_forward_01",
                )
                recorder.close()

        self.assertEqual(len(blueprint_calls), 1)
        blueprint_text = str(blueprint_calls[0][0])
        self.assertIn("/live/motors/motor_01/status", blueprint_text)
        self.assertIn("Expected vs Actual Velocity", blueprint_text)
        self.assertIn("/live/motors/motor_02/signals/velocity_error", blueprint_text)
        self.assertNotIn("/live/overview/current_state", blueprint_text)
        self.assertNotIn("/summary/", blueprint_text)

        latest_payload_by_path: dict[str, object] = {}
        for path, payload, _static in logged_entries:
            latest_payload_by_path[path] = payload

        self.assertAlmostEqual(
            float(latest_payload_by_path["live/motors/motor_01/signals/torque"]["values"][0]),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(latest_payload_by_path["live/motors/motor_02/signals/torque"]["values"][0]),
            1.25,
            places=6,
        )
        self.assertAlmostEqual(
            float(latest_payload_by_path["live/motors/motor_03/signals/torque"]["values"][0]),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(latest_payload_by_path["live/motors/motor_02/signals/torque_error"]["values"][0]),
            0.5,
            places=6,
        )
        self.assertAlmostEqual(
            float(latest_payload_by_path["live/motors/motor_02/signals/velocity"]["values"][0]),
            0.2,
            places=6,
        )
        self.assertAlmostEqual(
            float(latest_payload_by_path["live/motors/motor_02/signals/velocity"]["values"][1]),
            0.3,
            places=6,
        )
        self.assertAlmostEqual(
            float(latest_payload_by_path["live/motors/motor_02/signals/velocity_error"]["values"][0]),
            -0.1,
            places=6,
        )
        self.assertIn(
            "- sent_command: `1.250000`",
            str(latest_payload_by_path["live/motors/motor_02/status"]["text"]),
        )
        self.assertIn(
            "- torque_error: `0.500000`",
            str(latest_payload_by_path["live/motors/motor_02/status"]["text"]),
        )
        self.assertIn(
            "- expected_velocity: `0.300000`",
            str(latest_payload_by_path["live/motors/motor_02/status"]["text"]),
        )
        self.assertIn(
            "- velocity_error: `-0.100000`",
            str(latest_payload_by_path["live/motors/motor_02/status"]["text"]),
        )


if __name__ == "__main__":
    unittest.main()
