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
from friction_identification_core.config import DEFAULT_CONFIG_PATH, apply_overrides, load_config
from friction_identification_core.controller import SingleMotorController
from friction_identification_core.identification import identify_motor_friction, identify_motor_friction_lugre
from friction_identification_core.models import MotorCompensationParameters, ReferenceSample, RoundCapture
from friction_identification_core.pipeline import _perform_zeroing, run_compensation_validation, run_sequential_identification
from friction_identification_core.serial_protocol import (
    COMMAND_PAYLOAD_STRUCT,
    FeedbackFrame,
    RECV_FRAME_HEAD,
    RECV_FRAME_SIZE,
    RECV_FRAME_STRUCT,
    SerialFrameParser,
    SingleMotorCommandAdapter,
    calculate_xor_checksum,
)
from friction_identification_core.trajectory import build_reference_trajectory
from friction_identification_core.visualization import RerunRecorder


README_PATH = DEFAULT_CONFIG_PATH.parents[1] / "README.md"


class ClosedLoopFakeTransport:
    def __init__(
        self,
        motor_ids: tuple[int, ...],
        *,
        step_dt: float = 0.005,
        initial_position: float = 0.12,
        command_gain: float = 8.0,
        velocity_damping: float = 1.8,
        position_stiffness: float = 2.8,
        trip_motor_id: int | None = None,
        trip_after_target_frames: int = 1,
        trip_velocity: float | None = None,
        trip_torque: float | None = None,
        trip_position: float | None = None,
        static_friction: float = 0.0,
        initial_commands: tuple[float, ...] | None = None,
    ) -> None:
        self._motor_ids = tuple(int(motor_id) for motor_id in motor_ids)
        self._step_dt = float(step_dt)
        self._pending = bytearray()
        self._last_commands = np.zeros(7, dtype=np.float64)
        if initial_commands is not None:
            initial_command_values = np.asarray(initial_commands, dtype=np.float64).reshape(-1)
            if initial_command_values.size > self._last_commands.size:
                raise ValueError("initial_commands cannot exceed the 7-slot adapter width.")
            self._last_commands[: initial_command_values.size] = initial_command_values
        self._command_gain = float(command_gain)
        self._velocity_damping = float(velocity_damping)
        self._position_stiffness = float(position_stiffness)
        self._states = {
            int(motor_id): {
                "position": float(initial_position),
                "velocity": 0.0,
                "torque": 0.0,
                "temperature": 30.0 + float(motor_id),
            }
            for motor_id in self._motor_ids
        }
        self._target_frame_count = 0
        self._trip_motor_id = None if trip_motor_id is None else int(trip_motor_id)
        self._trip_after_target_frames = max(int(trip_after_target_frames), 1)
        self._trip_velocity = None if trip_velocity is None else float(trip_velocity)
        self._trip_torque = None if trip_torque is None else float(trip_torque)
        self._trip_position = None if trip_position is None else float(trip_position)
        self._static_friction = max(float(static_friction), 0.0)
        self.writes: list[bytes] = []
        self.closed = False

    def _advance_state(self, motor_id: int) -> tuple[float, float, float, float]:
        state = self._states[int(motor_id)]
        command = float(self._last_commands[int(motor_id) - 1])
        if abs(command) <= self._static_friction and abs(float(state["velocity"])) < 1.0e-6:
            effective_command = 0.0
        elif command > 0.0:
            effective_command = command - self._static_friction
        elif command < 0.0:
            effective_command = command + self._static_friction
        else:
            effective_command = 0.0
        acceleration = (
            self._command_gain * effective_command
            - self._velocity_damping * float(state["velocity"])
            - self._position_stiffness * float(state["position"])
        )
        state["velocity"] = float(state["velocity"]) + self._step_dt * acceleration
        state["position"] = float(state["position"]) + self._step_dt * float(state["velocity"])
        state["torque"] = effective_command + 0.04 * float(state["velocity"])
        if int(motor_id) == int(self._trip_motor_id or -1):
            self._target_frame_count += 1
            if self._target_frame_count >= self._trip_after_target_frames:
                if self._trip_velocity is not None:
                    state["velocity"] = float(self._trip_velocity)
                if self._trip_torque is not None:
                    state["torque"] = float(self._trip_torque)
                if self._trip_position is not None:
                    state["position"] = float(self._trip_position)
        return (
            float(state["position"]),
            float(state["velocity"]),
            float(state["torque"]),
            float(state["temperature"]),
        )

    def _build_cycle_bytes(self) -> bytes:
        frames = bytearray()
        for motor_id in self._motor_ids:
            position, velocity, torque, temperature = self._advance_state(int(motor_id))
            frames.extend(
                RECV_FRAME_STRUCT.pack(
                    RECV_FRAME_HEAD,
                    int(motor_id),
                    1,
                    position,
                    velocity,
                    torque,
                    temperature,
                )
            )
        return bytes(frames)

    def read(self, size: int) -> bytes:
        while len(self._pending) < int(size):
            self._pending.extend(self._build_cycle_bytes())
        chunk = bytes(self._pending[:size])
        del self._pending[:size]
        return chunk

    def write(self, payload: bytes) -> int:
        self.writes.append(bytes(payload))
        values = COMMAND_PAYLOAD_STRUCT.unpack(payload[2:30])
        self._last_commands[:] = np.asarray(values, dtype=np.float64)
        return len(payload)

    def reset_input_buffer(self) -> None:
        self._pending.clear()

    def close(self) -> None:
        self.closed = True


class CommandResponseFakeTransport(ClosedLoopFakeTransport):
    """Only returns feedback that was triggered by a command write."""

    def read(self, size: int) -> bytes:
        if not self._pending:
            return b""
        chunk = bytes(self._pending[:size])
        del self._pending[:size]
        return chunk

    def write(self, payload: bytes) -> int:
        written = super().write(payload)
        self._pending.extend(self._build_cycle_bytes())
        return written


class OneFrameReadTransport(ClosedLoopFakeTransport):
    """Returns a single UART frame per read to mimic streaming serial delivery."""

    def __init__(self, *args, read_sleep_s: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._read_sleep_s = max(float(read_sleep_s), 0.0)

    def read(self, size: int) -> bytes:
        _ = size
        if self._read_sleep_s > 0.0:
            time.sleep(self._read_sleep_s)
        while len(self._pending) < RECV_FRAME_SIZE:
            self._pending.extend(self._build_cycle_bytes())
        chunk = bytes(self._pending[:RECV_FRAME_SIZE])
        del self._pending[:RECV_FRAME_SIZE]
        return chunk


def _identification_config(**kwargs: object):
    config = load_config(DEFAULT_CONFIG_PATH).identification
    overrides = {
        "min_samples": 40,
        "min_direction_samples": 10,
        "min_motion_span": 0.01,
        "savgol_window": 11,
        "savgol_polyorder": 3,
        "validation_warmup_samples": 5,
    }
    overrides.update(kwargs)
    return replace(config, **overrides)


def _static_torque(velocity: np.ndarray) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=np.float64)
    return 0.82 * np.tanh(velocity / 0.05) + 0.11 * velocity - 0.02


def _lugre_torque(
    velocity: np.ndarray,
    time: np.ndarray,
    *,
    fc: float = 0.70,
    fs: float = 0.95,
    vs: float = 0.06,
    sigma0: float = 8.0,
    sigma1: float = 0.08,
    sigma2: float = 0.12,
    offset: float = -0.01,
) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    if time.size <= 1:
        dt = np.full(time.size, 1.0e-3, dtype=np.float64)
    else:
        dt = np.diff(np.concatenate(([time[0]], time)))
        dt[0] = dt[1] if dt.size > 1 else 1.0e-3
    tau = np.zeros_like(velocity)
    z = 0.0
    for index, (v_i, dt_i) in enumerate(zip(velocity, dt)):
        g = fc + (fs - fc) * np.exp(-((float(v_i) / max(float(vs), 1.0e-8)) ** 2))
        z_dot = float(v_i) - abs(float(v_i)) * z / max(float(g), 1.0e-8)
        tau[index] = sigma0 * z + sigma1 * z_dot + sigma2 * float(v_i) + offset
        z = z + max(float(dt_i), 1.0e-6) * z_dot
    return tau


def _build_capture_from_trajectory(
    *,
    dynamic: bool = False,
    max_velocity: float = 0.8,
    max_torque: float = 5.0,
) -> RoundCapture:
    base_config = load_config(DEFAULT_CONFIG_PATH)
    excitation = replace(
        base_config.excitation,
        sample_rate=120.0,
        hold_start=0.10,
        hold_end=0.10,
        base_frequency=1.0,
        steady_cycles=6,
        fade_in_cycles=1,
        fade_out_cycles=1,
        harmonic_multipliers=(1, 2, 3, 4),
        harmonic_weights=(1.0, 0.6, 0.4, 0.25),
    )
    trajectory = build_reference_trajectory(excitation, max_velocity=float(max_velocity))
    time_axis = np.asarray(trajectory.time, dtype=np.float64)
    position = np.asarray(trajectory.position_cmd, dtype=np.float64) + 0.002 * np.sin(2.0 * np.pi * 0.7 * time_axis)
    velocity = np.asarray(trajectory.velocity_cmd, dtype=np.float64) + 0.003 * np.sin(2.0 * np.pi * 0.9 * time_axis)
    torque = _lugre_torque(velocity, time_axis) if dynamic else _static_torque(velocity)
    command_raw = torque + 0.02 * np.tanh(velocity / 0.04)
    command = np.clip(command_raw, -float(max_torque) * 0.7, float(max_torque) * 0.7)
    return RoundCapture(
        group_index=1,
        round_index=1,
        target_motor_id=1,
        motor_name="motor_01",
        time=time_axis,
        motor_id=np.ones(time_axis.size, dtype=np.int64),
        position=position,
        velocity=velocity,
        torque_feedback=torque,
        command_raw=command_raw,
        command=command,
        position_cmd=np.asarray(trajectory.position_cmd, dtype=np.float64),
        velocity_cmd=np.asarray(trajectory.velocity_cmd, dtype=np.float64),
        acceleration_cmd=np.asarray(trajectory.acceleration_cmd, dtype=np.float64),
        phase_name=np.asarray(trajectory.phase_name),
        state=np.ones(time_axis.size, dtype=np.uint8),
        mos_temperature=np.full(time_axis.size, 35.0, dtype=np.float64),
        id_match_ok=np.ones(time_axis.size, dtype=bool),
        metadata={
            "synced_before_capture": True,
            "sequence_error_count": 0,
            "sequence_error_ratio": 0.0,
            "target_frame_count": int(time_axis.size),
            "target_frame_ratio": 1.0,
            "target_max_torque": float(max_torque),
            "target_max_velocity": float(max_velocity),
            "planned_duration_s": float(trajectory.duration_s),
            "actual_capture_duration_s": float(trajectory.duration_s),
            "round_total_duration_s": float(trajectory.duration_s),
        },
    )


def _write_runtime_summary(
    path: Path,
    *,
    motor_ids: tuple[int, ...],
    recommended_motor_ids: tuple[int, ...] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    recommended_set = set(motor_ids if recommended_motor_ids is None else recommended_motor_ids)
    motor_id_array = np.asarray(motor_ids, dtype=np.int64)
    motor_names = np.asarray([f"motor_{motor_id:02d}" for motor_id in motor_ids])
    recommended = np.asarray([motor_id in recommended_set for motor_id in motor_ids], dtype=bool)
    base = np.asarray([0.20 + 0.01 * index for index, _motor_id in enumerate(motor_ids)], dtype=np.float64)
    np.savez(
        path,
        motor_ids=motor_id_array,
        motor_names=motor_names,
        recommended_for_runtime=recommended,
        coulomb=base.copy(),
        viscous=0.02 * base,
        offset=np.zeros(len(motor_ids), dtype=np.float64),
        velocity_scale=np.full(len(motor_ids), 0.05, dtype=np.float64),
        validation_rmse=np.full(len(motor_ids), 0.01, dtype=np.float64),
        validation_r2=np.full(len(motor_ids), 0.95, dtype=np.float64),
    )
    return path


def _write_identify_run_manifest(
    results_dir: Path,
    *,
    run_label: str,
    end_time: str | None,
    summary_path: Path | None = None,
) -> Path:
    run_dir = results_dir / "runs" / run_label
    summary_dir = run_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    summary_files = {}
    if summary_path is not None:
        summary_files = {
            "run_summary_path": str(summary_path.resolve()),
            "run_summary_csv_path": str((summary_dir / "hardware_identification_summary.csv").resolve()),
            "run_summary_report_path": str((summary_dir / "hardware_identification_summary.md").resolve()),
            "root_summary_path": str((results_dir / "hardware_identification_summary.npz").resolve()),
            "root_summary_csv_path": str((results_dir / "hardware_identification_summary.csv").resolve()),
            "root_summary_report_path": str((results_dir / "hardware_identification_summary.md").resolve()),
        }
    manifest = {
        "run_label": run_label,
        "mode": "identify",
        "start_time": "2026-04-22T00:00:00+00:00",
        "end_time": end_time,
        "group_count": 1,
        "motor_order": [1, 3],
        "capture_files": [],
        "identification_files": [],
        "dynamic_identification_files": [],
        "summary_files": summary_files,
        "rerun_recording_path": str((run_dir / "identify.rrd").resolve()),
        "config_path": str((results_dir / "test_config.yaml").resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


class SequentialIdentificationTests(unittest.TestCase):
    def test_default_config_exposes_multisine_defaults(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)

        self.assertEqual(config.excitation.curve_type, "multisine")
        self.assertEqual(config.excitation.position_limit, 2.5)
        self.assertEqual(config.excitation.velocity_utilization, 0.85)
        self.assertEqual(config.enabled_motor_ids, (1, 2, 3, 4))
        self.assertEqual(config.control.position_gain.tolist(), [4.0, 4.0, 4.0, 4.0, 0.05, 0.05, 0.05])
        self.assertEqual(config.control.velocity_gain.tolist(), [2.0, 2.0, 2.0, 2.0, 0.025, 0.025, 0.025])
        self.assertEqual(config.control.zeroing_position_gain.tolist(), [4.0, 4.0, 4.0, 4.0, 0.4, 0.3, 0.3])
        self.assertEqual(config.control.zeroing_velocity_gain.tolist(), [2.0, 2.0, 2.0, 2.0, 0.2, 0.14, 0.14])
        self.assertEqual(config.control.speed_abort_ratio.tolist(), [1.2] * 7)
        self.assertEqual(config.identification.validation_velocity_band_edges_ratio, (0.05, 0.12, 0.25, 0.4, 0.6, 0.85))

    def test_motor_override_only_selects_from_config_enabled_subset(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)

        self.assertEqual(apply_overrides(config, motors="all").enabled_motor_ids, (1, 2, 3, 4))
        self.assertEqual(apply_overrides(config, motors="2,4").enabled_motor_ids, (2, 4))
        with self.assertRaisesRegex(ValueError, r"config motors.enabled"):
            apply_overrides(config, motors="5")

    def test_config_accepts_grouped_pd_gain_vectors(self) -> None:
        payload = """
control:
  position_gain: [4.0, 4.0, 3.0, 3.0, 2.0, 2.0, 2.0]
  velocity_gain: [2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0]
  zeroing_position_gain: [5.0, 5.0, 4.0, 4.0, 3.0, 3.0, 3.0]
  zeroing_velocity_gain: [2.5, 2.5, 2.0, 2.0, 1.5, 1.5, 1.5]
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(payload, encoding="utf-8")
            config = load_config(config_path)

        self.assertEqual(config.control.position_gain.tolist(), [4.0, 4.0, 3.0, 3.0, 2.0, 2.0, 2.0])
        self.assertEqual(config.control.velocity_gain.tolist(), [2.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.0])
        self.assertEqual(config.control.zeroing_position_gain.tolist(), [5.0, 5.0, 4.0, 4.0, 3.0, 3.0, 3.0])
        self.assertEqual(config.control.zeroing_velocity_gain.tolist(), [2.5, 2.5, 2.0, 2.0, 1.5, 1.5, 1.5])

    def test_config_rejects_legacy_fields(self) -> None:
        cases = {
            "legacy_velocity_p_gain": """
control:
  velocity_p_gain: 0.1
""",
            "legacy_torque_limits": """
control:
  torque_limits: [1, 1, 1, 1, 1, 1, 1]
""",
            "legacy_platforms": """
excitation:
  platforms:
    - speed_ratio: 0.2
""",
            "legacy_transition_duration": """
excitation:
  transition_duration: 0.1
""",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            for payload in cases.values():
                config_path.write_text(payload, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_config(config_path)

    def test_readme_and_cli_match_runtime_modes(self) -> None:
        parser = build_parser()
        mode_action = next(action for action in parser._actions if action.dest == "mode")
        self.assertEqual(tuple(mode_action.choices), ("identify", "compensate", "default", "sequential"))

        readme = README_PATH.read_text(encoding="utf-8")
        self.assertIn("`multisine`", readme)
        self.assertIn("velocity-band", readme)
        self.assertIn("zeroing", readme)

    def test_main_routes_modes(self) -> None:
        loaded_config = object()
        configured = object()

        with (
            mock.patch("friction_identification_core.__main__.load_config", return_value=loaded_config),
            mock.patch("friction_identification_core.__main__.apply_overrides", return_value=configured),
            mock.patch("friction_identification_core.__main__.run_sequential_identification") as identify_mock,
            mock.patch("friction_identification_core.__main__.run_compensation_validation") as compensate_mock,
        ):
            main([])
            main(["--mode", "compensate"])

        identify_mock.assert_called_once_with(configured, show_rerun_viewer=True)
        compensate_mock.assert_called_once_with(configured, show_rerun_viewer=True)

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

    def test_controller_uses_absolute_pd_and_compensation_stays_feedforward(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)
        controller = SingleMotorController(config)
        reference = ReferenceSample(
            position_cmd=0.5,
            velocity_cmd=0.2,
            acceleration_cmd=0.0,
            phase_name="excitation_cycle_01",
        )
        feedback = FeedbackFrame(
            motor_id=1,
            state=1,
            position=0.1,
            velocity=-0.1,
            torque=0.0,
            mos_temperature=30.0,
        )

        raw_command, limited_command = controller.update(1, reference, feedback)
        expected = float(config.control.position_gain[0]) * 0.4 + float(config.control.velocity_gain[0]) * 0.3
        self.assertAlmostEqual(raw_command, expected, places=6)
        self.assertAlmostEqual(limited_command, expected, places=6)

        params = MotorCompensationParameters(
            motor_id=1,
            motor_name="motor_01",
            coulomb=0.8,
            viscous=0.2,
            offset=0.1,
            velocity_scale=0.05,
        )
        feedforward_raw, feedforward_limited = controller.update(1, reference, feedback, compensation=params)
        self.assertAlmostEqual(feedforward_raw, params.feedforward_torque(float(feedback.velocity)), places=6)
        self.assertAlmostEqual(feedforward_raw, feedforward_limited, places=6)

    def test_multisine_trajectory_is_zero_centered_bounded_and_named_by_cycle(self) -> None:
        base_config = load_config(DEFAULT_CONFIG_PATH)
        excitation = replace(
            base_config.excitation,
            sample_rate=120.0,
            hold_start=0.20,
            hold_end=0.20,
            base_frequency=1.0,
            steady_cycles=4,
            fade_in_cycles=1,
            fade_out_cycles=1,
            harmonic_multipliers=(1, 2, 3),
            harmonic_weights=(1.0, 0.5, 0.3),
        )
        trajectory = build_reference_trajectory(excitation, max_velocity=1.0)
        ordered_phases: list[str] = []
        for phase in trajectory.phase_name.tolist():
            if ordered_phases and ordered_phases[-1] == str(phase):
                continue
            ordered_phases.append(str(phase))

        self.assertEqual(
            ordered_phases,
            [
                "hold_start",
                "fade_in",
                "excitation_cycle_01",
                "excitation_cycle_02",
                "excitation_cycle_03",
                "excitation_cycle_04",
                "fade_out",
                "hold_end",
            ],
        )
        self.assertLessEqual(float(np.max(np.abs(trajectory.position_cmd))), 2.5 + 1.0e-8)
        self.assertLessEqual(float(np.max(np.abs(trajectory.velocity_cmd))), 0.85 + 1.0e-8)
        excitation_mask = np.char.startswith(trajectory.phase_name.astype(str), "excitation_cycle_")
        self.assertAlmostEqual(float(np.mean(trajectory.position_cmd[excitation_mask])), 0.0, delta=0.05)
        self.assertEqual(int(np.count_nonzero(excitation_mask)), int(4 * 120))

    def test_static_identification_uses_excitation_cycles_and_velocity_band_holdout(self) -> None:
        capture = _build_capture_from_trajectory(dynamic=False)
        result = identify_motor_friction(_identification_config(), capture, max_velocity=0.8, max_torque=5.0)

        self.assertTrue(result.identified)
        self.assertTrue(np.array_equal(result.sample_mask, result.sample_mask & result.identification_window_mask))
        used_phases = {str(phase) for phase in capture.phase_name[result.identification_window_mask]}
        self.assertTrue(all(phase.startswith("excitation_cycle_") for phase in used_phases))
        self.assertEqual(result.metadata["validation_mode"], "velocity_band_holdout")
        self.assertTrue(result.metadata["train_velocity_bands"])
        self.assertTrue(result.metadata["valid_velocity_bands"])

    def test_lugre_identification_improves_holdout_rmse_on_synthetic_data(self) -> None:
        capture = _build_capture_from_trajectory(dynamic=True)
        static_result = identify_motor_friction(_identification_config(), capture, max_velocity=0.8, max_torque=5.0)
        dynamic_result = identify_motor_friction_lugre(_identification_config(), capture, static_result)

        self.assertTrue(static_result.identified)
        self.assertTrue(dynamic_result.identified)
        self.assertEqual(dynamic_result.metadata["validation_mode"], "cycle_holdout")
        self.assertLess(float(dynamic_result.valid_rmse), float(static_result.valid_rmse))
        self.assertTrue(np.isfinite(dynamic_result.fc))
        self.assertTrue(np.isfinite(dynamic_result.sigma0))

    def test_identify_mode_runs_zeroing_and_writes_static_and_dynamic_summaries(self) -> None:
        base_config = load_config(DEFAULT_CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1,)),
                excitation=replace(
                    base_config.excitation,
                    sample_rate=60.0,
                    hold_start=0.10,
                    hold_end=0.10,
                    base_frequency=1.0,
                    steady_cycles=3,
                    fade_in_cycles=1,
                    fade_out_cycles=1,
                    harmonic_multipliers=(1, 2, 3),
                    harmonic_weights=(1.0, 0.5, 0.3),
                ),
                control=replace(
                    base_config.control,
                    max_velocity=np.full_like(base_config.control.max_velocity, 0.7),
                    position_gain=np.full_like(base_config.control.position_gain, 0.8),
                    velocity_gain=np.full_like(base_config.control.velocity_gain, 0.18),
                    zeroing_required_frames=4,
                    zeroing_timeout=2.0,
                ),
                identification=replace(
                    base_config.identification,
                    min_samples=10,
                    min_direction_samples=4,
                    min_motion_span=0.0,
                    min_target_frame_ratio=0.1,
                    validation_warmup_samples=2,
                ),
                serial=replace(
                    base_config.serial,
                    read_chunk_size=1024,
                    read_timeout=0.001,
                    sync_timeout=0.5,
                    sync_cycles_required=1,
                ),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )

            transport = ClosedLoopFakeTransport(config.motor_ids, initial_position=0.10)
            result = run_sequential_identification(config, transport_factory=lambda: transport)

            self.assertEqual(len(result.artifacts), 1)
            self.assertTrue(result.summary_paths.root_summary_path.exists())
            self.assertTrue(result.summary_paths.dynamic_root_summary_path.exists())
            self.assertTrue((result.summary_paths.run_summary_path.parent.parent / "group_01" / "motor_01" / "lugre_identification.npz").exists())

            manifest = json.loads(result.summary_paths.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["mode"], "identify")
            self.assertEqual(len(manifest["capture_files"]), 1)
            self.assertEqual(len(manifest["identification_files"]), 1)
            self.assertEqual(len(manifest["dynamic_identification_files"]), 1)
            zero_packet = SingleMotorCommandAdapter(motor_count=7).pack(1, 0.0)
            self.assertEqual(transport.writes[-1], zero_packet)

            with np.load(result.summary_paths.root_summary_path, allow_pickle=False) as summary:
                self.assertIn("train_velocity_bands", summary.files)
                self.assertIn("valid_velocity_bands", summary.files)
                self.assertIn("recommended_for_runtime", summary.files)

            with np.load(result.summary_paths.dynamic_root_summary_path, allow_pickle=False) as summary:
                self.assertIn("fc", summary.files)
                self.assertIn("fs", summary.files)
                self.assertIn("valid_cycles", summary.files)

            with np.load(Path(manifest["identification_files"][0]), allow_pickle=False) as identification:
                self.assertIn("identification_window_mask", identification.files)

    def test_identify_mode_abort_writes_zero_packet_and_manifest(self) -> None:
        base_config = load_config(DEFAULT_CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1,)),
                excitation=replace(
                    base_config.excitation,
                    sample_rate=60.0,
                    hold_start=0.10,
                    hold_end=0.10,
                    base_frequency=1.0,
                    steady_cycles=3,
                    fade_in_cycles=1,
                    fade_out_cycles=1,
                    harmonic_multipliers=(1, 2, 3),
                    harmonic_weights=(1.0, 0.5, 0.3),
                ),
                control=replace(
                    base_config.control,
                    max_velocity=np.full_like(base_config.control.max_velocity, 0.6),
                    position_gain=np.full_like(base_config.control.position_gain, 0.8),
                    velocity_gain=np.full_like(base_config.control.velocity_gain, 0.18),
                    zeroing_required_frames=4,
                    zeroing_timeout=2.0,
                ),
                identification=replace(
                    base_config.identification,
                    min_samples=10,
                    min_direction_samples=4,
                    min_motion_span=0.0,
                    min_target_frame_ratio=0.1,
                ),
                serial=replace(
                    base_config.serial,
                    read_chunk_size=1024,
                    read_timeout=0.001,
                    sync_timeout=0.5,
                    sync_cycles_required=1,
                ),
                output=replace(base_config.output, results_dir=results_dir),
            )

            transport = ClosedLoopFakeTransport(
                config.motor_ids,
                initial_position=0.10,
                trip_motor_id=1,
                trip_after_target_frames=400,
                trip_velocity=1.0,
            )

            with self.assertRaisesRegex(ValueError, r"reason=velocity_limit_exceeded"):
                run_sequential_identification(config, transport_factory=lambda: transport)

            zero_packet = SingleMotorCommandAdapter(motor_count=7).pack(1, 0.0)
            self.assertEqual(transport.writes[-1], zero_packet)

            manifest_path = next((results_dir / "runs").glob("*_identify/run_manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["abort_event"]["reason"], "velocity_limit_exceeded")
            self.assertTrue(str(manifest["abort_event"]["phase_name"]).startswith("excitation_cycle_"))

    def test_identify_mode_zeroing_uses_dedicated_gains_to_break_static_friction(self) -> None:
        base_config = load_config(DEFAULT_CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1,)),
                excitation=replace(
                    base_config.excitation,
                    sample_rate=60.0,
                    hold_start=0.10,
                    hold_end=0.10,
                    base_frequency=1.0,
                    steady_cycles=3,
                    fade_in_cycles=1,
                    fade_out_cycles=1,
                    harmonic_multipliers=(1, 2, 3),
                    harmonic_weights=(1.0, 0.5, 0.3),
                ),
                control=replace(
                    base_config.control,
                    max_velocity=np.full_like(base_config.control.max_velocity, 0.7),
                    position_gain=np.full_like(base_config.control.position_gain, 0.02),
                    velocity_gain=np.full_like(base_config.control.velocity_gain, 0.01),
                    zeroing_position_gain=np.full_like(base_config.control.zeroing_position_gain, 0.8),
                    zeroing_velocity_gain=np.full_like(base_config.control.zeroing_velocity_gain, 0.18),
                    zeroing_required_frames=4,
                    zeroing_timeout=2.0,
                ),
                identification=replace(
                    base_config.identification,
                    min_samples=10,
                    min_direction_samples=4,
                    min_motion_span=0.0,
                    min_target_frame_ratio=0.1,
                    validation_warmup_samples=2,
                ),
                serial=replace(
                    base_config.serial,
                    read_chunk_size=1024,
                    read_timeout=0.001,
                    sync_timeout=0.5,
                    sync_cycles_required=1,
                ),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )

            transport = ClosedLoopFakeTransport(config.motor_ids, initial_position=0.10, static_friction=0.03)
            result = run_sequential_identification(config, transport_factory=lambda: transport)

            self.assertEqual(len(result.artifacts), 1)
            self.assertTrue(result.summary_paths.manifest_path.exists())

    def test_identify_mode_zeroing_allows_coarse_return_velocity_above_near_zero_limit(self) -> None:
        base_config = load_config(DEFAULT_CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1,)),
                excitation=replace(
                    base_config.excitation,
                    sample_rate=60.0,
                    hold_start=0.10,
                    hold_end=0.10,
                    base_frequency=1.0,
                    steady_cycles=3,
                    fade_in_cycles=1,
                    fade_out_cycles=1,
                    harmonic_multipliers=(1, 2, 3),
                    harmonic_weights=(1.0, 0.5, 0.3),
                ),
                control=replace(
                    base_config.control,
                    max_velocity=np.full_like(base_config.control.max_velocity, 0.7),
                    position_gain=np.full_like(base_config.control.position_gain, 0.8),
                    velocity_gain=np.full_like(base_config.control.velocity_gain, 0.18),
                    zeroing_position_gain=np.full_like(base_config.control.zeroing_position_gain, 0.8),
                    zeroing_velocity_gain=np.full_like(base_config.control.zeroing_velocity_gain, 0.18),
                    zeroing_velocity_limit=np.full_like(base_config.control.zeroing_velocity_limit, 0.4),
                    low_speed_abort_limit=np.full_like(base_config.control.low_speed_abort_limit, 0.45),
                    speed_abort_ratio=np.full_like(base_config.control.speed_abort_ratio, 1.5),
                    zeroing_required_frames=4,
                    zeroing_timeout=2.0,
                ),
                identification=replace(
                    base_config.identification,
                    min_samples=10,
                    min_direction_samples=4,
                    min_motion_span=0.0,
                    min_target_frame_ratio=0.1,
                    validation_warmup_samples=2,
                ),
                serial=replace(
                    base_config.serial,
                    read_chunk_size=1024,
                    read_timeout=0.001,
                    sync_timeout=0.5,
                    sync_cycles_required=1,
                ),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )

            transport = ClosedLoopFakeTransport(
                config.motor_ids,
                initial_position=0.30,
                command_gain=8.0,
            )
            result = run_sequential_identification(config, transport_factory=lambda: transport)

            self.assertEqual(len(result.artifacts), 1)
            manifest = json.loads(result.summary_paths.manifest_path.read_text(encoding="utf-8"))
            self.assertNotIn("abort_event", manifest)

    def test_identify_mode_zeroing_uses_near_zero_velocity_limit_for_locking_not_abort(self) -> None:
        base_config = load_config(DEFAULT_CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1,)),
                excitation=replace(
                    base_config.excitation,
                    sample_rate=60.0,
                    hold_start=0.10,
                    hold_end=0.10,
                    base_frequency=1.0,
                    steady_cycles=3,
                    fade_in_cycles=1,
                    fade_out_cycles=1,
                    harmonic_multipliers=(1, 2, 3),
                    harmonic_weights=(1.0, 0.5, 0.3),
                ),
                control=replace(
                    base_config.control,
                    max_velocity=np.full_like(base_config.control.max_velocity, 0.7),
                    position_gain=np.full_like(base_config.control.position_gain, 0.8),
                    velocity_gain=np.full_like(base_config.control.velocity_gain, 0.18),
                    zeroing_position_gain=np.full_like(base_config.control.zeroing_position_gain, 0.8),
                    zeroing_velocity_gain=np.full_like(base_config.control.zeroing_velocity_gain, 0.18),
                    zeroing_velocity_limit=np.full_like(base_config.control.zeroing_velocity_limit, 0.4),
                    low_speed_abort_limit=np.full_like(base_config.control.low_speed_abort_limit, 0.45),
                    speed_abort_ratio=np.full_like(base_config.control.speed_abort_ratio, 1.5),
                    zeroing_required_frames=4,
                    zeroing_timeout=0.2,
                ),
                identification=replace(
                    base_config.identification,
                    min_samples=10,
                    min_direction_samples=4,
                    min_motion_span=0.0,
                    min_target_frame_ratio=0.1,
                    validation_warmup_samples=2,
                ),
                serial=replace(
                    base_config.serial,
                    read_chunk_size=1024,
                    read_timeout=0.001,
                    sync_timeout=0.5,
                    sync_cycles_required=1,
                ),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )

            transport = ClosedLoopFakeTransport(
                config.motor_ids,
                initial_position=0.05,
                trip_motor_id=1,
                trip_after_target_frames=1,
                trip_velocity=0.6,
                trip_position=0.05,
            )

            with self.assertRaisesRegex(ValueError, r"reason=zeroing_timeout"):
                run_sequential_identification(config, transport_factory=lambda: transport)

            manifest_path = next((Path(tmpdir) / "runs").glob("*_identify/run_manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["abort_event"]["reason"], "zeroing_timeout")
            self.assertEqual(manifest["abort_event"]["stage"], "zeroing")

    def test_identify_mode_sends_zero_command_before_zeroing_reads_feedback(self) -> None:
        base_config = load_config(DEFAULT_CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1,)),
                excitation=replace(
                    base_config.excitation,
                    sample_rate=60.0,
                    hold_start=0.10,
                    hold_end=0.10,
                    base_frequency=1.0,
                    steady_cycles=3,
                    fade_in_cycles=1,
                    fade_out_cycles=1,
                    harmonic_multipliers=(1, 2, 3),
                    harmonic_weights=(1.0, 0.5, 0.3),
                ),
                control=replace(
                    base_config.control,
                    max_velocity=np.full_like(base_config.control.max_velocity, 0.7),
                    position_gain=np.full_like(base_config.control.position_gain, 0.02),
                    velocity_gain=np.full_like(base_config.control.velocity_gain, 0.01),
                    zeroing_position_gain=np.full_like(base_config.control.zeroing_position_gain, 0.8),
                    zeroing_velocity_gain=np.full_like(base_config.control.zeroing_velocity_gain, 0.18),
                    zeroing_required_frames=4,
                    zeroing_timeout=2.0,
                ),
                identification=replace(
                    base_config.identification,
                    min_samples=10,
                    min_direction_samples=4,
                    min_motion_span=0.0,
                    min_target_frame_ratio=0.1,
                    validation_warmup_samples=2,
                ),
                serial=replace(
                    base_config.serial,
                    read_chunk_size=1024,
                    read_timeout=0.001,
                    sync_timeout=0.5,
                    sync_cycles_required=1,
                ),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )

            transport = ClosedLoopFakeTransport(
                config.motor_ids,
                initial_position=0.0,
                initial_commands=(12.0,),
                static_friction=0.03,
            )
            result = run_sequential_identification(config, transport_factory=lambda: transport)

            self.assertEqual(len(result.artifacts), 1)
            self.assertIsNotNone(result.summary_paths)
            zero_packet = SingleMotorCommandAdapter(motor_count=7).pack(1, 0.0)
            self.assertGreaterEqual(len(transport.writes), 1)
            self.assertEqual(transport.writes[0], zero_packet)

    def test_identify_mode_zeroing_retries_feedback_request_for_command_response_transport(self) -> None:
        base_config = load_config(DEFAULT_CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1,)),
                excitation=replace(
                    base_config.excitation,
                    sample_rate=60.0,
                    hold_start=0.10,
                    hold_end=0.10,
                    base_frequency=1.0,
                    steady_cycles=3,
                    fade_in_cycles=1,
                    fade_out_cycles=1,
                    harmonic_multipliers=(1, 2, 3),
                    harmonic_weights=(1.0, 0.5, 0.3),
                ),
                control=replace(
                    base_config.control,
                    max_velocity=np.full_like(base_config.control.max_velocity, 0.7),
                    position_gain=np.full_like(base_config.control.position_gain, 0.02),
                    velocity_gain=np.full_like(base_config.control.velocity_gain, 0.01),
                    zeroing_position_gain=np.full_like(base_config.control.zeroing_position_gain, 0.8),
                    zeroing_velocity_gain=np.full_like(base_config.control.zeroing_velocity_gain, 0.18),
                    zeroing_required_frames=4,
                    zeroing_timeout=2.0,
                ),
                identification=replace(
                    base_config.identification,
                    min_samples=10,
                    min_direction_samples=4,
                    min_motion_span=0.0,
                    min_target_frame_ratio=0.1,
                    validation_warmup_samples=2,
                ),
                serial=replace(
                    base_config.serial,
                    read_chunk_size=1024,
                    read_timeout=0.001,
                    sync_timeout=0.5,
                    sync_cycles_required=1,
                ),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )

            transport = CommandResponseFakeTransport(
                config.motor_ids,
                initial_position=0.0,
                initial_commands=(12.0,),
                static_friction=0.03,
            )
            result = run_sequential_identification(config, transport_factory=lambda: transport)

            self.assertEqual(len(result.artifacts), 1)
            self.assertIsNotNone(result.summary_paths)

    def test_zeroing_feedback_retry_reuses_last_command_during_streaming_reads(self) -> None:
        base_config = load_config(DEFAULT_CONFIG_PATH)
        config = replace(
            base_config,
            motors=replace(base_config.motors, enabled_ids=(1,)),
            control=replace(
                base_config.control,
                zeroing_required_frames=4,
                zeroing_timeout=0.03,
            ),
            serial=replace(
                base_config.serial,
                read_chunk_size=RECV_FRAME_SIZE,
                read_timeout=0.001,
            ),
        )

        transport = OneFrameReadTransport(
            (1, 2),
            initial_position=0.10,
            static_friction=0.03,
            read_sleep_s=0.002,
        )
        parser = SerialFrameParser(max_motor_id=max(config.motor_ids))
        command_adapter = SingleMotorCommandAdapter(
            motor_count=max(config.motor_ids),
            torque_limits=config.control.max_torque,
        )
        controller = SingleMotorController(config)

        class _Recorder:
            def log_zeroing_event(self, **_kwargs) -> None:
                return None

            def log_zeroing_sample(self, **_kwargs) -> None:
                return None

            def log_live_command_packet(self, **_kwargs) -> None:
                return None

        with self.assertRaisesRegex(ValueError, r"reason=zeroing_timeout"):
            _perform_zeroing(
                config=config,
                transport=transport,
                parser=parser,
                command_adapter=command_adapter,
                controller=controller,
                rerun_recorder=_Recorder(),
            )

        self.assertGreaterEqual(len(transport.writes), 3)
        motor_1_commands = [float(COMMAND_PAYLOAD_STRUCT.unpack(payload[2:30])[0]) for payload in transport.writes]
        self.assertGreater(abs(motor_1_commands[1]), 1.0e-6)
        self.assertTrue(
            any(
                abs(previous) > 1.0e-6 and abs(current - previous) <= 1.0e-6
                for previous, current in zip(motor_1_commands[1:], motor_1_commands[2:])
            )
        )

    def test_zeroing_timeout_manifest_captures_last_filtered_state(self) -> None:
        base_config = load_config(DEFAULT_CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1,)),
                control=replace(
                    base_config.control,
                    position_gain=np.full_like(base_config.control.position_gain, 0.02),
                    velocity_gain=np.full_like(base_config.control.velocity_gain, 0.01),
                    zeroing_position_gain=np.full_like(base_config.control.zeroing_position_gain, 0.02),
                    zeroing_velocity_gain=np.full_like(base_config.control.zeroing_velocity_gain, 0.01),
                    zeroing_required_frames=4,
                    zeroing_timeout=0.2,
                ),
                identification=replace(
                    base_config.identification,
                    min_samples=10,
                    min_direction_samples=4,
                    min_motion_span=0.0,
                    min_target_frame_ratio=0.1,
                ),
                serial=replace(
                    base_config.serial,
                    read_chunk_size=1024,
                    read_timeout=0.001,
                    sync_timeout=0.5,
                    sync_cycles_required=1,
                ),
                output=replace(base_config.output, results_dir=results_dir),
            )

            transport = ClosedLoopFakeTransport(config.motor_ids, initial_position=0.10, static_friction=0.03)

            with self.assertRaisesRegex(ValueError, r"reason=zeroing_timeout"):
                run_sequential_identification(config, transport_factory=lambda: transport)

            manifest_path = next((results_dir / "runs").glob("*_identify/run_manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            detail = str(manifest["abort_event"]["detail"])
            self.assertIn("filtered_position=", detail)
            self.assertIn("filtered_velocity=", detail)
            self.assertIn("success_count=", detail)

    def test_compensation_mode_uses_summary_parameters_and_skips_identification_files(self) -> None:
        base_config = load_config(DEFAULT_CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            params_path = _write_runtime_summary(
                Path(tmpdir) / "summary.npz",
                motor_ids=(1,),
                recommended_motor_ids=(1,),
            )
            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1,)),
                excitation=replace(
                    base_config.excitation,
                    sample_rate=60.0,
                    hold_start=0.10,
                    hold_end=0.10,
                    base_frequency=1.0,
                    steady_cycles=3,
                    fade_in_cycles=1,
                    fade_out_cycles=1,
                ),
                control=replace(
                    base_config.control,
                    max_velocity=np.full_like(base_config.control.max_velocity, 0.7),
                    low_speed_abort_limit=np.full_like(base_config.control.low_speed_abort_limit, 0.45),
                    speed_abort_ratio=np.full_like(base_config.control.speed_abort_ratio, 1.5),
                ),
                identification=replace(
                    base_config.identification,
                    min_target_frame_ratio=0.1,
                ),
                serial=replace(
                    base_config.serial,
                    read_chunk_size=1024,
                    read_timeout=0.001,
                    sync_timeout=0.5,
                    sync_cycles_required=1,
                ),
                output=replace(base_config.output, results_dir=Path(tmpdir)),
            )

            result = run_compensation_validation(
                config,
                transport_factory=lambda: ClosedLoopFakeTransport(
                    config.motor_ids,
                    initial_position=0.0,
                    command_gain=2.0,
                    velocity_damping=2.5,
                    position_stiffness=3.0,
                ),
                parameters_path=params_path,
            )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["mode"], "compensate")
            self.assertEqual(manifest["identification_files"], [])
            self.assertEqual(manifest["dynamic_identification_files"], [])
            self.assertEqual(len(manifest["capture_files"]), 1)

    def test_compensation_mode_prefers_latest_completed_identify_summary(self) -> None:
        base_config = load_config(DEFAULT_CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)
            root_summary_path = _write_runtime_summary(
                results_dir / base_config.output.summary_filename,
                motor_ids=(1, 3),
                recommended_motor_ids=(1, 3),
            )
            older_summary_path = _write_runtime_summary(
                results_dir / "runs" / "20260422_090000_identify" / "summary" / base_config.output.summary_filename,
                motor_ids=(1, 3),
                recommended_motor_ids=(1, 3),
            )
            newer_summary_path = _write_runtime_summary(
                results_dir / "runs" / "20260422_100000_identify" / "summary" / base_config.output.summary_filename,
                motor_ids=(1, 3),
                recommended_motor_ids=(1, 3),
            )
            _write_identify_run_manifest(
                results_dir,
                run_label="20260422_090000_identify",
                end_time="2026-04-22T01:00:00+00:00",
                summary_path=older_summary_path,
            )
            _write_identify_run_manifest(
                results_dir,
                run_label="20260422_100000_identify",
                end_time="2026-04-22T02:00:00+00:00",
                summary_path=newer_summary_path,
            )
            _write_identify_run_manifest(
                results_dir,
                run_label="20260422_110000_identify",
                end_time=None,
                summary_path=root_summary_path,
            )

            config = replace(
                base_config,
                motors=replace(base_config.motors, enabled_ids=(1, 3)),
                output=replace(base_config.output, results_dir=results_dir),
                excitation=replace(
                    base_config.excitation,
                    sample_rate=60.0,
                    hold_start=0.10,
                    hold_end=0.10,
                    base_frequency=1.0,
                    steady_cycles=3,
                    fade_in_cycles=1,
                    fade_out_cycles=1,
                ),
                control=replace(
                    base_config.control,
                    low_speed_abort_limit=np.full_like(base_config.control.low_speed_abort_limit, 0.45),
                    speed_abort_ratio=np.full_like(base_config.control.speed_abort_ratio, 1.5),
                ),
                serial=replace(
                    base_config.serial,
                    read_chunk_size=1024,
                    read_timeout=0.001,
                    sync_timeout=0.2,
                    sync_cycles_required=1,
                ),
                identification=replace(base_config.identification, min_target_frame_ratio=0.1),
            )

            result = run_compensation_validation(
                config,
                transport_factory=lambda: ClosedLoopFakeTransport(
                    config.motor_ids,
                    initial_position=0.0,
                    command_gain=2.0,
                    velocity_damping=2.5,
                    position_stiffness=3.0,
                ),
            )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["compensation_parameters_path"], str(newer_summary_path.resolve()))

    def test_rerun_recorder_logs_overview_zeroing_and_dynamic_paths(self) -> None:
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
            blueprint=SimpleNamespace(
                Vertical=lambda *args, **kwargs: ("Vertical", args, kwargs),
                Horizontal=lambda *args, **kwargs: ("Horizontal", args, kwargs),
                Tabs=lambda *args, **kwargs: ("Tabs", args, kwargs),
                TextDocumentView=lambda **kwargs: ("TextDocumentView", kwargs),
                TimeSeriesView=lambda **kwargs: ("TimeSeriesView", kwargs),
                Blueprint=lambda *args, **kwargs: ("Blueprint", args, kwargs),
            ),
        )

        capture = _build_capture_from_trajectory(dynamic=True)
        static_result = identify_motor_friction(_identification_config(), capture, max_velocity=0.8, max_torque=5.0)
        dynamic_result = identify_motor_friction_lugre(_identification_config(), capture, static_result)

        with tempfile.TemporaryDirectory() as tmpdir:
            static_summary_path = Path(tmpdir) / "static.npz"
            static_report_path = Path(tmpdir) / "static.md"
            dynamic_summary_path = Path(tmpdir) / "dynamic.npz"
            dynamic_report_path = Path(tmpdir) / "dynamic.md"
            np.savez(static_summary_path, motor_ids=np.asarray([1], dtype=np.int64))
            np.savez(dynamic_summary_path, motor_ids=np.asarray([1], dtype=np.int64))
            static_report_path.write_text("# static\n", encoding="utf-8")
            dynamic_report_path.write_text("# dynamic\n", encoding="utf-8")

            with mock.patch("friction_identification_core.visualization.rr", fake_rr):
                recorder = RerunRecorder(
                    static_summary_path.with_suffix(".rrd"),
                    motor_ids=(1,),
                    motor_names={1: capture.motor_name},
                    show_viewer=True,
                )
                recorder.log_zeroing_event(event="zeroing_start", motor_id=1, detail="begin")
                recorder.log_zeroing_sample(
                    motor_id=1,
                    raw_position=0.1,
                    raw_velocity=0.05,
                    filtered_position=0.02,
                    filtered_velocity=0.01,
                    position_error=-0.02,
                    velocity_error=-0.01,
                    success_count=2,
                    required_frames=4,
                    inside_entry_band=True,
                    inside_exit_band=True,
                    command_raw=-0.1,
                    command=-0.1,
                    feedback_torque=0.05,
                    torque_limit=1.0,
                    velocity_limit=0.2,
                    position_limit=2.5,
                )
                recorder.log_identification(capture, static_result, dynamic_result)
                recorder.log_abort_event({"reason": "velocity_limit_exceeded"})
                recorder.log_summary(
                    summary_path=static_summary_path,
                    report_path=static_report_path,
                    dynamic_summary_path=dynamic_summary_path,
                    dynamic_report_path=dynamic_report_path,
                )
                recorder.close()

        expected_paths = {
            "live/overview/current_state",
            "live/zeroing/motor_01/events",
            "live/zeroing/motor_01/signals/success_count",
            "rounds/group_01/motor_01/round_01/masks",
            "rounds/group_01/motor_01/round_01/static_residual",
            "rounds/group_01/motor_01/round_01/dynamic_residual",
            "summary/static",
            "summary/dynamic",
            "live/overview/abort_event",
        }
        self.assertEqual(spawn_calls, [{"connect": True, "detach_process": True}])
        self.assertTrue(expected_paths.issubset(set(logged_paths)))

    def test_rerun_overview_keeps_active_motor_position_when_idle_frames_arrive(self) -> None:
        latest_payload_by_path: dict[str, object] = {}

        class _FakeRecordingStream:
            def __init__(self, _name: str) -> None:
                self.saved_path = None

            def save(self, path: Path) -> None:
                self.saved_path = Path(path)

            def send_blueprint(self, *_args, **_kwargs) -> None:
                return None

            def log(self, path: str, payload, static: bool = False) -> None:
                _ = static
                latest_payload_by_path[path] = payload

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
            blueprint=SimpleNamespace(
                Vertical=lambda *args, **kwargs: ("Vertical", args, kwargs),
                Horizontal=lambda *args, **kwargs: ("Horizontal", args, kwargs),
                Tabs=lambda *args, **kwargs: ("Tabs", args, kwargs),
                TextDocumentView=lambda **kwargs: ("TextDocumentView", kwargs),
                TimeSeriesView=lambda **kwargs: ("TimeSeriesView", kwargs),
                Blueprint=lambda *args, **kwargs: ("Blueprint", args, kwargs),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("friction_identification_core.visualization.rr", fake_rr):
                recorder = RerunRecorder(
                    Path(tmpdir) / "live.rrd",
                    motor_ids=(1, 2),
                    motor_names={1: "motor_01", 2: "motor_02"},
                )
                recorder.log_live_motor_sample(
                    group_index=1,
                    round_index=1,
                    active_motor_id=1,
                    motor_id=1,
                    position=1.1,
                    velocity=0.2,
                    feedback_torque=0.3,
                    command_raw=0.4,
                    command=0.5,
                    reference_position=1.23,
                    reference_velocity=0.33,
                    reference_acceleration=0.44,
                    velocity_limit=0.9,
                    torque_limit=1.0,
                    position_limit=2.5,
                    phase_name="excitation_cycle_01",
                    stage="identify",
                    safety_margin_text="ok",
                )
                recorder.log_live_motor_sample(
                    group_index=1,
                    round_index=1,
                    active_motor_id=1,
                    motor_id=2,
                    position=2.1,
                    velocity=0.0,
                    feedback_torque=0.0,
                    command_raw=0.0,
                    command=0.0,
                    reference_position=0.0,
                    reference_velocity=0.0,
                    reference_acceleration=0.0,
                    velocity_limit=0.1,
                    torque_limit=1.0,
                    position_limit=2.5,
                    phase_name="idle",
                    stage="identify",
                    safety_margin_text="ok",
                )
                recorder.close()

        overview_text = str(latest_payload_by_path["live/overview/current_state"]["text"])
        self.assertIn("- active_phase: `excitation_cycle_01`", overview_text)
        self.assertIn("- reference_position: `1.230000`", overview_text)
        self.assertIn("- reference_velocity: `0.330000`", overview_text)
        self.assertIn("- reference_acceleration: `0.440000`", overview_text)
        self.assertIn("- dynamic_velocity_threshold: `0.900000`", overview_text)

    def test_rerun_command_packets_refresh_position_series(self) -> None:
        latest_payload_by_path: dict[str, object] = {}

        class _FakeRecordingStream:
            def __init__(self, _name: str) -> None:
                self.saved_path = None

            def save(self, path: Path) -> None:
                self.saved_path = Path(path)

            def send_blueprint(self, *_args, **_kwargs) -> None:
                return None

            def log(self, path: str, payload, static: bool = False) -> None:
                _ = static
                latest_payload_by_path[path] = payload

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
            blueprint=SimpleNamespace(
                Vertical=lambda *args, **kwargs: ("Vertical", args, kwargs),
                Horizontal=lambda *args, **kwargs: ("Horizontal", args, kwargs),
                Tabs=lambda *args, **kwargs: ("Tabs", args, kwargs),
                TextDocumentView=lambda **kwargs: ("TextDocumentView", kwargs),
                TimeSeriesView=lambda **kwargs: ("TimeSeriesView", kwargs),
                Blueprint=lambda *args, **kwargs: ("Blueprint", args, kwargs),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("friction_identification_core.visualization.rr", fake_rr):
                recorder = RerunRecorder(
                    Path(tmpdir) / "live.rrd",
                    motor_ids=(1,),
                    motor_names={1: "motor_01"},
                )
                recorder.log_live_motor_sample(
                    group_index=1,
                    round_index=1,
                    active_motor_id=1,
                    motor_id=1,
                    position=1.1,
                    velocity=0.2,
                    feedback_torque=0.3,
                    command_raw=0.4,
                    command=0.5,
                    reference_position=1.23,
                    reference_velocity=0.33,
                    reference_acceleration=0.44,
                    velocity_limit=0.9,
                    torque_limit=1.0,
                    position_limit=2.5,
                    phase_name="excitation_cycle_01",
                    stage="identify",
                    safety_margin_text="ok",
                )
                recorder.log_live_command_packet(
                    sent_commands=np.asarray([0.6], dtype=np.float64),
                    expected_positions=np.asarray([1.5], dtype=np.float64),
                    expected_velocities=np.asarray([0.35], dtype=np.float64),
                )
                recorder.close()

        self.assertEqual(
            latest_payload_by_path["live/motors/motor_01/signals/position"]["values"],
            (1.1, 1.5),
        )
        status_text = str(latest_payload_by_path["live/motors/motor_01/status"]["text"])
        self.assertIn("- reference_position: `1.500000`", status_text)
        overview_text = str(latest_payload_by_path["live/overview/current_state"]["text"])
        self.assertIn("- reference_position: `1.500000`", overview_text)


if __name__ == "__main__":
    unittest.main()
