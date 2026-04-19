from __future__ import annotations

from pathlib import Path

import numpy as np

from friction_identification_core.models import MotorIdentificationResult, RoundCapture

try:
    import rerun as rr
except ImportError:  # pragma: no cover - optional at import time
    rr = None


class RerunRecorder:
    def __init__(self, recording_path: Path) -> None:
        self.recording_path = Path(recording_path)
        self._recording = None if rr is None else rr.RecordingStream("friction_identification")
        self._initialized_paths: set[str] = set()
        self._last_phase_by_round: dict[str, str] = {}
        if self._recording is None:
            return
        self._recording.save(self.recording_path)
        self._recording.log(
            "run/overview",
            rr.TextDocument(
                "\n".join(
                    [
                        "# Sequential Friction Identification",
                        "",
                        "- `signals/position`: feedback vs reference",
                        "- `signals/velocity`: feedback vs reference",
                        "- `signals/torque`: raw command vs sent command vs feedback torque",
                        "- `identification/torque_fit`: identification target vs prediction",
                    ]
                ),
                media_type="text/markdown",
            ),
        )

    @property
    def enabled(self) -> bool:
        return self._recording is not None

    def _round_root(self, *, group_index: int, round_index: int, motor_id: int) -> str:
        return f"rounds/group_{int(group_index):02d}/motor_{int(motor_id):02d}/round_{int(round_index):02d}"

    def _ensure_round_series(self, round_root: str) -> None:
        if self._recording is None:
            return

        position_path = f"{round_root}/signals/position"
        if position_path not in self._initialized_paths:
            self._recording.log(
                position_path,
                rr.SeriesLines(names=["feedback", "reference"]),
                static=True,
            )
            self._initialized_paths.add(position_path)

        velocity_path = f"{round_root}/signals/velocity"
        if velocity_path not in self._initialized_paths:
            self._recording.log(
                velocity_path,
                rr.SeriesLines(names=["feedback", "reference"]),
                static=True,
            )
            self._initialized_paths.add(velocity_path)

        torque_path = f"{round_root}/signals/torque"
        if torque_path not in self._initialized_paths:
            self._recording.log(
                torque_path,
                rr.SeriesLines(names=["raw_command", "sent_command", "feedback"]),
                static=True,
            )
            self._initialized_paths.add(torque_path)

        fit_path = f"{round_root}/identification/torque_fit"
        if fit_path not in self._initialized_paths:
            self._recording.log(
                fit_path,
                rr.SeriesLines(names=["target", "prediction"]),
                static=True,
            )
            self._initialized_paths.add(fit_path)

    def log_live_sample(
        self,
        *,
        group_index: int,
        round_index: int,
        motor_id: int,
        motor_name: str,
        sample_index: int,
        elapsed_s: float,
        position: float,
        position_cmd: float,
        velocity: float,
        velocity_cmd: float,
        command_raw: float,
        command: float,
        torque_feedback: float,
        phase_name: str,
    ) -> None:
        if self._recording is None:
            return

        round_root = self._round_root(group_index=group_index, round_index=round_index, motor_id=motor_id)
        self._ensure_round_series(round_root)
        self._recording.set_time("round_sample", sequence=int(sample_index))
        self._recording.set_time("round_time_s", duration=float(elapsed_s))
        self._recording.log(f"{round_root}/signals/position", rr.Scalars([float(position), float(position_cmd)]))
        self._recording.log(f"{round_root}/signals/velocity", rr.Scalars([float(velocity), float(velocity_cmd)]))
        self._recording.log(
            f"{round_root}/signals/torque",
            rr.Scalars([float(command_raw), float(command), float(torque_feedback)]),
        )

        previous_phase = self._last_phase_by_round.get(round_root)
        if previous_phase != phase_name:
            self._recording.log(
                f"{round_root}/events",
                rr.TextLog(f"{motor_name}: phase -> {phase_name}"),
            )
            self._last_phase_by_round[round_root] = str(phase_name)

    def log_identification(self, capture: RoundCapture, result: MotorIdentificationResult) -> None:
        if self._recording is None:
            return

        round_root = self._round_root(
            group_index=capture.group_index,
            round_index=capture.round_index,
            motor_id=capture.target_motor_id,
        )
        self._ensure_round_series(round_root)
        sample_count = min(capture.sample_count, int(result.torque_target.size), int(result.torque_pred.size))
        for sample_index in range(sample_count):
            self._recording.set_time("round_sample", sequence=int(sample_index))
            self._recording.set_time("round_time_s", duration=float(capture.time[sample_index]))
            self._recording.log(
                f"{round_root}/identification/torque_fit",
                rr.Scalars(
                    [
                        float(result.torque_target[sample_index]),
                        float(result.torque_pred[sample_index]),
                    ]
                ),
            )

        status = str(result.metadata.get("status", "unknown"))
        summary_lines = [
            f"# Motor {capture.target_motor_id:02d} Identification",
            "",
            f"- motor_name: `{capture.motor_name}`",
            f"- group_index: `{capture.group_index}`",
            f"- round_index: `{capture.round_index}`",
            f"- status: `{status}`",
            f"- identified: `{bool(result.identified)}`",
            f"- coulomb: `{float(result.coulomb):.6f}`",
            f"- viscous: `{float(result.viscous):.6f}`",
            f"- offset: `{float(result.offset):.6f}`",
            f"- velocity_scale: `{float(result.velocity_scale):.6f}`",
            f"- train_rmse: `{float(result.train_rmse):.6f}`",
            f"- valid_rmse: `{float(result.valid_rmse):.6f}`",
            f"- train_r2: `{float(result.train_r2):.6f}`",
            f"- valid_r2: `{float(result.valid_r2):.6f}`",
            f"- valid_sample_ratio: `{float(result.valid_sample_ratio):.4f}`",
            f"- sample_count: `{int(result.sample_count)}`",
        ]
        self._recording.log(
            f"{round_root}/identification/summary",
            rr.TextDocument("\n".join(summary_lines), media_type="text/markdown"),
        )

    def log_summary(self, *, summary_path: Path, report_path: Path) -> None:
        if self._recording is None:
            return

        with np.load(summary_path, allow_pickle=False) as summary:
            motor_ids = np.asarray(summary["motor_ids"], dtype=np.float64)
            for metric in ("coulomb", "viscous", "offset", "velocity_scale", "validation_rmse", "validation_r2"):
                self._recording.log(
                    f"summary/{metric}",
                    rr.BarChart(np.asarray(summary[metric], dtype=np.float64), abscissa=motor_ids),
                )

        self._recording.log(
            "summary/report",
            rr.TextDocument(report_path.read_text(encoding="utf-8"), media_type="text/markdown"),
        )

    def close(self) -> None:
        if self._recording is None:
            return
        self._recording.disconnect()

