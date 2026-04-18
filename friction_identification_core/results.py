from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from friction_identification_core.config import Config
from friction_identification_core.models import MotorIdentificationResult, RoundCapture
from friction_identification_core.runtime import ensure_directory, filesystem_timestamp, utc_now_iso8601, write_json


@dataclass(frozen=True)
class RoundArtifact:
    capture: RoundCapture
    identification: MotorIdentificationResult
    capture_path: Path
    identification_path: Path


@dataclass(frozen=True)
class SummaryPaths:
    run_summary_path: Path
    run_summary_csv_path: Path
    run_summary_report_path: Path
    root_summary_path: Path
    root_summary_csv_path: Path
    root_summary_report_path: Path
    manifest_path: Path


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    return value


def _json_scalar(payload: dict[str, Any]) -> np.ndarray:
    return np.asarray(json.dumps(_normalize_json_value(payload), ensure_ascii=False))


def _finite_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _finite_std(values: list[float]) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.std(finite))


class ResultStore:
    def __init__(self, config: Config) -> None:
        self._config = config
        self.results_dir = ensure_directory(config.results_dir)
        self.run_label = f"{filesystem_timestamp()}_sequential"
        self.run_dir = ensure_directory(self.results_dir / "runs" / self.run_label)
        self.summary_dir = ensure_directory(self.run_dir / "summary")
        self.manifest_path = self.run_dir / "run_manifest.json"
        self._manifest: dict[str, Any] = {
            "run_label": self.run_label,
            "mode": "sequential",
            "start_time": utc_now_iso8601(),
            "end_time": None,
            "group_count": int(config.group_count),
            "motor_order": list(config.enabled_motor_ids),
            "capture_files": [],
            "identification_files": [],
            "summary_files": {},
            "config_path": str(config.config_path),
        }
        self._write_manifest()

    def _write_manifest(self) -> None:
        write_json(self.manifest_path, self._manifest)

    def _motor_dir(self, group_index: int, motor_id: int) -> Path:
        return ensure_directory(self.run_dir / f"group_{int(group_index):02d}" / f"motor_{int(motor_id):02d}")

    def save_capture(self, capture: RoundCapture) -> Path:
        path = self._motor_dir(capture.group_index, capture.target_motor_id) / "capture.npz"
        np.savez(
            path,
            time=np.asarray(capture.time, dtype=np.float64),
            motor_id=np.asarray(capture.motor_id, dtype=np.int64),
            position=np.asarray(capture.position, dtype=np.float64),
            velocity=np.asarray(capture.velocity, dtype=np.float64),
            torque_feedback=np.asarray(capture.torque_feedback, dtype=np.float64),
            command=np.asarray(capture.command, dtype=np.float64),
            position_cmd=np.asarray(capture.position_cmd, dtype=np.float64),
            velocity_cmd=np.asarray(capture.velocity_cmd, dtype=np.float64),
            acceleration_cmd=np.asarray(capture.acceleration_cmd, dtype=np.float64),
            phase_name=np.asarray(capture.phase_name),
            state=np.asarray(capture.state, dtype=np.uint8),
            mos_temperature=np.asarray(capture.mos_temperature, dtype=np.float64),
            id_match_ok=np.asarray(capture.id_match_ok, dtype=bool),
            metadata=_json_scalar(capture.metadata),
        )
        self._manifest["capture_files"].append(str(path))
        self._write_manifest()
        return path

    def save_identification(self, capture: RoundCapture, result: MotorIdentificationResult) -> Path:
        path = self._motor_dir(capture.group_index, capture.target_motor_id) / "identification.npz"
        np.savez(
            path,
            motor_id=np.asarray(int(result.motor_id), dtype=np.int64),
            coulomb=np.asarray(float(result.coulomb), dtype=np.float64),
            viscous=np.asarray(float(result.viscous), dtype=np.float64),
            offset=np.asarray(float(result.offset), dtype=np.float64),
            velocity_scale=np.asarray(float(result.velocity_scale), dtype=np.float64),
            torque_pred=np.asarray(result.torque_pred, dtype=np.float64),
            torque_target=np.asarray(result.torque_target, dtype=np.float64),
            sample_mask=np.asarray(result.sample_mask, dtype=bool),
            train_mask=np.asarray(result.train_mask, dtype=bool),
            valid_mask=np.asarray(result.valid_mask, dtype=bool),
            train_rmse=np.asarray(float(result.train_rmse), dtype=np.float64),
            valid_rmse=np.asarray(float(result.valid_rmse), dtype=np.float64),
            train_r2=np.asarray(float(result.train_r2), dtype=np.float64),
            valid_r2=np.asarray(float(result.valid_r2), dtype=np.float64),
            identified=np.asarray(bool(result.identified), dtype=bool),
            metadata=_json_scalar(result.metadata),
        )
        self._manifest["identification_files"].append(str(path))
        self._write_manifest()
        return path

    def save_summary(self, artifacts: list[RoundArtifact]) -> SummaryPaths:
        motor_ids = list(self._config.motor_ids)
        motor_names = [self._config.motors.name_for(motor_id) for motor_id in motor_ids]
        identified_mask = np.zeros(len(motor_ids), dtype=bool)
        coulomb = np.full(len(motor_ids), np.nan, dtype=np.float64)
        viscous = np.full(len(motor_ids), np.nan, dtype=np.float64)
        offset = np.full(len(motor_ids), np.nan, dtype=np.float64)
        velocity_scale = np.full(len(motor_ids), np.nan, dtype=np.float64)
        validation_rmse = np.full(len(motor_ids), np.nan, dtype=np.float64)
        validation_r2 = np.full(len(motor_ids), np.nan, dtype=np.float64)
        sample_count = np.zeros(len(motor_ids), dtype=np.int64)
        valid_sample_ratio = np.full(len(motor_ids), np.nan, dtype=np.float64)
        coulomb_std = np.full(len(motor_ids), np.nan, dtype=np.float64)
        viscous_std = np.full(len(motor_ids), np.nan, dtype=np.float64)
        offset_std = np.full(len(motor_ids), np.nan, dtype=np.float64)
        round_count = np.zeros(len(motor_ids), dtype=np.int64)
        history: dict[str, list[dict[str, Any]]] = {}

        for index, motor_id in enumerate(motor_ids):
            motor_artifacts = [artifact for artifact in artifacts if artifact.capture.target_motor_id == motor_id]
            round_count[index] = len(motor_artifacts)
            history[str(motor_id)] = [
                {
                    "group_index": artifact.capture.group_index,
                    "round_index": artifact.capture.round_index,
                    "identified": bool(artifact.identification.identified),
                    "coulomb": float(artifact.identification.coulomb),
                    "viscous": float(artifact.identification.viscous),
                    "offset": float(artifact.identification.offset),
                    "velocity_scale": float(artifact.identification.velocity_scale),
                    "valid_rmse": float(artifact.identification.valid_rmse),
                    "valid_r2": float(artifact.identification.valid_r2),
                    "sample_count": int(artifact.identification.sample_count),
                    "valid_sample_ratio": float(artifact.identification.valid_sample_ratio),
                    "capture_path": str(artifact.capture_path),
                    "identification_path": str(artifact.identification_path),
                }
                for artifact in motor_artifacts
            ]

            identified = [artifact.identification for artifact in motor_artifacts if artifact.identification.identified]
            identified_mask[index] = bool(identified)
            if identified:
                coulomb_values = [float(item.coulomb) for item in identified]
                viscous_values = [float(item.viscous) for item in identified]
                offset_values = [float(item.offset) for item in identified]
                velocity_scale_values = [float(item.velocity_scale) for item in identified]
                validation_rmse_values = [float(item.valid_rmse) for item in identified]
                validation_r2_values = [float(item.valid_r2) for item in identified]
                coulomb[index] = _finite_mean(coulomb_values)
                viscous[index] = _finite_mean(viscous_values)
                offset[index] = _finite_mean(offset_values)
                velocity_scale[index] = _finite_mean(velocity_scale_values)
                validation_rmse[index] = _finite_mean(validation_rmse_values)
                validation_r2[index] = _finite_mean(validation_r2_values)
                coulomb_std[index] = _finite_std(coulomb_values)
                viscous_std[index] = _finite_std(viscous_values)
                offset_std[index] = _finite_std(offset_values)

            mean_sample_count = _finite_mean([float(item.identification.sample_count) for item in motor_artifacts])
            sample_count[index] = 0 if not np.isfinite(mean_sample_count) else int(round(mean_sample_count))
            valid_sample_ratio[index] = _finite_mean([float(item.identification.valid_sample_ratio) for item in motor_artifacts])

        summary_payload = {
            "motor_ids": np.asarray(motor_ids, dtype=np.int64),
            "motor_names": np.asarray(motor_names),
            "identified_mask": identified_mask,
            "coulomb": coulomb,
            "viscous": viscous,
            "offset": offset,
            "velocity_scale": velocity_scale,
            "validation_rmse": validation_rmse,
            "validation_r2": validation_r2,
            "sample_count": sample_count,
            "valid_sample_ratio": valid_sample_ratio,
            "coulomb_std": coulomb_std,
            "viscous_std": viscous_std,
            "offset_std": offset_std,
            "round_count": round_count,
            "history_json": np.asarray(json.dumps(history, ensure_ascii=False)),
        }

        run_summary_path = self.summary_dir / self._config.output.summary_filename
        run_summary_csv_path = self.summary_dir / self._config.output.summary_csv_filename
        run_summary_report_path = self.summary_dir / self._config.output.summary_report_filename
        np.savez(run_summary_path, **summary_payload)
        self._write_summary_csv(run_summary_csv_path, summary_payload)
        self._write_summary_report(run_summary_report_path, summary_payload)

        root_summary_path = self.results_dir / self._config.output.summary_filename
        root_summary_csv_path = self.results_dir / self._config.output.summary_csv_filename
        root_summary_report_path = self.results_dir / self._config.output.summary_report_filename
        shutil.copyfile(run_summary_path, root_summary_path)
        shutil.copyfile(run_summary_csv_path, root_summary_csv_path)
        shutil.copyfile(run_summary_report_path, root_summary_report_path)

        self._manifest["end_time"] = utc_now_iso8601()
        self._manifest["summary_files"] = {
            "run_summary_path": str(run_summary_path),
            "run_summary_csv_path": str(run_summary_csv_path),
            "run_summary_report_path": str(run_summary_report_path),
            "root_summary_path": str(root_summary_path),
            "root_summary_csv_path": str(root_summary_csv_path),
            "root_summary_report_path": str(root_summary_report_path),
        }
        self._write_manifest()

        return SummaryPaths(
            run_summary_path=run_summary_path,
            run_summary_csv_path=run_summary_csv_path,
            run_summary_report_path=run_summary_report_path,
            root_summary_path=root_summary_path,
            root_summary_csv_path=root_summary_csv_path,
            root_summary_report_path=root_summary_report_path,
            manifest_path=self.manifest_path,
        )

    def _write_summary_csv(self, path: Path, payload: dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "motor_id",
                    "motor_name",
                    "identified",
                    "coulomb",
                    "viscous",
                    "offset",
                    "velocity_scale",
                    "validation_rmse",
                    "validation_r2",
                    "sample_count",
                    "valid_sample_ratio",
                    "coulomb_std",
                    "viscous_std",
                    "offset_std",
                    "round_count",
                ]
            )
            for index, motor_id in enumerate(payload["motor_ids"]):
                writer.writerow(
                    [
                        int(motor_id),
                        str(payload["motor_names"][index]),
                        bool(payload["identified_mask"][index]),
                        float(payload["coulomb"][index]),
                        float(payload["viscous"][index]),
                        float(payload["offset"][index]),
                        float(payload["velocity_scale"][index]),
                        float(payload["validation_rmse"][index]),
                        float(payload["validation_r2"][index]),
                        int(payload["sample_count"][index]),
                        float(payload["valid_sample_ratio"][index]),
                        float(payload["coulomb_std"][index]),
                        float(payload["viscous_std"][index]),
                        float(payload["offset_std"][index]),
                        int(payload["round_count"][index]),
                    ]
                )

    def _write_summary_report(self, path: Path, payload: dict[str, Any]) -> None:
        lines = [
            "# Sequential Motor Identification Summary",
            "",
            f"- run: `{self.run_label}`",
            f"- groups: `{self._config.group_count}`",
            f"- motor order: `{','.join(str(motor_id) for motor_id in self._config.enabled_motor_ids)}`",
            "",
            "| motor_id | name | identified | coulomb | viscous | offset | valid_rmse | valid_r2 | samples | valid_ratio |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for index, motor_id in enumerate(payload["motor_ids"]):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(int(motor_id)),
                        str(payload["motor_names"][index]),
                        "yes" if bool(payload["identified_mask"][index]) else "no",
                        f"{float(payload['coulomb'][index]):.6f}",
                        f"{float(payload['viscous'][index]):.6f}",
                        f"{float(payload['offset'][index]):.6f}",
                        f"{float(payload['validation_rmse'][index]):.6f}",
                        f"{float(payload['validation_r2'][index]):.6f}",
                        str(int(payload["sample_count"][index])),
                        f"{float(payload['valid_sample_ratio'][index]):.4f}",
                    ]
                )
                + " |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
