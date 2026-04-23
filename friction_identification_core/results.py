from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from friction_identification_core.config import Config
from friction_identification_core.models import (
    MotorDynamicIdentificationResult,
    MotorIdentificationResult,
    RoundCapture,
)
from friction_identification_core.runtime import ensure_directory, filesystem_timestamp, utc_now_iso8601, write_json


DYNAMIC_SUMMARY_FILENAME = "hardware_dynamic_identification_summary.npz"
DYNAMIC_SUMMARY_CSV_FILENAME = "hardware_dynamic_identification_summary.csv"
DYNAMIC_SUMMARY_REPORT_FILENAME = "hardware_dynamic_identification_summary.md"


@dataclass(frozen=True)
class RoundArtifact:
    capture: RoundCapture
    identification: MotorIdentificationResult | None
    dynamic_identification: MotorDynamicIdentificationResult | None
    capture_path: Path
    identification_path: Path | None
    dynamic_identification_path: Path | None


@dataclass(frozen=True)
class SummaryPaths:
    run_summary_path: Path
    run_summary_csv_path: Path
    run_summary_report_path: Path
    root_summary_path: Path
    root_summary_csv_path: Path
    root_summary_report_path: Path
    dynamic_run_summary_path: Path
    dynamic_run_summary_csv_path: Path
    dynamic_run_summary_report_path: Path
    dynamic_root_summary_path: Path
    dynamic_root_summary_csv_path: Path
    dynamic_root_summary_report_path: Path
    manifest_path: Path
    rerun_recording_path: Path


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


def _unique_strings_join(values: list[str]) -> str:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text.lower() == "nan" or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return "; ".join(ordered)


def _worst_conclusion(values: list[str]) -> str:
    ranking = {
        "not_run": 0,
        "recommended": 1,
        "caution": 2,
        "reject": 3,
    }
    selected = "not_run"
    selected_rank = -1
    for value in values:
        label = str(value).strip().lower() or "not_run"
        rank = ranking.get(label, ranking["reject"])
        if rank > selected_rank:
            selected = label
            selected_rank = rank
    return selected


class ResultStore:
    def __init__(self, config: Config, *, mode: str) -> None:
        self._config = config
        self._mode = str(mode)
        self.results_dir = ensure_directory(config.results_dir)
        self.run_label = f"{filesystem_timestamp()}_{self._mode}"
        self.run_dir = ensure_directory(self.results_dir / "runs" / self.run_label)
        self.summary_dir = ensure_directory(self.run_dir / "summary")
        self.rerun_recording_path = self.run_dir / f"{self._mode}.rrd"
        self.manifest_path = self.run_dir / "run_manifest.json"
        self._manifest: dict[str, Any] = {
            "run_label": self.run_label,
            "mode": self._mode,
            "start_time": utc_now_iso8601(),
            "end_time": None,
            "group_count": int(config.group_count),
            "motor_order": list(config.enabled_motor_ids),
            "capture_files": [],
            "identification_files": [],
            "dynamic_identification_files": [],
            "summary_files": {},
            "rerun_recording_path": str(self.rerun_recording_path),
            "config_path": str(config.config_path),
        }
        self._write_manifest()

    def _write_manifest(self) -> None:
        write_json(self.manifest_path, self._manifest)

    def record_abort_event(self, payload: dict[str, Any]) -> None:
        self._manifest["abort_event"] = _normalize_json_value(payload)
        self._write_manifest()

    def finalize(self, *, compensation_parameters_path: Path | None = None) -> None:
        self._manifest["end_time"] = utc_now_iso8601()
        if compensation_parameters_path is not None:
            self._manifest["compensation_parameters_path"] = str(compensation_parameters_path)
        self._write_manifest()

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
            command_raw=np.asarray(capture.command_raw, dtype=np.float64),
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
            identification_window_mask=np.asarray(result.identification_window_mask, dtype=bool),
            tracking_ok_mask=np.asarray(result.tracking_ok_mask, dtype=bool),
            saturation_ok_mask=np.asarray(result.saturation_ok_mask, dtype=bool),
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

    def save_dynamic_identification(self, capture: RoundCapture, result: MotorDynamicIdentificationResult) -> Path:
        path = self._motor_dir(capture.group_index, capture.target_motor_id) / "lugre_identification.npz"
        np.savez(
            path,
            motor_id=np.asarray(int(result.motor_id), dtype=np.int64),
            fc=np.asarray(float(result.fc), dtype=np.float64),
            fs=np.asarray(float(result.fs), dtype=np.float64),
            vs=np.asarray(float(result.vs), dtype=np.float64),
            sigma0=np.asarray(float(result.sigma0), dtype=np.float64),
            sigma1=np.asarray(float(result.sigma1), dtype=np.float64),
            sigma2=np.asarray(float(result.sigma2), dtype=np.float64),
            offset=np.asarray(float(result.offset), dtype=np.float64),
            torque_pred=np.asarray(result.torque_pred, dtype=np.float64),
            torque_target=np.asarray(result.torque_target, dtype=np.float64),
            sample_mask=np.asarray(result.sample_mask, dtype=bool),
            train_mask=np.asarray(result.train_mask, dtype=bool),
            valid_mask=np.asarray(result.valid_mask, dtype=bool),
            validation_warmup_mask=np.asarray(result.validation_warmup_mask, dtype=bool),
            train_rmse=np.asarray(float(result.train_rmse), dtype=np.float64),
            valid_rmse=np.asarray(float(result.valid_rmse), dtype=np.float64),
            train_r2=np.asarray(float(result.train_r2), dtype=np.float64),
            valid_r2=np.asarray(float(result.valid_r2), dtype=np.float64),
            identified=np.asarray(bool(result.identified), dtype=bool),
            metadata=_json_scalar(result.metadata),
        )
        self._manifest["dynamic_identification_files"].append(str(path))
        self._write_manifest()
        return path

    def save_summary(self, artifacts: list[RoundArtifact]) -> SummaryPaths:
        static_payload = self._build_static_summary_payload(artifacts)
        dynamic_payload = self._build_dynamic_summary_payload(artifacts)

        run_summary_path = self.summary_dir / self._config.output.summary_filename
        run_summary_csv_path = self.summary_dir / self._config.output.summary_csv_filename
        run_summary_report_path = self.summary_dir / self._config.output.summary_report_filename
        np.savez(run_summary_path, **static_payload)
        self._write_static_summary_csv(run_summary_csv_path, static_payload)
        self._write_static_summary_report(run_summary_report_path, static_payload)

        dynamic_run_summary_path = self.summary_dir / DYNAMIC_SUMMARY_FILENAME
        dynamic_run_summary_csv_path = self.summary_dir / DYNAMIC_SUMMARY_CSV_FILENAME
        dynamic_run_summary_report_path = self.summary_dir / DYNAMIC_SUMMARY_REPORT_FILENAME
        np.savez(dynamic_run_summary_path, **dynamic_payload)
        self._write_dynamic_summary_csv(dynamic_run_summary_csv_path, dynamic_payload)
        self._write_dynamic_summary_report(dynamic_run_summary_report_path, dynamic_payload)

        root_summary_path = self.results_dir / self._config.output.summary_filename
        root_summary_csv_path = self.results_dir / self._config.output.summary_csv_filename
        root_summary_report_path = self.results_dir / self._config.output.summary_report_filename
        shutil.copyfile(run_summary_path, root_summary_path)
        shutil.copyfile(run_summary_csv_path, root_summary_csv_path)
        shutil.copyfile(run_summary_report_path, root_summary_report_path)

        dynamic_root_summary_path = self.results_dir / DYNAMIC_SUMMARY_FILENAME
        dynamic_root_summary_csv_path = self.results_dir / DYNAMIC_SUMMARY_CSV_FILENAME
        dynamic_root_summary_report_path = self.results_dir / DYNAMIC_SUMMARY_REPORT_FILENAME
        shutil.copyfile(dynamic_run_summary_path, dynamic_root_summary_path)
        shutil.copyfile(dynamic_run_summary_csv_path, dynamic_root_summary_csv_path)
        shutil.copyfile(dynamic_run_summary_report_path, dynamic_root_summary_report_path)

        self._manifest["summary_files"] = {
            "run_summary_path": str(run_summary_path),
            "run_summary_csv_path": str(run_summary_csv_path),
            "run_summary_report_path": str(run_summary_report_path),
            "root_summary_path": str(root_summary_path),
            "root_summary_csv_path": str(root_summary_csv_path),
            "root_summary_report_path": str(root_summary_report_path),
            "dynamic_run_summary_path": str(dynamic_run_summary_path),
            "dynamic_run_summary_csv_path": str(dynamic_run_summary_csv_path),
            "dynamic_run_summary_report_path": str(dynamic_run_summary_report_path),
            "dynamic_root_summary_path": str(dynamic_root_summary_path),
            "dynamic_root_summary_csv_path": str(dynamic_root_summary_csv_path),
            "dynamic_root_summary_report_path": str(dynamic_root_summary_report_path),
        }
        self.finalize()

        return SummaryPaths(
            run_summary_path=run_summary_path,
            run_summary_csv_path=run_summary_csv_path,
            run_summary_report_path=run_summary_report_path,
            root_summary_path=root_summary_path,
            root_summary_csv_path=root_summary_csv_path,
            root_summary_report_path=root_summary_report_path,
            dynamic_run_summary_path=dynamic_run_summary_path,
            dynamic_run_summary_csv_path=dynamic_run_summary_csv_path,
            dynamic_run_summary_report_path=dynamic_run_summary_report_path,
            dynamic_root_summary_path=dynamic_root_summary_path,
            dynamic_root_summary_csv_path=dynamic_root_summary_csv_path,
            dynamic_root_summary_report_path=dynamic_root_summary_report_path,
            manifest_path=self.manifest_path,
            rerun_recording_path=self.rerun_recording_path,
        )

    def _base_history(self, artifact: RoundArtifact) -> dict[str, Any]:
        return {
            "group_index": int(artifact.capture.group_index),
            "round_index": int(artifact.capture.round_index),
            "capture_path": str(artifact.capture_path),
            "identification_path": "-" if artifact.identification_path is None else str(artifact.identification_path),
            "dynamic_identification_path": (
                "-"
                if artifact.dynamic_identification_path is None
                else str(artifact.dynamic_identification_path)
            ),
            "sequence_error_count": int(artifact.capture.metadata.get("sequence_error_count", 0)),
            "sequence_error_ratio": float(artifact.capture.metadata.get("sequence_error_ratio", 0.0)),
            "target_frame_count": int(artifact.capture.metadata.get("target_frame_count", artifact.capture.sample_count)),
            "target_frame_ratio": float(artifact.capture.metadata.get("target_frame_ratio", 0.0)),
            "planned_duration_s": float(artifact.capture.metadata.get("planned_duration_s", np.nan)),
            "actual_capture_duration_s": float(artifact.capture.metadata.get("actual_capture_duration_s", np.nan)),
            "round_total_duration_s": float(artifact.capture.metadata.get("round_total_duration_s", np.nan)),
            "synced_before_capture": bool(artifact.capture.metadata.get("synced_before_capture", False)),
        }

    def _build_static_summary_payload(self, artifacts: list[RoundArtifact]) -> dict[str, np.ndarray]:
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
        sequence_error_count = np.zeros(len(motor_ids), dtype=np.int64)
        sequence_error_ratio = np.full(len(motor_ids), np.nan, dtype=np.float64)
        target_frame_count = np.zeros(len(motor_ids), dtype=np.int64)
        target_frame_ratio = np.full(len(motor_ids), np.nan, dtype=np.float64)
        planned_duration_s = np.full(len(motor_ids), np.nan, dtype=np.float64)
        actual_capture_duration_s = np.full(len(motor_ids), np.nan, dtype=np.float64)
        round_total_duration_s = np.full(len(motor_ids), np.nan, dtype=np.float64)
        synced_before_capture = np.zeros(len(motor_ids), dtype=bool)
        coulomb_std = np.full(len(motor_ids), np.nan, dtype=np.float64)
        viscous_std = np.full(len(motor_ids), np.nan, dtype=np.float64)
        offset_std = np.full(len(motor_ids), np.nan, dtype=np.float64)
        round_count = np.zeros(len(motor_ids), dtype=np.int64)
        status = np.full(len(motor_ids), "not_run", dtype="<U256")
        validation_mode = np.full(len(motor_ids), "-", dtype="<U64")
        validation_reason = np.full(len(motor_ids), "-", dtype="<U512")
        train_velocity_bands = np.full(len(motor_ids), "-", dtype="<U1024")
        valid_velocity_bands = np.full(len(motor_ids), "-", dtype="<U1024")
        recommended_for_runtime = np.zeros(len(motor_ids), dtype=bool)
        conclusion_level = np.full(len(motor_ids), "not_run", dtype="<U64")
        conclusion_text = np.full(len(motor_ids), "-", dtype="<U1024")
        saturation_ratio = np.full(len(motor_ids), np.nan, dtype=np.float64)
        tracking_error_ratio = np.full(len(motor_ids), np.nan, dtype=np.float64)
        history: dict[str, list[dict[str, Any]]] = {}

        for index, motor_id in enumerate(motor_ids):
            motor_artifacts = [artifact for artifact in artifacts if artifact.capture.target_motor_id == motor_id]
            static_artifacts = [artifact for artifact in motor_artifacts if artifact.identification is not None]
            round_count[index] = len(motor_artifacts)
            history[str(motor_id)] = []
            for artifact in motor_artifacts:
                history[str(motor_id)].append(
                    {
                        **self._base_history(artifact),
                        "identified": bool(artifact.identification is not None and artifact.identification.identified),
                        "coulomb": float(np.nan if artifact.identification is None else artifact.identification.coulomb),
                        "viscous": float(np.nan if artifact.identification is None else artifact.identification.viscous),
                        "offset": float(np.nan if artifact.identification is None else artifact.identification.offset),
                        "velocity_scale": float(
                            np.nan if artifact.identification is None else artifact.identification.velocity_scale
                        ),
                        "validation_rmse": float(
                            np.nan if artifact.identification is None else artifact.identification.valid_rmse
                        ),
                        "validation_r2": float(
                            np.nan if artifact.identification is None else artifact.identification.valid_r2
                        ),
                        "sample_count": int(0 if artifact.identification is None else artifact.identification.sample_count),
                        "status": str(
                            "not_run"
                            if artifact.identification is None
                            else artifact.identification.metadata.get("status", "unknown")
                        ),
                        "validation_mode": str(
                            ""
                            if artifact.identification is None
                            else artifact.identification.metadata.get("validation_mode", "")
                        ),
                        "validation_reason": str(
                            ""
                            if artifact.identification is None
                            else artifact.identification.metadata.get("validation_reason", "")
                        ),
                        "train_velocity_bands": list(
                            []
                            if artifact.identification is None
                            else artifact.identification.metadata.get("train_velocity_bands", [])
                        ),
                        "valid_velocity_bands": list(
                            []
                            if artifact.identification is None
                            else artifact.identification.metadata.get("valid_velocity_bands", [])
                        ),
                        "recommended_for_runtime": bool(
                            False
                            if artifact.identification is None
                            else artifact.identification.metadata.get("recommended_for_runtime", False)
                        ),
                        "conclusion_level": str(
                            "not_run"
                            if artifact.identification is None
                            else artifact.identification.metadata.get("conclusion_level", "reject")
                        ),
                        "conclusion_text": str(
                            ""
                            if artifact.identification is None
                            else artifact.identification.metadata.get("conclusion_text", "")
                        ),
                        "saturation_ratio": float(
                            np.nan
                            if artifact.identification is None
                            else artifact.identification.metadata.get("saturation_ratio", np.nan)
                        ),
                        "tracking_error_ratio": float(
                            np.nan
                            if artifact.identification is None
                            else artifact.identification.metadata.get("tracking_error_ratio", np.nan)
                        ),
                    }
                )

            identified = [artifact.identification for artifact in static_artifacts if artifact.identification.identified]
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

            mean_sample_count = _finite_mean([float(item.identification.sample_count) for item in static_artifacts if item.identification is not None])
            sample_count[index] = 0 if not np.isfinite(mean_sample_count) else int(round(mean_sample_count))
            valid_sample_ratio[index] = _finite_mean([float(item.identification.valid_sample_ratio) for item in static_artifacts if item.identification is not None])
            mean_sequence_error_count = _finite_mean([float(item.capture.metadata.get("sequence_error_count", np.nan)) for item in motor_artifacts])
            sequence_error_count[index] = 0 if not np.isfinite(mean_sequence_error_count) else int(round(mean_sequence_error_count))
            sequence_error_ratio[index] = _finite_mean([float(item.capture.metadata.get("sequence_error_ratio", np.nan)) for item in motor_artifacts])
            mean_target_frame_count = _finite_mean([float(item.capture.metadata.get("target_frame_count", np.nan)) for item in motor_artifacts])
            target_frame_count[index] = 0 if not np.isfinite(mean_target_frame_count) else int(round(mean_target_frame_count))
            target_frame_ratio[index] = _finite_mean([float(item.capture.metadata.get("target_frame_ratio", np.nan)) for item in motor_artifacts])
            planned_duration_s[index] = _finite_mean([float(item.capture.metadata.get("planned_duration_s", np.nan)) for item in motor_artifacts])
            actual_capture_duration_s[index] = _finite_mean([float(item.capture.metadata.get("actual_capture_duration_s", np.nan)) for item in motor_artifacts])
            round_total_duration_s[index] = _finite_mean([float(item.capture.metadata.get("round_total_duration_s", np.nan)) for item in motor_artifacts])
            synced_before_capture[index] = bool(motor_artifacts) and all(bool(item.capture.metadata.get("synced_before_capture", False)) for item in motor_artifacts)
            status[index] = _unique_strings_join([str(item.identification.metadata.get("status", "unknown")) for item in static_artifacts if item.identification is not None])
            validation_mode[index] = _unique_strings_join([str(item.identification.metadata.get("validation_mode", "")) for item in static_artifacts if item.identification is not None])
            validation_reason[index] = _unique_strings_join([str(item.identification.metadata.get("validation_reason", "")) for item in static_artifacts if item.identification is not None])
            train_velocity_bands[index] = _unique_strings_join([",".join(item.identification.metadata.get("train_velocity_bands", [])) for item in static_artifacts if item.identification is not None])
            valid_velocity_bands[index] = _unique_strings_join([",".join(item.identification.metadata.get("valid_velocity_bands", [])) for item in static_artifacts if item.identification is not None])
            recommended_for_runtime[index] = bool(static_artifacts) and all(bool(item.identification.metadata.get("recommended_for_runtime", False)) for item in static_artifacts if item.identification is not None)
            conclusion_level[index] = _worst_conclusion([str(item.identification.metadata.get("conclusion_level", "reject")) for item in static_artifacts if item.identification is not None])
            conclusion_text[index] = _unique_strings_join([str(item.identification.metadata.get("conclusion_text", "")) for item in static_artifacts if item.identification is not None]) or "-"
            saturation_ratio[index] = _finite_mean([float(item.identification.metadata.get("saturation_ratio", np.nan)) for item in static_artifacts if item.identification is not None])
            tracking_error_ratio[index] = _finite_mean([float(item.identification.metadata.get("tracking_error_ratio", np.nan)) for item in static_artifacts if item.identification is not None])

        return {
            "motor_ids": np.asarray(motor_ids, dtype=np.int64),
            "motor_names": np.asarray(motor_names),
            "identified_mask": identified_mask,
            "status": status,
            "coulomb": coulomb,
            "viscous": viscous,
            "offset": offset,
            "velocity_scale": velocity_scale,
            "validation_rmse": validation_rmse,
            "validation_r2": validation_r2,
            "sample_count": sample_count,
            "valid_sample_ratio": valid_sample_ratio,
            "sequence_error_count": sequence_error_count,
            "sequence_error_ratio": sequence_error_ratio,
            "target_frame_count": target_frame_count,
            "target_frame_ratio": target_frame_ratio,
            "planned_duration_s": planned_duration_s,
            "actual_capture_duration_s": actual_capture_duration_s,
            "round_total_duration_s": round_total_duration_s,
            "synced_before_capture": synced_before_capture,
            "validation_mode": validation_mode,
            "validation_reason": validation_reason,
            "train_velocity_bands": train_velocity_bands,
            "valid_velocity_bands": valid_velocity_bands,
            "recommended_for_runtime": recommended_for_runtime,
            "conclusion_level": conclusion_level,
            "conclusion_text": conclusion_text,
            "saturation_ratio": saturation_ratio,
            "tracking_error_ratio": tracking_error_ratio,
            "coulomb_std": coulomb_std,
            "viscous_std": viscous_std,
            "offset_std": offset_std,
            "round_count": round_count,
            "history_json": np.asarray(json.dumps(history, ensure_ascii=False)),
        }

    def _build_dynamic_summary_payload(self, artifacts: list[RoundArtifact]) -> dict[str, np.ndarray]:
        motor_ids = list(self._config.motor_ids)
        motor_names = [self._config.motors.name_for(motor_id) for motor_id in motor_ids]
        identified_mask = np.zeros(len(motor_ids), dtype=bool)
        fc = np.full(len(motor_ids), np.nan, dtype=np.float64)
        fs = np.full(len(motor_ids), np.nan, dtype=np.float64)
        vs = np.full(len(motor_ids), np.nan, dtype=np.float64)
        sigma0 = np.full(len(motor_ids), np.nan, dtype=np.float64)
        sigma1 = np.full(len(motor_ids), np.nan, dtype=np.float64)
        sigma2 = np.full(len(motor_ids), np.nan, dtype=np.float64)
        offset = np.full(len(motor_ids), np.nan, dtype=np.float64)
        validation_rmse = np.full(len(motor_ids), np.nan, dtype=np.float64)
        validation_r2 = np.full(len(motor_ids), np.nan, dtype=np.float64)
        static_validation_rmse = np.full(len(motor_ids), np.nan, dtype=np.float64)
        sample_count = np.zeros(len(motor_ids), dtype=np.int64)
        valid_sample_ratio = np.full(len(motor_ids), np.nan, dtype=np.float64)
        round_count = np.zeros(len(motor_ids), dtype=np.int64)
        status = np.full(len(motor_ids), "not_run", dtype="<U256")
        validation_mode = np.full(len(motor_ids), "-", dtype="<U64")
        validation_reason = np.full(len(motor_ids), "-", dtype="<U512")
        train_cycles = np.full(len(motor_ids), "-", dtype="<U1024")
        valid_cycles = np.full(len(motor_ids), "-", dtype="<U1024")
        history: dict[str, list[dict[str, Any]]] = {}

        for index, motor_id in enumerate(motor_ids):
            motor_artifacts = [artifact for artifact in artifacts if artifact.capture.target_motor_id == motor_id]
            dynamic_artifacts = [artifact for artifact in motor_artifacts if artifact.dynamic_identification is not None]
            round_count[index] = len(motor_artifacts)
            history[str(motor_id)] = []
            for artifact in motor_artifacts:
                history[str(motor_id)].append(
                    {
                        **self._base_history(artifact),
                        "identified": bool(
                            artifact.dynamic_identification is not None and artifact.dynamic_identification.identified
                        ),
                        "fc": float(np.nan if artifact.dynamic_identification is None else artifact.dynamic_identification.fc),
                        "fs": float(np.nan if artifact.dynamic_identification is None else artifact.dynamic_identification.fs),
                        "vs": float(np.nan if artifact.dynamic_identification is None else artifact.dynamic_identification.vs),
                        "sigma0": float(np.nan if artifact.dynamic_identification is None else artifact.dynamic_identification.sigma0),
                        "sigma1": float(np.nan if artifact.dynamic_identification is None else artifact.dynamic_identification.sigma1),
                        "sigma2": float(np.nan if artifact.dynamic_identification is None else artifact.dynamic_identification.sigma2),
                        "offset": float(np.nan if artifact.dynamic_identification is None else artifact.dynamic_identification.offset),
                        "validation_rmse": float(np.nan if artifact.dynamic_identification is None else artifact.dynamic_identification.valid_rmse),
                        "validation_r2": float(np.nan if artifact.dynamic_identification is None else artifact.dynamic_identification.valid_r2),
                        "sample_count": int(0 if artifact.dynamic_identification is None else artifact.dynamic_identification.sample_count),
                        "status": str(
                            "not_run"
                            if artifact.dynamic_identification is None
                            else artifact.dynamic_identification.metadata.get("status", "unknown")
                        ),
                        "validation_mode": str(
                            ""
                            if artifact.dynamic_identification is None
                            else artifact.dynamic_identification.metadata.get("validation_mode", "")
                        ),
                        "validation_reason": str(
                            ""
                            if artifact.dynamic_identification is None
                            else artifact.dynamic_identification.metadata.get("validation_reason", "")
                        ),
                        "train_cycles": list(
                            []
                            if artifact.dynamic_identification is None
                            else artifact.dynamic_identification.metadata.get("train_cycles", [])
                        ),
                        "valid_cycles": list(
                            []
                            if artifact.dynamic_identification is None
                            else artifact.dynamic_identification.metadata.get("valid_cycles", [])
                        ),
                        "static_validation_rmse": float(
                            np.nan
                            if artifact.dynamic_identification is None
                            else artifact.dynamic_identification.metadata.get("static_validation_rmse", np.nan)
                        ),
                    }
                )

            identified = [artifact.dynamic_identification for artifact in dynamic_artifacts if artifact.dynamic_identification.identified]
            identified_mask[index] = bool(identified)
            if identified:
                fc[index] = _finite_mean([float(item.fc) for item in identified])
                fs[index] = _finite_mean([float(item.fs) for item in identified])
                vs[index] = _finite_mean([float(item.vs) for item in identified])
                sigma0[index] = _finite_mean([float(item.sigma0) for item in identified])
                sigma1[index] = _finite_mean([float(item.sigma1) for item in identified])
                sigma2[index] = _finite_mean([float(item.sigma2) for item in identified])
                offset[index] = _finite_mean([float(item.offset) for item in identified])
                validation_rmse[index] = _finite_mean([float(item.valid_rmse) for item in identified])
                validation_r2[index] = _finite_mean([float(item.valid_r2) for item in identified])
                static_validation_rmse[index] = _finite_mean(
                    [float(item.metadata.get("static_validation_rmse", np.nan)) for item in identified]
                )

            mean_sample_count = _finite_mean([float(item.dynamic_identification.sample_count) for item in dynamic_artifacts if item.dynamic_identification is not None])
            sample_count[index] = 0 if not np.isfinite(mean_sample_count) else int(round(mean_sample_count))
            valid_sample_ratio[index] = _finite_mean([float(item.dynamic_identification.valid_sample_ratio) for item in dynamic_artifacts if item.dynamic_identification is not None])
            status[index] = _unique_strings_join([str(item.dynamic_identification.metadata.get("status", "unknown")) for item in dynamic_artifacts if item.dynamic_identification is not None])
            validation_mode[index] = _unique_strings_join([str(item.dynamic_identification.metadata.get("validation_mode", "")) for item in dynamic_artifacts if item.dynamic_identification is not None])
            validation_reason[index] = _unique_strings_join([str(item.dynamic_identification.metadata.get("validation_reason", "")) for item in dynamic_artifacts if item.dynamic_identification is not None])
            train_cycles[index] = _unique_strings_join([",".join(str(value) for value in item.dynamic_identification.metadata.get("train_cycles", [])) for item in dynamic_artifacts if item.dynamic_identification is not None])
            valid_cycles[index] = _unique_strings_join([",".join(str(value) for value in item.dynamic_identification.metadata.get("valid_cycles", [])) for item in dynamic_artifacts if item.dynamic_identification is not None])

        return {
            "motor_ids": np.asarray(motor_ids, dtype=np.int64),
            "motor_names": np.asarray(motor_names),
            "identified_mask": identified_mask,
            "status": status,
            "fc": fc,
            "fs": fs,
            "vs": vs,
            "sigma0": sigma0,
            "sigma1": sigma1,
            "sigma2": sigma2,
            "offset": offset,
            "validation_rmse": validation_rmse,
            "validation_r2": validation_r2,
            "static_validation_rmse": static_validation_rmse,
            "sample_count": sample_count,
            "valid_sample_ratio": valid_sample_ratio,
            "validation_mode": validation_mode,
            "validation_reason": validation_reason,
            "train_cycles": train_cycles,
            "valid_cycles": valid_cycles,
            "round_count": round_count,
            "history_json": np.asarray(json.dumps(history, ensure_ascii=False)),
        }

    def _write_static_summary_csv(self, path: Path, payload: dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "motor_id",
                    "motor_name",
                    "status",
                    "identified",
                    "synced_before_capture",
                    "sequence_error_count",
                    "sequence_error_ratio",
                    "target_frame_count",
                    "target_frame_ratio",
                    "planned_duration_s",
                    "actual_capture_duration_s",
                    "round_total_duration_s",
                    "validation_mode",
                    "validation_reason",
                    "train_velocity_bands",
                    "valid_velocity_bands",
                    "recommended_for_runtime",
                    "conclusion_level",
                    "conclusion_text",
                    "saturation_ratio",
                    "tracking_error_ratio",
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
                        str(payload["status"][index]),
                        bool(payload["identified_mask"][index]),
                        bool(payload["synced_before_capture"][index]),
                        int(payload["sequence_error_count"][index]),
                        float(payload["sequence_error_ratio"][index]),
                        int(payload["target_frame_count"][index]),
                        float(payload["target_frame_ratio"][index]),
                        float(payload["planned_duration_s"][index]),
                        float(payload["actual_capture_duration_s"][index]),
                        float(payload["round_total_duration_s"][index]),
                        str(payload["validation_mode"][index]),
                        str(payload["validation_reason"][index]),
                        str(payload["train_velocity_bands"][index]),
                        str(payload["valid_velocity_bands"][index]),
                        bool(payload["recommended_for_runtime"][index]),
                        str(payload["conclusion_level"][index]),
                        str(payload["conclusion_text"][index]),
                        float(payload["saturation_ratio"][index]),
                        float(payload["tracking_error_ratio"][index]),
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

    def _write_dynamic_summary_csv(self, path: Path, payload: dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "motor_id",
                    "motor_name",
                    "status",
                    "identified",
                    "validation_mode",
                    "validation_reason",
                    "train_cycles",
                    "valid_cycles",
                    "fc",
                    "fs",
                    "vs",
                    "sigma0",
                    "sigma1",
                    "sigma2",
                    "offset",
                    "validation_rmse",
                    "validation_r2",
                    "static_validation_rmse",
                    "sample_count",
                    "valid_sample_ratio",
                    "round_count",
                ]
            )
            for index, motor_id in enumerate(payload["motor_ids"]):
                writer.writerow(
                    [
                        int(motor_id),
                        str(payload["motor_names"][index]),
                        str(payload["status"][index]),
                        bool(payload["identified_mask"][index]),
                        str(payload["validation_mode"][index]),
                        str(payload["validation_reason"][index]),
                        str(payload["train_cycles"][index]),
                        str(payload["valid_cycles"][index]),
                        float(payload["fc"][index]),
                        float(payload["fs"][index]),
                        float(payload["vs"][index]),
                        float(payload["sigma0"][index]),
                        float(payload["sigma1"][index]),
                        float(payload["sigma2"][index]),
                        float(payload["offset"][index]),
                        float(payload["validation_rmse"][index]),
                        float(payload["validation_r2"][index]),
                        float(payload["static_validation_rmse"][index]),
                        int(payload["sample_count"][index]),
                        float(payload["valid_sample_ratio"][index]),
                        int(payload["round_count"][index]),
                    ]
                )

    def _write_static_summary_report(self, path: Path, payload: dict[str, Any]) -> None:
        lines = [
            "# 曲线辨识窗口 / 速度带验证 Summary",
            "",
            f"- run: `{self.run_label}`",
            f"- groups: `{self._config.group_count}`",
            f"- motor order: `{','.join(str(motor_id) for motor_id in self._config.enabled_motor_ids)}`",
            "",
            "| motor_id | name | conclusion | recommended_for_runtime | status | valid_rmse |",
            "| --- | --- | --- | --- | --- | ---: |",
        ]
        for index, motor_id in enumerate(payload["motor_ids"]):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(int(motor_id)),
                        str(payload["motor_names"][index]),
                        str(payload["conclusion_level"][index]),
                        "true" if bool(payload["recommended_for_runtime"][index]) else "false",
                        str(payload["status"][index]),
                        f"{float(payload['validation_rmse'][index]):.6f}",
                    ]
                )
                + " |"
            )
        lines.extend(["", "## Velocity-Band Coverage", ""])
        for index, motor_id in enumerate(payload["motor_ids"]):
            lines.extend(
                [
                    f"### Motor {int(motor_id):02d} {str(payload['motor_names'][index])}",
                    "",
                    f"- train_velocity_bands: `{str(payload['train_velocity_bands'][index]) or '-'}`",
                    f"- valid_velocity_bands: `{str(payload['valid_velocity_bands'][index]) or '-'}`",
                    f"- validation_mode: `{str(payload['validation_mode'][index]) or '-'}`",
                    f"- validation_reason: `{str(payload['validation_reason'][index]) or '-'}`",
                    f"- conclusion_level: `{str(payload['conclusion_level'][index]) or '-'}`",
                    f"- conclusion_text: `{str(payload['conclusion_text'][index]) or '-'}`",
                    "",
                ]
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_dynamic_summary_report(self, path: Path, payload: dict[str, Any]) -> None:
        lines = [
            "# LuGre 动态辨识 / 周期留出验证 Summary",
            "",
            f"- run: `{self.run_label}`",
            f"- groups: `{self._config.group_count}`",
            f"- motor order: `{','.join(str(motor_id) for motor_id in self._config.enabled_motor_ids)}`",
            "",
            "| motor_id | name | status | valid_rmse | static_valid_rmse | valid_cycles |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
        for index, motor_id in enumerate(payload["motor_ids"]):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(int(motor_id)),
                        str(payload["motor_names"][index]),
                        str(payload["status"][index]),
                        f"{float(payload['validation_rmse'][index]):.6f}",
                        f"{float(payload['static_validation_rmse'][index]):.6f}",
                        str(payload["valid_cycles"][index]),
                    ]
                )
                + " |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
