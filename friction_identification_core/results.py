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
    identification: MotorIdentificationResult | None
    capture_path: Path
    identification_path: Path | None


@dataclass(frozen=True)
class SummaryPaths:
    run_summary_path: Path
    run_summary_csv_path: Path
    run_summary_report_path: Path
    root_summary_path: Path
    root_summary_csv_path: Path
    root_summary_report_path: Path
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


def _finite_max(values: list[float]) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def _finite_min(values: list[float]) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.min(finite))


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
            "summary_files": {},
            "rerun_recording_path": str(self.rerun_recording_path),
            "config_path": str(config.config_path),
        }
        self._write_manifest()

    def _write_manifest(self) -> None:
        write_json(self.manifest_path, self._manifest)

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
            steady_state_mask=np.asarray(result.steady_state_mask, dtype=bool),
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
        train_platforms = np.full(len(motor_ids), "-", dtype="<U1024")
        valid_platforms = np.full(len(motor_ids), "-", dtype="<U1024")
        recommended_for_runtime = np.zeros(len(motor_ids), dtype=bool)
        conclusion_level = np.full(len(motor_ids), "not_run", dtype="<U64")
        conclusion_text = np.full(len(motor_ids), "-", dtype="<U1024")
        saturation_ratio = np.full(len(motor_ids), np.nan, dtype=np.float64)
        tracking_error_ratio = np.full(len(motor_ids), np.nan, dtype=np.float64)
        high_speed_platform_count = np.zeros(len(motor_ids), dtype=np.int64)
        high_speed_valid_rmse = np.full(len(motor_ids), np.nan, dtype=np.float64)
        history: dict[str, list[dict[str, Any]]] = {}

        for index, motor_id in enumerate(motor_ids):
            motor_artifacts = [artifact for artifact in artifacts if artifact.capture.target_motor_id == motor_id]
            identified_artifacts = [artifact for artifact in motor_artifacts if artifact.identification is not None]
            round_count[index] = len(motor_artifacts)
            history[str(motor_id)] = [
                {
                    "group_index": artifact.capture.group_index,
                    "round_index": artifact.capture.round_index,
                    "identified": bool(artifact.identification is not None and artifact.identification.identified),
                    "coulomb": float(np.nan if artifact.identification is None else artifact.identification.coulomb),
                    "viscous": float(np.nan if artifact.identification is None else artifact.identification.viscous),
                    "offset": float(np.nan if artifact.identification is None else artifact.identification.offset),
                    "velocity_scale": float(
                        np.nan if artifact.identification is None else artifact.identification.velocity_scale
                    ),
                    "valid_rmse": float(np.nan if artifact.identification is None else artifact.identification.valid_rmse),
                    "valid_r2": float(np.nan if artifact.identification is None else artifact.identification.valid_r2),
                    "sample_count": int(0 if artifact.identification is None else artifact.identification.sample_count),
                    "valid_sample_ratio": float(
                        np.nan if artifact.identification is None else artifact.identification.valid_sample_ratio
                    ),
                    "status": str(
                        "not_run"
                        if artifact.identification is None
                        else artifact.identification.metadata.get("status", "unknown")
                    ),
                    "sequence_error_count": int(artifact.capture.metadata.get("sequence_error_count", 0)),
                    "sequence_error_ratio": float(artifact.capture.metadata.get("sequence_error_ratio", 0.0)),
                    "target_frame_count": int(artifact.capture.metadata.get("target_frame_count", artifact.capture.sample_count)),
                    "target_frame_ratio": float(artifact.capture.metadata.get("target_frame_ratio", 0.0)),
                    "planned_duration_s": float(artifact.capture.metadata.get("planned_duration_s", np.nan)),
                    "actual_capture_duration_s": float(
                        artifact.capture.metadata.get("actual_capture_duration_s", np.nan)
                    ),
                    "round_total_duration_s": float(artifact.capture.metadata.get("round_total_duration_s", np.nan)),
                    "synced_before_capture": bool(artifact.capture.metadata.get("synced_before_capture", False)),
                    "validation_mode": str(
                        "" if artifact.identification is None else artifact.identification.metadata.get("validation_mode", "")
                    ),
                    "validation_reason": str(
                        ""
                        if artifact.identification is None
                        else artifact.identification.metadata.get("validation_reason", "")
                    ),
                    "train_platforms": list(
                        [] if artifact.identification is None else artifact.identification.metadata.get("train_platforms", [])
                    ),
                    "valid_platforms": list(
                        [] if artifact.identification is None else artifact.identification.metadata.get("valid_platforms", [])
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
                        "" if artifact.identification is None else artifact.identification.metadata.get("conclusion_text", "")
                    ),
                    "saturation_ratio": float(
                        np.nan if artifact.identification is None else artifact.identification.metadata.get("saturation_ratio", np.nan)
                    ),
                    "tracking_error_ratio": float(
                        np.nan
                        if artifact.identification is None
                        else artifact.identification.metadata.get("tracking_error_ratio", np.nan)
                    ),
                    "high_speed_platform_count": int(
                        0 if artifact.identification is None else artifact.identification.metadata.get("high_speed_platform_count", 0)
                    ),
                    "high_speed_valid_rmse": float(
                        np.nan
                        if artifact.identification is None
                        else artifact.identification.metadata.get("high_speed_valid_rmse", np.nan)
                    ),
                    "dropped_platforms": list(
                        [] if artifact.identification is None else artifact.identification.metadata.get("dropped_platforms", [])
                    ),
                    "capture_path": str(artifact.capture_path),
                    "identification_path": "-" if artifact.identification_path is None else str(artifact.identification_path),
                }
                for artifact in motor_artifacts
            ]

            identified = [artifact.identification for artifact in identified_artifacts if artifact.identification.identified]
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
            mean_sequence_error_count = _finite_mean(
                [float(item.capture.metadata.get("sequence_error_count", np.nan)) for item in motor_artifacts]
            )
            sequence_error_count[index] = (
                0 if not np.isfinite(mean_sequence_error_count) else int(round(mean_sequence_error_count))
            )
            sequence_error_ratio[index] = _finite_mean(
                [float(item.capture.metadata.get("sequence_error_ratio", np.nan)) for item in motor_artifacts]
            )
            mean_target_frame_count = _finite_mean(
                [float(item.capture.metadata.get("target_frame_count", np.nan)) for item in motor_artifacts]
            )
            target_frame_count[index] = 0 if not np.isfinite(mean_target_frame_count) else int(round(mean_target_frame_count))
            target_frame_ratio[index] = _finite_mean(
                [float(item.capture.metadata.get("target_frame_ratio", np.nan)) for item in motor_artifacts]
            )
            planned_duration_s[index] = _finite_mean(
                [float(item.capture.metadata.get("planned_duration_s", np.nan)) for item in motor_artifacts]
            )
            actual_capture_duration_s[index] = _finite_mean(
                [float(item.capture.metadata.get("actual_capture_duration_s", np.nan)) for item in motor_artifacts]
            )
            round_total_duration_s[index] = _finite_mean(
                [float(item.capture.metadata.get("round_total_duration_s", np.nan)) for item in motor_artifacts]
            )
            synced_before_capture[index] = bool(motor_artifacts) and all(
                bool(item.capture.metadata.get("synced_before_capture", False)) for item in motor_artifacts
            )
            status[index] = _unique_strings_join([
                str(item.identification.metadata.get("status", "unknown"))
                for item in identified_artifacts
                if item.identification is not None
            ])
            validation_mode[index] = _unique_strings_join([
                str(item.identification.metadata.get("validation_mode", ""))
                for item in identified_artifacts
                if item.identification is not None
            ])
            validation_reason[index] = _unique_strings_join([
                str(item.identification.metadata.get("validation_reason", ""))
                for item in identified_artifacts
                if item.identification is not None
            ])
            train_platforms[index] = _unique_strings_join([
                ",".join(item.identification.metadata.get("train_platforms", []))
                for item in identified_artifacts
                if item.identification is not None
            ])
            valid_platforms[index] = _unique_strings_join([
                ",".join(item.identification.metadata.get("valid_platforms", []))
                for item in identified_artifacts
                if item.identification is not None
            ])
            conclusion_level_values = [
                str(item.identification.metadata.get("conclusion_level", "reject")) for item in motor_artifacts
                if item.identification is not None
            ]
            conclusion_level[index] = _worst_conclusion(conclusion_level_values)
            recommended_for_runtime[index] = bool(identified_artifacts) and all(
                bool(item.identification.metadata.get("recommended_for_runtime", False))
                for item in identified_artifacts
                if item.identification is not None
            )
            conclusion_text[index] = _unique_strings_join(
                [
                    str(item.identification.metadata.get("conclusion_text", ""))
                    for item in identified_artifacts
                    if item.identification is not None
                ]
            ) or "-"
            saturation_ratio[index] = _finite_max(
                [
                    float(item.identification.metadata.get("saturation_ratio", np.nan))
                    for item in identified_artifacts
                    if item.identification is not None
                ]
            )
            tracking_error_ratio[index] = _finite_max(
                [
                    float(item.identification.metadata.get("tracking_error_ratio", np.nan))
                    for item in identified_artifacts
                    if item.identification is not None
                ]
            )
            min_high_speed_count = _finite_min(
                [
                    float(item.identification.metadata.get("high_speed_platform_count", np.nan))
                    for item in identified_artifacts
                    if item.identification is not None
                ]
            )
            high_speed_platform_count[index] = (
                0 if not np.isfinite(min_high_speed_count) else int(round(min_high_speed_count))
            )
            high_speed_valid_rmse[index] = _finite_max(
                [
                    float(item.identification.metadata.get("high_speed_valid_rmse", np.nan))
                    for item in identified_artifacts
                    if item.identification is not None
                ]
            )

        summary_payload = {
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
            "train_platforms": train_platforms,
            "valid_platforms": valid_platforms,
            "recommended_for_runtime": recommended_for_runtime,
            "conclusion_level": conclusion_level,
            "conclusion_text": conclusion_text,
            "saturation_ratio": saturation_ratio,
            "tracking_error_ratio": tracking_error_ratio,
            "high_speed_platform_count": high_speed_platform_count,
            "high_speed_valid_rmse": high_speed_valid_rmse,
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

        self._manifest["summary_files"] = {
            "run_summary_path": str(run_summary_path),
            "run_summary_csv_path": str(run_summary_csv_path),
            "run_summary_report_path": str(run_summary_report_path),
            "root_summary_path": str(root_summary_path),
            "root_summary_csv_path": str(root_summary_csv_path),
            "root_summary_report_path": str(root_summary_report_path),
        }
        self.finalize()

        return SummaryPaths(
            run_summary_path=run_summary_path,
            run_summary_csv_path=run_summary_csv_path,
            run_summary_report_path=run_summary_report_path,
            root_summary_path=root_summary_path,
            root_summary_csv_path=root_summary_csv_path,
            root_summary_report_path=root_summary_report_path,
            manifest_path=self.manifest_path,
            rerun_recording_path=self.rerun_recording_path,
        )

    def _write_summary_csv(self, path: Path, payload: dict[str, Any]) -> None:
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
                    "recommended_for_runtime",
                    "conclusion_level",
                    "conclusion_text",
                    "train_platforms",
                    "valid_platforms",
                    "saturation_ratio",
                    "tracking_error_ratio",
                    "high_speed_platform_count",
                    "high_speed_valid_rmse",
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
                        bool(payload["recommended_for_runtime"][index]),
                        str(payload["conclusion_level"][index]),
                        str(payload["conclusion_text"][index]),
                        str(payload["train_platforms"][index]),
                        str(payload["valid_platforms"][index]),
                        float(payload["saturation_ratio"][index]),
                        float(payload["tracking_error_ratio"][index]),
                        int(payload["high_speed_platform_count"][index]),
                        float(payload["high_speed_valid_rmse"][index]),
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
            "| motor_id | name | conclusion | recommended_for_runtime | status | high_speed_platform_count | high_speed_valid_rmse | saturation_ratio | tracking_error_ratio | valid_rmse |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
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
                        str(int(payload["high_speed_platform_count"][index])),
                        f"{float(payload['high_speed_valid_rmse'][index]):.6f}",
                        f"{float(payload['saturation_ratio'][index]):.4f}",
                        f"{float(payload['tracking_error_ratio'][index]):.4f}",
                        f"{float(payload['validation_rmse'][index]):.6f}",
                    ]
                )
                + " |"
            )
        lines.extend(["", "## Runtime Conclusions", ""])
        for index, motor_id in enumerate(payload["motor_ids"]):
            lines.extend(
                [
                    f"### Motor {int(motor_id):02d} {str(payload['motor_names'][index])}",
                    "",
                    f"- recommended_for_runtime: `{'true' if bool(payload['recommended_for_runtime'][index]) else 'false'}`",
                    f"- conclusion_level: `{str(payload['conclusion_level'][index])}`",
                    f"- conclusion_text: `{str(payload['conclusion_text'][index])}`",
                    f"- planned_duration_s: `{float(payload['planned_duration_s'][index]):.6f}`",
                    f"- actual_capture_duration_s: `{float(payload['actual_capture_duration_s'][index]):.6f}`",
                    f"- round_total_duration_s: `{float(payload['round_total_duration_s'][index]):.6f}`",
                    (
                        f"- core_parameters: `coulomb={float(payload['coulomb'][index]):.6f}, "
                        f"viscous={float(payload['viscous'][index]):.6f}, "
                        f"offset={float(payload['offset'][index]):.6f}, "
                        f"velocity_scale={float(payload['velocity_scale'][index]):.6f}`"
                    ),
                    f"- high_speed_platform_count: `{int(payload['high_speed_platform_count'][index])}`",
                    f"- high_speed_valid_rmse: `{float(payload['high_speed_valid_rmse'][index]):.6f}`",
                    "",
                ]
            )
        lines.extend(["## Platform Coverage", ""])
        for index, motor_id in enumerate(payload["motor_ids"]):
            lines.extend(
                [
                    f"### Motor {int(motor_id):02d} {str(payload['motor_names'][index])}",
                    "",
                    f"- train_platforms: `{str(payload['train_platforms'][index]) or '-'}`",
                    f"- valid_platforms: `{str(payload['valid_platforms'][index]) or '-'}`",
                    f"- validation_mode: `{str(payload['validation_mode'][index]) or '-'}`",
                    f"- validation_reason: `{str(payload['validation_reason'][index]) or '-'}`",
                    f"- recommended_for_runtime: `{'true' if bool(payload['recommended_for_runtime'][index]) else 'false'}`",
                    f"- conclusion_level: `{str(payload['conclusion_level'][index])}`",
                    f"- sequence_error_count: `{int(payload['sequence_error_count'][index])}`",
                    f"- target_frame_count: `{int(payload['target_frame_count'][index])}`",
                    f"- planned_duration_s: `{float(payload['planned_duration_s'][index]):.6f}`",
                    f"- actual_capture_duration_s: `{float(payload['actual_capture_duration_s'][index]):.6f}`",
                    f"- round_total_duration_s: `{float(payload['round_total_duration_s'][index]):.6f}`",
                    f"- saturation_ratio: `{float(payload['saturation_ratio'][index]):.4f}`",
                    f"- tracking_error_ratio: `{float(payload['tracking_error_ratio'][index]):.4f}`",
                    "",
                ]
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
