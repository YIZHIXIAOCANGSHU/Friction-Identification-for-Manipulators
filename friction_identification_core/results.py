from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from friction_identification_core.config import Config
from friction_identification_core.models import CollectedData, FrictionIdentificationResult
from friction_identification_core.runtime import ensure_directory, log_info, write_json


@dataclass(frozen=True)
class ResultPaths:
    npz_path: Path
    json_path: Path | None = None
    report_path: Path | None = None


@dataclass(frozen=True)
class JointResult:
    joint_index: int
    joint_name: str
    coulomb: float
    viscous: float
    offset: float
    velocity_scale: float
    validation_rmse: float
    validation_r2: float
    sample_count: int
    valid_sample_ratio: float
    identified: bool = True


@dataclass(frozen=True)
class IdentificationResults:
    timestamp: str
    batch_count: int
    joint_results: list[JointResult]
    metadata: dict[str, Any]
    batch_coulomb: np.ndarray | None = None
    batch_viscous: np.ndarray | None = None
    batch_offset: np.ndarray | None = None
    batch_validation_rmse: np.ndarray | None = None
    batch_validation_r2: np.ndarray | None = None
    batch_valid_sample_ratio: np.ndarray | None = None
    batch_sample_count: np.ndarray | None = None


def build_validation_mask(num_samples: int) -> np.ndarray:
    mask = np.zeros(num_samples, dtype=bool)
    if num_samples > 0:
        mask[::5] = True
        mask[: min(20, num_samples)] = False
        if np.all(mask):
            mask[-1] = False
    return mask


def _now_iso8601() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _json_blob(payload: dict[str, Any]) -> np.ndarray:
    return np.asarray(json.dumps(_normalize_json_value(payload), ensure_ascii=False))


def _safe_nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite_count = np.sum(np.isfinite(values), axis=axis)
    summed = np.nansum(values, axis=axis)
    mean = np.divide(
        summed,
        np.maximum(finite_count, 1),
        out=np.zeros_like(summed, dtype=np.float64),
        where=np.maximum(finite_count, 1) > 0,
    )
    mean = np.asarray(mean, dtype=np.float64)
    mean[finite_count == 0] = np.nan
    return mean


def _safe_nanstd(values: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mean = _safe_nanmean(values, axis=axis)
    centered = values - np.expand_dims(mean, axis=axis)
    centered[~np.isfinite(values)] = np.nan
    variance = _safe_nanmean(centered * centered, axis=axis)
    return np.sqrt(np.nan_to_num(variance, nan=0.0))


class ResultsManager:
    """Batch-oriented result storage for hardware parallel identification."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.results_dir = ensure_directory(config.results_dir)

    def capture_batch_path(self, batch_index: int) -> Path:
        return self.results_dir / f"{self.config.output.hardware_capture_prefix}_batch_{batch_index:02d}.npz"

    def identification_batch_path(self, batch_index: int) -> Path:
        return self.results_dir / f"{self.config.output.hardware_ident_prefix}_batch_{batch_index:02d}.npz"

    def compensation_validation_path(self) -> Path:
        return self.config.compensation_validation_path

    def save_collection(
        self,
        data: CollectedData,
        *,
        batch_index: int,
        total_batches: int,
    ) -> ResultPaths:
        if data.mode == "compensate":
            path = self.compensation_validation_path()
        else:
            path = self.capture_batch_path(batch_index)
        payload = self._serialize_collection(data, batch_index=batch_index, total_batches=total_batches)
        np.savez(path, **payload)
        log_info(f"批次数据已保存: {path}")
        return ResultPaths(npz_path=path)

    def save_identification(
        self,
        data: CollectedData,
        result: FrictionIdentificationResult | None,
        *,
        batch_index: int,
        total_batches: int,
    ) -> ResultPaths | None:
        if result is None:
            return None
        path = self.identification_batch_path(batch_index)
        payload = self._serialize_identification(
            data,
            result,
            batch_index=batch_index,
            total_batches=total_batches,
        )
        np.savez(path, **payload)
        log_info(f"批次辨识结果已保存: {path}")
        return ResultPaths(npz_path=path)

    def save_summary(self, batches: list[Any]) -> ResultPaths:
        timestamp = _now_iso8601()
        identified_batches = [batch for batch in batches if batch.identification is not None]
        joint_count = self.config.joint_count
        batch_count = len(batches)

        batch_coulomb = np.full((len(identified_batches), joint_count), np.nan, dtype=np.float64)
        batch_viscous = np.full_like(batch_coulomb, np.nan)
        batch_offset = np.full_like(batch_coulomb, np.nan)
        batch_rmse = np.full_like(batch_coulomb, np.nan)
        batch_r2 = np.full_like(batch_coulomb, np.nan)
        batch_sample_count = np.zeros((batch_count, joint_count), dtype=np.int64)
        batch_valid_ratio = np.zeros((batch_count, joint_count), dtype=np.float64)

        for batch_idx, batch in enumerate(batches):
            clean_mask = np.asarray(batch.data.clean_mask, dtype=bool) if batch.data.clean_mask is not None else None
            retained = int(np.count_nonzero(clean_mask)) if clean_mask is not None else batch.data.sample_count
            valid_ratio = retained / max(batch.data.sample_count, 1)
            batch_sample_count[batch_idx, self.config.active_joint_indices] = retained
            batch_valid_ratio[batch_idx, self.config.active_joint_indices] = valid_ratio

        for fit_idx, batch in enumerate(identified_batches):
            fit_joint_indices = list(batch.data.metadata.get("fit_joint_indices", self.config.identification.active_joints))
            for local_idx, joint_idx in enumerate(fit_joint_indices):
                if local_idx >= len(batch.identification.parameters):
                    break
                params = batch.identification.parameters[local_idx]
                batch_coulomb[fit_idx, joint_idx] = float(params.coulomb)
                batch_viscous[fit_idx, joint_idx] = float(params.viscous)
                batch_offset[fit_idx, joint_idx] = float(params.offset)
                batch_rmse[fit_idx, joint_idx] = float(batch.identification.validation_rmse[local_idx])
                batch_r2[fit_idx, joint_idx] = float(batch.identification.validation_r2[local_idx])

        coulomb = _safe_nanmean(batch_coulomb, axis=0) if identified_batches else np.zeros(joint_count, dtype=np.float64)
        viscous = _safe_nanmean(batch_viscous, axis=0) if identified_batches else np.zeros(joint_count, dtype=np.float64)
        offset = _safe_nanmean(batch_offset, axis=0) if identified_batches else np.zeros(joint_count, dtype=np.float64)
        validation_rmse = _safe_nanmean(batch_rmse, axis=0) if identified_batches else np.full(joint_count, np.nan)
        validation_r2 = _safe_nanmean(batch_r2, axis=0) if identified_batches else np.full(joint_count, np.nan)
        coulomb_std = _safe_nanstd(batch_coulomb, axis=0) if identified_batches else np.zeros(joint_count, dtype=np.float64)
        viscous_std = _safe_nanstd(batch_viscous, axis=0) if identified_batches else np.zeros(joint_count, dtype=np.float64)
        offset_std = _safe_nanstd(batch_offset, axis=0) if identified_batches else np.zeros(joint_count, dtype=np.float64)
        sample_count = np.max(batch_sample_count, axis=0) if batch_count > 0 else np.zeros(joint_count, dtype=np.int64)
        valid_sample_ratio = np.mean(batch_valid_ratio, axis=0) if batch_count > 0 else np.zeros(joint_count, dtype=np.float64)
        identified_mask = np.zeros(joint_count, dtype=bool)
        identified_mask[self.config.active_joint_indices] = np.isfinite(coulomb[self.config.active_joint_indices])

        summary_payload = {
            "metadata": _json_blob(
                {
                    "timestamp": timestamp,
                    "batch_count": batch_count,
                    "joint_names": list(self.config.robot.joint_names),
                    "active_joints": list(self.config.identification.active_joints),
                    "config_path": str(self.config.config_path),
                    "config_snapshot": _normalize_json_value(asdict(self.config)),
                }
            ),
            "identified_mask": identified_mask,
            "coulomb": np.nan_to_num(coulomb, nan=0.0),
            "viscous": np.nan_to_num(viscous, nan=0.0),
            "offset": np.nan_to_num(offset, nan=0.0),
            "validation_rmse": validation_rmse,
            "validation_r2": validation_r2,
            "sample_count": sample_count,
            "valid_sample_ratio": valid_sample_ratio,
            "coulomb_consistency_std": np.nan_to_num(coulomb_std, nan=0.0),
            "viscous_consistency_std": np.nan_to_num(viscous_std, nan=0.0),
            "offset_consistency_std": np.nan_to_num(offset_std, nan=0.0),
            "batch_coulomb": batch_coulomb,
            "batch_viscous": batch_viscous,
            "batch_offset": batch_offset,
            "batch_validation_rmse": batch_rmse,
            "batch_validation_r2": batch_r2,
            "batch_sample_count": batch_sample_count,
            "batch_valid_sample_ratio": batch_valid_ratio,
        }
        np.savez(self.config.summary_path, **summary_payload)

        legacy_json = {
            "timestamp": timestamp,
            "batch_count": batch_count,
            "joint_names": list(self.config.robot.joint_names),
            "active_joints": list(self.config.identification.active_joints),
            "estimated_coulomb": np.nan_to_num(coulomb, nan=0.0).tolist(),
            "estimated_viscous": np.nan_to_num(viscous, nan=0.0).tolist(),
            "estimated_offset": np.nan_to_num(offset, nan=0.0).tolist(),
            "validation_rmse": _normalize_json_value(validation_rmse),
            "validation_r2": _normalize_json_value(validation_r2),
            "valid_sample_ratio": _normalize_json_value(valid_sample_ratio),
            "coulomb_consistency_std": _normalize_json_value(coulomb_std),
            "viscous_consistency_std": _normalize_json_value(viscous_std),
            "offset_consistency_std": _normalize_json_value(offset_std),
        }
        json_path = write_json(self.results_dir / self.config.output.legacy_summary_filename, legacy_json)
        report_path = self._write_summary_report(
            timestamp=timestamp,
            batch_count=batch_count,
            coulomb=coulomb,
            viscous=viscous,
            offset=offset,
            validation_rmse=validation_rmse,
            validation_r2=validation_r2,
            valid_sample_ratio=valid_sample_ratio,
            coulomb_std=coulomb_std,
            viscous_std=viscous_std,
            offset_std=offset_std,
        )
        log_info(f"汇总结果已保存: {self.config.summary_path}")
        return ResultPaths(
            npz_path=self.config.summary_path,
            json_path=json_path,
            report_path=report_path,
        )

    def load_summary(self, path: Path | None = None) -> IdentificationResults:
        target = path or self.config.summary_path
        if not Path(target).exists():
            raise FileNotFoundError(target)
        with np.load(target, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
            joint_results = []
            joint_names = metadata.get("joint_names", list(self.config.robot.joint_names))
            batch_coulomb = np.asarray(payload["batch_coulomb"], dtype=np.float64) if "batch_coulomb" in payload else None
            batch_viscous = np.asarray(payload["batch_viscous"], dtype=np.float64) if "batch_viscous" in payload else None
            batch_offset = np.asarray(payload["batch_offset"], dtype=np.float64) if "batch_offset" in payload else None
            batch_validation_rmse = (
                np.asarray(payload["batch_validation_rmse"], dtype=np.float64)
                if "batch_validation_rmse" in payload
                else None
            )
            batch_validation_r2 = (
                np.asarray(payload["batch_validation_r2"], dtype=np.float64)
                if "batch_validation_r2" in payload
                else None
            )
            batch_valid_sample_ratio = (
                np.asarray(payload["batch_valid_sample_ratio"], dtype=np.float64)
                if "batch_valid_sample_ratio" in payload
                else None
            )
            batch_sample_count = (
                np.asarray(payload["batch_sample_count"], dtype=np.int64) if "batch_sample_count" in payload else None
            )
            for joint_idx, joint_name in enumerate(joint_names):
                joint_results.append(
                    JointResult(
                        joint_index=joint_idx,
                        joint_name=str(joint_name),
                        coulomb=float(np.asarray(payload["coulomb"], dtype=np.float64)[joint_idx]),
                        viscous=float(np.asarray(payload["viscous"], dtype=np.float64)[joint_idx]),
                        offset=float(np.asarray(payload["offset"], dtype=np.float64)[joint_idx]),
                        velocity_scale=float(self.config.fitting.velocity_scale),
                        validation_rmse=float(np.asarray(payload["validation_rmse"], dtype=np.float64)[joint_idx]),
                        validation_r2=float(np.asarray(payload["validation_r2"], dtype=np.float64)[joint_idx]),
                        sample_count=int(np.asarray(payload["sample_count"], dtype=np.int64)[joint_idx]),
                        valid_sample_ratio=float(
                            np.asarray(payload["valid_sample_ratio"], dtype=np.float64)[joint_idx]
                        ),
                        identified=bool(np.asarray(payload["identified_mask"], dtype=bool)[joint_idx]),
                    )
                )
        return IdentificationResults(
            timestamp=str(metadata.get("timestamp", "")),
            batch_count=int(metadata.get("batch_count", 0)),
            joint_results=joint_results,
            metadata=metadata,
            batch_coulomb=batch_coulomb,
            batch_viscous=batch_viscous,
            batch_offset=batch_offset,
            batch_validation_rmse=batch_validation_rmse,
            batch_validation_r2=batch_validation_r2,
            batch_valid_sample_ratio=batch_valid_sample_ratio,
            batch_sample_count=batch_sample_count,
        )

    def _serialize_collection(
        self,
        data: CollectedData,
        *,
        batch_index: int,
        total_batches: int,
    ) -> dict[str, np.ndarray]:
        payload: dict[str, np.ndarray] = {
            "metadata": _json_blob(
                {
                    "source": data.source,
                    "mode": data.mode,
                    "batch_index": batch_index,
                    "total_batches": total_batches,
                    "config_path": str(self.config.config_path),
                    "joint_names": list(self.config.robot.joint_names),
                    "sample_count": int(data.sample_count),
                    **_normalize_json_value(data.metadata),
                }
            ),
            "time": np.asarray(data.time),
            "q": np.asarray(data.q),
            "qd": np.asarray(data.qd),
            "q_cmd": np.asarray(data.q_cmd),
            "qd_cmd": np.asarray(data.qd_cmd),
            "tau_command": np.asarray(data.tau_command),
            "tau_measured": np.asarray(data.tau_measured),
        }
        optional_arrays = {
            "qdd": data.qdd,
            "qdd_cmd": data.qdd_cmd,
            "tau_track_ff": data.tau_track_ff,
            "tau_track_fb": data.tau_track_fb,
            "tau_friction_comp": data.tau_friction_comp,
            "tau_rigid": data.tau_rigid,
            "tau_residual": data.tau_residual,
            "tau_friction": data.tau_friction,
            "clean_mask": data.clean_mask,
            "rotation_state": data.rotation_state,
            "range_ratio": data.range_ratio,
            "limit_margin_remaining": data.limit_margin_remaining,
            "batch_index": data.batch_index,
            "phase_name": data.phase_name,
            "ee_pos": data.ee_pos,
            "ee_quat": data.ee_quat,
            "mos_temperature": data.mos_temperature,
            "coil_temperature": data.coil_temperature,
            "uart_cycle_hz": data.uart_cycle_hz,
            "uart_latency_ms": data.uart_latency_ms,
            "uart_transfer_kbps": data.uart_transfer_kbps,
        }
        for key, value in optional_arrays.items():
            if value is None:
                continue
            if key == "clean_mask":
                payload[key] = np.asarray(value, dtype=bool)
            else:
                payload[key] = np.asarray(value)
        return payload

    def _serialize_identification(
        self,
        data: CollectedData,
        result: FrictionIdentificationResult,
        *,
        batch_index: int,
        total_batches: int,
    ) -> dict[str, np.ndarray]:
        payload = self._serialize_collection(data, batch_index=batch_index, total_batches=total_batches)
        fit_joint_indices = list(data.metadata.get("fit_joint_indices", self.config.identification.active_joints))
        payload.update(
            {
                "fit_joint_indices": np.asarray(fit_joint_indices, dtype=np.int64),
                "coulomb": np.asarray([param.coulomb for param in result.parameters], dtype=np.float64),
                "viscous": np.asarray([param.viscous for param in result.parameters], dtype=np.float64),
                "offset": np.asarray([param.offset for param in result.parameters], dtype=np.float64),
                "velocity_scale": np.asarray([param.velocity_scale for param in result.parameters], dtype=np.float64),
                "predicted_torque": np.asarray(result.predicted_torque, dtype=np.float64),
                "measured_torque_fit": np.asarray(result.measured_torque, dtype=np.float64),
                "train_mask": np.asarray(result.train_mask, dtype=bool),
                "validation_mask": np.asarray(result.validation_mask, dtype=bool),
                "train_rmse": np.asarray(result.train_rmse, dtype=np.float64),
                "validation_rmse": np.asarray(result.validation_rmse, dtype=np.float64),
                "train_r2": np.asarray(result.train_r2, dtype=np.float64),
                "validation_r2": np.asarray(result.validation_r2, dtype=np.float64),
            }
        )
        return payload

    def _write_summary_report(
        self,
        *,
        timestamp: str,
        batch_count: int,
        coulomb: np.ndarray,
        viscous: np.ndarray,
        offset: np.ndarray,
        validation_rmse: np.ndarray,
        validation_r2: np.ndarray,
        valid_sample_ratio: np.ndarray,
        coulomb_std: np.ndarray,
        viscous_std: np.ndarray,
        offset_std: np.ndarray,
    ) -> Path:
        lines = [
            "# Hardware Parallel Friction Identification Report",
            "",
            f"- Timestamp: {timestamp}",
            f"- Batches: {batch_count}",
            f"- Active joints: {list(self.config.identification.active_joints)}",
            "",
            "| Joint | Coulomb | Viscous | Offset | Val RMSE | Val R2 | Valid Ratio | Coulomb Std | Viscous Std | Offset Std |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for joint_idx, joint_name in enumerate(self.config.robot.joint_names):
            lines.append(
                f"| {joint_name} | {coulomb[joint_idx]:.6f} | {viscous[joint_idx]:.6f} | "
                f"{offset[joint_idx]:.6f} | {validation_rmse[joint_idx]:.6f} | {validation_r2[joint_idx]:.4f} | "
                f"{valid_sample_ratio[joint_idx]:.4f} | {coulomb_std[joint_idx]:.6f} | "
                f"{viscous_std[joint_idx]:.6f} | {offset_std[joint_idx]:.6f} |"
            )
        report_path = self.config.report_path
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path


ResultStore = ResultsManager


__all__ = [
    "IdentificationResults",
    "JointResult",
    "ResultPaths",
    "ResultsManager",
    "ResultStore",
    "build_validation_mask",
]
