from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from friction_identification_core.config import Config
from friction_identification_core.models import CollectedData, FrictionIdentificationResult, JointFrictionParameters
from friction_identification_core.runtime import ensure_directory, log_info, write_json


RUNS_DIRNAME = "runs"
COMPARISONS_DIRNAME = "comparisons"
RUN_MANIFEST_FILENAME = "run_manifest.json"
LATEST_COLLECT_MANIFEST_FILENAME = "latest_collect_manifest.json"
LATEST_COMPARISON_MANIFEST_FILENAME = "latest_comparison_manifest.json"


@dataclass(frozen=True)
class ResultPaths:
    npz_path: Path | None = None
    json_path: Path | None = None
    report_path: Path | None = None
    csv_path: Path | None = None
    manifest_path: Path | None = None
    archive_dir: Path | None = None


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


@dataclass(frozen=True)
class ArchivedRunSummary:
    run_label: str
    mode: str
    timestamp: str
    batch_count: int
    archive_dir: Path
    summary_path: Path
    report_path: Path | None
    csv_path: Path | None
    manifest_path: Path
    summary: IdentificationResults


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


def _filesystem_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


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


def _mean_of_finite(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _summary_csv_filename(config: Config) -> str:
    return f"{Path(config.output.hardware_summary_filename).stem}.csv"


def _summary_readable_json_filename(config: Config) -> str:
    return config.output.legacy_summary_filename


class ResultsManager:
    """Batch-oriented result storage for hardware parallel identification."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.results_dir = ensure_directory(config.results_dir)
        self.archive_root = ensure_directory(self.results_dir / RUNS_DIRNAME)
        self.comparison_root = ensure_directory(self.results_dir / COMPARISONS_DIRNAME)
        self.run_label = _filesystem_timestamp()
        self.run_started_at = _now_iso8601()
        self.run_mode: str | None = None
        self.run_dir: Path | None = None
        self.manifest_path: Path | None = None
        self._manifest: dict[str, Any] = {}

    def begin_run(self, *, mode: str, total_batches: int) -> Path:
        if self.run_dir is not None:
            return self.run_dir
        if mode == "sequential":
            resumed = self._resume_incomplete_sequential_run(total_batches=total_batches)
            if resumed is not None:
                return resumed
        if mode in {"collect", "sequential"}:
            self._archive_existing_latest_collect_if_needed()
        self.run_mode = str(mode)
        self.run_dir = ensure_directory(self.archive_root / f"{self.run_label}_{mode}")
        self.manifest_path = self.run_dir / RUN_MANIFEST_FILENAME
        self._manifest = {
            "run_label": self.run_label,
            "mode": self.run_mode,
            "started_at": self.run_started_at,
            "total_batches": int(total_batches),
            "config_path": str(self.config.config_path),
            "results_root": str(self.results_dir),
            "archive_dir": str(self.run_dir),
            "summary_npz_path": None,
            "summary_report_path": None,
            "summary_csv_path": None,
            "summary_json_path": None,
            "comparison_report_path": None,
            "comparison_csv_path": None,
            "collection_batches": [],
            "identification_batches": [],
            "compensation_validation_path": None,
        }
        self._write_manifest()
        log_info(f"本次运行已归档到: {self.run_dir}")
        return self.run_dir

    def _resume_incomplete_sequential_run(self, *, total_batches: int) -> Path | None:
        manifest_payloads: list[tuple[Path, dict[str, Any]]] = []
        for manifest_path in sorted(self.archive_root.glob(f"*_sequential/{RUN_MANIFEST_FILENAME}")):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(payload.get("mode", "")).strip().lower() != "sequential":
                continue
            if payload.get("finished_at"):
                continue
            if int(payload.get("total_batches", total_batches)) != int(total_batches):
                continue
            if payload.get("config_path") and str(payload.get("config_path")) != str(self.config.config_path):
                continue
            manifest_payloads.append((manifest_path, payload))

        if not manifest_payloads:
            return None

        manifest_path, payload = manifest_payloads[-1]
        self.run_label = str(payload.get("run_label", manifest_path.parent.name.removesuffix("_sequential")))
        self.run_started_at = str(payload.get("started_at", self.run_started_at))
        self.run_mode = "sequential"
        self.run_dir = manifest_path.parent
        self.manifest_path = manifest_path
        self._manifest = payload
        log_info(f"检测到未完成的逐电机运行，继续在原目录断点续跑: {self.run_dir}")
        return self.run_dir

    def capture_batch_path(self, batch_index: int) -> Path:
        return self._require_run_dir() / f"{self.config.output.hardware_capture_prefix}_batch_{batch_index:02d}.npz"

    def identification_batch_path(self, batch_index: int) -> Path:
        return self._require_run_dir() / f"{self.config.output.hardware_ident_prefix}_batch_{batch_index:02d}.npz"

    def compensation_validation_path(self) -> Path:
        return self._require_run_dir() / self.config.output.hardware_compensation_filename

    def sequential_group_dir(self, group_index: int) -> Path:
        return ensure_directory(self._require_run_dir() / f"group_{int(group_index):02d}")

    def sequential_capture_path(self, *, group_index: int, joint_index: int) -> Path:
        return self.sequential_group_dir(group_index) / f"joint_{int(joint_index)}_capture.npz"

    def sequential_identification_path(self, *, group_index: int, joint_index: int) -> Path:
        return self.sequential_group_dir(group_index) / f"joint_{int(joint_index)}_identification.npz"

    def active_summary_dir(self) -> Path:
        run_dir = self._require_run_dir()
        if self.run_mode == "sequential":
            return ensure_directory(run_dir / "summary")
        return run_dir

    def active_summary_path(self) -> Path:
        return self.active_summary_dir() / self.config.output.hardware_summary_filename

    def active_report_path(self) -> Path:
        return self.active_summary_dir() / self.config.output.hardware_report_filename

    def active_summary_csv_path(self) -> Path:
        return self.active_summary_dir() / _summary_csv_filename(self.config)

    def active_summary_json_path(self) -> Path:
        return self.active_summary_dir() / _summary_readable_json_filename(self.config)

    def latest_collect_manifest_path(self) -> Path:
        return self.results_dir / LATEST_COLLECT_MANIFEST_FILENAME

    def latest_comparison_manifest_path(self) -> Path:
        return self.results_dir / LATEST_COMPARISON_MANIFEST_FILENAME

    def latest_summary_csv_path(self) -> Path:
        return self.results_dir / _summary_csv_filename(self.config)

    def save_collection(
        self,
        data: CollectedData,
        *,
        batch_index: int,
        total_batches: int,
        group_index: int | None = None,
        joint_index: int | None = None,
    ) -> ResultPaths:
        self.begin_run(mode=data.mode, total_batches=total_batches)
        if data.mode == "compensate":
            path = self.compensation_validation_path()
        elif data.mode == "sequential":
            if group_index is None or joint_index is None:
                raise ValueError("sequential collection requires group_index and joint_index.")
            path = self.sequential_capture_path(group_index=group_index, joint_index=joint_index)
        else:
            path = self.capture_batch_path(batch_index)
        payload = self._serialize_collection(data, batch_index=batch_index, total_batches=total_batches)
        np.savez(path, **payload)
        log_info(f"批次数据已保存: {path}")
        if data.mode == "compensate":
            latest_validation = self.results_dir / self.config.output.hardware_compensation_filename
            self._mirror_file(path, latest_validation)
            self._manifest["compensation_validation_path"] = str(path)
        else:
            collection_batches = list(self._manifest.get("collection_batches", []))
            collection_batches.append(str(path))
            self._manifest["collection_batches"] = collection_batches
        self._write_manifest()
        return ResultPaths(npz_path=path, manifest_path=self.manifest_path, archive_dir=self.run_dir)

    def save_identification(
        self,
        data: CollectedData,
        result: FrictionIdentificationResult | None,
        *,
        batch_index: int,
        total_batches: int,
        group_index: int | None = None,
        joint_index: int | None = None,
    ) -> ResultPaths | None:
        if result is None:
            return None
        self.begin_run(mode=data.mode, total_batches=total_batches)
        if data.mode == "sequential":
            if group_index is None or joint_index is None:
                raise ValueError("sequential identification requires group_index and joint_index.")
            path = self.sequential_identification_path(group_index=group_index, joint_index=joint_index)
        else:
            path = self.identification_batch_path(batch_index)
        payload = self._serialize_identification(
            data,
            result,
            batch_index=batch_index,
            total_batches=total_batches,
        )
        np.savez(path, **payload)
        log_info(f"批次辨识结果已保存: {path}")
        identification_batches = list(self._manifest.get("identification_batches", []))
        identification_batches.append(str(path))
        self._manifest["identification_batches"] = identification_batches
        self._write_manifest()
        return ResultPaths(npz_path=path, manifest_path=self.manifest_path, archive_dir=self.run_dir)

    def save_summary(self, batches: list[Any]) -> ResultPaths:
        summary_mode = self.run_mode or (str(batches[0].data.mode) if batches else "collect")
        self.begin_run(mode=summary_mode, total_batches=len(batches))
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
            joint_clean_mask = (
                np.asarray(batch.data.joint_clean_mask, dtype=bool)
                if batch.data.joint_clean_mask is not None
                else None
            )
            if joint_clean_mask is not None and joint_clean_mask.shape == (batch.data.sample_count, joint_count):
                retained_per_joint = np.count_nonzero(joint_clean_mask, axis=0)
                valid_ratio_per_joint = retained_per_joint / max(batch.data.sample_count, 1)
                batch_sample_count[batch_idx] = retained_per_joint
                batch_valid_ratio[batch_idx] = valid_ratio_per_joint
                continue

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

        summary_path = self.active_summary_path()
        csv_path = self.active_summary_csv_path()
        json_path = self.active_summary_json_path()
        report_path = self.active_report_path()

        summary_payload = {
            "metadata": _json_blob(
                {
                    "timestamp": timestamp,
                    "run_label": self.run_label,
                    "mode": self.run_mode,
                    "batch_count": batch_count,
                    "joint_names": list(self.config.robot.joint_names),
                    "active_joints": list(self.config.identification.active_joints),
                    "config_path": str(self.config.config_path),
                    "results_root": str(self.results_dir),
                    "archive_dir": str(self.run_dir),
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
        np.savez(summary_path, **summary_payload)

        joint_rows = self._build_joint_rows(
            coulomb=coulomb,
            viscous=viscous,
            offset=offset,
            validation_rmse=validation_rmse,
            validation_r2=validation_r2,
            valid_sample_ratio=valid_sample_ratio,
            sample_count=sample_count,
            identified_mask=identified_mask,
            coulomb_std=coulomb_std,
            viscous_std=viscous_std,
            offset_std=offset_std,
        )
        summary_json = self._build_summary_json(
            timestamp=timestamp,
            batch_count=batch_count,
            joint_rows=joint_rows,
            summary_path=summary_path,
            report_path=report_path,
            csv_path=csv_path,
            batch_coulomb=batch_coulomb,
            batch_viscous=batch_viscous,
            batch_offset=batch_offset,
            batch_validation_rmse=batch_rmse,
            batch_validation_r2=batch_r2,
            batch_valid_ratio=batch_valid_ratio,
        )
        json_path = write_json(json_path, summary_json)
        csv_path = self._write_summary_csv(csv_path, joint_rows)
        report_path = self._write_summary_report(
            timestamp=timestamp,
            batch_count=batch_count,
            summary_path=summary_path,
            json_path=json_path,
            csv_path=csv_path,
            joint_rows=joint_rows,
        )

        latest_summary_path = self._mirror_file(summary_path, self.config.summary_path)
        latest_csv_path = self._mirror_file(csv_path, self.latest_summary_csv_path())
        latest_report_path = self._mirror_file(report_path, self.config.report_path)
        latest_json_payload = dict(summary_json)
        latest_json_payload.update(
            {
                "summary_npz_path": str(latest_summary_path),
                "report_path": str(latest_report_path),
                "csv_path": str(latest_csv_path),
                "latest_collect_manifest_path": str(self.latest_collect_manifest_path()),
            }
        )
        latest_json_path = write_json(self.results_dir / self.config.output.legacy_summary_filename, latest_json_payload)

        self._manifest.update(
            {
                "summary_npz_path": str(summary_path),
                "summary_report_path": str(report_path),
                "summary_csv_path": str(csv_path),
                "summary_json_path": str(json_path),
                "latest_summary_npz_path": str(latest_summary_path),
                "latest_summary_report_path": str(latest_report_path),
                "latest_summary_csv_path": str(latest_csv_path),
                "latest_summary_json_path": str(latest_json_path),
                "finished_at": _now_iso8601(),
            }
        )
        self._write_manifest()
        write_json(
            self.latest_collect_manifest_path(),
            {
                "run_label": self.run_label,
                "mode": self.run_mode,
                "archive_dir": str(self.run_dir),
                "manifest_path": str(self.manifest_path),
                "summary_npz_path": str(latest_summary_path),
                "report_path": str(latest_report_path),
                "csv_path": str(latest_csv_path),
                "json_path": str(latest_json_path),
                "timestamp": timestamp,
            },
        )

        log_info(f"汇总结果已保存: {summary_path}")
        log_info(f"可读 Markdown 报告: {report_path}")
        log_info(f"可读 CSV 汇总: {csv_path}")
        return ResultPaths(
            npz_path=summary_path,
            json_path=json_path,
            report_path=report_path,
            csv_path=csv_path,
            manifest_path=self.manifest_path,
            archive_dir=self.run_dir,
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

    def load_collection_artifact(self, path: Path) -> CollectedData:
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item())) if "metadata" in payload else {}

            def _optional_array(key: str, *, dtype: Any | None = None) -> np.ndarray | None:
                if key not in payload:
                    return None
                return np.asarray(payload[key], dtype=dtype) if dtype is not None else np.asarray(payload[key])

            return CollectedData(
                source=str(metadata.get("source", "hardware")),
                mode=str(metadata.get("mode", "collect")),
                time=np.asarray(payload["time"], dtype=np.float64),
                q=np.asarray(payload["q"], dtype=np.float64),
                qd=np.asarray(payload["qd"], dtype=np.float64),
                q_cmd=np.asarray(payload["q_cmd"], dtype=np.float64),
                qd_cmd=np.asarray(payload["qd_cmd"], dtype=np.float64),
                tau_command=np.asarray(payload["tau_command"], dtype=np.float64),
                tau_measured=np.asarray(payload["tau_measured"], dtype=np.float64),
                qdd=_optional_array("qdd", dtype=np.float64),
                qdd_cmd=_optional_array("qdd_cmd", dtype=np.float64),
                tau_track_ff=_optional_array("tau_track_ff", dtype=np.float64),
                tau_track_fb=_optional_array("tau_track_fb", dtype=np.float64),
                tau_friction_comp=_optional_array("tau_friction_comp", dtype=np.float64),
                tau_rigid=_optional_array("tau_rigid", dtype=np.float64),
                tau_residual=_optional_array("tau_residual", dtype=np.float64),
                tau_friction=_optional_array("tau_friction", dtype=np.float64),
                clean_mask=_optional_array("clean_mask", dtype=bool),
                joint_refresh_mask=_optional_array("joint_refresh_mask", dtype=bool),
                joint_clean_mask=_optional_array("joint_clean_mask", dtype=bool),
                rotation_state=_optional_array("rotation_state", dtype=np.int8),
                range_ratio=_optional_array("range_ratio", dtype=np.float64),
                limit_margin_remaining=_optional_array("limit_margin_remaining", dtype=np.float64),
                batch_index=_optional_array("batch_index", dtype=np.int64),
                phase_name=_optional_array("phase_name"),
                ee_pos=_optional_array("ee_pos", dtype=np.float64),
                ee_quat=_optional_array("ee_quat", dtype=np.float64),
                mos_temperature=_optional_array("mos_temperature", dtype=np.float64),
                coil_temperature=_optional_array("coil_temperature", dtype=np.float64),
                uart_cycle_hz=_optional_array("uart_cycle_hz", dtype=np.float64),
                uart_latency_ms=_optional_array("uart_latency_ms", dtype=np.float64),
                uart_transfer_kbps=_optional_array("uart_transfer_kbps", dtype=np.float64),
                metadata=metadata,
            )

    def load_identification_artifact(self, path: Path) -> FrictionIdentificationResult:
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item())) if "metadata" in payload else {}
            fit_joint_indices = (
                np.asarray(payload["fit_joint_indices"], dtype=np.int64)
                if "fit_joint_indices" in payload
                else np.arange(np.asarray(payload["coulomb"], dtype=np.float64).size, dtype=np.int64)
            )
            all_joint_names = list(metadata.get("joint_names", list(self.config.robot.joint_names)))
            joint_names = [
                all_joint_names[int(joint_idx)]
                for joint_idx in fit_joint_indices
                if 0 <= int(joint_idx) < len(all_joint_names)
            ]
            return FrictionIdentificationResult(
                joint_names=joint_names,
                parameters=[
                    JointFrictionParameters(
                        coulomb=float(coulomb),
                        viscous=float(viscous),
                        offset=float(offset),
                        velocity_scale=float(velocity_scale),
                    )
                    for coulomb, viscous, offset, velocity_scale in zip(
                        np.asarray(payload["coulomb"], dtype=np.float64),
                        np.asarray(payload["viscous"], dtype=np.float64),
                        np.asarray(payload["offset"], dtype=np.float64),
                        np.asarray(payload["velocity_scale"], dtype=np.float64),
                    )
                ],
                predicted_torque=np.asarray(payload["predicted_torque"], dtype=np.float64),
                measured_torque=np.asarray(payload["measured_torque_fit"], dtype=np.float64),
                train_mask=np.asarray(payload["train_mask"], dtype=bool),
                validation_mask=np.asarray(payload["validation_mask"], dtype=bool),
                train_rmse=np.asarray(payload["train_rmse"], dtype=np.float64),
                validation_rmse=np.asarray(payload["validation_rmse"], dtype=np.float64),
                train_r2=np.asarray(payload["train_r2"], dtype=np.float64),
                validation_r2=np.asarray(payload["validation_r2"], dtype=np.float64),
            )

    def load_existing_sequential_runs(self) -> list[dict[str, Any]]:
        if self.run_mode != "sequential" or self.run_dir is None:
            return []

        runs: list[dict[str, Any]] = []
        for capture_path in sorted(self.run_dir.glob("group_*/joint_*_capture.npz")):
            try:
                data = self.load_collection_artifact(capture_path)
            except Exception:
                continue

            termination_reason = str(data.metadata.get("termination_reason", "completed")).strip().lower()
            if termination_reason in {"interrupted", "error"}:
                continue

            group_index = int(data.metadata.get("group_index", 1))
            joint_index = int(data.metadata.get("target_joint_index", data.metadata.get("joint_index", -1)))
            batch_index = int(data.metadata.get("batch_index", 0))
            identification_path = self.sequential_identification_path(
                group_index=group_index,
                joint_index=joint_index,
            )
            identification = None
            if identification_path.exists():
                try:
                    identification = self.load_identification_artifact(identification_path)
                except Exception:
                    identification = None

            runs.append(
                {
                    "group_index": group_index,
                    "joint_index": joint_index,
                    "batch_index": batch_index,
                    "data": data,
                    "identification": identification,
                    "collection_path": capture_path,
                    "identification_path": identification_path if identification_path.exists() else None,
                }
            )

        return sorted(runs, key=lambda item: (int(item["batch_index"]), int(item["joint_index"])))

    def compare_archived_runs(
        self,
        *,
        limit: int = 5,
        compare_all: bool = False,
    ) -> ResultPaths:
        archived_runs = self.list_archived_runs(mode="collect")
        if not archived_runs:
            raise FileNotFoundError(f"在 {self.archive_root} 和 {self.results_dir} 中都没有找到可比较的辨识结果。")

        ordered_runs = sorted(archived_runs, key=lambda item: (item.timestamp, item.run_label))
        selected_runs = ordered_runs if compare_all else ordered_runs[-max(int(limit), 1) :]
        stamp = _filesystem_timestamp()
        report_path = self.comparison_root / f"identification_compare_{stamp}.md"
        csv_path = self.comparison_root / f"identification_compare_{stamp}.csv"

        self._write_comparison_csv(csv_path, selected_runs)
        self._write_comparison_report(report_path, selected_runs)

        latest_report_path = self._mirror_file(report_path, self.comparison_root / "identification_compare_latest.md")
        latest_csv_path = self._mirror_file(csv_path, self.comparison_root / "identification_compare_latest.csv")
        manifest_path = write_json(
            self.latest_comparison_manifest_path(),
            {
                "generated_at": _now_iso8601(),
                "report_path": str(latest_report_path),
                "csv_path": str(latest_csv_path),
                "selected_run_labels": [item.run_label for item in selected_runs],
                "selected_archive_dirs": [str(item.archive_dir) for item in selected_runs],
            },
        )

        log_info(f"跨运行对比报告已生成: {report_path}")
        log_info(f"跨运行对比 CSV 已生成: {csv_path}")
        return ResultPaths(
            report_path=report_path,
            csv_path=csv_path,
            manifest_path=manifest_path,
            archive_dir=self.comparison_root,
        )

    def list_archived_runs(self, *, mode: str | None = None) -> list[ArchivedRunSummary]:
        runs: list[ArchivedRunSummary] = []
        for manifest_path in sorted(self.archive_root.glob(f"*/{RUN_MANIFEST_FILENAME}")):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            run_mode = str(payload.get("mode", ""))
            if mode is not None and run_mode != mode:
                continue
            summary_path_raw = payload.get("summary_npz_path")
            if not summary_path_raw:
                continue
            summary_path = Path(summary_path_raw)
            if not summary_path.exists():
                continue
            try:
                summary = self.load_summary(summary_path)
            except Exception:
                continue
            runs.append(
                ArchivedRunSummary(
                    run_label=str(payload.get("run_label", manifest_path.parent.name)),
                    mode=run_mode or "collect",
                    timestamp=str(summary.timestamp or payload.get("started_at", "")),
                    batch_count=int(summary.batch_count),
                    archive_dir=manifest_path.parent,
                    summary_path=summary_path,
                    report_path=Path(payload["summary_report_path"]) if payload.get("summary_report_path") else None,
                    csv_path=Path(payload["summary_csv_path"]) if payload.get("summary_csv_path") else None,
                    manifest_path=manifest_path,
                    summary=summary,
                )
            )

        if runs:
            return runs

        if mode in {None, "collect"} and self.config.summary_path.exists():
            summary = self.load_summary(self.config.summary_path)
            runs.append(
                ArchivedRunSummary(
                    run_label="latest_root",
                    mode="collect",
                    timestamp=str(summary.timestamp),
                    batch_count=int(summary.batch_count),
                    archive_dir=self.results_dir,
                    summary_path=self.config.summary_path,
                    report_path=self.config.report_path if self.config.report_path.exists() else None,
                    csv_path=self.latest_summary_csv_path() if self.latest_summary_csv_path().exists() else None,
                    manifest_path=self.latest_collect_manifest_path(),
                    summary=summary,
                )
            )
        return runs

    def _require_run_dir(self) -> Path:
        if self.run_dir is None:
            raise RuntimeError("Result archive has not been initialized. Call begin_run() first.")
        return self.run_dir

    def _write_manifest(self) -> None:
        if self.manifest_path is None:
            return
        write_json(self.manifest_path, self._manifest)

    def _mirror_file(self, source: Path | None, destination: Path | None) -> Path | None:
        if source is None or destination is None:
            return None
        source = Path(source)
        destination = Path(destination)
        if not source.exists():
            return None
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination

    def _archive_existing_latest_collect_if_needed(self) -> None:
        if self.latest_collect_manifest_path().exists():
            try:
                payload = json.loads(self.latest_collect_manifest_path().read_text(encoding="utf-8"))
                archive_dir = Path(payload.get("archive_dir", ""))
                if archive_dir.exists():
                    return
            except Exception:
                pass

        summary_path = self.config.summary_path
        if not summary_path.exists():
            return

        import_label = f"{_filesystem_timestamp()}_legacy_collect"
        import_dir = ensure_directory(self.archive_root / import_label)
        imported_summary_path = import_dir / self.config.output.hardware_summary_filename
        imported_report_path = import_dir / self.config.output.hardware_report_filename
        imported_csv_path = import_dir / _summary_csv_filename(self.config)
        imported_json_path = import_dir / self.config.output.legacy_summary_filename

        self._mirror_file(summary_path, imported_summary_path)
        if self.config.report_path.exists():
            self._mirror_file(self.config.report_path, imported_report_path)
        if self.latest_summary_csv_path().exists():
            self._mirror_file(self.latest_summary_csv_path(), imported_csv_path)
        if (self.results_dir / self.config.output.legacy_summary_filename).exists():
            self._mirror_file(self.results_dir / self.config.output.legacy_summary_filename, imported_json_path)

        manifest_path = import_dir / RUN_MANIFEST_FILENAME
        write_json(
            manifest_path,
            {
                "run_label": import_label,
                "mode": "collect",
                "started_at": _now_iso8601(),
                "archive_dir": str(import_dir),
                "config_path": str(self.config.config_path),
                "imported_from_latest_root": True,
                "summary_npz_path": str(imported_summary_path),
                "summary_report_path": str(imported_report_path) if imported_report_path.exists() else None,
                "summary_csv_path": str(imported_csv_path) if imported_csv_path.exists() else None,
                "summary_json_path": str(imported_json_path) if imported_json_path.exists() else None,
            },
        )
        write_json(
            self.latest_collect_manifest_path(),
            {
                "run_label": import_label,
                "archive_dir": str(import_dir),
                "manifest_path": str(manifest_path),
                "summary_npz_path": str(summary_path),
                "report_path": str(self.config.report_path) if self.config.report_path.exists() else None,
                "csv_path": str(self.latest_summary_csv_path()) if self.latest_summary_csv_path().exists() else None,
                "json_path": str(self.results_dir / self.config.output.legacy_summary_filename)
                if (self.results_dir / self.config.output.legacy_summary_filename).exists()
                else None,
            },
        )
        log_info(f"已将现有最新辨识结果归档为历史运行: {import_dir}")

    def _build_joint_rows(
        self,
        *,
        coulomb: np.ndarray,
        viscous: np.ndarray,
        offset: np.ndarray,
        validation_rmse: np.ndarray,
        validation_r2: np.ndarray,
        valid_sample_ratio: np.ndarray,
        sample_count: np.ndarray,
        identified_mask: np.ndarray,
        coulomb_std: np.ndarray,
        viscous_std: np.ndarray,
        offset_std: np.ndarray,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for joint_idx, joint_name in enumerate(self.config.robot.joint_names):
            rows.append(
                {
                    "joint_index": int(joint_idx),
                    "joint_name": str(joint_name),
                    "identified": bool(identified_mask[joint_idx]),
                    "coulomb": float(coulomb[joint_idx]),
                    "viscous": float(viscous[joint_idx]),
                    "offset": float(offset[joint_idx]),
                    "validation_rmse": float(validation_rmse[joint_idx]),
                    "validation_r2": float(validation_r2[joint_idx]),
                    "valid_sample_ratio": float(valid_sample_ratio[joint_idx]),
                    "sample_count": int(sample_count[joint_idx]),
                    "coulomb_std": float(coulomb_std[joint_idx]),
                    "viscous_std": float(viscous_std[joint_idx]),
                    "offset_std": float(offset_std[joint_idx]),
                }
            )
        return rows

    def _build_summary_json(
        self,
        *,
        timestamp: str,
        batch_count: int,
        joint_rows: list[dict[str, Any]],
        summary_path: Path,
        report_path: Path,
        csv_path: Path,
        batch_coulomb: np.ndarray,
        batch_viscous: np.ndarray,
        batch_offset: np.ndarray,
        batch_validation_rmse: np.ndarray,
        batch_validation_r2: np.ndarray,
        batch_valid_ratio: np.ndarray,
    ) -> dict[str, Any]:
        active_rows = [row for row in joint_rows if row["identified"]]
        best_r2_joint = max(active_rows, key=lambda item: item["validation_r2"], default=None)
        worst_rmse_joint = max(active_rows, key=lambda item: item["validation_rmse"], default=None)
        lowest_valid_joint = min(active_rows, key=lambda item: item["valid_sample_ratio"], default=None)
        return {
            "run_label": self.run_label,
            "mode": self.run_mode,
            "timestamp": timestamp,
            "batch_count": int(batch_count),
            "config_path": str(self.config.config_path),
            "archive_dir": str(self.run_dir),
            "summary_npz_path": str(summary_path),
            "report_path": str(report_path),
            "csv_path": str(csv_path),
            "joint_names": list(self.config.robot.joint_names),
            "active_joints": list(self.config.identification.active_joints),
            "estimated_coulomb": [row["coulomb"] for row in joint_rows],
            "estimated_viscous": [row["viscous"] for row in joint_rows],
            "estimated_offset": [row["offset"] for row in joint_rows],
            "validation_rmse": [row["validation_rmse"] for row in joint_rows],
            "validation_r2": [row["validation_r2"] for row in joint_rows],
            "valid_sample_ratio": [row["valid_sample_ratio"] for row in joint_rows],
            "coulomb_consistency_std": [row["coulomb_std"] for row in joint_rows],
            "viscous_consistency_std": [row["viscous_std"] for row in joint_rows],
            "offset_consistency_std": [row["offset_std"] for row in joint_rows],
            "joint_results": joint_rows,
            "run_highlights": {
                "mean_validation_rmse": _mean_of_finite(row["validation_rmse"] for row in active_rows),
                "mean_validation_r2": _mean_of_finite(row["validation_r2"] for row in active_rows),
                "mean_valid_sample_ratio": _mean_of_finite(row["valid_sample_ratio"] for row in active_rows),
                "best_r2_joint": best_r2_joint,
                "worst_rmse_joint": worst_rmse_joint,
                "lowest_valid_ratio_joint": lowest_valid_joint,
            },
            "batch_metrics": {
                "batch_coulomb": _normalize_json_value(batch_coulomb),
                "batch_viscous": _normalize_json_value(batch_viscous),
                "batch_offset": _normalize_json_value(batch_offset),
                "batch_validation_rmse": _normalize_json_value(batch_validation_rmse),
                "batch_validation_r2": _normalize_json_value(batch_validation_r2),
                "batch_valid_sample_ratio": _normalize_json_value(batch_valid_ratio),
            },
        }

    def _write_summary_csv(self, path: Path, joint_rows: list[dict[str, Any]]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "joint_index",
                    "joint_name",
                    "identified",
                    "coulomb",
                    "viscous",
                    "offset",
                    "validation_rmse",
                    "validation_r2",
                    "valid_sample_ratio",
                    "sample_count",
                    "coulomb_std",
                    "viscous_std",
                    "offset_std",
                ],
            )
            writer.writeheader()
            for row in joint_rows:
                writer.writerow(row)
        return path

    def _write_summary_report(
        self,
        *,
        timestamp: str,
        batch_count: int,
        summary_path: Path,
        json_path: Path,
        csv_path: Path,
        joint_rows: list[dict[str, Any]],
    ) -> Path:
        active_rows = [row for row in joint_rows if row["identified"]]
        best_r2_joint = max(active_rows, key=lambda item: item["validation_r2"], default=None)
        worst_rmse_joint = max(active_rows, key=lambda item: item["validation_rmse"], default=None)
        lowest_valid_joint = min(active_rows, key=lambda item: item["valid_sample_ratio"], default=None)
        mode_label = "Sequential" if self.run_mode == "sequential" else "Parallel"
        run_unit_label = "Joint Runs" if self.run_mode == "sequential" else "Batches"
        lines = [
            f"# Hardware {mode_label} Friction Identification Report",
            "",
            "## Run Overview",
            "",
            f"- Run Label: `{self.run_label}`",
            f"- Mode: `{self.run_mode}`",
            f"- Timestamp: `{timestamp}`",
            f"- {run_unit_label}: `{batch_count}`",
            f"- Active Joints: `{list(self.config.identification.active_joints)}`",
            f"- Archive Dir: `{self.run_dir}`",
            "",
            "## Quick Read",
            "",
            f"- Mean Validation RMSE: `{_mean_of_finite(row['validation_rmse'] for row in active_rows):.6f}`",
            f"- Mean Validation R2: `{_mean_of_finite(row['validation_r2'] for row in active_rows):.4f}`",
            f"- Mean Valid Sample Ratio: `{_mean_of_finite(row['valid_sample_ratio'] for row in active_rows):.4f}`",
            f"- Best R2 Joint: `{best_r2_joint['joint_name']}` ({best_r2_joint['validation_r2']:.4f})"
            if best_r2_joint is not None
            else "- Best R2 Joint: `N/A`",
            f"- Worst RMSE Joint: `{worst_rmse_joint['joint_name']}` ({worst_rmse_joint['validation_rmse']:.6f})"
            if worst_rmse_joint is not None
            else "- Worst RMSE Joint: `N/A`",
            f"- Lowest Valid Ratio Joint: `{lowest_valid_joint['joint_name']}` ({lowest_valid_joint['valid_sample_ratio']:.4f})"
            if lowest_valid_joint is not None
            else "- Lowest Valid Ratio Joint: `N/A`",
            "",
            "## Files",
            "",
            f"- Summary NPZ: `{summary_path}`",
            f"- Readable JSON: `{json_path}`",
            f"- Readable CSV: `{csv_path}`",
            f"- Latest Root Summary: `{self.config.summary_path}`",
            f"- Compare Command: `./run.sh compare`",
            "",
            "## Joint Table",
            "",
            "| Joint | Identified | Coulomb | Viscous | Offset | Val RMSE | Val R2 | Valid Ratio | Sample Count | Coulomb Std | Viscous Std | Offset Std |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in joint_rows:
            lines.append(
                f"| {row['joint_name']} | {row['identified']} | {row['coulomb']:.6f} | {row['viscous']:.6f} | "
                f"{row['offset']:.6f} | {row['validation_rmse']:.6f} | {row['validation_r2']:.4f} | "
                f"{row['valid_sample_ratio']:.4f} | {row['sample_count']} | {row['coulomb_std']:.6f} | "
                f"{row['viscous_std']:.6f} | {row['offset_std']:.6f} |"
            )
        report_path = self.active_report_path()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    def _write_comparison_csv(self, path: Path, runs: list[ArchivedRunSummary]) -> Path:
        metric_names = [
            "coulomb",
            "viscous",
            "offset",
            "validation_rmse",
            "validation_r2",
            "valid_sample_ratio",
            "sample_count",
        ]
        fieldnames = [
            "run_label",
            "timestamp",
            "batch_count",
            "archive_dir",
            "summary_path",
            "report_path",
            "csv_path",
        ]
        for joint_idx, joint_name in enumerate(self.config.robot.joint_names, start=1):
            for metric in metric_names:
                fieldnames.append(f"J{joint_idx}_{metric}")
                fieldnames.append(f"{joint_name}_{metric}")

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for run in runs:
                row = {
                    "run_label": run.run_label,
                    "timestamp": run.timestamp,
                    "batch_count": run.batch_count,
                    "archive_dir": str(run.archive_dir),
                    "summary_path": str(run.summary_path),
                    "report_path": str(run.report_path) if run.report_path is not None else "",
                    "csv_path": str(run.csv_path) if run.csv_path is not None else "",
                }
                for joint_result in run.summary.joint_results:
                    joint_key = f"J{joint_result.joint_index + 1}"
                    joint_name_key = joint_result.joint_name
                    metric_values = {
                        "coulomb": joint_result.coulomb,
                        "viscous": joint_result.viscous,
                        "offset": joint_result.offset,
                        "validation_rmse": joint_result.validation_rmse,
                        "validation_r2": joint_result.validation_r2,
                        "valid_sample_ratio": joint_result.valid_sample_ratio,
                        "sample_count": joint_result.sample_count,
                    }
                    for metric_name, value in metric_values.items():
                        row[f"{joint_key}_{metric_name}"] = value
                        row[f"{joint_name_key}_{metric_name}"] = value
                writer.writerow(row)
        return path

    def _write_comparison_report(self, path: Path, runs: list[ArchivedRunSummary]) -> Path:
        lines = [
            "# Hardware Identification Cross-Run Comparison",
            "",
            f"- Generated At: `{_now_iso8601()}`",
            f"- Compared Runs: `{len(runs)}`",
            f"- Results Root: `{self.results_dir}`",
            "",
            "## Compared Runs",
            "",
            "| Run | Timestamp | Batches | Archive Dir | Summary | Report |",
            "|---|---|---:|---|---|---|",
        ]
        for run in runs:
            lines.append(
                f"| {run.run_label} | {run.timestamp} | {run.batch_count} | `{run.archive_dir}` | "
                f"`{run.summary_path.name}` | `{run.report_path.name if run.report_path is not None else ''}` |"
            )

        if len(runs) >= 2:
            latest = runs[-1]
            previous = runs[-2]
            lines.extend(
                [
                    "",
                    f"## Latest vs Previous",
                    "",
                    f"- Latest: `{latest.run_label}`",
                    f"- Previous: `{previous.run_label}`",
                    "",
                    "| Joint | Delta Coulomb | Delta Viscous | Delta Offset | Delta RMSE | Delta R2 | Delta Valid Ratio |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for latest_joint, previous_joint in zip(latest.summary.joint_results, previous.summary.joint_results):
                lines.append(
                    f"| {latest_joint.joint_name} | "
                    f"{latest_joint.coulomb - previous_joint.coulomb:+.6f} | "
                    f"{latest_joint.viscous - previous_joint.viscous:+.6f} | "
                    f"{latest_joint.offset - previous_joint.offset:+.6f} | "
                    f"{latest_joint.validation_rmse - previous_joint.validation_rmse:+.6f} | "
                    f"{latest_joint.validation_r2 - previous_joint.validation_r2:+.4f} | "
                    f"{latest_joint.valid_sample_ratio - previous_joint.valid_sample_ratio:+.4f} |"
                )

        for joint_idx, joint_name in enumerate(self.config.robot.joint_names):
            lines.extend(
                [
                    "",
                    f"## J{joint_idx + 1} {joint_name}",
                    "",
                    "| Run | Coulomb | Viscous | Offset | Val RMSE | Val R2 | Valid Ratio | Sample Count |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for run in runs:
                joint = run.summary.joint_results[joint_idx]
                lines.append(
                    f"| {run.run_label} | {joint.coulomb:.6f} | {joint.viscous:.6f} | {joint.offset:.6f} | "
                    f"{joint.validation_rmse:.6f} | {joint.validation_r2:.4f} | {joint.valid_sample_ratio:.4f} | "
                    f"{joint.sample_count} |"
                )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

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
                    "run_label": self.run_label,
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
            "joint_refresh_mask": data.joint_refresh_mask,
            "joint_clean_mask": data.joint_clean_mask,
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
            if key in {"clean_mask", "joint_refresh_mask", "joint_clean_mask"}:
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


def compare_saved_runs(
    config: Config,
    *,
    limit: int = 5,
    compare_all: bool = False,
) -> ResultPaths:
    return ResultsManager(config).compare_archived_runs(limit=limit, compare_all=compare_all)


ResultStore = ResultsManager


__all__ = [
    "ArchivedRunSummary",
    "IdentificationResults",
    "JointResult",
    "ResultPaths",
    "ResultsManager",
    "ResultStore",
    "build_validation_mask",
    "compare_saved_runs",
]
