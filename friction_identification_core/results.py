from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from friction_identification_core.config import Config
from friction_identification_core.models import CollectedData, FrictionIdentificationResult
from friction_identification_core.runtime import ensure_directory, log_info


_RESERVED_KEYS = {
    "metadata",
    "config_snapshot",
    "identified_mask",
    "coulomb",
    "viscous",
    "offset",
    "velocity_scale",
    "validation_rmse",
    "validation_r2",
    "sample_count",
}


@dataclass(frozen=True)
class ResultPaths:
    npz_path: Path
    json_path: Path | None = None


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
    identified: bool = True


@dataclass
class IdentificationResults:
    source_type: str
    timestamp: str
    config_snapshot: dict[str, Any]
    joints: dict[int, JointResult]
    raw_data: dict[str, np.ndarray]
    metadata: dict[str, Any]


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


def _sanitize_runtime_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in {"raw_batch", "clean_batch"}:
            continue
        if isinstance(value, np.ndarray) and value.size > 64:
            continue
        if hasattr(value, "__dataclass_fields__") and not isinstance(value, (str, bytes)):
            continue
        sanitized[str(key)] = _normalize_json_value(value)
    return sanitized


def _read_json_blob(payload: np.lib.npyio.NpzFile, key: str) -> dict[str, Any]:
    if key not in payload.files:
        return {}
    encoded = payload[key]
    if np.ndim(encoded) == 0:
        raw = encoded.item()
    else:
        raw = encoded.reshape(-1)[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _read_vector(payload: np.lib.npyio.NpzFile, key: str, joint_count: int, *, fill: float) -> np.ndarray:
    if key not in payload.files:
        return np.full(joint_count, fill, dtype=np.float64)
    values = np.asarray(payload[key], dtype=np.float64).reshape(-1)
    if values.size != joint_count:
        return np.full(joint_count, fill, dtype=np.float64)
    return values


class ResultsManager:
    """Single-file results persistence for simulation and hardware runs."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.results_dir = ensure_directory(config.results_dir)

    def save(self, results: IdentificationResults, path: Path) -> ResultPaths:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self._serialize(results)
        np.savez(target, **payload)
        return ResultPaths(npz_path=target)

    def load(self, path: Path) -> IdentificationResults:
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(target)

        with np.load(target, allow_pickle=False) as payload:
            metadata = _read_json_blob(payload, "metadata")
            config_snapshot = _read_json_blob(payload, "config_snapshot")
            joint_names = list(
                metadata.get(
                    "joint_names",
                    [f"joint_{idx + 1}" for idx in range(int(metadata.get("joint_count", 0)))],
                )
            )
            joint_count = len(joint_names)
            identified_mask = (
                np.asarray(payload["identified_mask"], dtype=bool).reshape(-1)
                if "identified_mask" in payload.files
                else np.zeros(joint_count, dtype=bool)
            )
            if joint_count == 0:
                joint_count = int(identified_mask.size)
                joint_names = [f"joint_{idx + 1}" for idx in range(joint_count)]

            coulomb = _read_vector(payload, "coulomb", joint_count, fill=0.0)
            viscous = _read_vector(payload, "viscous", joint_count, fill=0.0)
            offset = _read_vector(payload, "offset", joint_count, fill=0.0)
            velocity_scale = _read_vector(payload, "velocity_scale", joint_count, fill=np.nan)
            validation_rmse = _read_vector(payload, "validation_rmse", joint_count, fill=np.nan)
            validation_r2 = _read_vector(payload, "validation_r2", joint_count, fill=np.nan)
            sample_count = _read_vector(payload, "sample_count", joint_count, fill=0.0).astype(np.int64)

            joints = {
                joint_idx: JointResult(
                    joint_index=joint_idx,
                    joint_name=joint_names[joint_idx],
                    coulomb=float(coulomb[joint_idx]),
                    viscous=float(viscous[joint_idx]),
                    offset=float(offset[joint_idx]),
                    velocity_scale=float(velocity_scale[joint_idx]),
                    validation_rmse=float(validation_rmse[joint_idx]),
                    validation_r2=float(validation_r2[joint_idx]),
                    sample_count=int(sample_count[joint_idx]),
                    identified=bool(identified_mask[joint_idx]),
                )
                for joint_idx in range(joint_count)
            }

            raw_data = {
                key: np.asarray(payload[key])
                for key in payload.files
                if key not in _RESERVED_KEYS
            }

        source_type = str(metadata.get("source_type", target.stem.replace("_results", "")))
        timestamp = str(metadata.get("timestamp", ""))
        return IdentificationResults(
            source_type=source_type,
            timestamp=timestamp,
            config_snapshot=config_snapshot,
            joints=joints,
            raw_data=raw_data,
            metadata=metadata,
        )

    def append_joint(
        self,
        path: Path,
        joint_result: JointResult,
        *,
        raw_data: dict[str, np.ndarray] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IdentificationResults:
        target = Path(path)
        if target.exists():
            results = self.load(target)
        else:
            results = self._empty_results(self._infer_source_type(target))

        results.joints[joint_result.joint_index] = joint_result
        if raw_data:
            results.raw_data.update(raw_data)
        if metadata:
            results.metadata.update(_normalize_json_value(metadata))
        results.metadata["timestamp"] = _now_iso8601()
        results.metadata["identified_joints"] = [
            idx for idx, joint in sorted(results.joints.items()) if joint.identified
        ]
        results.timestamp = str(results.metadata["timestamp"])
        self.save(results, target)
        return self.load(target)

    def get_summary(self, path: Path) -> dict[str, Any]:
        results = self.load(path)
        ordered = [results.joints[idx] for idx in sorted(results.joints)]
        return {
            "source_type": results.source_type,
            "timestamp": results.timestamp,
            "identified_joints": [joint.joint_index for joint in ordered if joint.identified],
            "coulomb": [joint.coulomb for joint in ordered],
            "viscous": [joint.viscous for joint in ordered],
            "offset": [joint.offset for joint in ordered],
            "validation_rmse": [joint.validation_rmse for joint in ordered],
            "validation_r2": [joint.validation_r2 for joint in ordered],
            "sample_count": [joint.sample_count for joint in ordered],
        }

    def save_run(
        self,
        data: CollectedData,
        result: FrictionIdentificationResult | None = None,
    ) -> ResultPaths:
        path = self.path_for_source(data.source)
        existing = self.load(path) if path.exists() else self._empty_results(data.source)
        updated = self._merge(existing, data, result)
        saved = self.save(updated, path)
        log_info(f"{data.source} 结果已保存到聚合文件: {saved.npz_path}")
        return saved

    def save_collection(
        self,
        data: CollectedData,
        *,
        result: FrictionIdentificationResult | None = None,
    ) -> ResultPaths:
        return self.save_run(data, result if data.source == "simulation" else None)

    def save_identification(
        self,
        data: CollectedData,
        result: FrictionIdentificationResult | None,
    ) -> ResultPaths | None:
        if result is None:
            return None
        return self.save_run(data, result)

    def path_for_source(self, source_type: str) -> Path:
        normalized = source_type.strip().lower()
        if normalized == "simulation":
            return self.config.simulation_results_path
        if normalized == "hardware":
            return self.config.hardware_results_path
        raise ValueError(f"Unsupported source type: {source_type}")

    def _empty_results(self, source_type: str) -> IdentificationResults:
        metadata = {
            "source_type": source_type,
            "timestamp": "",
            "joint_names": list(self.config.robot.joint_names),
            "joint_count": int(self.config.joint_count),
            "identified_joints": [],
        }
        joints = {
            joint_idx: JointResult(
                joint_index=joint_idx,
                joint_name=self.config.robot.joint_names[joint_idx],
                coulomb=0.0,
                viscous=0.0,
                offset=0.0,
                velocity_scale=float("nan"),
                validation_rmse=float("nan"),
                validation_r2=float("nan"),
                sample_count=0,
                identified=False,
            )
            for joint_idx in range(self.config.joint_count)
        }
        return IdentificationResults(
            source_type=source_type,
            timestamp="",
            config_snapshot=_normalize_json_value(asdict(self.config)),
            joints=joints,
            raw_data={},
            metadata=metadata,
        )

    def _infer_source_type(self, path: Path) -> str:
        if "hardware" in path.stem:
            return "hardware"
        return "simulation"

    def _merge(
        self,
        existing: IdentificationResults,
        data: CollectedData,
        result: FrictionIdentificationResult | None,
    ) -> IdentificationResults:
        timestamp = _now_iso8601()
        config_snapshot = _normalize_json_value(asdict(self.config))
        metadata = {
            "source_type": data.source,
            "timestamp": timestamp,
            "latest_mode": data.mode,
            "config_path": str(self.config.config_path),
            "joint_names": list(self.config.robot.joint_names),
            "joint_count": int(self.config.joint_count),
            "target_joint": int(self.config.target_joint),
            "target_joint_name": self.config.target_joint_name,
            "sample_count": int(data.sample_count),
            "identification_available": bool(result is not None),
        }
        metadata.update(_sanitize_runtime_metadata(data.metadata))

        raw_data = self._serialize_collected_data(data)
        if result is not None:
            raw_data.update(self._serialize_identification_outputs(data, result))

        joints = dict(existing.joints)
        if result is not None:
            fit_joint_indices = list(data.metadata.get("fit_joint_indices", [self.config.target_joint]))
            clean_sample_count = (
                int(np.count_nonzero(data.clean_mask))
                if data.clean_mask is not None
                else int(result.measured_torque.shape[0])
            )
            for local_idx, params in enumerate(result.parameters):
                if local_idx >= len(fit_joint_indices):
                    break
                joint_idx = int(fit_joint_indices[local_idx])
                if not 0 <= joint_idx < self.config.joint_count:
                    continue
                joint_name = self.config.robot.joint_names[joint_idx]
                joints[joint_idx] = JointResult(
                    joint_index=joint_idx,
                    joint_name=joint_name,
                    coulomb=float(params.coulomb),
                    viscous=float(params.viscous),
                    offset=float(params.offset),
                    velocity_scale=float(params.velocity_scale),
                    validation_rmse=float(result.validation_rmse[local_idx]),
                    validation_r2=float(result.validation_r2[local_idx]),
                    sample_count=clean_sample_count,
                    identified=True,
                )

        metadata["identified_joints"] = [
            idx for idx, joint in sorted(joints.items()) if joint.identified
        ]
        return IdentificationResults(
            source_type=data.source,
            timestamp=timestamp,
            config_snapshot=config_snapshot,
            joints=joints,
            raw_data=raw_data,
            metadata=metadata,
        )

    def _serialize(self, results: IdentificationResults) -> dict[str, Any]:
        ordered = [results.joints[idx] for idx in sorted(results.joints)]
        payload: dict[str, Any] = {
            "metadata": np.asarray(json.dumps(_normalize_json_value(results.metadata), ensure_ascii=False)),
            "config_snapshot": np.asarray(
                json.dumps(_normalize_json_value(results.config_snapshot), ensure_ascii=False)
            ),
            "identified_mask": np.asarray([joint.identified for joint in ordered], dtype=bool),
            "coulomb": np.asarray([joint.coulomb for joint in ordered], dtype=np.float64),
            "viscous": np.asarray([joint.viscous for joint in ordered], dtype=np.float64),
            "offset": np.asarray([joint.offset for joint in ordered], dtype=np.float64),
            "velocity_scale": np.asarray([joint.velocity_scale for joint in ordered], dtype=np.float64),
            "validation_rmse": np.asarray([joint.validation_rmse for joint in ordered], dtype=np.float64),
            "validation_r2": np.asarray([joint.validation_r2 for joint in ordered], dtype=np.float64),
            "sample_count": np.asarray([joint.sample_count for joint in ordered], dtype=np.int64),
        }
        payload.update(results.raw_data)
        return payload

    def _serialize_collected_data(self, data: CollectedData) -> dict[str, np.ndarray]:
        payload = {
            "time": np.asarray(data.time, dtype=np.float64),
            "q": np.asarray(data.q, dtype=np.float64),
            "qd": np.asarray(data.qd, dtype=np.float64),
            "q_cmd": np.asarray(data.q_cmd, dtype=np.float64),
            "qd_cmd": np.asarray(data.qd_cmd, dtype=np.float64),
            "tau_command": np.asarray(data.tau_command, dtype=np.float64),
            "tau_measured": np.asarray(data.tau_measured, dtype=np.float64),
        }

        optional_arrays = {
            "qdd": data.qdd,
            "qdd_cmd": data.qdd_cmd,
            "tau_feedforward": data.tau_feedforward,
            "tau_feedback": data.tau_feedback,
            "tau_rigid": data.tau_rigid,
            "tau_passive": data.tau_passive,
            "tau_constraint": data.tau_constraint,
            "tau_friction": data.tau_friction,
            "clean_mask": data.clean_mask,
            "ee_pos": data.ee_pos,
            "ee_quat": data.ee_quat,
            "ee_pos_cmd": data.ee_pos_cmd,
            "ee_quat_cmd": data.ee_quat_cmd,
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
                payload[key] = np.asarray(value, dtype=np.float64)

        if "true_coulomb" in data.metadata:
            payload["true_coulomb"] = np.asarray(data.metadata["true_coulomb"], dtype=np.float64)
        if "true_viscous" in data.metadata:
            payload["true_viscous"] = np.asarray(data.metadata["true_viscous"], dtype=np.float64)
        return payload

    def _serialize_identification_outputs(
        self,
        data: CollectedData,
        result: FrictionIdentificationResult,
    ) -> dict[str, np.ndarray]:
        payload = {
            "tau_predicted": np.asarray(result.predicted_torque, dtype=np.float64),
            "train_mask": np.asarray(result.train_mask, dtype=bool),
            "validation_mask": np.asarray(result.validation_mask, dtype=bool),
            "measured_torque_fit": np.asarray(result.measured_torque, dtype=np.float64),
        }

        target_indices = list(data.metadata.get("fit_joint_indices", [self.config.target_joint]))
        if (
            data.tau_friction is not None
            and data.clean_mask is not None
            and result.predicted_torque.ndim == 2
            and len(target_indices) == result.predicted_torque.shape[1]
        ):
            predicted_full = np.zeros_like(np.asarray(data.tau_friction, dtype=np.float64))
            clean_mask = np.asarray(data.clean_mask, dtype=bool).reshape(-1)
            for local_idx, joint_idx in enumerate(target_indices):
                predicted_full[clean_mask, int(joint_idx)] = result.predicted_torque[:, local_idx]
            payload["tau_predicted_full"] = predicted_full
        return payload


ResultStore = ResultsManager


__all__ = [
    "IdentificationResults",
    "JointResult",
    "ResultPaths",
    "ResultsManager",
    "ResultStore",
    "build_validation_mask",
]
