from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from friction_identification_core.config import Config
from friction_identification_core.controller import FrictionIdentificationController, SafetyGuard
from friction_identification_core.estimator import fit_multijoint_friction
from friction_identification_core.models import CollectedData, DataSource, FrictionIdentificationResult
from friction_identification_core.results import ResultPaths, ResultStore
from friction_identification_core.runtime import log_info
from friction_identification_core.sources import build_source


@dataclass(frozen=True)
class BatchRunArtifact:
    batch_index: int
    data: CollectedData
    identification: FrictionIdentificationResult | None
    collection_paths: ResultPaths
    identification_paths: ResultPaths | None


@dataclass(frozen=True)
class JointRunArtifact:
    joint_index: int
    joint_name: str
    group_index: int
    run_index: int
    data: CollectedData
    identification: FrictionIdentificationResult | None
    collection_paths: ResultPaths
    identification_paths: ResultPaths | None


@dataclass(frozen=True)
class PipelineRunResult:
    source: str
    mode: str
    batches: tuple[BatchRunArtifact, ...]
    summary_paths: ResultPaths | None = None


@dataclass(frozen=True)
class SequentialPipelineResult:
    source: str
    mode: str
    joint_runs: tuple[JointRunArtifact, ...]
    summary_paths: ResultPaths | None = None


class _PipelineBase:
    def __init__(self, config: Config, source: DataSource) -> None:
        self.config = config
        self.source = source
        self.safety = SafetyGuard(config, active_joint_mask=config.active_joint_mask)
        self.controller = FrictionIdentificationController(
            config,
            source.inverse_dynamics_backend,
            safety=self.safety,
        )
        self.results = ResultStore(config)

    def _sleep_with_interrupt(self, delay_s: float, *, message: str) -> bool:
        if delay_s <= 0.0:
            return True
        log_info(message)
        try:
            time.sleep(delay_s)
        except KeyboardInterrupt:
            log_info("检测到人工中断，停止后续任务并保留已完成结果。")
            return False
        return True

    def _fit_data(self, data: CollectedData) -> FrictionIdentificationResult | None:
        identification_inputs = self.source.prepare_identification(data)
        if identification_inputs is None:
            return None
        result = fit_multijoint_friction(
            velocity=identification_inputs.velocity,
            torque=identification_inputs.torque,
            joint_names=identification_inputs.joint_names,
            sample_mask=identification_inputs.sample_mask,
            velocity_scale=self.config.fitting.velocity_scale,
            regularization=self.config.fitting.regularization,
            max_iterations=self.config.fitting.max_iterations,
            huber_delta=self.config.fitting.huber_delta,
            min_velocity_threshold=self.config.fitting.min_velocity_threshold,
            true_coulomb=identification_inputs.true_coulomb,
            true_viscous=identification_inputs.true_viscous,
        )
        if not any(np.isfinite(param.coulomb) and np.isfinite(param.viscous) for param in result.parameters):
            log_info("各关节有效样本仍不足以稳定辨识，本轮跳过辨识结果保存。")
            return None
        return result

    def _publish_identification(self, data: CollectedData, result: FrictionIdentificationResult | None) -> None:
        if result is None:
            return
        publish_fn = getattr(self.source, "publish_identification_result", None)
        if callable(publish_fn):
            publish_fn(data, result)


class IdentificationPipeline(_PipelineBase):
    """Hardware-only parallel friction identification pipeline."""

    def run(self, *, mode: str) -> PipelineRunResult:
        if mode not in {"collect", "compensate"}:
            raise ValueError("mode must be 'collect' or 'compensate'.")

        reference = self.source.build_reference() if mode == "collect" else None
        batches: list[BatchRunArtifact] = []
        summary_paths = None
        total_batches = self.config.batch_collection.num_batches if mode == "collect" else 1
        self.results.begin_run(mode=mode, total_batches=total_batches)

        try:
            for batch_index in range(1, total_batches + 1):
                if mode == "collect":
                    log_info(f"开始批次 {batch_index}/{total_batches} 并行采集与辨识。")
                data = self.source.collect(
                    mode=mode,
                    reference=reference,
                    controller=self.controller,
                    safety=self.safety,
                    batch_index=batch_index,
                    total_batches=total_batches,
                )
                result = self._fit_data(data) if self.source.supports_identification(mode) else None
                self._publish_identification(data, result)
                collection_paths = self.results.save_collection(
                    data,
                    batch_index=batch_index,
                    total_batches=total_batches,
                )
                identification_paths = self.results.save_identification(
                    data,
                    result,
                    batch_index=batch_index,
                    total_batches=total_batches,
                )
                batches.append(
                    BatchRunArtifact(
                        batch_index=batch_index,
                        data=data,
                        identification=result,
                        collection_paths=collection_paths,
                        identification_paths=identification_paths,
                    )
                )
                termination_reason = str(data.metadata.get("termination_reason", "completed")).strip().lower()
                if termination_reason == "interrupted":
                    log_info("当前批次被人工中断，停止后续批次并汇总已完成结果。")
                    break

                if mode != "collect" or batch_index >= total_batches:
                    continue

                delay_s = max(float(self.config.batch_collection.inter_batch_delay), 0.0)
                if not self._sleep_with_interrupt(
                    delay_s,
                    message=f"批次 {batch_index} 完成，等待 {delay_s:.1f} s 后开始下一批。",
                ):
                    break

            if mode == "collect" and batches:
                summary_paths = self.results.save_summary(batches)
                self.source.publish_summary(self.results.load_summary(summary_paths.npz_path))
            return PipelineRunResult(
                source=self.source.source_name,
                mode=mode,
                batches=tuple(batches),
                summary_paths=summary_paths,
            )
        finally:
            last_data = batches[-1].data if batches else None
            last_result = batches[-1].identification if batches else None
            self.source.finalize(last_data, last_result)


class SequentialIdentificationPipeline(_PipelineBase):
    """Sequential single-joint friction identification pipeline."""

    def run(self) -> SequentialPipelineResult:
        active_joint_indices = list(self.config.identification.active_joints)
        total_groups = int(self.config.identification.sequential.num_groups)
        total_runs = len(active_joint_indices) * total_groups
        joint_runs: list[JointRunArtifact] = []
        summary_paths = None
        self.results.begin_run(mode="sequential", total_batches=total_runs)
        completed_run_keys: set[tuple[int, int]] = set()

        existing_runs = getattr(self.results, "load_existing_sequential_runs", lambda: [])()
        for saved_run in existing_runs:
            joint_index = int(saved_run["joint_index"])
            group_index = int(saved_run["group_index"])
            if joint_index not in active_joint_indices or not 1 <= group_index <= total_groups:
                continue
            completed_run_keys.add((group_index, joint_index))
            joint_runs.append(
                JointRunArtifact(
                    joint_index=joint_index,
                    joint_name=self.config.robot.joint_names[joint_index],
                    group_index=group_index,
                    run_index=int(saved_run["batch_index"]),
                    data=saved_run["data"],
                    identification=saved_run["identification"],
                    collection_paths=ResultPaths(
                        npz_path=saved_run["collection_path"],
                        manifest_path=self.results.manifest_path,
                        archive_dir=self.results.run_dir,
                    ),
                    identification_paths=(
                        ResultPaths(
                            npz_path=saved_run["identification_path"],
                            manifest_path=self.results.manifest_path,
                            archive_dir=self.results.run_dir,
                        )
                        if saved_run["identification_path"] is not None
                        else None
                    ),
                )
            )
        if joint_runs:
            log_info(
                f"已恢复 {len(joint_runs)}/{total_runs} 个已完成的逐电机轮次，"
                "本次只继续未完成部分。"
            )

        try:
            stop_requested = False
            reporter = getattr(self.source, "reporter", None)

            for group_index in range(1, total_groups + 1):
                for joint_offset, joint_index in enumerate(active_joint_indices, start=1):
                    run_index = (group_index - 1) * len(active_joint_indices) + joint_offset
                    if (group_index, joint_index) in completed_run_keys:
                        continue
                    joint_name = self.config.robot.joint_names[joint_index]
                    log_info(
                        "开始逐电机辨识: "
                        f"group={group_index}/{total_groups}, "
                        f"joint=J{joint_index + 1} ({joint_name}), "
                        f"run={run_index}/{total_runs}"
                    )
                    if reporter is not None and hasattr(reporter, "log_sequential_progress"):
                        reporter.log_sequential_progress(
                            group_index=group_index,
                            total_groups=total_groups,
                            joint_index=joint_index,
                            active_joint_indices=active_joint_indices,
                            completed_joint_indices=active_joint_indices[: max(joint_offset - 1, 0)],
                        )

                    reference = self.source.build_reference(joint_index=joint_index)
                    data = self.source.collect(
                        mode="sequential",
                        reference=reference,
                        controller=self.controller,
                        safety=self.safety,
                        batch_index=run_index,
                        total_batches=total_runs,
                        target_joint_index=joint_index,
                        group_index=group_index,
                        total_groups=total_groups,
                    )
                    result = self._fit_data(data) if self.source.supports_identification("sequential") else None
                    self._publish_identification(data, result)
                    collection_paths = self.results.save_collection(
                        data,
                        batch_index=run_index,
                        total_batches=total_runs,
                        group_index=group_index,
                        joint_index=joint_index,
                    )
                    identification_paths = self.results.save_identification(
                        data,
                        result,
                        batch_index=run_index,
                        total_batches=total_runs,
                        group_index=group_index,
                        joint_index=joint_index,
                    )
                    joint_runs.append(
                        JointRunArtifact(
                            joint_index=joint_index,
                            joint_name=joint_name,
                            group_index=group_index,
                            run_index=run_index,
                            data=data,
                            identification=result,
                            collection_paths=collection_paths,
                            identification_paths=identification_paths,
                        )
                    )
                    completed_run_keys.add((group_index, joint_index))
                    termination_reason = str(data.metadata.get("termination_reason", "completed")).strip().lower()
                    if termination_reason == "interrupted":
                        log_info("当前关节采集被人工中断，停止后续逐电机任务并汇总已完成结果。")
                        stop_requested = True
                        break

                    remaining_group_pending = any(
                        (group_index, future_joint_index) not in completed_run_keys
                        for future_joint_index in active_joint_indices[joint_offset:]
                    )
                    if not remaining_group_pending:
                        continue
                    if not self._sleep_with_interrupt(
                        max(float(self.config.identification.sequential.inter_joint_delay), 0.0),
                        message=(
                            f"J{joint_index + 1} 完成，等待 "
                            f"{float(self.config.identification.sequential.inter_joint_delay):.1f} s 后切换下一关节。"
                        ),
                    ):
                        stop_requested = True
                        break

                if stop_requested:
                    break
                remaining_global_pending = any(
                    (future_group_index, future_joint_index) not in completed_run_keys
                    for future_group_index in range(group_index + 1, total_groups + 1)
                    for future_joint_index in active_joint_indices
                )
                if not remaining_global_pending:
                    continue
                if not self._sleep_with_interrupt(
                    max(float(self.config.identification.sequential.inter_group_delay), 0.0),
                    message=(
                        f"第 {group_index} 组完成，等待 "
                        f"{float(self.config.identification.sequential.inter_group_delay):.1f} s 后开始下一组。"
                    ),
                ):
                    break

            if joint_runs:
                joint_runs.sort(key=lambda item: item.run_index)
                summary_paths = self.results.save_summary(joint_runs)
                self.source.publish_summary(self.results.load_summary(summary_paths.npz_path))
            return SequentialPipelineResult(
                source=self.source.source_name,
                mode="sequential",
                joint_runs=tuple(joint_runs),
                summary_paths=summary_paths,
            )
        finally:
            last_data = joint_runs[-1].data if joint_runs else None
            last_result = joint_runs[-1].identification if joint_runs else None
            self.source.finalize(last_data, last_result)


def run_hardware(config: Config, *, mode: str) -> PipelineRunResult | SequentialPipelineResult:
    source = build_source(config)
    if mode == "sequential":
        return SequentialIdentificationPipeline(config, source).run()
    return IdentificationPipeline(config, source).run(mode=mode)


__all__ = [
    "BatchRunArtifact",
    "IdentificationPipeline",
    "JointRunArtifact",
    "PipelineRunResult",
    "SequentialIdentificationPipeline",
    "SequentialPipelineResult",
    "run_hardware",
]
