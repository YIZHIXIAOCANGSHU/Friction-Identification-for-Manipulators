from __future__ import annotations

import time
from dataclasses import dataclass

from friction_identification_core.config import Config
from friction_identification_core.controller import FrictionIdentificationController, SafetyGuard
from friction_identification_core.estimator import fit_multijoint_friction
from friction_identification_core.models import CollectedData, DataSource, FrictionIdentificationResult
from friction_identification_core.results import ResultPaths, ResultStore, build_validation_mask
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
class PipelineRunResult:
    source: str
    mode: str
    batches: tuple[BatchRunArtifact, ...]
    summary_paths: ResultPaths | None = None


class IdentificationPipeline:
    """Hardware-only parallel friction identification pipeline."""

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

    def _fit_batch(self, data: CollectedData) -> FrictionIdentificationResult | None:
        identification_inputs = self.source.prepare_identification(data)
        if identification_inputs is None:
            return None
        return fit_multijoint_friction(
            velocity=identification_inputs.velocity,
            torque=identification_inputs.torque,
            joint_names=identification_inputs.joint_names,
            validation_mask=build_validation_mask(identification_inputs.velocity.shape[0]),
            velocity_scale=self.config.fitting.velocity_scale,
            regularization=self.config.fitting.regularization,
            max_iterations=self.config.fitting.max_iterations,
            huber_delta=self.config.fitting.huber_delta,
            min_velocity_threshold=self.config.fitting.min_velocity_threshold,
            true_coulomb=identification_inputs.true_coulomb,
            true_viscous=identification_inputs.true_viscous,
        )

    def run(self, *, mode: str) -> PipelineRunResult:
        if mode not in {"collect", "compensate"}:
            raise ValueError("mode must be 'collect' or 'compensate'.")

        reference = self.source.build_reference() if mode == "collect" else None
        batches: list[BatchRunArtifact] = []
        summary_paths = None
        total_batches = self.config.batch_collection.num_batches if mode == "collect" else 1

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
                result = self._fit_batch(data) if self.source.supports_identification(mode) else None
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

                if mode != "collect":
                    continue
                if batch_index >= total_batches:
                    continue

                delay_s = max(float(self.config.batch_collection.inter_batch_delay), 0.0)
                if delay_s > 0.0:
                    log_info(f"批次 {batch_index} 完成，等待 {delay_s:.1f} s 后开始下一批。")
                    time.sleep(delay_s)

            if mode == "collect":
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


def run_hardware(config: Config, *, mode: str) -> PipelineRunResult:
    return IdentificationPipeline(config, build_source(config)).run(mode=mode)


__all__ = [
    "BatchRunArtifact",
    "IdentificationPipeline",
    "PipelineRunResult",
    "run_hardware",
]
