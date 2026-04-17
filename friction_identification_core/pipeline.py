from __future__ import annotations

from dataclasses import dataclass

from friction_identification_core.config import Config
from friction_identification_core.controller import FrictionIdentificationController, SafetyGuard
from friction_identification_core.estimator import fit_multijoint_friction
from friction_identification_core.models import CollectedData, DataSource, FrictionIdentificationResult
from friction_identification_core.results import ResultPaths, ResultStore, build_validation_mask
from friction_identification_core.sources import build_source


@dataclass(frozen=True)
class PipelineRunResult:
    source: str
    mode: str
    data: CollectedData
    identification: FrictionIdentificationResult | None
    collection_paths: ResultPaths
    identification_paths: ResultPaths | None


class IdentificationPipeline:
    """Shared orchestration layer for simulation and hardware data sources."""

    def __init__(self, config: Config, source: DataSource) -> None:
        self.config = config
        self.source = source
        self.safety = SafetyGuard(config, active_joint_mask=config.target_joint_mask)
        self.controller = FrictionIdentificationController(
            config,
            source.inverse_dynamics_backend,
            safety=self.safety,
        )
        self.results = ResultStore(config)

    def run(self, *, mode: str) -> PipelineRunResult:
        reference = None
        if mode in {"collect", "full_feedforward"}:
            reference = self.source.build_reference()

        data = None
        result = None
        try:
            data = self.source.collect(
                mode=mode,
                reference=reference,
                controller=self.controller,
                safety=self.safety,
            )

            if self.source.supports_identification(mode):
                identification_inputs = self.source.prepare_identification(data)
                if identification_inputs is not None:
                    result = fit_multijoint_friction(
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

            collection_paths = self.results.save_collection(data, result=result)
            identification_paths = self.results.save_identification(data, result)
            if data.source == "simulation" and result is not None:
                identification_paths = collection_paths

            return PipelineRunResult(
                source=data.source,
                mode=mode,
                data=data,
                identification=result,
                collection_paths=collection_paths,
                identification_paths=identification_paths,
            )
        finally:
            self.source.finalize(data, result)


def run_simulation(config: Config, *, mode: str = "collect") -> PipelineRunResult:
    return IdentificationPipeline(config, build_source(config, "sim")).run(mode=mode)


def run_hardware(config: Config, *, mode: str) -> PipelineRunResult:
    return IdentificationPipeline(config, build_source(config, "hw")).run(mode=mode)


__all__ = [
    "IdentificationPipeline",
    "PipelineRunResult",
    "run_hardware",
    "run_simulation",
]
