from friction_identification_core.config import Config
from friction_identification_core.sources.hardware import HardwareSource
from friction_identification_core.sources.simulation import SimulationSource


def build_source(config: Config, source: str):
    normalized = source.strip().lower()
    if normalized in {"sim", "simulation"}:
        return SimulationSource(config)
    if normalized in {"hw", "hardware", "real"}:
        return HardwareSource(config)
    raise ValueError(f"Unsupported source: {source}")


__all__ = ["HardwareSource", "SimulationSource", "build_source"]
