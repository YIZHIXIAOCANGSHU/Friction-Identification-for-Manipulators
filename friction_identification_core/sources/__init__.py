from friction_identification_core.config import Config
from friction_identification_core.sources.hardware import HardwareSource


def build_source(config: Config) -> HardwareSource:
    return HardwareSource(config)


__all__ = ["HardwareSource", "build_source"]
