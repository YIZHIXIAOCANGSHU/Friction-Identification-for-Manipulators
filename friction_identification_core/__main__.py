from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from friction_identification_core.config import DEFAULT_CONFIG_PATH, Config, load_config


def _default_config_argument() -> str:
    return str(DEFAULT_CONFIG_PATH.relative_to(DEFAULT_CONFIG_PATH.parents[1]))


def _apply_overrides(config: Config, *, output: str | None) -> Config:
    if not output:
        return config
    output_path = config.resolve_project_path(output)
    return replace(
        config,
        output=replace(config.output, results_dir=Path(output_path).resolve()),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hardware-only parallel friction-identification CLI.")
    parser.add_argument(
        "--config",
        default=_default_config_argument(),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--mode",
        choices=("collect", "compensate"),
        default="collect",
        help="Run parallel collection/identification or friction-only compensation validation.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output directory override.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config = _apply_overrides(config, output=args.output)

    from friction_identification_core.pipeline import run_hardware

    run_hardware(config, mode=args.mode)


if __name__ == "__main__":
    main()
