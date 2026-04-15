#!/usr/bin/env python3

from __future__ import annotations

import argparse

from friction_identification_core.config import DEFAULT_CONFIG_PATH, load_config
from friction_identification_core.simulation.runner import run_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run simulation-based friction identification.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH.relative_to(DEFAULT_CONFIG_PATH.parents[2])),
        help="Path to YAML config file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    run_simulation(config)


if __name__ == "__main__":
    main()
