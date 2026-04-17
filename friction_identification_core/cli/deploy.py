#!/usr/bin/env python3

from __future__ import annotations

import argparse

from friction_identification_core.config import DEFAULT_CONFIG_PATH, load_config
from friction_identification_core.hardware.runner import run_hardware


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hardware friction collection or compensation.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH.relative_to(DEFAULT_CONFIG_PATH.parents[2])),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--mode",
        choices=("collect", "compensate", "full_feedforward"),
        default="collect",
        help="Hardware execution mode: collect, compensate, or full_feedforward (g+c+friction).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    run_hardware(config, mode=args.mode)


if __name__ == "__main__":
    main()
