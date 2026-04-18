from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from friction_identification_core.config import DEFAULT_CONFIG_PATH, Config, load_config
from friction_identification_core.runtime import log_info


def _default_config_argument() -> str:
    return str(DEFAULT_CONFIG_PATH.relative_to(DEFAULT_CONFIG_PATH.parents[1]))


def _parse_joint_override(raw: str | None, joint_count: int) -> tuple[int, ...] | None:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    if text == "all":
        return tuple(range(joint_count))

    joints: list[int] = []
    seen: set[int] = set()
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        joint_idx = int(token)
        if not 0 <= joint_idx < joint_count:
            raise ValueError(f"joint index {joint_idx} is outside [0, {joint_count - 1}].")
        if joint_idx in seen:
            continue
        joints.append(joint_idx)
        seen.add(joint_idx)
    if not joints:
        raise ValueError("--joints 未解析出任何有效关节索引。")
    return tuple(joints)


def _apply_overrides(config: Config, *, output: str | None, joints: str | None) -> Config:
    if not output:
        updated = config
    else:
        output_path = config.resolve_project_path(output)
        updated = replace(
            config,
            output=replace(config.output, results_dir=Path(output_path).resolve()),
        )

    joint_override = _parse_joint_override(joints, updated.joint_count)
    if joint_override is None:
        return updated
    return replace(
        updated,
        identification=replace(updated.identification, active_joints=joint_override),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hardware friction-identification CLI.")
    parser.add_argument(
        "--config",
        default=_default_config_argument(),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--mode",
        choices=("collect", "sequential", "compensate", "compare"),
        default="sequential",
        help="sequential: 逐电机辨识, collect: 并行辨识, compensate: 补偿验证, compare: 历史结果对比。",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--joints",
        default=None,
        help="Specify active joints, for example '0,2,4' or 'all'.",
    )
    parser.add_argument(
        "--compare-limit",
        type=int,
        default=5,
        help="Number of archived collect runs to include in compare mode.",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Compare all archived collect runs instead of only the latest ones.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    try:
        parser = build_parser()
        args = parser.parse_args(argv)

        config = load_config(args.config)
        config = _apply_overrides(config, output=args.output, joints=args.joints)

        if args.mode == "compare":
            from friction_identification_core.results import compare_saved_runs

            try:
                compare_saved_runs(
                    config,
                    limit=args.compare_limit,
                    compare_all=bool(args.compare_all),
                )
            except FileNotFoundError as exc:
                raise SystemExit(f"[ERROR] {exc}") from exc
            return

        from friction_identification_core.pipeline import run_hardware

        run_hardware(config, mode=args.mode)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
    except KeyboardInterrupt:
        log_info("已收到 Ctrl+C，中止当前运行。")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
