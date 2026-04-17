from __future__ import annotations

import numpy as np


def _rotation_label(state: int) -> str:
    labels = {1: "Forward", 0: "Hold", -1: "Reverse"}
    return labels.get(int(state), "Unknown")


def compute_rotation_state(velocity: np.ndarray, *, velocity_eps: float) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=np.float64)
    state = np.zeros_like(velocity, dtype=np.int8)
    state[velocity > float(velocity_eps)] = 1
    state[velocity < -float(velocity_eps)] = -1
    return state


def compute_range_ratio(q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    lower_b = np.broadcast_to(lower, q.shape)
    upper_b = np.broadcast_to(upper, q.shape)
    finite_window = np.isfinite(lower_b) & np.isfinite(upper_b) & (upper_b > lower_b)
    ratio = np.full(q.shape, 0.5, dtype=np.float64)
    if np.any(finite_window):
        span = np.maximum(upper_b[finite_window] - lower_b[finite_window], 1e-9)
        ratio[finite_window] = np.clip((q[finite_window] - lower_b[finite_window]) / span, 0.0, 1.0)
    return ratio


def compute_limit_margin_remaining(q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    lower_b = np.broadcast_to(lower, q.shape)
    upper_b = np.broadcast_to(upper, q.shape)
    finite_window = np.isfinite(lower_b) & np.isfinite(upper_b)
    margin = np.full(q.shape, np.nan, dtype=np.float64)
    if np.any(finite_window):
        margin[finite_window] = np.minimum(q[finite_window] - lower_b[finite_window], upper_b[finite_window] - q[finite_window])
    return margin


def format_joint_motion_summary(
    joint_names: list[str],
    rotation_state: np.ndarray,
    range_ratio: np.ndarray,
    velocity: np.ndarray,
) -> str:
    rotation_state = np.asarray(rotation_state, dtype=np.int8).reshape(-1)
    range_ratio = np.asarray(range_ratio, dtype=np.float64).reshape(-1)
    velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
    lines = [
        "## Joint Motion",
        "",
        "| Joint | Direction | Range Ratio | Velocity |",
        "|---|---|---:|---:|",
    ]
    for idx, joint_name in enumerate(joint_names):
        lines.append(
            f"| J{idx + 1} `{joint_name}` | {_rotation_label(rotation_state[idx])} | "
            f"{range_ratio[idx]:.2f} | {velocity[idx]:+.3f} |"
        )
    return "\n".join(lines)


def format_feedback_cycle_summary(
    active_joint_ids: list[int] | np.ndarray,
    feedback_cycle_joint_ids: list[int] | np.ndarray,
) -> str:
    active_joint_ids = [int(item) for item in np.asarray(active_joint_ids, dtype=np.int64).reshape(-1)]
    feedback_cycle_joint_ids = [int(item) for item in np.asarray(feedback_cycle_joint_ids, dtype=np.int64).reshape(-1)]
    active_text = ", ".join(f"J{joint_id}" for joint_id in active_joint_ids) if active_joint_ids else "N/A"
    cycle_text = (
        " -> ".join(f"J{joint_id}" for joint_id in feedback_cycle_joint_ids)
        if feedback_cycle_joint_ids
        else "N/A"
    )
    return "\n".join(
        [
            "## Feedback Cycle",
            "",
            f"- Active joints: `{active_text}`",
            f"- Trigger order: `{cycle_text}`",
            f"- Coverage: `{len(feedback_cycle_joint_ids)}/{len(active_joint_ids)}`",
        ]
    )


def format_runtime_status(
    *,
    batch_index: int,
    total_batches: int,
    step_index: int,
    phase_name: str,
    valid_sample_ratio: float,
    uart_cycle_hz: float,
    uart_latency_ms: float,
    uart_transfer_kbps: float,
) -> str:
    return "\n".join(
        [
            "## Runtime",
            "",
            f"- Batch: `{int(batch_index)}/{int(total_batches)}`",
            f"- Step: `{int(step_index)}`",
            f"- Phase: `{phase_name or 'unknown'}`",
            f"- Valid sample ratio: `{float(valid_sample_ratio):.3f}`",
            f"- UART cycle rate: `{float(uart_cycle_hz):.2f} Hz`",
            f"- UART cycle period: `{float(uart_latency_ms):.2f} ms`",
            f"- UART throughput: `{float(uart_transfer_kbps):.2f} kbps`",
        ]
    )


__all__ = [
    "compute_limit_margin_remaining",
    "compute_range_ratio",
    "compute_rotation_state",
    "format_feedback_cycle_summary",
    "format_joint_motion_summary",
    "format_runtime_status",
]
