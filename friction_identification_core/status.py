from __future__ import annotations

import numpy as np


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
    labels = {1: "Forward", 0: "Hold", -1: "Reverse"}
    lines = []
    for idx, joint_name in enumerate(joint_names):
        lines.append(
            f"{joint_name}: {labels[int(rotation_state[idx])]:<7} | "
            f"ratio={range_ratio[idx]:.2f} | qd={velocity[idx]:+.3f}"
        )
    return "\n".join(lines)


__all__ = [
    "compute_limit_margin_remaining",
    "compute_range_ratio",
    "compute_rotation_state",
    "format_joint_motion_summary",
]
