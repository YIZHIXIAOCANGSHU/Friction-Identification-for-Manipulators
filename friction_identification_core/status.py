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
    span = np.maximum(upper - lower, 1e-9)
    return np.clip((q - lower) / span, 0.0, 1.0)


def compute_limit_margin_remaining(q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    return np.minimum(q - lower, upper - q)


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
