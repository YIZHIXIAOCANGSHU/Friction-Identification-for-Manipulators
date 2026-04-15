from __future__ import annotations

"""Legacy-style module exports that mirror the default robot model config."""

from .config import DEFAULT_FRICTION_CONFIG


# Bind once so downstream imports can reuse the same default robot settings.
MODEL_CONFIG = DEFAULT_FRICTION_CONFIG.model

URDF_PATH = MODEL_CONFIG.urdf_path
JOINT_NAMES = list(MODEL_CONFIG.joint_names)
TORQUE_LIMITS = MODEL_CONFIG.torque_limits.copy()
JOINT_LIMITS = MODEL_CONFIG.joint_limits.copy()
HOME_QPOS = MODEL_CONFIG.home_qpos.copy()
TCP_OFFSET = MODEL_CONFIG.tcp_offset.copy()
END_EFFECTOR_BODY = MODEL_CONFIG.end_effector_body
FRICTION_LOSS = MODEL_CONFIG.friction_loss.copy()
DAMPING = MODEL_CONFIG.damping.copy()
