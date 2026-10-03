"""ilc_jump_sim's index maps and small helpers, without its ROS imports (identical definitions)."""
from __future__ import annotations

import math

import numpy as np

from ilc_quad.sim_quad_model import CANONICAL_JOINT_NAMES

NU = len(CANONICAL_JOINT_NAMES)
PLANAR_TO_CANONICAL = [
    [CANONICAL_JOINT_NAMES.index(f"{leg}_{j}_joint") for leg in legs]
    for legs in (("FL", "FR"), ("RL", "RR")) for j in ("thigh", "calf")
]
HIP_IDX = [CANONICAL_JOINT_NAMES.index(f"{leg}_hip_joint") for leg in ("FL", "FR", "RL", "RR")]


def expand(planar) -> np.ndarray:
    """Planar (4,) joint quantity -> canonical (12,), both legs of a pair alike, hips 0."""
    full = np.zeros(NU)
    for i, idx in enumerate(PLANAR_TO_CANONICAL):
        full[idx] = planar[i]
    return full


def collapse(full) -> np.ndarray:
    """Canonical (12,) -> planar (4,), the mean over each leg pair."""
    return np.array([np.mean(np.asarray(full)[idx]) for idx in PLANAR_TO_CANONICAL])


def planar_pitch(q):
    """Nose-up pitch (the planar model's theta) from a (w, x, y, z) quaternion."""
    w, x, y, z = q
    return -math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
