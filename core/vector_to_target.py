"""
Target-centric vector to B, and the errors that matter through a detour.
Siddharth's Week 5 analysis core (portable, NumPy-only).

The controller drives along the vector from the drone to target B, computed in
the WORLD frame:

    v_to_B = B - position

B is a map constant (surveyed, fixed); `position` is the drifting/corrected
estimate. If localization is good, the estimated vector matches the true one and
the drone heads correctly to B. If drift corrupts the estimate -- which an
aggressive obstacle detour tends to cause -- the vector points the wrong way and
the drone misses B *even after avoiding the obstacle*. Week 5 verifies that the
landmark corrector keeps this vector correct through the maneuver.

Two errors are tracked:
  * position error   ||est_pos - true_pos||               [m]
  * DIRECTION error  angle between (B - est_pos) and       [deg]
                     (B - true_pos)  -- the one that actually steers the drone
"""
from __future__ import annotations

import numpy as np


def vector_to_target(position, B) -> np.ndarray:
    return np.asarray(B, dtype=float) - np.asarray(position, dtype=float)


def heading_to_target(position, B) -> float:
    """Absolute heading (radians) the drone would fly to reach B from `position`."""
    v = vector_to_target(position, B)
    return float(np.arctan2(v[1], v[0]))


def position_error(est_pos, true_pos) -> float:
    return float(np.linalg.norm(np.asarray(est_pos, float) - np.asarray(true_pos, float)))


def direction_error_deg(est_pos, true_pos, B) -> float:
    """Angle between the estimated and true vectors to B. This is the quantity
    the target-centric controller is sensitive to: a large value means the drone
    is being steered off-target, regardless of how close it is to B."""
    ve = vector_to_target(est_pos, B)
    vt = vector_to_target(true_pos, B)
    ne, nt = np.linalg.norm(ve), np.linalg.norm(vt)
    if ne < 1e-9 or nt < 1e-9:
        return 0.0
    cos = float(np.clip(np.dot(ve, vt) / (ne * nt), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))
