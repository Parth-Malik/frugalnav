"""
Landmark corrector (AVL) -- Siddharth's Week 3 core contribution.

Role in the architecture (plan section 1): the "Landmark corrector (AVL)"
module. On a marker sighting, look up the marker's known world pose, turn the
camera-frame observation into an ABSOLUTE world-frame position fix for the
drone, and emit a LandmarkFix. A separate re-anchor step merges that fix into
the drifting estimate.

(Tight fusion of the fix into the VIO/EKF state is Parth's `state_fusion`
module. `reanchor()` below is the decoupled, independently-testable reference
version -- the plan's "decoupled modules, fused estimate" split, section 1.)

The geometry
------------
Frames:  W world | B drone body | C camera | M marker.
Known:
    T_WM   from the landmark map           (marker in world)      -- surveyed
    T_CM   from ArUco pose estimation      (marker in camera)     -- the sighting
    T_BC   fixed camera-mount extrinsic    (camera in body)       -- config
Want:
    T_WB   drone body in world             -- the absolute fix

    T_WB = T_WM . T_MC . T_CB
         = T_WM . inv(T_CM) . inv(T_BC)

Read the chain right-to-left: body -> camera -> marker -> world. The marker
frame cancels, leaving body -> world. The drone's world (x, y) is the
translation part of T_WB.
"""
from __future__ import annotations

import numpy as np

from core.geometry import T_from_rvec_tvec, inv_T, xy_of, yaw_of
from core.types import LandmarkFix, MarkerSighting, NavState

# Map a pose's reprojection error (px) to the fix's position std (m). A clean
# sighting is trusted to a couple of centimetres; a blurry / grazing one less.
_BASE_FIX_STD_M = 0.02
_PX_TO_STD_M = 0.01


class LandmarkCorrector:
    def __init__(self, landmark_map):
        self.map = landmark_map

    def correct(self, sighting: MarkerSighting):
        """MarkerSighting -> LandmarkFix, or None if the marker is not mapped."""
        entry = self.map.get(sighting.marker_id)
        if entry is None:
            return None                        # unknown marker: cannot anchor to it

        T_CM = T_from_rvec_tvec(sighting.rvec, sighting.tvec)
        T_WB = entry.T_WM @ inv_T(T_CM) @ inv_T(self.map.T_BC)

        std = _BASE_FIX_STD_M + _PX_TO_STD_M * float(sighting.reproj_error_px)
        cov = np.eye(2) * (std ** 2)
        return LandmarkFix(
            xy=xy_of(T_WB),
            yaw=yaw_of(T_WB),
            covariance=cov,
            marker_id=int(sighting.marker_id),
            timestamp=sighting.timestamp,
        )

    def correct_all(self, sightings):
        """Fixes for every mapped marker in a frame (unmapped ones dropped)."""
        return [f for f in (self.correct(s) for s in sightings) if f is not None]


def reanchor(state: NavState, fix: LandmarkFix, gain=None) -> NavState:
    """Merge an absolute fix into the drifting estimate.

    Decoupled reference fusion: a Kalman gain K = P (P + R)^-1 optimally blends
    the prior estimate covariance P with the fix covariance R. Pass gain=None
    for that blend, or a float in [0, 1] to force a fixed gain (gain=1.0 snaps
    fully onto the fix -- the clearest "money-shot" behaviour).

    Target B is NOT touched here: B is a world-frame constant. Only the drone's
    estimate moves, so the recomputed vector (B - state.xy) is corrected for
    free. This is exactly why correcting drift never moves the target.
    """
    P = np.asarray(state.covariance, dtype=float).reshape(2, 2)
    R = np.asarray(fix.covariance, dtype=float).reshape(2, 2)
    if gain is None:
        K = P @ np.linalg.inv(P + R)
    else:
        K = np.eye(2) * float(gain)
    innovation = fix.xy - state.xy
    new_xy = state.xy + K @ innovation
    new_P = (np.eye(2) - K) @ P
    return NavState(xy=new_xy, yaw=fix.yaw, covariance=new_P, timestamp=fix.timestamp)
