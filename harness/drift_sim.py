"""
Drift-injection simulator (harness -- throwaway).

Implements the plan's Week-1/2 fallback: a VIO drift model (systematic bias +
random walk) so the corrector and scheduler can be built without a compiled
VIO. Crucially it also synthesizes *physically consistent* ArUco sightings from
ground truth + the map, so the REAL landmark corrector runs on real geometry,
not on hand-typed numbers.

Scenario: a drone flies an L-shaped path in the world XY plane at fixed altitude
with a downward camera. Markers lie flat on the ground (z=0). This keeps the
picture legible while still exercising the full 3D corrector math (T_CM built
from ground truth, T_WB recovered by the corrector).
"""
from __future__ import annotations

import numpy as np

from core.geometry import inv_T, make_T, rot_z, rvec_from_rot
from core.types import MarkerSighting, NavState
from core.landmark_corrector import reanchor


# ----------------------------------------------------------------------------
# Ground truth
# ----------------------------------------------------------------------------
def build_scenario(step_len: float = 1.0, altitude: float = 5.0):
    """An L-shaped ground-truth flight: north 40 m, then east 30 m.

    Returns a dict with times, true (x, y), true yaw, true body poses (4x4), and
    the flight altitude. The path is chosen to sweep over the mapped markers.
    """
    # North leg: (0,0) -> (0,40), heading +Y (yaw = +90 deg).
    north_y = np.arange(0.0, 40.0 + 1e-9, step_len)
    north = np.column_stack([np.zeros_like(north_y), north_y])
    north_yaw = np.full(len(north), np.pi / 2)

    # East leg: (0,40) -> (30,40), heading +X (yaw = 0).
    east_x = np.arange(step_len, 30.0 + 1e-9, step_len)
    east = np.column_stack([east_x, np.full_like(east_x, 40.0)])
    east_yaw = np.zeros(len(east))

    true_xy = np.vstack([north, east])
    true_yaw = np.concatenate([north_yaw, east_yaw])
    n = len(true_xy)
    times = np.arange(n, dtype=float)

    true_T = []
    for (x, y), yaw in zip(true_xy, true_yaw):
        true_T.append(make_T(rot_z(yaw), [x, y, altitude]))

    return {
        "times": times,
        "true_xy": true_xy,
        "true_yaw": true_yaw,
        "true_T": true_T,
        "altitude": altitude,
    }


# ----------------------------------------------------------------------------
# Synthetic sightings (ground truth + map -> what the camera really sees)
# ----------------------------------------------------------------------------
def world_pose_to_sighting(T_WB_true, entry, lmap, rng, t=0.0,
                           tvec_noise_m=0.01, rvec_noise=0.004):
    """Build the MarkerSighting the downward camera would produce for `entry`.

    T_CM = inv(T_WC) @ T_WM  with  T_WC = T_WB_true @ T_BC.  We then add small
    Gaussian noise so the corrector faces a realistic (imperfect) observation.
    """
    T_WC = T_WB_true @ lmap.T_BC
    T_CM = inv_T(T_WC) @ entry.T_WM

    rvec = rvec_from_rot(T_CM[:3, :3]) + rng.normal(0.0, rvec_noise, 3)
    tvec = T_CM[:3, 3] + rng.normal(0.0, tvec_noise_m, 3)

    # A plausible reprojection error: grows with range and with the injected
    # noise. Purely for the corrector's covariance weighting.
    rng_m = float(np.linalg.norm(T_CM[:3, 3]))
    reproj_px = 0.3 + 0.05 * rng_m + rng.uniform(0.0, 0.3)
    return MarkerSighting(int(entry.marker_id), rvec, tvec, t, reproj_px)


def markers_in_range(T_WB_true, lmap, sensing_radius):
    """Mapped markers whose ground position is within `sensing_radius` of the
    drone's ground track (i.e. inside the downward camera footprint)."""
    x, y = T_WB_true[0, 3], T_WB_true[1, 3]
    hits = []
    for mid in lmap.ids():
        mx, my = lmap.marker_world_xy(mid)
        d = np.hypot(x - mx, y - my)
        if d <= sensing_radius:
            hits.append((d, lmap.get(mid)))
    hits.sort(key=lambda dm: dm[0])            # closest first (best geometry)
    return [entry for _, entry in hits]


# ----------------------------------------------------------------------------
# Drift model  (relative-motion error that integrates into VIO drift)
# ----------------------------------------------------------------------------
class DriftModel:
    """Corrupts each true motion increment with a slowly-varying rotation +
    scale error (random walk) plus white noise. Integrating the corrupted
    increments reproduces the characteristic curved growth of VIO drift."""

    def __init__(self, rng, yaw_bias=0.003, yaw_rw=0.004,
                 scale_rw=0.002, add_noise=0.008):
        self.rng = rng
        self.yaw_bias = yaw_bias
        self.yaw_rw = yaw_rw
        self.scale_rw = scale_rw
        self.add_noise = add_noise
        self.yaw_err = 0.0
        self.log_scale = 0.0

    def corrupt(self, delta_xy):
        self.yaw_err += self.yaw_bias + self.rng.normal(0.0, self.yaw_rw)
        self.log_scale += self.rng.normal(0.0, self.scale_rw)
        s = np.exp(self.log_scale)
        c, sn = np.cos(self.yaw_err), np.sin(self.yaw_err)
        R = np.array([[c, -sn], [sn, c]])
        return s * (R @ delta_xy) + self.rng.normal(0.0, self.add_noise, 2)


# ----------------------------------------------------------------------------
# Run the estimator over the scenario, optionally correcting at markers
# ----------------------------------------------------------------------------
def run_estimator(scenario, lmap, corrector, correct=True, gain=1.0, seed=0,
                  sensing_radius=3.0, detect_prob=0.95, once_per_marker=True):
    """Integrate drifted odometry along the scenario.

    When `correct` and a mapped marker is inside the camera footprint, generate
    the sighting, run the REAL corrector to get a LandmarkFix, and re-anchor the
    estimate onto it. Returns per-step estimates plus bookkeeping for metrics.

    `once_per_marker` (the natural Week-3 baseline): fix once per marker pass,
    not on every frame a marker is visible. Deciding *which* of the many visible
    frames to actually spend a fix on is exactly the Week-4 scheduler's job.
    """
    rng = np.random.default_rng(seed)
    drift = DriftModel(rng)

    true_xy = scenario["true_xy"]
    true_T = scenario["true_T"]
    n = len(true_xy)

    est = np.zeros((n, 2))
    est[0] = true_xy[0].copy()
    state = NavState(xy=est[0].copy(), yaw=scenario["true_yaw"][0])

    correction_steps = []
    fix_points = []
    n_marker_frames = 0
    n_detected = 0
    used_markers = set()

    for k in range(1, n):
        # 1. propagate with drifted relative motion
        true_delta = true_xy[k] - true_xy[k - 1]
        state.xy = state.xy + drift.corrupt(true_delta)

        # 2. opportunistically correct at a mapped marker
        if correct:
            in_range = markers_in_range(true_T[k], lmap, sensing_radius)
            entry = next((e for e in in_range
                          if not (once_per_marker and e.marker_id in used_markers)), None)
            if entry is not None:
                n_marker_frames += 1
                if rng.random() < detect_prob:      # detection can miss
                    sighting = world_pose_to_sighting(true_T[k], entry, lmap, rng, t=k)
                    fix = corrector.correct(sighting)
                    if fix is not None:
                        n_detected += 1
                        used_markers.add(entry.marker_id)
                        state = reanchor(state, fix, gain=gain)
                        correction_steps.append(k)
                        fix_points.append(fix.xy.copy())

        est[k] = state.xy.copy()

    return {
        "est_xy": est,
        "correction_steps": correction_steps,
        "fix_points": np.array(fix_points) if fix_points else np.empty((0, 2)),
        "n_marker_frames": n_marker_frames,
        "n_detected": n_detected,
    }
