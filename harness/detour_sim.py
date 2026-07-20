"""
Closed-loop detour simulator (harness -- throwaway).

Head to B -> detour around an obstacle -> resume -> arrive. The drone STEERS BY
ITS ESTIMATE (target-centric: fly along B - est_pos), while obstacle avoidance is
reactive on TRUE geometry (Rohan's real optical-flow module doesn't depend on the
drifting estimate). That split is the whole point of Week 5:

    * avoidance stays safe even when localization drifts (uses real sensing), but
    * ARRIVING AT B depends on the estimate -- so if drift corrupts the estimate
      during the maneuver, the drone clears the obstacle yet misses the target.

Siddharth's landmark corrector (reused from Week 3) is what keeps the estimate --
and therefore the vector to B -- correct through the detour. This module runs the
loop with and without that correction so the difference can be measured.

Attribution: the controller merge (seek + evade) and the obstacle standoff are
minimal stand-ins for Parth's controller and Rohan's obstacle module. Siddharth
owns the verification layered on top (vector-to-B integrity, tracking continuity).
"""
from __future__ import annotations

import json

import numpy as np

from core.geometry import inv_T, make_T, rot_z, rvec_from_rot
from core.landmark_corrector import reanchor
from core.landmark_map import LandmarkMap
from core.types import MarkerSighting, NavState
from core.vector_to_target import direction_error_deg, position_error


# ----------------------------------------------------------------------------
def load_scenario(path):
    """Returns (LandmarkMap, obstacle dict, sim dict)."""
    lmap = LandmarkMap.from_json(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return lmap, cfg["obstacle"], cfg["sim"]


# --- reused sighting synthesis (same geometry as week3) ---------------------
def simulate_sighting(T_WB_true, entry, lmap, rng, t=0.0,
                      tvec_noise=0.01, rvec_noise=0.004):
    T_CM = inv_T(T_WB_true @ lmap.T_BC) @ entry.T_WM
    rvec = rvec_from_rot(T_CM[:3, :3]) + rng.normal(0.0, rvec_noise, 3)
    tvec = T_CM[:3, 3] + rng.normal(0.0, tvec_noise, 3)
    rng_m = float(np.linalg.norm(T_CM[:3, 3]))
    reproj = 0.3 + 0.05 * rng_m + rng.uniform(0.0, 0.3)
    return MarkerSighting(int(entry.marker_id), rvec, tvec, t, reproj)


def markers_in_range(true_pos, lmap, radius):
    hits = []
    for mid in lmap.ids():
        mx, my = lmap.marker_world_xy(mid)
        d = float(np.hypot(true_pos[0] - mx, true_pos[1] - my))
        if d <= radius:
            hits.append((d, lmap.get(mid)))
    hits.sort(key=lambda dm: dm[0])
    return [entry for _, entry in hits]


# --- VIO drift model (integrated relative-motion error) ---------------------
class DriftModel:
    def __init__(self, rng, yaw_bias=0.003, yaw_rw=0.004, scale_rw=0.002, add_noise=0.008):
        self.rng = rng
        self.yaw_bias, self.yaw_rw = yaw_bias, yaw_rw
        self.scale_rw, self.add_noise = scale_rw, add_noise
        self.yaw_err = 0.0
        self.log_scale = 0.0

    def corrupt(self, delta_xy, rate_mult=1.0):
        self.yaw_err += (self.yaw_bias + self.rng.normal(0.0, self.yaw_rw)) * rate_mult
        self.log_scale += self.rng.normal(0.0, self.scale_rw) * rate_mult
        s = np.exp(self.log_scale)
        c, sn = np.cos(self.yaw_err), np.sin(self.yaw_err)
        R = np.array([[c, -sn], [sn, c]])
        return s * (R @ delta_xy) + self.rng.normal(0.0, self.add_noise, 2) * rate_mult


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


# ----------------------------------------------------------------------------
def run_detour(lmap, obstacle, sim, corrector, correct=True, seed=0):
    """Run the closed loop once. Returns trajectories + per-step verification
    signals + arrival metrics."""
    rng = np.random.default_rng(seed)
    drift = DriftModel(rng)

    B = lmap.target_B
    start = np.asarray(sim["start_xy"], dtype=float)
    obs_c = np.asarray(obstacle["center"], dtype=float)
    obs_r = float(obstacle["radius"])
    standoff = obs_r + float(obstacle["standoff_margin"])
    route_axis = _unit(B - start)

    v = float(sim["speed"])
    alt = float(sim["altitude"])
    seek_gain = float(sim["seek_gain"])
    evade_gain = float(sim["evade_gain"])
    man_mult = float(sim["maneuver_drift_mult"])
    sensing_r = float(sim["sensing_radius"])
    arrive_r = float(sim["arrive_radius"])
    max_steps = int(sim["max_steps"])
    feat_nom = float(sim["feature_nominal"])
    feat_evd = float(sim["feature_evade"])

    true_pos = start.copy()
    state = NavState(xy=start.copy())
    used = set()

    H = {k: [] for k in ("true_xy", "est_xy", "mode", "feat", "pos_err", "dir_err")}
    corrections = []
    min_true_obstacle_clearance = np.inf

    for k in range(max_steps):
        seek_dir = _unit(B - state.xy)                 # target-centric, ESTIMATE-based

        # obstacle avoidance uses TRUE geometry (real sensor, not the estimate)
        to_obs = true_pos - obs_c
        d_obs = float(np.linalg.norm(to_obs))
        along = float(np.dot(to_obs, route_axis))       # progress past the obstacle
        evading = (d_obs < standoff) and (along < obs_r)

        if evading:
            perp = np.array([-seek_dir[1], seek_dir[0]])       # rotate seek 90 deg
            if np.dot(perp, to_obs) < 0:
                perp = -perp                                   # push AWAY from obstacle
            cmd = _unit(seek_gain * seek_dir + evade_gain * perp)
            rate, feat, mode = man_mult, feat_evd + rng.normal(0, 3), "evade"
        else:
            cmd = seek_dir
            rate, feat, mode = 1.0, feat_nom + rng.normal(0, 5), "seek"

        # advance truth, then propagate the drifted estimate by the SAME motion
        prev = true_pos.copy()
        true_pos = true_pos + v * cmd
        min_true_obstacle_clearance = min(min_true_obstacle_clearance,
                                          float(np.linalg.norm(true_pos - obs_c)) - obs_r)
        state.xy = state.xy + drift.corrupt(true_pos - prev, rate)

        # opportunistic landmark correction (once per marker pass)
        if correct:
            yaw = float(np.arctan2(cmd[1], cmd[0]))
            T_WB_true = make_T(rot_z(yaw), [true_pos[0], true_pos[1], alt])
            entry = next((e for e in markers_in_range(true_pos, lmap, sensing_r)
                          if e.marker_id not in used), None)
            if entry is not None:
                fix = corrector.correct(simulate_sighting(T_WB_true, entry, lmap, rng, t=k))
                if fix is not None:
                    state = reanchor(state, fix, gain=1.0)
                    used.add(entry.marker_id)
                    corrections.append(k)

        H["true_xy"].append(true_pos.copy())
        H["est_xy"].append(state.xy.copy())
        H["mode"].append(mode)
        H["feat"].append(max(0.0, feat))
        H["pos_err"].append(position_error(state.xy, true_pos))
        H["dir_err"].append(direction_error_deg(state.xy, true_pos, B))

        if float(np.linalg.norm(state.xy - B)) < arrive_r:   # drone BELIEVES it arrived
            break

    true_xy = np.array(H["true_xy"])
    est_xy = np.array(H["est_xy"])
    arrival_miss = float(np.linalg.norm(true_xy[-1] - B))     # where it ACTUALLY ended vs B
    hit_obstacle = min_true_obstacle_clearance < 0.0

    return {
        "correct": correct,
        "true_xy": true_xy,
        "est_xy": est_xy,
        "mode": H["mode"],
        "feat": np.array(H["feat"]),
        "pos_err": np.array(H["pos_err"]),
        "dir_err": np.array(H["dir_err"]),
        "corrections": corrections,
        "arrival_miss_m": arrival_miss,
        "obstacle_hit": hit_obstacle,
        "min_clearance_m": float(min_true_obstacle_clearance),
        "B": B, "obstacle": (obs_c, obs_r, standoff), "start": start,
    }
