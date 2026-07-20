"""
harness/integrated_sim.py  (throwaway scaffolding, plan section 6)
------------------------------------------------------------------
The END-TO-END money-shot. Everything the plan promises, in one closed loop,
driven by the real portable core (`core/navigator.py`) -- not a shortcut sim:

    homing toward a fixed target B  ->  VIO drift accumulates (worse in a
    feature-poor "hard patch")  ->  the uncertainty scheduler fires a landmark
    fix only when it needs to  ->  state fusion snaps the estimate back  ->  an
    obstacle looms and optical-flow TTC drives a reactive detour  ->  the drone
    resumes and arrives at B.

The sim owns only the WORLD and the SENSING (truth, drift, marker sightings,
time-to-contact). Every DECISION -- when to correct, how to fuse, where to steer,
whether to evade -- is made by the core. Swapping the core's scheduler is the only
difference between the three policies compared here.

Three policies over the identical world + seed:
    none         : corrector disabled -> pure-VIO drift (the baseline)
    fixed        : correct every P marker passes, blind to conditions
    uncertainty  : correct only when U crosses the threshold  <- the contribution
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.controller import ControllerConfig, TargetCentricController
from core.geometry import inv_T, make_T, rot_z, rvec_from_rot
from core.landmark_map import LandmarkMap, MarkerEntry
from core.navigator import SensorInput, build_navigator
from core.obstacle_avoidance import AvoidanceConfig, ObstacleAvoidance, ttc_from_range
from core.state_fusion import FusionConfig, StateFusion
from core.types import MarkerSighting
from core.uncertainty_scheduler import SchedulerConfig, UncertaintyScheduler

# reuse the validated relative-motion drift model from Week 5
from harness.detour_sim import DriftModel


# ============================== the world ===================================
@dataclass
class World:
    B: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    start: np.ndarray = field(default_factory=lambda: np.array([58.0, 24.0]))
    marker_fracs: tuple = (0.14, 0.28, 0.42, 0.60, 0.75, 0.88)  # along the path
    marker_offset: float = 0.0        # markers sit on the nominal path
    obstacle_frac: float = 0.52       # obstacle straddles the path here
    obstacle_lateral: float = 1.5     # nudge off-axis so a real go-around is needed
    obstacle_radius: float = 2.3
    hard_frac: float = 0.34           # centre of the feature-poor zone
    hard_radius: float = 9.0
    hard_peak: float = 0.9            # difficulty in [0,1] at the centre
    # sim
    dt: float = 1.0 / 30.0
    speed: float = 2.0                # m/s cruise
    sensing_radius: float = 4.0       # marker "in view" range
    obstacle_sense: float = 6.0       # start seeing the obstacle within this range
    arrive_radius: float = 1.0
    max_steps: int = 3000
    marker_size_m: float = 0.30
    # scenario toggles (used by the A/B/C evaluation, harness/eval_scenarios.py)
    has_markers: bool = True     # A: False (open, no markers) ; B/C: True
    has_obstacle: bool = True    # A/B: False ; C: True (detour)


def _pt(world: World, frac: float, lateral: float = 0.0) -> np.ndarray:
    """A point at fraction `frac` along start->B, pushed `lateral` metres left."""
    seg = world.B - world.start
    axis = seg / np.linalg.norm(seg)
    left = np.array([-axis[1], axis[0]])
    return world.start + frac * seg + lateral * left


def build_landmark_map(world: World) -> LandmarkMap:
    """Build the world-frame landmark map inline (markers, camera, target B)."""
    markers = []
    fracs = world.marker_fracs if world.has_markers else ()
    for i, f in enumerate(fracs):
        p = _pt(world, f, world.marker_offset)
        T_WM = make_T(rot_z(0.0), [p[0], p[1], 0.0])
        markers.append(MarkerEntry(i, T_WM, world.marker_size_m))
    K = np.array([[800.0, 0, 640.0], [0, 800.0, 360.0], [0, 0, 1.0]])
    dist = np.zeros(5)
    T_BC = make_T(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]]), [0, 0, 0])  # cam down
    return LandmarkMap(markers, world.B, K, dist, T_BC, world.marker_size_m)


def difficulty(world: World, pos) -> float:
    """[0,1] feature-poorness at a point (one Gaussian-ish hard patch)."""
    c = _pt(world, world.hard_frac, 0.0)
    d = float(np.linalg.norm(np.asarray(pos) - c))
    if d >= world.hard_radius:
        return 0.0
    return world.hard_peak * (1.0 - d / world.hard_radius)


def synth_sighting(true_pos, heading, entry, lmap, rng, t, d):
    """Synthesize a physically-consistent ArUco sighting of `entry` from the drone's
    TRUE pose (same geometry as Week 5). Noise grows with difficulty (blurrier)."""
    T_WB_true = make_T(rot_z(heading), [true_pos[0], true_pos[1], 1.2])
    T_CM = inv_T(T_WB_true @ lmap.T_BC) @ entry.T_WM
    tvec_noise = 0.01 + 0.04 * d
    rvec_noise = 0.004 + 0.02 * d
    rvec = rvec_from_rot(T_CM[:3, :3]) + rng.normal(0.0, rvec_noise, 3)
    tvec = T_CM[:3, 3] + rng.normal(0.0, tvec_noise, 3)
    rng_m = float(np.linalg.norm(T_CM[:3, 3]))
    reproj = 0.3 + 0.05 * rng_m + 3.0 * d + rng.uniform(0.0, 0.3)
    return MarkerSighting(int(entry.marker_id), rvec, tvec, t, reproj)


# ===================== a fixed-period scheduler shim ========================
class PeriodicScheduler:
    """Drop-in for UncertaintyScheduler.compute(): fires every `period` calls,
    ignoring U. Lets the navigator run a 'fixed-period' policy unchanged."""

    def __init__(self, period=45):
        self.period = int(period)
        self._k = 0
        self._since = period
        self.last_components = {}

    def compute(self, cues):
        self._k += 1
        self._since += 1
        trigger = self._since >= self.period
        return 0.0, trigger, "fixed_period" if trigger else "none", {}

    def reset_after_fix(self):
        self._since = 0


# ============================== the run =====================================
def _obstacle_cue(world, true_pos, seek_hat, obs_c):
    """Return (ttc, bearing) while the obstacle is within sensing range and still
    ahead (not yet passed), else None. Reactive sensing on TRUE geometry.

    The looming cue TRIGGERS the detour; a geometric 'still ahead and near' test
    HOLDS it until the drone is clear -- otherwise, as the drone turns away, the
    closing speed collapses, TTC shoots up, and a pure-TTC evader would release and
    drift straight back into the obstacle (the classic reactive-avoidance chatter)."""
    to_obs = obs_c - true_pos
    d_center = float(np.linalg.norm(to_obs))
    d_surface = d_center - world.obstacle_radius
    if d_surface > world.obstacle_sense:
        return None
    to_hat = to_obs / max(d_center, 1e-9)
    along = float(np.dot(seek_hat, to_hat))                 # >0: obstacle ahead
    if along < -0.2:                                         # obstacle is behind us -> clear
        return None
    # closing speed floored so TTC stays finite (thus evasion stays engaged) while
    # we are beside the obstacle mid-detour, not only while driving straight at it.
    closing = max(along, 0.35) * world.speed
    ttc = ttc_from_range(d_surface, closing)
    cross = float(seek_hat[0] * to_hat[1] - seek_hat[1] * to_hat[0])  # + = obstacle left
    dot = float(np.clip(along, -1.0, 1.0))
    bearing = float(np.arctan2(cross, dot))
    return ttc, bearing


def run(world: World, policy: str = "uncertainty", seed: int = 0,
        fixed_period: int = 45):
    """Run one closed-loop flight under `policy`. Returns a history dict."""
    rng = np.random.default_rng(seed)
    # Low, honest BASE drift everywhere (~1% of distance), with the bulk of the
    # error CONCENTRATED in the feature-poor hard patch via rate_mult below. That
    # concentration is what lets an adaptive scheduler beat a fixed timer: spend
    # fixes where drift actually grows, skip the easy stretches. (Uniform drift
    # would make every policy equivalent -- see the Week 6 evaluation for the
    # controlled A/B/C version of this argument.)
    drift = DriftModel(rng, yaw_bias=0.00003, yaw_rw=0.00020,
                       scale_rw=0.00010, add_noise=0.0018)

    lmap = build_landmark_map(world)
    obs_c = _pt(world, world.obstacle_frac, world.obstacle_lateral)

    # --- assemble the core for this policy ---
    ctrl = TargetCentricController(
        target_B=world.B, cfg=ControllerConfig(kp=0.6, v_max=world.speed,
                                               arrive_tol=world.arrive_radius))
    fusion = StateFusion(init_xy=world.start, init_std=0.05,
                         cfg=FusionConfig(q_per_metre=0.03))
    # gain ~2.5x cruise so that, after adding to the forward seek and clamping to
    # v_max, the perpendicular dominates the command and the drone turns hard and
    # early enough for a comfortable standoff margin.
    avoider = ObstacleAvoidance(AvoidanceConfig(ttc_trigger=3.6, ttc_release=5.0,
                                                ttc_min=0.5, gain=5.0))
    if policy == "none":
        nav = build_navigator(world.B, landmark_map=None, start_xy=world.start,
                              controller=ctrl, fusion=fusion, avoider=avoider)
    elif policy == "fixed":
        nav = build_navigator(world.B, landmark_map=lmap, start_xy=world.start,
                              controller=ctrl, fusion=fusion, avoider=avoider)
        nav.scheduler = PeriodicScheduler(period=fixed_period)
    elif policy == "uncertainty":
        nav = build_navigator(world.B, landmark_map=lmap, start_xy=world.start,
                              controller=ctrl, fusion=fusion, avoider=avoider,
                              scheduler=UncertaintyScheduler(SchedulerConfig(tau=0.45)))
    else:
        raise ValueError(policy)

    true_pos = world.start.copy().astype(float)
    prev_true = true_pos.copy()
    prev_features = 150.0
    used = set()

    H = {k: [] for k in ("true_xy", "est_xy", "U", "sigma_pos", "trigger",
                         "corrected", "evading", "feat", "err", "dir_err", "mode")}
    corrections, invocations = [], 0

    for k in range(world.max_steps):
        d = difficulty(world, true_pos)
        seek = world.B - fusion.state.xy
        seek_hat = seek / max(np.linalg.norm(seek), 1e-9)
        heading = float(np.arctan2(seek_hat[1], seek_hat[0]))

        # obstacle sensed this step? (drives the detour AND a maneuver drift bump)
        obstacle = _obstacle_cue(world, true_pos, seek_hat, obs_c) if world.has_obstacle else None
        maneuvering = obstacle is not None

        # --- SENSING: build the SensorInput the core consumes ---
        # Drift is amplified sharply inside the hard patch (rate_mult up to ~8x) and
        # again during the obstacle maneuver (Week-5 point: an aggressive detour
        # itself injects drift, plan constraint 4). Most error is earned in these two
        # zones, which is exactly where the adaptive scheduler must spend its fixes.
        man = 3.5 if maneuvering else 1.0
        vio_delta = drift.corrupt(true_pos - prev_true, rate_mult=(1.0 + 8.0 * d) * man)

        feat = max(0.0, 150.0 * (1.0 - 0.85 * d) + rng.normal(0, 3))
        feature_loss = max(0.0, prev_features - feat) / world.dt
        prev_features = feat
        blur = max(5.0, 300.0 * (1.0 - 0.85 * d) + rng.normal(0, 8))
        # extra process variance in the hard patch AND the maneuver so the fused
        # sigma_pos honestly SPIKES where real drift spikes -- the scheduler reads it.
        cues = dict(feature_loss=feature_loss, blur=blur, imu_bias=0.02 + 0.05 * d,
                    sigma_head=0.01, active_features=feat,
                    extra_var=0.16 * d + (0.10 if maneuvering else 0.0))

        # marker in view? (nearest unused mapped marker within sensing radius)
        sighting = None
        for mid in lmap.ids():
            if mid in used:
                continue
            mx, my = lmap.marker_world_xy(mid)
            if np.hypot(true_pos[0] - mx, true_pos[1] - my) <= world.sensing_radius:
                sighting = synth_sighting(true_pos, heading, lmap.get(mid), lmap, rng, k, d)
                break

        si = SensorInput(t=k * world.dt, vio_delta=vio_delta, cues=cues,
                         sighting=sighting, obstacle=obstacle)

        # --- DECIDE + COMMAND (all core) ---
        out = nav.step(si)
        if out.trigger and sighting is not None and nav.corrector is not None:
            invocations += 1                      # a real detector/corrector run
            if out.corrected:
                used.add(sighting.marker_id)
                corrections.append(k)
                # A marker gives absolute ORIENTATION as well as position, so it
                # re-anchors the VIO's heading/scale drift -- not just the fused xy.
                # (Same effect as drift_scaffold's `heading_bias *= 0.1` on a fix.)
                # Without this, position snaps back but the accumulated heading error
                # keeps rotating every subsequent VIO delta and error ramps straight
                # back up -- the estimator would be badly overconfident.
                drift.yaw_err *= 0.1
                drift.log_scale *= 0.1

        # --- ACTUATE: truth follows the command; markers can be re-approached ---
        prev_true = true_pos.copy()
        v = np.array([out.cmd.vx, out.cmd.vy])
        true_pos = true_pos + v * world.dt

        err = float(np.linalg.norm(true_pos - out.est_xy))
        vt = world.B - true_pos
        ve = world.B - out.est_xy
        nb = np.linalg.norm(vt) * np.linalg.norm(ve)
        dir_err = 0.0 if nb < 1e-9 else float(np.degrees(
            np.arccos(np.clip(np.dot(vt, ve) / nb, -1, 1))))

        H["true_xy"].append(true_pos.copy())
        H["est_xy"].append(out.est_xy.copy())
        H["U"].append(out.U)
        H["sigma_pos"].append(out.sigma_pos)
        H["trigger"].append(out.trigger)
        H["corrected"].append(out.corrected)
        H["evading"].append(out.evading)
        H["feat"].append(feat)
        H["err"].append(err)
        H["dir_err"].append(dir_err)
        H["mode"].append("evade" if out.evading else "seek")

        if nav.arrived():
            break

    true_xy = np.array(H["true_xy"])
    est_xy = np.array(H["est_xy"])
    arrival_miss = float(np.linalg.norm(true_xy[-1] - world.B))
    # true clearance to the obstacle over the whole flight (safety check)
    if world.has_obstacle:
        clr = float(np.min(np.linalg.norm(true_xy - obs_c, axis=1)) - world.obstacle_radius)
    else:
        clr = float("inf")

    return {
        "policy": policy,
        "true_xy": true_xy, "est_xy": est_xy,
        "U": np.array(H["U"]), "sigma_pos": np.array(H["sigma_pos"]),
        "trigger": np.array(H["trigger"]), "corrected": np.array(H["corrected"]),
        "evading": np.array(H["evading"]), "feat": np.array(H["feat"]),
        "err": np.array(H["err"]), "dir_err": np.array(H["dir_err"]),
        "mode": H["mode"], "corrections": corrections, "invocations": invocations,
        "arrival_miss_m": arrival_miss, "min_clearance_m": float(clr),
        "obstacle_hit": bool(clr < 0.0),
        "lmap": lmap, "obstacle": (obs_c, world.obstacle_radius),
        "B": world.B, "start": world.start, "hard_center": _pt(world, world.hard_frac),
        "hard_radius": world.hard_radius, "dt": world.dt,
        "peak_err": float(np.max(H["err"])),
    }


def run_all(seed: int = 0, fixed_period: int = 45):
    """Run all three policies over the same world + seed."""
    world = World()
    return {p: run(world, policy=p, seed=seed, fixed_period=fixed_period)
            for p in ("none", "fixed", "uncertainty")}


if __name__ == "__main__":
    res = run_all(seed=1)
    print(f"{'policy':<14}{'arrival':>9}{'peak_err':>10}{'AVL':>6}{'fixes':>7}{'clear':>8}")
    for p in ("none", "fixed", "uncertainty"):
        r = res[p]
        print(f"{p:<14}{r['arrival_miss_m']:>8.2f}m{r['peak_err']:>9.2f}m"
              f"{r['invocations']:>6}{len(r['corrections']):>7}{r['min_clearance_m']:>7.2f}m")
