"""
frugalnav_hf.py
---------------
FrugalNav-HF: a higher-fidelity closed-loop simulator for the TCOM study
"When Is a Correction Worth Its Cost?".

What is reused UNCHANGED from FrugalNav (vendored in ./core):
  * core.uncertainty_scheduler.UncertaintyScheduler  (the adaptive trigger, U)
  * core.state_fusion.StateFusion                    (the fused estimate + Kalman update)

What is new here (stands in for PX4 SITL, which was not available to us):
  * planar rigid-body quadrotor: first-order attitude (tilt) lag, tilt limit,
    linear rotor drag + quadratic body drag, cascaded velocity controller,
    battery draw model;
  * wind: steady component + Dryden-like first-order Gauss-Markov gusts;
  * camera: auto-exposure under changing light, lens blur, velocity- and
    rate-dependent motion blur, sensor noise, random dropout; marker detection
    probability and fix error come from a lookup table MEASURED with the real
    OpenCV ArUco detector (camera_calibration.py);
  * VIO error model whose noise rises with blur, low texture, haze and body rate;
  * three worlds (open arena, city block, rocky canyon) with haze patches;
  * three policies: always-correct, fixed timer, adaptive (FrugalNav's U).

Everything is seeded; policies share seeds so comparisons are paired.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from core.state_fusion import FusionConfig, StateFusion            # noqa: E402
from core.types import LandmarkFix                                  # noqa: E402
from core.uncertainty_scheduler import (CueBounds, SchedulerConfig,  # noqa: E402
                                        UncertaintyScheduler)

G = 9.81
DT = 0.02            # physics step, 50 Hz
NAV_EVERY = 5        # navigator / camera at 10 Hz
F_PX = 800.0         # FrugalNav camera focal length
DETECT_MS = None     # filled from results/detector_timing.json


# ------------------------------------------------------------------ camera LUT
class CameraModel:
    """Interpolates the measured ArUco calibration grid."""

    def __init__(self, path=os.path.join(HERE, "results", "camera_lut.npz")):
        rows = np.load(path)["rows"]
        self.axes = [np.unique(rows[:, i]) for i in range(5)]
        shape = [len(a) for a in self.axes]
        idx = [np.searchsorted(self.axes[i], rows[:, i]) for i in range(5)]
        pdet = np.zeros(shape)
        err = np.zeros(shape)
        lap = np.zeros(shape)
        for r, *ix in zip(rows, *idx):
            pdet[tuple(ix)] = r[5]
            err[tuple(ix)] = r[6] if np.isfinite(r[6]) else 0.10
            lap[tuple(ix)] = r[8]
        kw = dict(bounds_error=False, fill_value=None)
        self._p = RegularGridInterpolator(self.axes, pdet, **kw)
        self._e = RegularGridInterpolator(self.axes, err, **kw)
        self._l = RegularGridInterpolator(self.axes, lap, **kw)
        # sharpness reference: clean image at each altitude
        self.lap_clean = {a: float(self._l([[a, 0, 0, 0, 1]])[0]) for a in self.axes[0]}

    def _q(self, alt, lens, motion, noise, gain):
        x = [np.clip(alt, self.axes[0][0], self.axes[0][-1]),
             np.clip(lens, 0, self.axes[1][-1]), np.clip(motion, 0, self.axes[2][-1]),
             np.clip(noise, 0, self.axes[3][-1]),
             np.clip(gain, self.axes[4][0], self.axes[4][-1])]
        return [x]

    def p_detect(self, *a):
        return float(np.clip(self._p(self._q(*a))[0], 0, 1))

    def fix_err(self, *a):
        return float(max(self._e(self._q(*a))[0], 0.0))

    def sharpness(self, alt, *a):
        """Noise-compensated Laplacian variance relative to a clean frame, on a log
        scale, in [0,1]. Raw variance falls ~30x for 1 px of blur, so a linear
        ratio saturates; log(lap)/log(lap_clean) stays graded."""
        ref = np.interp(alt, self.axes[0], [self.lap_clean[v] for v in self.axes[0]])
        lap = max(float(self._l(self._q(alt, *a))[0]), 1.0)
        return float(np.clip(math.log(lap) / math.log(ref), 0, 1))


# ------------------------------------------------------------------- worlds
@dataclass
class World:
    name: str
    route: np.ndarray          # waypoints (N,2)
    marker_spacing: float      # markers along the route every s metres
    marker_lateral: float      # std of marker lateral offset from route (m)
    occlusion: float           # prob. a marker is hidden (buildings, shadow) per pass
    texture: callable          # s (arc length) -> texture quality [0,1]
    haze: list                 # [(s0, s1, contrast_factor, altitude)]
    gust_mult: float           # local turbulence amplification
    altitude: float = 5.0


def _polyline(pts):
    return np.asarray(pts, dtype=float)


def make_worlds():
    open_arena = World(
        "Open arena", _polyline([(0, 0), (0, 40), (30, 40)]),
        marker_spacing=10.0, marker_lateral=0.4, occlusion=0.0,
        texture=lambda s: 0.9, haze=[], gust_mult=1.0)
    city = World(
        "City block", _polyline([(0, 0), (0, 25), (20, 25), (20, 50), (40, 50)]),
        marker_spacing=8.0, marker_lateral=0.6, occlusion=0.35,
        texture=lambda s: 0.45 if (25 < s < 45 or 70 < s < 85) else 0.75,
        haze=[(50.0, 62.0, 0.6, 4.0)], gust_mult=1.4)
    canyon = World(
        "Rocky canyon", _polyline([(0, 0), (8, 20), (0, 40), (10, 60), (4, 80)]),
        marker_spacing=13.0, marker_lateral=0.5, occlusion=0.10,
        texture=lambda s: 0.8 if s < 35 else (0.5 if s < 65 else 0.8),
        haze=[(32.0, 66.0, 0.35, 3.0)], gust_mult=1.2)
    return [open_arena, city, canyon]


def arc_points(route, spacing, rng, lateral):
    """Survey markers every `spacing` m of arc length, jittered laterally."""
    seg = np.diff(route, axis=0)
    L = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0], np.cumsum(L)])
    out = []
    s = spacing * 0.6
    while s < cum[-1]:
        i = min(np.searchsorted(cum, s) - 1, len(L) - 1)
        u = (s - cum[i]) / L[i]
        p = route[i] + u * seg[i]
        n = np.array([-seg[i][1], seg[i][0]]) / L[i]
        out.append(p + n * rng.normal(0, lateral))
        s += spacing
    return np.array(out), cum


def arclen_of(p, route, cum):
    best, bs = 1e9, 0.0
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        ab = b - a
        u = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
        d = np.linalg.norm(p - (a + u * ab))
        if d < best:
            best, bs = d, cum[i] + u * np.linalg.norm(ab)
    return bs


# -------------------------------------------------------------- conditions
CONDITIONS = {
    #             wind_mean  gust_sigma  lens  t_exp_nom  noise0  light_var  dropout
    "Ideal":     dict(wind=0.0, gust=0.0, lens=0.0, texp=0.004, noise=0.0, light=0.0, drop=0.0),
    "Wind":      dict(wind=3.0, gust=1.5, lens=0.0, texp=0.004, noise=0.0, light=0.0, drop=0.0),
    "Degraded":  dict(wind=0.0, gust=0.0, lens=1.5, texp=0.008, noise=0.02, light=0.5, drop=0.10),
    "Combined":  dict(wind=3.0, gust=1.5, lens=1.5, texp=0.008, noise=0.02, light=0.5, drop=0.10),
}


# ---------------------------------------------------------------- policies
class Policy:
    name = "base"

    def wants(self, t, since_fix, U, trig):
        raise NotImplementedError


class AlwaysCorrect(Policy):
    name = "Always-correct"

    def wants(self, t, since_fix, U, trig):
        return True


class FixedTimer(Policy):
    def __init__(self, period_s=5.0):
        self.period = period_s
        self.name = "Fixed timer"

    def wants(self, t, since_fix, U, trig):
        return since_fix >= self.period


class Adaptive(Policy):
    name = "Adaptive (U)"

    def wants(self, t, since_fix, U, trig):
        return trig


class Repaired(Policy):
    """FrugalNav-R: repaired scheduler (core_ext.py) + honest process noise."""
    name = "Repaired (U*)"
    variant = "repaired"

    def __init__(self, budget_m=0.5, tau=0.45, refractory_ticks=15):
        self.budget = budget_m
        self.tau = tau
        self.refractory = refractory_ticks

    def wants(self, t, since_fix, U, trig):
        return trig


class NoCorrection(Policy):
    """Never corrects; used only to calibrate the honest process-noise model."""
    name = "No correction"
    variant = "repaired"

    def wants(self, t, since_fix, U, trig):
        return False


def scheduler_for(tau):
    # Blur bound re-expressed on the noise-compensated sharpness ratio (1 = clean
    # frame); sharpness is fed in as a 0..300 pseudo-Laplacian so the vendored
    # normaliser and weights are used exactly as shipped.
    cfg = SchedulerConfig(tau=tau, bounds=CueBounds(blur=(90.0, 270.0)))
    return UncertaintyScheduler(cfg)


# --------------------------------------------------------------- one flight
# Honest process noise (repaired variant): correlated VIO errors (scale, heading,
# bias) make drift grow with the SQUARE of distance since the last fix, and poor
# imagery adds more. Coefficients calibrated on held-out seeds (tune_repaired.py).
Q_QUAD = 8.0e-4      # m^2 per m^2 of distance-since-fix (tuned, results/repaired_tuning.json)
Q_QUAL = 0.02        # m^2 per m flown at zero image quality (tuned)


def fly(world: World, cond_name: str, policy: Policy, seed: int, tau=0.45,
        cam: CameraModel | None = None, log=False):
    repaired = getattr(policy, "variant", "shipped") == "repaired"
    c = CONDITIONS[cond_name]
    rs = np.random.SeedSequence([seed, sum(map(ord, world.name))])
    r_world, r_wind, r_vio, r_cam, r_light = [np.random.default_rng(s) for s in rs.spawn(5)]

    markers, cum = arc_points(world.route, world.marker_spacing, r_world, world.marker_lateral)
    hidden = r_world.random(len(markers)) < world.occlusion
    target = world.route[-1]

    # --- state
    p = world.route[0].copy()
    v = np.zeros(2)
    tilt = np.zeros(2)                      # tilt "acceleration" state (m/s^2 equiv.)
    yaw_err = 0.0                           # VIO heading error (rad)
    gyro_bias = r_vio.normal(0, 0.004)      # rad/s heading drift
    vbias = np.zeros(2)                     # VIO velocity bias (m/s)
    scale_err = r_vio.normal(0, 0.015)
    wind_dir = r_wind.uniform(0, 2 * np.pi)
    wind_mean = c["wind"] * np.array([math.cos(wind_dir), math.sin(wind_dir)])
    gust = np.zeros(2)
    light_phase = r_light.uniform(0, 2 * np.pi)

    fus = StateFusion(init_xy=p, init_std=0.05, cfg=FusionConfig(q_per_metre=0.02))
    if repaired:
        from core_ext import RepairedScheduler
        sch = RepairedScheduler(getattr(policy, "budget", 1.0), tau=getattr(policy, "tau", 0.45),
                                refractory_ticks=getattr(policy, "refractory", 15))
    else:
        sch = scheduler_for(tau)
    marker_s = np.array([arclen_of(m, world.route, cum) for m in markers])
    D_fix = 0.0                             # distance flown since last fix (estimated)
    last_quality = 1.0
    head_var = 0.01 ** 2
    feats_prev = 120.0

    # --- vehicle / controller constants (1.2 kg class quad)
    TAU_ATT, A_MAX = 0.12, G * math.tan(math.radians(25))
    K_V, V_MAX = 2.0, 2.0
    C_LIN, C_QUAD, MASS = 0.30, 0.05, 1.2       # rotor drag (1/s), body drag (kg/m)
    P_HOVER = 180.0                              # W

    wp_i, t, since_fix = 1, 0.0, 99.0
    attempts = fixes = 0
    reasons = {}
    bits = []
    errs = []
    nees = []
    energy_J = 0.0
    trace = []
    arrived = False
    T_MAX = 160.0
    vio_vel = np.zeros(2)
    omega = 0.0

    step = 0
    while t < T_MAX:
        s_true = arclen_of(p, world.route, cum)
        alt, contrast = world.altitude, 1.0
        for (h0, h1, cf, ha) in world.haze:
            if h0 <= s_true <= h1:
                alt, contrast = ha, cf
        texture = world.texture(s_true) * contrast

        # ---------------- navigator + camera at 10 Hz
        if step % NAV_EVERY == 0:
            est = fus.state.xy.copy()
            # waypoint following on the ESTIMATE
            while wp_i < len(world.route) - 1 and np.linalg.norm(world.route[wp_i] - est) < 1.5:
                wp_i += 1
            goal = world.route[wp_i]
            d = goal - est
            dist = np.linalg.norm(d)
            if wp_i == len(world.route) - 1 and np.linalg.norm(target - est) < 0.5:
                arrived = True
                break
            v_cmd = d / max(dist, 1e-6) * min(V_MAX, 0.8 * dist + 0.3)

            # camera exposure and image quality
            light = 1.0 - c["light"] * (0.5 + 0.5 * math.sin(0.15 * t + light_phase))
            light = max(light * (0.75 if contrast < 1 else 1.0), 0.15)
            t_exp = min(c["texp"] / light, 0.030) if c["light"] > 0 else c["texp"]
            gain = min(1.0, light * t_exp / c["texp"]) if c["light"] > 0 else 1.0
            noise = c["noise"] / math.sqrt(max(gain, 0.2)) if c["noise"] > 0 else 0.0
            speed = np.linalg.norm(v)
            motion_px = (speed / alt + abs(omega)) * F_PX * t_exp
            lens = c["lens"] + (1.0 - contrast) * 1.5          # haze = extra low-pass
            sharp = cam.sharpness(alt, lens, motion_px, noise, gain * contrast)
            feats = 120.0 * texture * (0.4 + 0.6 * sharp) + r_cam.normal(0, 4)
            feat_loss = max(0.0, feats_prev - feats) * 10.0   # per second
            feats_prev = feats

            sig_cue = fus.sigma_pos()
            if repaired:
                # predicted sigma at the next mapped marker beyond the current footprint
                s_est = arclen_of(est, world.route, cum)
                ahead = marker_s[marker_s > s_est + alt * 0.8]
                gap = (ahead[0] if len(ahead) else cum[-1]) - s_est
                P_now = float(np.max(np.linalg.eigvalsh(fus.state.covariance)))
                grow = Q_QUAD * ((D_fix + gap) ** 2 - D_fix ** 2) +                     (0.02 + Q_QUAL * (1 - last_quality)) * gap
                sig_cue = math.sqrt(P_now + grow)
            cues = dict(sigma_pos=sig_cue, sigma_head=math.sqrt(head_var),
                        feature_loss=feat_loss, blur=90.0 + 180.0 * sharp,
                        imu_bias=abs(gyro_bias) * 10, active_features=feats)
            U, trig, reason, _ = sch.compute(cues)

            # marker in footprint? (half-width = alt * tan(HFOV/2), HFOV ~ 77 deg)
            foot = alt * 0.8
            dd = np.linalg.norm(markers - p, axis=1)
            vis = np.where((dd < foot) & ~hidden)[0]
            if len(vis) and policy.wants(t, since_fix, U, trig):
                attempts += 1
                reasons[reason] = reasons.get(reason, 0) + 1
                pd = cam.p_detect(alt, lens, motion_px, noise, gain * contrast)
                pd *= (1.0 - c["drop"])
                if r_cam.random() < pd:
                    sig = max(cam.fix_err(alt, lens, motion_px, noise, gain * contrast), 0.03) + 0.02
                    meas = p + r_cam.normal(0, sig, 2)
                    P_before = fus.state.covariance.copy()
                    fus.update(LandmarkFix(xy=meas, yaw=0.0, covariance=np.eye(2) * sig ** 2,
                                           marker_id=int(vis[0]), timestamp=t))
                    P_after = fus.state.covariance
                    bits.append(0.5 * math.log2(np.linalg.det(P_before) / np.linalg.det(P_after)))
                    yaw_err = r_cam.normal(0, 0.01)
                    head_var = 0.01 ** 2
                    vbias *= 0.3                        # fix also re-anchors VIO bias
                    fixes += 1
                    since_fix = 0.0
                    D_fix = 0.0
                    sch.reset_after_fix()

            # VIO: relative motion over the 0.1 s window, with degradation-driven noise
            dt_nav = DT * NAV_EVERY
            quality = max(texture * (0.3 + 0.7 * sharp), 0.05)
            sig_v = 0.02 + 0.06 * (1 - quality) + 0.04 * abs(omega)
            vbias += r_vio.normal(0, 0.004 * (1.5 - quality), 2) * math.sqrt(dt_nav)
            yaw_err += gyro_bias * dt_nav + r_vio.normal(0, 0.002 * (1.5 - quality))
            Ry = np.array([[math.cos(yaw_err), -math.sin(yaw_err)],
                           [math.sin(yaw_err), math.cos(yaw_err)]])
            vio_vel = (1 + scale_err) * (Ry @ v) + vbias + r_vio.normal(0, sig_v, 2)
            delta = vio_vel * dt_nav
            # filter inflates its own process noise from the cues it can see
            dd_m = float(np.linalg.norm(delta))
            if repaired:
                # quality as the drone can observe it: feature count and sharpness
                obs_q = max(min(feats / 120.0, 1.0) * (0.3 + 0.7 * sharp), 0.05)
                last_quality = obs_q
                extra = Q_QUAL * (1 - obs_q) * dd_m + Q_QUAD * ((D_fix + dd_m) ** 2 - D_fix ** 2)
            else:
                extra = (0.01 * (1 - quality)) * dd_m
            D_fix += dd_m
            fus.predict(delta, extra_var=extra)
            head_var += (0.002 * (1.5 - quality)) ** 2 + (0.004 * dt_nav) ** 2
            since_fix += dt_nav

            e = fus.state.xy - p
            errs.append(float(np.linalg.norm(e)))
            P = fus.state.covariance
            nees.append(float(e @ np.linalg.solve(P, e)))
            if log:
                trace.append((t, *p, *fus.state.xy, fus.sigma_pos(), U, sharp, alt,
                              int(fixes), errs[-1],
                              0.5 * math.log2((2 * math.pi * math.e) ** 2 * np.linalg.det(P))))

        # ---------------- physics at 50 Hz (autopilot tracks v_cmd using VIO velocity)
        a_des = K_V * (v_cmd - vio_vel)
        n = np.linalg.norm(a_des)
        if n > A_MAX:
            a_des *= A_MAX / n
        tilt_prev = tilt.copy()
        tilt += (a_des - tilt) * DT / TAU_ATT
        omega = float(np.linalg.norm(tilt - tilt_prev) / G / DT)    # body rate proxy, rad/s
        gust += (-gust / 2.0) * DT + c["gust"] * world.gust_mult * math.sqrt(2 * DT / 2.0) * r_wind.normal(0, 1, 2)
        w = wind_mean * world.gust_mult + gust
        vr = v - w
        acc = tilt - C_LIN * vr - (C_QUAD / MASS) * np.linalg.norm(vr) * vr
        v += acc * DT
        p += v * DT
        thrust_ratio = math.sqrt(1 + (np.linalg.norm(tilt) / G) ** 2)
        energy_J += P_HOVER * thrust_ratio ** 1.5 * DT
        t += DT
        step += 1

    e_arr = np.array(errs)
    det_ms = DETECT_MS or 25.0
    return dict(world=world.name, cond=cond_name, policy=policy.name, seed=seed,
                tau=getattr(policy, "tau", tau), refractory=getattr(policy, "refractory", np.nan),
                budget=getattr(policy, "budget", np.nan),
                period=getattr(policy, "period", np.nan),
                arrived=arrived, time_s=t, arrival_err=float(np.linalg.norm(p - target)),
                rmse=float(np.sqrt(np.mean(e_arr ** 2))), peak=float(e_arr.max()),
                attempts=attempts, fixes=fixes,
                bits_per_fix=float(np.mean(bits)) if bits else np.nan,
                bits_total=float(np.sum(bits)) if bits else 0.0,
                compute_share=attempts * det_ms / 1000.0 / t,
                energy_Wh=energy_J / 3600.0, nees=float(np.mean(nees)),
                trig_soft=reasons.get("soft_U", 0), trig_sigma=reasons.get("hard_floor_sigma", 0),
                trig_obs=reasons.get("observability_floor", 0),
                trace=np.array(trace) if log else None)
