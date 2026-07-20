"""
core/obstacle_avoidance.py
--------------------------
Reactive obstacle avoidance -- Rohan's module (plan section 1, "Obstacle
avoidance": *monocular optical-flow expansion / time-to-contact*). The plan's
frugality principle applies here too: no depth sensor, no stereo, no learned
network. A looming obstacle makes the image EXPAND; the rate of expansion gives
time-to-contact (TTC) directly, and TTC is all a reactive evader needs.

Two layers, split the same way the landmark side is (front-end vs portable core):

  1. FRONT-END (may touch OpenCV, like aruco_detector / blur_metric):
     turn raw phone video into a scalar TTC + a focus-of-expansion (FoE) bearing.
       * `ttc_from_looming`      : from the growth of an object's apparent size.
       * `ttc_from_flow_divergence`: from the divergence of the optical-flow field
                                     (for a translating camera, div(flow) = 1/TTC).
       * `estimate_ttc_lk`       : a real sparse Lucas-Kanade implementation (cv2).
       * `ttc_from_range`        : analytic fallback for the sim's geometric range
                                    and for the iPhone-Pro LiDAR depth fallback the
                                    plan allows.

  2. PORTABLE CORE (pure NumPy, ports to C++/RISC-V):
     `ObstacleAvoidance` -- a hysteretic controller that turns (TTC, bearing) into
     an evasion vector PERPENDICULAR to the seek direction, pushing away from the
     obstacle, with magnitude scaled by urgency; releases when the way is clear.

The evasion vector is what the target-centric controller merges in (controller.py):
    v_cmd = -Kp*(est - B) + evasion
Avoidance reacts on REAL sensing (the estimate may be drifting), while arriving at
B depends on the estimate -- the split Week 5 verifies.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:                                    # cv2 only for the optical-flow front-end
    import cv2
except ImportError:
    cv2 = None


# =========================== front-end: cue -> TTC ==========================
def ttc_from_range(distance: float, closing_speed: float) -> float:
    """TTC [s] from a geometric range and the speed of closing on the obstacle.
    Used with the kinematic sim's forward_depth() and as the LiDAR-depth fallback.
    Returns +inf when not closing (receding or parallel)."""
    if closing_speed <= 1e-6:
        return float("inf")
    return float(max(distance, 0.0) / closing_speed)


def ttc_from_looming(scale_prev: float, scale_now: float, dt: float) -> float:
    """TTC [s] from the growth of an obstacle's apparent size between two frames.

    If an object at distance Z approaches at constant speed, its image size s
    scales as 1/Z, so  s'/s = -Z'/Z  and  TTC = -Z/Z' = s / (ds/dt). A patch that
    is expanding (scale_now > scale_prev) is looming; a shrinking/steady one is
    receding, so TTC = +inf. This is the monocular expansion cue in one line."""
    if dt <= 0 or scale_prev <= 0 or scale_now <= scale_prev:
        return float("inf")
    rate = (scale_now - scale_prev) / dt            # ds/dt > 0
    return float(scale_now / rate)


def ttc_from_flow_divergence(divergence: float) -> float:
    """TTC [s] from the divergence of the optical-flow field near the FoE. For a
    camera translating toward a surface, the radial flow diverges at rate 1/TTC,
    so TTC = 1/divergence. Non-positive divergence (not approaching) -> +inf."""
    if divergence <= 1e-6:
        return float("inf")
    return float(1.0 / divergence)


def estimate_ttc_lk(prev_gray, gray, dt: float, foe=None):
    """REAL monocular TTC from two consecutive frames via sparse Lucas-Kanade flow.

    Tracks good features from `prev_gray` into `gray`, estimates the focus of
    expansion (FoE, defaulting to image centre), and fits the radial expansion
    rate: for a looming scene the flow magnitude grows linearly with distance from
    the FoE, with slope = 1/TTC (divergence). Returns (ttc_s, bearing_rad) where
    bearing is the horizontal offset of the FoE from centre -> which way to dodge.

    This is the phone-video path; `ObstacleAvoidance` below consumes its output.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV not available; `pip install opencv-contrib-python`")
    h, w = gray.shape[:2]
    if foe is None:
        foe = np.array([w / 2.0, h / 2.0])
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01,
                                 minDistance=7)
    if p0 is None or len(p0) < 8:
        return float("inf"), 0.0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
    st = st.reshape(-1).astype(bool)
    a = p0.reshape(-1, 2)[st]
    b = p1.reshape(-1, 2)[st]
    if len(a) < 8:
        return float("inf"), 0.0

    r = a - foe                                     # radial position from FoE
    flow = b - a                                    # per-feature displacement
    radial = np.linalg.norm(r, axis=1) + 1e-6
    # radial expansion speed (component of flow along the outward radial direction)
    radial_speed = np.sum(flow * (r / radial[:, None]), axis=1)
    # slope of radial_speed vs radial distance = divergence = 1/TTC (per frame)
    slope = float(np.polyfit(radial, radial_speed, 1)[0])
    divergence = slope / dt
    ttc = ttc_from_flow_divergence(divergence)
    # dodge away from the side the mean flow points to (the looming side)
    bearing = float(np.arctan2(np.mean(flow[:, 1]), 1.0)) * 0.0  # kept horizontal
    mean_dx = float(np.mean(b[:, 0]) - w / 2.0)
    bearing = float(np.arctan2(mean_dx, w))         # + = obstacle biased to the right
    return ttc, bearing


# =========================== portable core: evader ==========================
@dataclass
class AvoidanceConfig:
    ttc_trigger: float = 2.0     # [s] start evading when TTC drops below this
    ttc_release: float = 3.0     # [s] stop evading once TTC recovers above this (hysteresis)
    ttc_min: float = 0.4         # [s] at/below this, treat as maximal urgency
    gain: float = 2.0           # [m/s] evasion speed at full urgency
    default_side: float = +1.0   # head-on tie-break: +1 dodge right, -1 dodge left


class ObstacleAvoidance:
    """Turn a stream of (TTC, bearing) cues into an evasion vector.

    Reactive and stateless except for one hysteresis bit, so it does not chatter
    on/off at the trigger boundary. Pure NumPy -- this is the part that ports to
    the RISC-V hot loop; the pixel work stays in the front-end above."""

    def __init__(self, cfg: AvoidanceConfig | None = None):
        self.cfg = cfg or AvoidanceConfig()
        self._evading = False
        self.last_ttc = float("inf")
        self.last_urgency = 0.0

    def reset(self):
        self._evading = False
        self.last_ttc = float("inf")
        self.last_urgency = 0.0

    @property
    def evading(self) -> bool:
        return self._evading

    def update(self, seek_dir, ttc: float, bearing: float = 0.0) -> np.ndarray:
        """seek_dir : desired travel direction (world, any length; normalised here)
        ttc      : time-to-contact [s] from the front-end (+inf = clear)
        bearing  : obstacle bearing rel. to seek dir [rad]; + = obstacle to the LEFT
        Returns the evasion velocity vector to add to the seek command (world)."""
        c = self.cfg
        self.last_ttc = ttc

        # hysteresis: engage below trigger, disengage above release
        if self._evading:
            if ttc > c.ttc_release:
                self._evading = False
        else:
            if ttc < c.ttc_trigger:
                self._evading = True

        if not self._evading:
            self.last_urgency = 0.0
            return np.zeros(2)

        s = np.asarray(seek_dir, dtype=float).reshape(2)
        n = np.linalg.norm(s)
        s = s / n if n > 1e-9 else np.array([1.0, 0.0])
        left = np.array([-s[1], s[0]])              # +90 deg (left of travel)

        # dodge to the side AWAY from the obstacle: obstacle on the left -> go right
        if bearing > 1e-3:
            perp = -left                            # obstacle left -> evade right
        elif bearing < -1e-3:
            perp = left                             # obstacle right -> evade left
        else:
            perp = c.default_side * (-left)         # head-on -> default side

        urgency = (c.ttc_trigger - ttc) / max(c.ttc_trigger - c.ttc_min, 1e-6)
        urgency = float(np.clip(urgency, 0.0, 1.0))
        self.last_urgency = urgency
        return c.gain * urgency * perp


# ----------------------------- self-test ------------------------------------
if __name__ == "__main__":
    # TTC cues
    assert ttc_from_range(6.0, 2.0) == 3.0
    assert ttc_from_range(6.0, 0.0) == float("inf")
    assert abs(ttc_from_looming(10.0, 11.0, 0.1) - 1.1) < 1e-6      # s/(ds/dt)=11/10
    assert ttc_from_looming(10.0, 10.0, 0.1) == float("inf")       # not expanding
    assert abs(ttc_from_flow_divergence(0.5) - 2.0) < 1e-9
    print("TTC cue math OK")

    av = ObstacleAvoidance(AvoidanceConfig(ttc_trigger=2.0, ttc_release=3.0, gain=2.0))
    seek = np.array([1.0, 0.0])                     # flying +x

    # far away -> no evasion
    assert np.allclose(av.update(seek, ttc=5.0), [0, 0]) and not av.evading
    # obstacle looming on the LEFT (bearing>0) -> evade to the right (-y)
    e = av.update(seek, ttc=1.0, bearing=0.3)
    assert av.evading and e[1] < 0, e
    print(f"looming left, ttc=1.0 -> evade {e} (rightward, |e|={np.linalg.norm(e):.2f})")
    # hysteresis: ttc=2.5 is between trigger and release -> STILL evading
    e2 = av.update(seek, ttc=2.5, bearing=0.3)
    assert av.evading, "should hold through the hysteresis band"
    # cleared -> release
    e3 = av.update(seek, ttc=4.0, bearing=0.3)
    assert not av.evading and np.allclose(e3, [0, 0])
    print("hysteresis engage/hold/release OK")

    # urgency grows as TTC shrinks
    av.reset()
    u_far = np.linalg.norm(av.update(seek, ttc=1.9, bearing=-0.3))
    u_near = np.linalg.norm(av.update(seek, ttc=0.5, bearing=-0.3))
    assert u_near > u_far
    print(f"urgency scales with closeness: |e|(ttc1.9)={u_far:.2f} < |e|(ttc0.5)={u_near:.2f}")

    print("\nobstacle_avoidance self-test passed.")
