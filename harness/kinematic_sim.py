"""
harness/kinematic_sim.py
------------------------
Rohan's kinematic simulator. The plan lists this for Week 5, but build it NOW:
your Week-4 uncertainty metric needs a testbed with controllable drift and perfect
ground truth, and a real VIO (Week 2) may not even compile in time. This sim lets
your contribution proceed no matter what the rest of the team's VIO is doing.

It is a *harness* (throwaway) tool, NOT part of the portable core. It fakes the
things the real system will measure, so you can develop the scheduler against it.

World model (top-down 2D, fixed altitude):
  - Target B is the ORIGIN (0,0) in the world frame (matches the locked architecture).
  - The drone starts at some offset and homes toward (0,0): v = -Kp * (x_hat, y_hat).
  - TRUE state is integrated from the commanded velocity (perfect physics).
  - The VIO ESTIMATE is the true motion + injected noise + slow bias drift. This is
    our stand-in for OpenVINS until the real one is wired in -- the scheduler cannot
    tell the difference, because it only ever sees the cues.
  - "Difficulty" d(t) in [0,1] models hard patches (low texture / motion blur). Drift,
    feature-loss, and blur all worsen with d(t). This is what makes adaptive scheduling
    beat fixed-interval: corrections should cluster in the hard patches.
  - Markers: when the drone is within a marker's radius, an absolute fix is available.
  - Obstacles: a forward depth-cone min-distance is provided for Rohan's Week-5 module.

Everything is seeded, so runs are exactly reproducible.
"""
from dataclasses import dataclass, field
import math
import numpy as np


@dataclass
class Marker:
    id: int
    x: float
    y: float
    radius: float = 4.0      # the drone can get a fix within this distance


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float = 2.0


@dataclass
class World:
    start: tuple = (60.0, 25.0)          # drone start offset from target (m)
    markers: list = field(default_factory=lambda: [
        Marker(0, 45, 20), Marker(1, 30, 12), Marker(2, 15, 6),
    ])
    obstacles: list = field(default_factory=lambda: [Obstacle(35, 16, 2.5)])
    # Hard patches: (x_center, y_center, radius, peak_difficulty)
    hard_patches: list = field(default_factory=lambda: [
        (50, 22, 8, 0.9),    # a low-texture / blurry zone early in the flight
        (22, 9, 6, 0.7),
    ])


@dataclass
class SimConfig:
    dt: float = 1.0 / 30.0               # 30 Hz
    kp: float = 0.6                       # homing gain
    v_max: float = 2.0                    # m/s speed clamp (handheld-ish, KLT-comfortable)
    arrive_tol: float = 1.0               # m: 'arrived' when |(x,y)| < this
    max_steps: int = 4000
    # noise model for the fake VIO
    vel_noise_base: float = 0.01          # m/s std of per-axis velocity noise (easy terrain)
    vel_noise_hard: float = 0.20          # extra std at difficulty=1
    gyro_bias_walk: float = 0.0008        # rad/s per sqrt-step random walk on heading bias
    fix_noise: float = 0.10               # m std of an AVL marker fix
    seed: int = 0


class KinematicSim:
    def __init__(self, world: World | None = None, cfg: SimConfig | None = None):
        self.world = world or World()
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.reset()

    def reset(self):
        c, w = self.cfg, self.world
        self.t = 0.0
        self.true = np.array(w.start, dtype=float)     # true (x, y) offset from target
        self.est = self.true.copy()                    # VIO estimate (starts perfect)
        self.heading_bias = 0.0                         # accumulating heading-bias proxy
        self.P_pos = 0.01                               # covariance proxy (variance, m^2)
        self.prev_features = 120.0
        self.arrived = False
        self.steps = 0

    # ---- environment helpers -------------------------------------------------
    def difficulty(self, pos):
        """Max difficulty over all hard patches the point falls in. 0 = easy."""
        d = 0.0
        for (hx, hy, hr, peak) in self.world.hard_patches:
            dist = math.hypot(pos[0] - hx, pos[1] - hy)
            if dist < hr:
                d = max(d, peak * (1.0 - dist / hr))
        return d

    def forward_depth(self, pos, heading):
        """Analytic min-distance to an obstacle inside a +/-25 deg forward cone,
        max range 8 m. Feeds Rohan's Week-5 obstacle module. Returns (dist, bearing)."""
        best = (8.0, 0.0)
        for ob in self.world.obstacles:
            dx, dy = ob.x - pos[0], ob.y - pos[1]
            dist = math.hypot(dx, dy) - ob.radius
            bearing = math.atan2(dy, dx) - heading
            bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
            if abs(bearing) < math.radians(25) and 0 < dist < best[0]:
                best = (dist, bearing)
        return best

    def marker_in_view(self):
        """Return a Marker if the drone (true pos) is over one, else None."""
        for m in self.world.markers:
            if math.hypot(self.true[0] - m.x, self.true[1] - m.y) < m.radius:
                return m
        return None

    # ---- the cues the scheduler consumes -------------------------------------
    def cues(self):
        d = self.difficulty(self.true)
        feat = 120.0 * (1.0 - 0.8 * d) + self.rng.normal(0, 3)      # fewer features when hard
        feat = max(0.0, feat)
        feature_loss = max(0.0, self.prev_features - feat) / self.cfg.dt
        self.prev_features = feat
        blur_var = 300.0 * (1.0 - 0.85 * d) + self.rng.normal(0, 8) # low variance = blurry
        return dict(
            sigma_pos=math.sqrt(self.P_pos),
            sigma_head=abs(self.heading_bias) + 0.01,
            feature_loss=feature_loss,
            blur=max(5.0, blur_var),
            imu_bias=abs(self.heading_bias),
            active_features=feat,
        )

    # ---- one simulation step -------------------------------------------------
    def step(self, evasion=np.zeros(2)):
        c = self.cfg
        d = self.difficulty(self.true)

        # homing control off the ESTIMATE (this is the target-centric feedback loop:
        # estimate error -> command error). Merge any evasion vector from Thread 2.
        v_cmd = -c.kp * self.est + evasion
        speed = np.linalg.norm(v_cmd)
        if speed > c.v_max:
            v_cmd = v_cmd / speed * c.v_max

        # TRUE motion: clean integration.
        self.true = self.true + v_cmd * c.dt

        # VIO ESTIMATE: true motion + noise that grows with difficulty + bias drift.
        vel_noise = c.vel_noise_base + c.vel_noise_hard * d
        self.heading_bias += self.rng.normal(0, c.gyro_bias_walk)
        noisy_v = v_cmd + self.rng.normal(0, vel_noise, size=2)
        # rotate by the accumulated heading bias -> realistic curving drift
        cb, sb = math.cos(self.heading_bias), math.sin(self.heading_bias)
        noisy_v = np.array([cb * noisy_v[0] - sb * noisy_v[1],
                            sb * noisy_v[0] + cb * noisy_v[1]])
        self.est = self.est + noisy_v * c.dt

        # covariance proxy grows during prediction (more in hard patches)
        self.P_pos += (vel_noise ** 2) * (1.0 + 3.0 * d)

        self.t += c.dt
        self.steps += 1
        if np.linalg.norm(self.true) < c.arrive_tol:
            self.arrived = True

    def apply_fix(self):
        """Simulate an ArUco absolute fix: snap estimate toward truth, shrink covariance."""
        m = self.marker_in_view()
        if m is None:
            return False
        meas = self.true + self.rng.normal(0, self.cfg.fix_noise, size=2)
        self.est = meas.copy()          # (a real system fuses; snap is fine for the sim)
        self.heading_bias *= 0.1        # marker re-anchors heading too
        self.P_pos = self.cfg.fix_noise ** 2
        return True

    @property
    def error(self):
        """Current true error of the estimate (m)."""
        return float(np.linalg.norm(self.true - self.est))


# ----------------------------- self-test ------------------------------------
if __name__ == "__main__":
    # No correction: the drone homes on a drifting estimate, error should grow.
    sim = KinematicSim(cfg=SimConfig(seed=1))
    errs = []
    while not sim.arrived and sim.steps < sim.cfg.max_steps:
        sim.step()
        errs.append(sim.error)
    print(f"[no-correction] steps={sim.steps} arrived={sim.arrived} "
          f"final_est_error={errs[-1]:.2f} m  peak_error={max(errs):.2f} m")
    assert max(errs) > 0.3, "expected meaningful drift with no correction"

    # With marker fixes whenever one is in view: error should be bounded smaller.
    sim2 = KinematicSim(cfg=SimConfig(seed=1))
    errs2 = []
    while not sim2.arrived and sim2.steps < sim2.cfg.max_steps:
        sim2.step()
        sim2.apply_fix()
        errs2.append(sim2.error)
    print(f"[with-markers]  steps={sim2.steps} arrived={sim2.arrived} "
          f"final_est_error={errs2[-1]:.2f} m  peak_error={max(errs2):.2f} m")
    assert max(errs2) <= max(errs), "markers should not make drift worse"
    print("\nKinematic sim self-tests passed.")
