"""
harness/drift_scaffold.py
-------------------------
The "fallback VIO" the plan calls for -- and Rohan's Week-2 deliverable.

It takes a REAL EuRoC trajectory (ground-truth position + velocity + IMU) and produces
a DRIFTING estimate from it, the way a real VIO would: integrate a corrupted velocity
(noise + a slowly walking bias), so the estimate diverges from ground truth over time.
Because we start from real data, the drift plot is honest, and we can CALIBRATE the
kinematic sim's noise to match the real drift rate (this is what answers "isn't your
drift faked?").

It implements VioSource, so the scheduler runs on it exactly as it runs on the sim.

Glass-box signals (what the scheduler consumes), all derived from REAL data where possible:
  - sigma_pos : covariance proxy, grows since the last fix          (modeled)
  - imu_bias  : magnitude of the real ground-truth gyro bias        (REAL, from EuRoC)
  - blur      : Laplacian-variance proxy from real angular velocity (REAL IMU -> proxy)
  - feature_loss : proxy from real linear acceleration spikes       (REAL IMU -> proxy)
  - active_features : proxy that drops during fast motion
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.vio_adapter import VioSource, VioSignals


class EurocDriftSource(VioSource):
    def __init__(self, gt, imu, seed=0, vel_noise=0.04, bias_walk=0.0015,
                 fix_every_m=8.0):
        """gt, imu: dicts from EurocReader. fix_every_m: a virtual landmark every N metres
        of TRUE travel (stand-in for ArUco markers placed along the route)."""
        self.rng = np.random.default_rng(seed)
        self.vel_noise = vel_noise
        self.bias_walk = bias_walk
        self.fix_every_m = fix_every_m

        # target-centric: make the LAST ground-truth point the target origin (0,0)
        self.gt_xy = np.column_stack([gt["x"], gt["y"]])
        self.gt_xy = self.gt_xy - self.gt_xy[-1]          # shift so target B = (0,0)
        self.t = gt["t"]
        self.bias_mag = (np.abs(gt.get("bgx", np.zeros_like(self.t)))
                         + np.abs(gt.get("bgy", np.zeros_like(self.t)))
                         + np.abs(gt.get("bgz", np.zeros_like(self.t))))

        # interpolate IMU onto the GT timestamps for the blur/feature proxies
        self.gyro_mag = np.interp(self.t, imu["t"],
                                  np.sqrt(imu["wx"]**2 + imu["wy"]**2 + imu["wz"]**2))
        acc = np.sqrt(imu["ax"]**2 + imu["ay"]**2 + (imu["az"]-9.81)**2)
        self.acc_mag = np.interp(self.t, imu["t"], acc)

        self.reset()

    def reset(self):
        self.k = 0
        self.est = self.gt_xy[0].copy()
        self.heading_bias = 0.0
        self.P_pos = 0.01
        self.dist_since_fix = 0.0
        self.prev_features = 150.0

    def update(self):
        if self.k >= len(self.t) - 1:
            return None
        dt = max(1e-3, self.t[self.k + 1] - self.t[self.k])

        # true velocity from GT deltas
        true_v = (self.gt_xy[self.k + 1] - self.gt_xy[self.k]) / dt
        self.dist_since_fix += np.linalg.norm(true_v) * dt

        # corrupt it: walking heading bias + gaussian noise -> realistic curving drift
        self.heading_bias += self.rng.normal(0, self.bias_walk)
        c, s = np.cos(self.heading_bias), np.sin(self.heading_bias)
        nv = true_v + self.rng.normal(0, self.vel_noise, size=2)
        nv = np.array([c * nv[0] - s * nv[1], s * nv[0] + c * nv[1]])
        self.est = self.est + nv * dt
        self.P_pos += (self.vel_noise ** 2) * (1.0 + 2.0 * min(self.gyro_mag[self.k], 1.0))

        # glass-box signals (real where possible)
        blur_var = max(5.0, 300.0 - 250.0 * min(self.gyro_mag[self.k] / 1.5, 1.0))
        feat = max(0.0, 150.0 - 90.0 * min(self.acc_mag[self.k] / 5.0, 1.0))
        feature_loss = max(0.0, self.prev_features - feat) / dt
        self.prev_features = feat

        cues = dict(
            sigma_pos=np.sqrt(self.P_pos),
            sigma_head=abs(self.heading_bias) + 0.01,
            feature_loss=feature_loss,
            blur=blur_var,
            imu_bias=float(self.bias_mag[self.k]),
            active_features=feat,
        )
        sig = VioSignals(t=float(self.t[self.k]), est=tuple(self.est),
                         cues=cues, gt=tuple(self.gt_xy[self.k]))
        self.k += 1
        return sig

    def apply_fix(self):
        """A virtual landmark is 'in view' once we've travelled fix_every_m of true
        distance. Snapping the estimate to GT models an ArUco absolute fix."""
        if self.dist_since_fix < self.fix_every_m:
            return False
        self.est = self.gt_xy[min(self.k, len(self.gt_xy) - 1)].copy() \
            + self.rng.normal(0, 0.05, size=2)
        self.heading_bias *= 0.1
        self.P_pos = 0.05 ** 2
        self.dist_since_fix = 0.0
        return True

    def arrived(self):
        return self.k >= len(self.t) - 1


def calibrate_sim_noise(gt, imu, seed=0):
    """Run the drift scaffold with NO corrections over the whole real sequence and report
    the drift in the STANDARD VIO metric: drift as a % of distance travelled. Real VIO
    drift is dominated by heading/bias error that grows with distance, so there is no
    honest single-parameter closed form -- instead you tune the kinematic_sim's bias_walk
    until ITS drift-% matches this number. That keeps the sim grounded in real data."""
    src = EurocDriftSource(gt, imu, seed=seed)
    errs, ts, path = [], [], 0.0
    prev = None
    while True:
        sig = src.update()
        if sig is None:
            break
        errs.append(sig.error()); ts.append(sig.t)
        if prev is not None:
            path += ((sig.gt[0]-prev[0])**2 + (sig.gt[1]-prev[1])**2) ** 0.5
        prev = sig.gt
    errs = np.array(errs); T = ts[-1] - ts[0]
    final_drift = float(errs[-1])
    drift_pct = 100.0 * final_drift / max(path, 1e-6)
    return dict(sequence_duration_s=round(float(T), 1),
                path_length_m=round(float(path), 1),
                final_uncorrected_drift_m=round(final_drift, 2),
                drift_rate_m_per_s=round(float(final_drift / T), 3),
                drift_pct_of_distance=round(float(drift_pct), 2),
                tuning_hint="adjust kinematic_sim bias_walk until its drift_pct matches this")


if __name__ == "__main__":
    # self-test on a synthetic EuRoC trajectory (curved, 30 s)
    n = 2000
    t = np.linspace(0, 30, n)
    x = 40 * np.cos(np.linspace(0, 1.2, n)); y = 40 * np.sin(np.linspace(0, 1.2, n))
    vx = np.gradient(x, t); vy = np.gradient(y, t)
    gt = dict(t=t, x=x, y=y, z=np.zeros(n), vx=vx, vy=vy, vz=np.zeros(n),
              bgx=np.full(n, 0.003), bgy=np.full(n, 0.001), bgz=np.full(n, 0.002))
    imu = dict(t=t, wx=0.2*np.sin(t), wy=np.zeros(n), wz=0.3*np.cos(t),
               ax=np.zeros(n), ay=np.zeros(n), az=np.full(n, 9.81))

    src = EurocDriftSource(gt, imu, seed=1)
    peak = 0.0
    while True:
        sig = src.update()
        if sig is None:
            break
        peak = max(peak, sig.error())
    print(f"uncorrected peak drift over sequence: {peak:.2f} m")
    assert peak > 0.5
    print("calibration:", calibrate_sim_noise(gt, imu, seed=1))
    print("\ndrift_scaffold self-test passed.")
