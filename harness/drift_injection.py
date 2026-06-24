"""Drift-injection VIO fallback (harness — needs ground truth, so not portable).

Replays real ground truth as relative increments and injects a controllable,
constant-rate drift. With drift_rate=0 the integrated trajectory reconstructs
ground truth exactly; with drift_rate>0 it diverges at a known rate.
"""

import numpy as np

from core.interfaces import VioOutput, DTYPE
from core.se3 import make_se3, relative_se3


class DriftInjectionAdapter:
    def __init__(self, gt_timestamps, gt_poses, drift_rate_m_per_s=0.05,
                 drift_dir=(1.0, 1.0, 1.0), active_features=35, imu_bias_norm=0.05):
        if len(gt_timestamps) == 0:
            raise ValueError("DriftInjectionAdapter needs a non-empty ground-truth trajectory.")
        self.gt_timestamps = np.asarray(gt_timestamps, dtype=DTYPE)
        self.gt_poses = gt_poses
        self.drift_rate = float(drift_rate_m_per_s)
        d = np.asarray(drift_dir, dtype=DTYPE)
        self.drift_dir = d / (np.linalg.norm(d) + 1e-12)   # unit dir -> honest m/s
        self.active_features = int(active_features)
        self.imu_bias_norm = float(imu_bias_norm)
        self.reset()

    def reset(self):
        self.last_idx = None
        self.last_ts = None
        self.accumulated_drift_m = 0.0
        self._j = 0                       # monotonic forward pointer (O(N) total)

    def update(self, sensor_input):
        t = sensor_input.timestamp

        while (self._j + 1 < len(self.gt_timestamps)
               and abs(self.gt_timestamps[self._j + 1] - t)
                   <= abs(self.gt_timestamps[self._j] - t)):
            self._j += 1
        idx = self._j

        if self.last_idx is None:
            self.last_idx = idx

        delta = relative_se3(self.gt_poses[self.last_idx], self.gt_poses[idx])  # telescopes
        self.last_idx = idx

        dt = (t - self.last_ts) if (self.last_ts is not None and t > self.last_ts) else 0.0
        self.last_ts = t

        step = self.drift_dir * self.drift_rate * dt
        self.accumulated_drift_m += float(np.linalg.norm(step))
        modified_delta = delta @ make_se3(np.eye(3, dtype=DTYPE), step)

        return VioOutput(
            timestamp=t,
            delta_pose=modified_delta,
            pos_std_m=self.accumulated_drift_m,
            active_features=self.active_features,
            imu_bias_norm=self.imu_bias_norm,
        )
