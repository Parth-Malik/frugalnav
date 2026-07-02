"""Drift-injection VIO fallbacks (harness). Both emit VioOutput with `delta_pose`."""

import numpy as np

from core.interfaces import VioOutput, DTYPE
from core.se3 import make_se3, relative_se3


class DriftInjectionAdapter:
    """Week 2: pre-loaded GT arrays, consumed by the pipeline via update(SensorInput)."""

    def __init__(self, gt_timestamps, gt_poses, drift_rate_m_per_s=0.05,
                 drift_dir=(1.0, 1.0, 1.0), active_features=35, imu_bias_norm=0.05):
        if len(gt_timestamps) == 0:
            raise ValueError("DriftInjectionAdapter needs a non-empty ground-truth trajectory.")
        self.gt_timestamps = np.asarray(gt_timestamps, dtype=DTYPE)
        self.gt_poses = gt_poses
        self.drift_rate = float(drift_rate_m_per_s)
        d = np.asarray(drift_dir, dtype=DTYPE)
        self.drift_dir = d / (np.linalg.norm(d) + 1e-12)
        self.active_features = int(active_features)
        self.imu_bias_norm = float(imu_bias_norm)
        self.reset()

    def reset(self):
        self.last_idx = None
        self.last_ts = None
        self.accumulated_drift_m = 0.0
        self._j = 0

    def update(self, sensor_input):
        t = sensor_input.timestamp
        while (self._j + 1 < len(self.gt_timestamps)
               and abs(self.gt_timestamps[self._j + 1] - t)
                   <= abs(self.gt_timestamps[self._j] - t)):
            self._j += 1
        idx = self._j
        if self.last_idx is None:
            self.last_idx = idx
        delta = relative_se3(self.gt_poses[self.last_idx], self.gt_poses[idx])
        self.last_idx = idx
        dt = (t - self.last_ts) if (self.last_ts is not None and t > self.last_ts) else 0.0
        self.last_ts = t
        step = self.drift_dir * self.drift_rate * dt
        self.accumulated_drift_m += float(np.linalg.norm(step))
        modified_delta = delta @ make_se3(np.eye(3, dtype=DTYPE), step)
        vel_body = (delta[:3, 3] / dt) if dt > 0 else np.zeros(3)
        return VioOutput(timestamp=t, delta_pose=modified_delta,
                         pos_std_m=self.accumulated_drift_m,
                         active_features=self.active_features,
                         imu_bias_norm=self.imu_bias_norm, velocity_body=vel_body)


class PoseDriftAdapter:
    """Week 3/4: driven per-tick by (timestamp, gt_pose_world); reports clean velocity.

    pos_std_m is the per-tick injected drift magnitude, so the fused uncertainty that
    accumulates from it tracks the real drift the scheduler must respond to.
    active_features is reported (default healthy); an optional feature_dip window lets
    a demo exercise the observability floor.
    """

    def __init__(self, pos_bias=(0.0, 0.0, 0.0), random_walk_std=0.01, seed=0,
                 active_features=150, feature_dip=None):
        self.pos_bias = np.asarray(pos_bias, dtype=DTYPE)
        self.random_walk_std = float(random_walk_std)
        self._rng = np.random.default_rng(seed)
        self.active_features = int(active_features)
        self.feature_dip = feature_dip
        self.reset()

    def reset(self):
        self.last_gt_pose = None
        self.last_timestamp = None

    def _features_at(self, t):
        if self.feature_dip is not None:
            t0, t1, dipped = self.feature_dip
            if t0 <= t <= t1:
                return int(dipped)
        return self.active_features

    def update(self, timestamp, gt_pose_world):
        gt_pose_world = np.asarray(gt_pose_world, dtype=DTYPE)
        if self.last_gt_pose is None:
            self.last_gt_pose = gt_pose_world.copy()
            self.last_timestamp = timestamp
            return VioOutput(timestamp, np.eye(4, dtype=DTYPE), 0.0,
                             active_features=self._features_at(timestamp),
                             velocity_body=np.zeros(3))
        dt = max(timestamp - self.last_timestamp, 1e-6)
        delta_gt = relative_se3(self.last_gt_pose, gt_pose_world)
        noise = self._rng.normal(0, self.random_walk_std, 3)
        injected = (self.pos_bias * dt) + noise
        delta_noisy = delta_gt.copy()
        delta_noisy[:3, 3] += injected
        vel_body = delta_gt[:3, 3] / dt
        self.last_gt_pose = gt_pose_world.copy()
        self.last_timestamp = timestamp
        return VioOutput(
            timestamp=timestamp,
            delta_pose=delta_noisy,
            pos_std_m=float(np.linalg.norm(injected)),
            active_features=self._features_at(timestamp),
            velocity_body=vel_body,
        )
