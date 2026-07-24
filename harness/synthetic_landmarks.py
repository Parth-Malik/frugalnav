"""Synthetic landmark source (harness). Stand-in for the ArUco corrector."""

import numpy as np

from core.interfaces import LandmarkFix, DTYPE


class SyntheticLandmarkSource:
    def __init__(self, gt_timestamps, gt_poses, period_s=5.0, pos_std_m=0.05,
                 noise_m=0.03, seed=0):
        self.gt_timestamps = np.asarray(gt_timestamps, dtype=DTYPE)
        self.gt_poses = np.asarray(gt_poses, dtype=DTYPE)
        self.period_s = float(period_s)
        self.pos_std_m = float(pos_std_m)
        self.noise_m = float(noise_m)
        self._rng = np.random.default_rng(seed)
        self.last_fix_time = -1e9

    def try_fix(self, timestamp: float) -> LandmarkFix:
        if timestamp - self.last_fix_time < self.period_s:
            return LandmarkFix.invalid(timestamp)
        idx = int(np.argmin(np.abs(self.gt_timestamps - timestamp)))
        gt_pose = self.gt_poses[idx].copy()
        gt_pose[:3, 3] += self._rng.normal(0, self.noise_m, 3)
        self.last_fix_time = timestamp
        return LandmarkFix(valid=True, timestamp=timestamp, marker_id=1,
                           pose_world=gt_pose, pos_std_m=self.pos_std_m)
