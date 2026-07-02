"""Scalar-gain landmark fusion (portable core/)."""

import numpy as np
from typing import Optional

from core.interfaces import VioOutput, LandmarkFix, PoseEstimate, DTYPE
from core.se3 import project_to_SO3


class StateFusion:
    def __init__(self):
        self.reset()

    def reset(self, initial_pose: Optional[np.ndarray] = None) -> None:
        if initial_pose is None:
            self._current_pose = np.eye(4, dtype=DTYPE)
        else:
            self._current_pose = np.array(initial_pose, dtype=DTYPE).reshape(4, 4)
            self._current_pose[:3, :3] = project_to_SO3(self._current_pose[:3, :3])
        self._current_vel_world = np.zeros(3, dtype=DTYPE)
        self._sigma = 0.05
        self._last_timestamp = 0.0
        self.correction_count = 0

    def predict(self, vio_out: VioOutput) -> None:
        self._last_timestamp = vio_out.timestamp
        self._current_pose = self._current_pose @ vio_out.delta_pose
        self._current_pose[:3, :3] = project_to_SO3(self._current_pose[:3, :3])
        R_world_body = self._current_pose[:3, :3]
        self._current_vel_world = R_world_body @ vio_out.velocity_body
        self._sigma = float(np.sqrt(self._sigma ** 2 + vio_out.pos_std_m ** 2))

    def correct(self, fix: LandmarkFix) -> None:
        if not fix.valid or fix.pose_world is None:
            return
        sigma_sq = self._sigma ** 2
        fix_std_sq = max(fix.pos_std_m ** 2, 1e-9)
        K = sigma_sq / (sigma_sq + fix_std_sq)
        pos_pred = self._current_pose[:3, 3]
        pos_fix = fix.pose_world[:3, 3]
        self._current_pose[:3, 3] = pos_pred + K * (pos_fix - pos_pred)
        self._sigma = float(np.sqrt((1.0 - K) * sigma_sq))
        self.correction_count += 1

    @property
    def estimate(self) -> PoseEstimate:
        return PoseEstimate(timestamp=self._last_timestamp,
                            pose_world=self._current_pose.copy(),
                            velocity_world=self._current_vel_world.copy(),
                            pos_std_m=self._sigma)
