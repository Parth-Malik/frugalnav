"""The portable hot loop: SensorInput -> VioOutput -> integrated PoseEstimate.

core/ rule: imports only numpy + core. No cv2, no dataset paths, no plotting.
"""

import numpy as np
from typing import Iterable, List, Optional

from core.interfaces import SensorInput, PoseEstimate, DTYPE
from core.se3 import project_to_SO3


class FrugalPipeline:
    def __init__(self, vio_adapter):
        self.vio_adapter = vio_adapter
        self.correction_count = 0
        self.current_pose = np.eye(4, dtype=DTYPE)
        self.current_velocity = np.zeros(3, dtype=DTYPE)
        self._last_pos = None
        self._last_ts = None
        self._steps_since_projection = 0
        self._skip_velocity_derivative = False

    def seed_initial_pose(self, pose: np.ndarray):
        self.current_pose = np.array(pose, dtype=DTYPE).reshape(4, 4)

    def step(self, sensor_input: SensorInput) -> PoseEstimate:
        vio_out = self.vio_adapter.update(sensor_input)

        if vio_out is not None:
            if np.all(np.isfinite(vio_out.delta_pose)):
                self.current_pose = self.current_pose @ vio_out.delta_pose
                self._steps_since_projection += 1
                if self._steps_since_projection >= 100:
                    self.current_pose[:3, :3] = project_to_SO3(self.current_pose[:3, :3])
                    self._steps_since_projection = 0

        pos = self.current_pose[:3, 3]
        if self._last_pos is not None and self._last_ts is not None:
            dt = sensor_input.timestamp - self._last_ts
            if dt > 0 and not self._skip_velocity_derivative:
                self.current_velocity = (pos - self._last_pos) / dt

        self._last_pos = pos.copy()
        self._last_ts = sensor_input.timestamp
        self._skip_velocity_derivative = False

        # No .copy() here: PoseEstimate.__post_init__ owns the single copy.
        return PoseEstimate(
            timestamp=sensor_input.timestamp,
            pose_world=self.current_pose,
            velocity_world=self.current_velocity,
            pos_std_m=vio_out.pos_std_m if vio_out is not None else 0.0,
        )

    def replay(self, stream: Iterable[SensorInput],
               initial_pose: Optional[np.ndarray] = None) -> List[PoseEstimate]:
        """Harness/eval convenience: accumulates the whole trajectory.
        The embedded loop uses step() one tick at a time, not this."""
        self.vio_adapter.reset()
        if initial_pose is not None:
            self.seed_initial_pose(initial_pose)
        else:
            self.current_pose = np.eye(4, dtype=DTYPE)
        self.current_velocity = np.zeros(3, dtype=DTYPE)
        self._last_pos = None
        self._last_ts = None
        self.correction_count = 0
        self._steps_since_projection = 0

        trajectory = []
        for sensor_input in stream:
            trajectory.append(self.step(sensor_input))
        return trajectory
