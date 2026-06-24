"""The portable hot loop: SensorInput -> VioOutput -> integrated PoseEstimate."""

import numpy as np
from typing import Iterable, List, Optional
from core.interfaces import SensorInput, PoseEstimate, DTYPE

class FrugalPipeline:
    def __init__(self, vio_adapter):
        self.vio_adapter = vio_adapter
        self.correction_count = 0
        self.current_pose = np.eye(4, dtype=DTYPE)
        self.current_velocity = np.zeros(3, dtype=DTYPE)
        self._last_pos = None
        self._last_ts = None

    def step(self, sensor_input: SensorInput) -> PoseEstimate:
        vio_out = self.vio_adapter.update(sensor_input)
        if vio_out is not None:
            self.current_pose = self.current_pose @ vio_out.delta_pose

        pos = self.current_pose[:3, 3]
        if self._last_pos is not None and self._last_ts is not None:
            dt = sensor_input.timestamp - self._last_ts
            if dt > 0:
                self.current_velocity = (pos - self._last_pos) / dt
                
        self._last_pos = pos.copy()
        self._last_ts = sensor_input.timestamp

        return PoseEstimate(
            timestamp=sensor_input.timestamp,
            pose_world=self.current_pose.copy(),
            velocity_world=self.current_velocity.copy(),
            pos_std_m=vio_out.pos_std_m if vio_out is not None else 0.0,
        )

    def replay(self, stream: Iterable[SensorInput],
               initial_pose: Optional[np.ndarray] = None) -> List[PoseEstimate]:
        self.vio_adapter.reset()
        self.current_pose = (np.eye(4, dtype=DTYPE) if initial_pose is None
                             else np.array(initial_pose, dtype=DTYPE).reshape(4, 4))
        self.current_velocity = np.zeros(3, dtype=DTYPE)
        self._last_pos = None
        self._last_ts = None
        self.correction_count = 0

        trajectory = []
        for sensor_input in stream:
            trajectory.append(self.step(sensor_input))
        return trajectory
