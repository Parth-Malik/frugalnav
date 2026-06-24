import numpy as np
from typing import Protocol
from core.interfaces import SensorInput, VioOutput
from core.se3 import inv_se3

class VioAdapter(Protocol):
    def update(self, sensor_input: SensorInput) -> VioOutput: ...
    def reset(self) -> None: ...

class TrajectoryReplayAdapter:
    def __init__(self, timestamps, poses):
        self.timestamps = np.array(timestamps)
        self.poses = poses
        self.reset()

    def reset(self):
        self.idx = 0

    def update(self, sensor_input: SensorInput) -> VioOutput:
        if self.idx >= len(self.timestamps):
            return None
        
        target_ts = sensor_input.timestamp
        closest_idx = np.argmin(np.abs(self.timestamps - target_ts))
        
        if closest_idx == 0:
            delta = np.eye(4)
        else:
            delta = inv_se3(self.poses[closest_idx - 1]) @ self.poses[closest_idx]
            
        self.idx = closest_idx + 1
        
        return VioOutput(
            timestamp=target_ts,
            delta_pose=delta,
            pos_std_m=0.01 * (closest_idx ** 0.5),
            active_features=30,
            imu_bias_norm=0.1
        )
