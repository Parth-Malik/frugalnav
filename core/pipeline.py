import numpy as np
from typing import Iterable, List
from core.interfaces import SensorInput, PoseEstimate

class FrugalPipeline:
    def __init__(self, vio_adapter):
        self.vio_adapter = vio_adapter
        self.correction_count = 0
        self.current_pose = np.eye(4)
        self.current_velocity = np.zeros(3)

    def step(self, sensor_input: SensorInput) -> PoseEstimate:
        vio_out = self.vio_adapter.update(sensor_input)
        if vio_out is not None:
            self.current_pose = self.current_pose @ vio_out.delta_pose
        return PoseEstimate(
            timestamp=sensor_input.timestamp,
            pose_world=self.current_pose.copy(),
            velocity_world=self.current_velocity.copy(),
            pos_std_m=vio_out.pos_std_m if vio_out else 0.0
        )

    def replay(self, stream: Iterable[SensorInput]) -> List[PoseEstimate]:
        self.vio_adapter.reset()
        self.current_pose = np.eye(4)
        self.correction_count = 0
        trajectory = []
        for sensor_input in stream:
            estimate = self.step(sensor_input)
            trajectory.append(estimate)
        return trajectory
