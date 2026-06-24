"""The portable hot loop: SensorInput -> VioOutput -> integrated PoseEstimate."""

import numpy as np
from typing import Iterable, List, Optional
from core.interfaces import SensorInput, PoseEstimate, DTYPE
from core.se3 import project_to_SO3

class FrugalPipeline:
    def __init__(self, vio_adapter):
        self.vio_adapter = vio_adapter
        self.correction_count = 0
        self.current_pose = np.eye(4, dtype=DTYPE)
        self._steps_since_projection = 0

    def seed_initial_pose(self, pose: np.ndarray):
        """Allows embedded runners using step() directly to align the initial state."""
        self.current_pose = np.array(pose, dtype=DTYPE).reshape(4, 4)

    def step(self, sensor_input: SensorInput) -> PoseEstimate:
        vio_out = self.vio_adapter.update(sensor_input)
        
        # If VIO drops the frame, maintain current pose and assume zero velocity
        if vio_out is None:
             return PoseEstimate(sensor_input.timestamp, self.current_pose, 
                                 np.zeros(3), np.zeros(3), 0.0)

        if np.all(np.isfinite(vio_out.delta_pose)):
            self.current_pose = self.current_pose @ vio_out.delta_pose
            self._steps_since_projection += 1
            
            # Periodic SO(3) Projection
            if self._steps_since_projection >= 100:
                self.current_pose[:3, :3] = project_to_SO3(self.current_pose[:3, :3])
                self._steps_since_projection = 0

        # State Fusion (Week 3) will override self.current_pose here if corrected_this_tick is True

        return PoseEstimate(
            timestamp=sensor_input.timestamp,
            pose_world=self.current_pose, 
            velocity_world=vio_out.velocity_world,         # Natively passed
            angular_vel_world=vio_out.angular_vel_world,   # Natively passed
            pos_std_m=vio_out.pos_std_m,
        )

    def replay(self, stream: Iterable[SensorInput],
               initial_pose: Optional[np.ndarray] = None) -> List[PoseEstimate]:
        """HARNESS ONLY: Accumulates trajectory into an unbounded list."""
        self.vio_adapter.reset()
        if initial_pose is not None:
            self.seed_initial_pose(initial_pose)
        else:
            self.current_pose = np.eye(4, dtype=DTYPE)
            
        self.correction_count = 0
        self._steps_since_projection = 0

        trajectory = []
        for sensor_input in stream:
            trajectory.append(self.step(sensor_input))
        return trajectory
