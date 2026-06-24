"""Core data contracts for FrugalNav. Fixed-shape structs, ready for the C++/Eigen port.

DTYPE is the single precision knob: np.float64 for laptop prototyping,
np.float32 for the GAP9 RISC-V precision sweep (feasibility study).
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

DTYPE = np.float64

@dataclass
class SensorInput:
    """Raw sensor tick. IMU in body frame: accel m/s^2, gyro rad/s."""
    timestamp: float
    linear_accel: np.ndarray
    angular_vel: np.ndarray
    has_image: bool = False
    image_frame: Optional[str] = None

    def __post_init__(self):
        # Force a copy to prevent aliasing caller memory
        self.linear_accel = np.array(self.linear_accel, dtype=DTYPE).reshape(3)
        self.angular_vel = np.array(self.angular_vel, dtype=DTYPE).reshape(3)

@dataclass
class VioOutput:
    """Relative motion + glass-box internals from the VIO."""
    timestamp: float
    delta_pose: np.ndarray
    pos_std_m: float
    active_features: int
    imu_bias_norm: float
    blur: float = 0.0

    def __post_init__(self):
        self.delta_pose = np.array(self.delta_pose, dtype=DTYPE).reshape(4, 4)

@dataclass
class LandmarkFix:
    """Absolute fix from an ArUco marker, pose in the world frame."""
    valid: bool
    timestamp: float
    marker_id: int
    pose_world: Optional[np.ndarray]
    pos_std_m: float

    def __post_init__(self):
        if self.valid and self.pose_world is not None:
            self.pose_world = np.array(self.pose_world, dtype=DTYPE).reshape(4, 4)

    @classmethod
    def invalid(cls, timestamp: float) -> "LandmarkFix":
        return cls(valid=False, timestamp=timestamp, marker_id=-1, pose_world=None, pos_std_m=0.0)

@dataclass
class PoseEstimate:
    """The single fused belief of the UAV state, in the world frame."""
    timestamp: float
    pose_world: np.ndarray
    velocity_world: np.ndarray
    pos_std_m: float

    def __post_init__(self):
        self.pose_world = np.array(self.pose_world, dtype=DTYPE).reshape(4, 4)
        self.velocity_world = np.array(self.velocity_world, dtype=DTYPE).reshape(3)

@dataclass
class VelocityCmd:
    """Velocity command in the body frame."""
    timestamp: float
    linear_vel: np.ndarray
    yaw_rate: float

    def __post_init__(self):
        self.linear_vel = np.array(self.linear_vel, dtype=DTYPE).reshape(3)
