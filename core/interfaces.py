"""Core data contracts for FrugalNav. Fixed-shape structs, ready for the C++/Eigen port."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SensorInput:
    """Raw sensor tick. IMU in body frame: accel m/s^2, gyro rad/s."""

    timestamp: float
    linear_accel: np.ndarray
    angular_vel: np.ndarray
    has_image: bool = False
    image_frame: Optional[np.ndarray] = None

    def __post_init__(self):
        self.linear_accel = np.asarray(self.linear_accel, dtype=np.float64).reshape(3)
        self.angular_vel = np.asarray(self.angular_vel, dtype=np.float64).reshape(3)


@dataclass
class VioOutput:
    """Relative motion + glass-box internals from the VIO. Compose as: pose_new = pose_prev @ delta_pose."""

    timestamp: float
    delta_pose: np.ndarray
    pos_std_m: float
    active_features: int
    imu_bias_norm: float
    blur: float = 0.0

    def __post_init__(self):
        self.delta_pose = np.asarray(self.delta_pose, dtype=np.float64).reshape(4, 4)


@dataclass
class LandmarkFix:
    """Absolute fix from an ArUco marker, pose in the world (landmark map) frame."""

    valid: bool
    timestamp: float
    marker_id: int
    pose_world: Optional[np.ndarray]
    pos_std_m: float

    def __post_init__(self):
        if self.valid and self.pose_world is not None:
            self.pose_world = np.asarray(self.pose_world, dtype=np.float64).reshape(4, 4)

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
        self.pose_world = np.asarray(self.pose_world, dtype=np.float64).reshape(4, 4)
        self.velocity_world = np.asarray(self.velocity_world, dtype=np.float64).reshape(3)


@dataclass
class VelocityCmd:
    """Velocity command in the body frame. Controller rotates the world-frame error via the estimate's rotation."""

    timestamp: float
    linear_vel: np.ndarray
    yaw_rate: float

    def __post_init__(self):
        self.linear_vel = np.asarray(self.linear_vel, dtype=np.float64).reshape(3)
