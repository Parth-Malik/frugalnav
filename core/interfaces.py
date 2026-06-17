import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class SensorInput:
    """
    Raw sensor stream from the hardware or dataset.
    Frames: IMU data is in the Body frame.
    Units: m/s^2 for acceleration, rad/s for angular velocity.
    """
    timestamp: float
    linear_accel: np.ndarray      
    angular_vel: np.ndarray       
    has_image: bool = False
    image_frame: Optional[np.ndarray] = None  # Handle to the image payload

    def __post_init__(self):
        self.linear_accel = np.asarray(self.linear_accel, dtype=np.float64).reshape(3)
        self.angular_vel = np.asarray(self.angular_vel, dtype=np.float64).reshape(3)


@dataclass
class VioOutput:
    """
    The relative state increment and glass-box internals from the VIO frontend.
    Feeds the uncertainty scheduler and state fusion.
    Composition: pose_new = pose_prev @ delta_pose
    """
    timestamp: float
    delta_pose: np.ndarray        # shape (4, 4) SE(3), relative motion since last tick
    pos_std_m: float              # VIO's internal confidence metric (standard deviation in meters)
    active_features: int
    imu_bias_norm: float
    blur: float = 0.0             # Feeds the Uncertainty (U) metric

    def __post_init__(self):
        self.delta_pose = np.asarray(self.delta_pose, dtype=np.float64).reshape(4, 4)


@dataclass
class LandmarkFix:
    """
    Absolute position fix from the ArUco detector (AVL).
    Frames: pose_world is in the World (Landmark map) frame.
    """
    valid: bool                   # True if a marker was successfully detected this tick
    timestamp: float
    marker_id: int
    pose_world: np.ndarray        # shape (4, 4) SE(3)
    pos_std_m: float              # Measurement standard deviation in meters (for Kalman fusion weighting)

    def __post_init__(self):
        if self.valid:
            self.pose_world = np.asarray(self.pose_world, dtype=np.float64).reshape(4, 4)


@dataclass
class PoseEstimate:
    """
    The single fused belief of the UAV's state. 
    Written by state fusion, read by the target-centric controller.
    Frames: World frame.
    """
    timestamp: float
    pose_world: np.ndarray        # shape (4, 4) SE(3)
    velocity_world: np.ndarray    # shape (3,)
    pos_std_m: float              # Current fused estimate uncertainty (drives the scheduler)

    def __post_init__(self):
        self.pose_world = np.asarray(self.pose_world, dtype=np.float64).reshape(4, 4)
        self.velocity_world = np.asarray(self.velocity_world, dtype=np.float64).reshape(3)


@dataclass
class VelocityCmd:
    """
    Target-centric output command to drive the UAV.
    Frames: Body frame. The controller must extract the rotation matrix from the 
            fused PoseEstimate to rotate the World-frame error into this Body frame.
    Units: m/s for linear velocity, rad/s for yaw rate.
    """
    timestamp: float
    linear_vel: np.ndarray        # shape (3,)
    yaw_rate: float

    def __post_init__(self):
        self.linear_vel = np.asarray(self.linear_vel, dtype=np.float64).reshape(3)
