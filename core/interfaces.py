import numpy as np
from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class SensorInput:
    """
    Raw sensor stream from the hardware or dataset.
    Frames: IMU data is in the Body frame.
    Units: m/s^2 for acceleration, rad/s for angular velocity.
    """
    timestamp: float
    linear_accel: np.ndarray      # shape (3,)
    angular_vel: np.ndarray       # shape (3,)
    has_image: bool = False
    image_frame: Optional[Any] = None  # Handle to the image payload, no by-value copies

@dataclass
class VioOutput:
    """
    The relative state increment and glass-box internals from the VIO frontend.
    Feeds the uncertainty scheduler and state fusion.
    """
    timestamp: float
    delta_pose: np.ndarray        # shape (4, 4) SE(3), relative motion since last tick
    position_cov: float           # VIO's internal confidence metric
    active_features: int
    imu_bias_norm: float
    blur: float = 0.0             # Feeds the Uncertainty (U) metric

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
    position_cov: float           # Current fused estimate uncertainty (drives the scheduler)

@dataclass
class VelocityCmd:
    """
    Target-centric output command to drive the UAV.
    Frames: Body frame.
    Units: m/s for linear velocity, rad/s for yaw rate.
    """
    timestamp: float
    linear_vel: np.ndarray        # shape (3,), unified 3-vector convention
    yaw_rate: float
