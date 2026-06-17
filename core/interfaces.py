import numpy as np
from dataclasses import dataclass

@dataclass
class SensorInput:
    """
    Represents the continuous stream of data feeding the VIO front-end 
    and the uncertainty scheduler.
    """
    timestamp: float
    # IMU Data: Fixed 3D vectors
    linear_accel: np.ndarray  # shape (3,)
    angular_vel: np.ndarray   # shape (3,)
    # Glass-box VIO signals for the scheduler
    position_cov: float = 0.0
    active_features: int = 0
    imu_bias_norm: float = 0.0

@dataclass
class LandmarkFix:
    """
    Represents an absolute position fix from the ArUco detector (AVL).
    Merged into the state fusion module.
    """
    timestamp: float
    marker_id: int
    # 4x4 Homogeneous Transformation Matrix representing pose in the world frame
    pose_world: np.ndarray  # shape (4, 4)
    confidence: float       # e.g., based on marker scale/blur

@dataclass
class VelocityCmd:
    """
    Target-centric output command to drive the UAV.
    """
    timestamp: float
    v_x: float  # Lateral X command
    v_y: float  # Lateral Y command
    v_z: float  # Vertical command
    yaw_rate: float
