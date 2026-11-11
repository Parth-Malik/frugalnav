"""
Data structures that cross module boundaries.

The plan's design rule (section 6): "Sensor structs in, command struct out.
Fixed-size buffers, no dynamic allocation in the hot loop." In Python these are
small dataclasses; the C++ port replaces each with a POD struct of the same
shape. Keeping the shapes tiny and fixed is what makes the RISC-V memory budget
(section 2, constraint 3) hold.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MarkerSighting:
    """One ArUco marker observed in one camera frame.

    The pose is the marker expressed in the CAMERA frame (T_CM) -- the raw
    output of ArUco pose estimation. rvec/tvec follow OpenCV's convention, in
    meters. This is what the detector emits and the corrector consumes.
    """
    marker_id: int
    rvec: np.ndarray                       # (3,) axis-angle, marker -> camera
    tvec: np.ndarray                       # (3,) marker origin in camera frame [m]
    timestamp: float = 0.0
    reproj_error_px: float = 0.0           # pose quality; higher = trust less


@dataclass
class LandmarkFix:
    """An absolute world-frame position fix from the landmark corrector.

    This is the "AVL fix" of the architecture: the scheduler decides whether to
    consume it, and state fusion merges it into the VIO estimate.
    """
    xy: np.ndarray                         # (2,) drone position in WORLD frame [m]
    yaw: float                             # heading in world frame [rad]
    covariance: np.ndarray                 # (2,2) position covariance [m^2]
    marker_id: int
    timestamp: float = 0.0


@dataclass
class NavState:
    """The fused navigation estimate the rest of the system reads."""
    xy: np.ndarray                         # (2,) estimated world position [m]
    yaw: float = 0.0
    covariance: np.ndarray = field(default_factory=lambda: np.eye(2) * 0.01)
    timestamp: float = 0.0


@dataclass
class VelocityCmd:
    """Target-centric velocity command (Parth's controller I/O). Present here so
    Siddharth's modules compile against the shared interface set."""
    vx: float
    vy: float
