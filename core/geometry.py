"""
SE(3) rigid-transform helpers -- pure NumPy, no OpenCV.

A pose is a 4x4 homogeneous matrix  T = [[R, t], [0, 1]]  where R is a 3x3
rotation and t is a 3-vector translation. T maps a point in the CHILD frame to
the PARENT frame:  p_parent = T @ [p_child; 1].

Naming convention used across the whole project:
    T_AB       = pose of frame B expressed in frame A
               = the transform that takes B-frame coords into A-frame coords.
    inv_T(T_AB) = T_BA
    T_AB @ T_BC = T_AC          (the shared B cancels: A<-B, B<-C  =>  A<-C)

This module deliberately avoids OpenCV so it maps cleanly onto the portable
core that ports to C++/Eigen / RISC-V. NumPy only.
"""
from __future__ import annotations

import numpy as np


def rot_from_rvec(rvec) -> np.ndarray:
    """Rodrigues: axis-angle 3-vector -> 3x3 rotation matrix."""
    rvec = np.asarray(rvec, dtype=float).reshape(3)
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K = np.array([
        [0.0,  -k[2],  k[1]],
        [k[2],  0.0,  -k[0]],
        [-k[1], k[0],  0.0],
    ])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def rvec_from_rot(R) -> np.ndarray:
    """Inverse Rodrigues: 3x3 rotation -> axis-angle 3-vector.

    Implemented via a rotation->quaternion conversion (Shepperd's method), which
    is numerically robust for the WHOLE angle range including near 180 deg, where
    the naive off-diagonal formula divides by sin(theta) -> 0 and loses the axis.
    (The earlier off-diagonal version failed for headings ~180 deg -- e.g. a drone
    homing on a target at the world origin -- giving metre-level fix errors.)
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    tr = float(np.trace(R))
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0                      # S = 4*qw
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0   # S = 4*qx
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0   # S = 4*qy
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0   # S = 4*qz
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    q = np.array([qw, qx, qy, qz], dtype=float)
    q /= np.linalg.norm(q)
    vec_norm = float(np.linalg.norm(q[1:]))
    if vec_norm < 1e-12:                                  # no rotation
        return np.zeros(3)
    theta = 2.0 * np.arctan2(vec_norm, q[0])              # in [0, 2*pi)
    axis = q[1:] / vec_norm
    if theta > np.pi:                                    # canonical angle in [0, pi]
        theta = 2.0 * np.pi - theta
        axis = -axis
    return axis * theta


def rot_z(yaw: float) -> np.ndarray:
    """Rotation about the world Z (up) axis by `yaw` radians."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def rpy_to_R(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Roll-pitch-yaw (degrees, ZYX order) -> 3x3 rotation. Used for the fixed
    camera mount extrinsic (a downward camera is roll=180)."""
    r, p, y = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_T(R=None, t=None) -> np.ndarray:
    """Assemble a 4x4 homogeneous transform from a rotation and/or translation."""
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def T_from_rvec_tvec(rvec, tvec) -> np.ndarray:
    """Build T from OpenCV-style (rvec, tvec) -- e.g. an ArUco pose estimate."""
    return make_T(rot_from_rvec(rvec), tvec)


def inv_T(T) -> np.ndarray:
    """Inverse of a homogeneous transform (cheaper + more stable than np.inv)."""
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def translation_of(T) -> np.ndarray:
    return np.asarray(T, dtype=float)[:3, 3].copy()


def xy_of(T) -> np.ndarray:
    """The (x, y) ground-plane position of a pose."""
    return translation_of(T)[:2].copy()


def yaw_of(T) -> float:
    """Heading (rotation about world Z) extracted from a pose."""
    R = np.asarray(T, dtype=float)[:3, :3]
    return float(np.arctan2(R[1, 0], R[0, 0]))
