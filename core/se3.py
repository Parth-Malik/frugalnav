"""SE(3) / SO(3) math. Poses are 4x4 homogeneous matrices: [[R, t], [0, 1]]."""

import numpy as np
from core.interfaces import DTYPE

def skew(w):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]], dtype=DTYPE)

def exp_so3(w):
    """Rodrigues: axis-angle vector (3,) -> rotation matrix (3, 3)."""
    w = np.asarray(w, dtype=DTYPE)
    theta = float(np.linalg.norm(w))
    W = skew(w)
    
    if theta < 1e-8:
        return np.eye(3, dtype=DTYPE) + W
        
    return (np.eye(3, dtype=DTYPE)
            + (np.sin(theta) / theta) * W
            + ((1 - np.cos(theta)) / (theta ** 2)) * (W @ W))

def make_se3(R, t):
    T = np.eye(4, dtype=DTYPE)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def inv_se3(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=DTYPE)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def relative_se3(T_prev, T_curr):
    """Body-frame increment such that T_curr = T_prev @ relative_se3(T_prev, T_curr)."""
    return inv_se3(T_prev) @ T_curr

def q_to_R(q):
    """Quaternion (w, x, y, z) -> rotation matrix. Normalizes first."""
    q = np.asarray(q, dtype=DTYPE)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ValueError("Cannot convert near-zero quaternion to rotation matrix.")
        
    w, x, y, z = q / n
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ], dtype=DTYPE)

def project_to_SO3(R):
    """
    SVD-based projection back onto the SO(3) manifold.
    Essential for mitigating float32 roundoff error during integration.
    """
    U, _, Vt = np.linalg.svd(R)
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0:
        U[:, -1] *= -1
        R_proj = U @ Vt
    return R_proj
