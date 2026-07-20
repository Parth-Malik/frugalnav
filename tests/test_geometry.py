"""
Unit tests for core/geometry.py. No pytest needed:
    python tests/test_geometry.py
(They are also plain `test_*` functions, so `pytest` works too if installed.)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.geometry import (
    inv_T, make_T, rot_from_rvec, rot_z, rpy_to_R, rvec_from_rot,
    T_from_rvec_tvec, xy_of, yaw_of,
)

RNG = np.random.default_rng(0)


def test_rvec_rot_roundtrip():
    for _ in range(200):
        rvec = RNG.normal(0, 1, 3)
        R = rot_from_rvec(rvec)
        # R must be a proper rotation
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert abs(np.linalg.det(R) - 1.0) < 1e-9
        # rvec -> R -> rvec -> R round-trips exactly
        assert np.allclose(rot_from_rvec(rvec_from_rot(R)), R, atol=1e-8)


def test_rvec_rot_roundtrip_near_180():
    # Regression: rvec_from_rot must stay exact as the rotation angle approaches
    # 180 deg (the naive off-diagonal formula divides by sin(theta)->0 and fails).
    # This is the regime a drone homing on a target at the world origin hits.
    for theta in np.linspace(np.pi - 0.3, np.pi - 1e-4, 40):
        for _ in range(20):
            axis = RNG.normal(0, 1, 3)
            axis /= np.linalg.norm(axis)
            R = rot_from_rvec(axis * theta)
            assert np.allclose(rot_from_rvec(rvec_from_rot(R)), R, atol=1e-9), theta


def test_inverse_transform():
    for _ in range(200):
        T = make_T(rot_from_rvec(RNG.normal(0, 1, 3)), RNG.normal(0, 5, 3))
        assert np.allclose(inv_T(T) @ T, np.eye(4), atol=1e-9)
        assert np.allclose(T @ inv_T(T), np.eye(4), atol=1e-9)


def test_chain_cancels():
    # T_AC = T_AB @ T_BC ; and inv chains: inv(T_AB) @ T_AC == T_BC
    T_AB = make_T(rot_from_rvec(RNG.normal(0, 1, 3)), RNG.normal(0, 5, 3))
    T_BC = make_T(rot_from_rvec(RNG.normal(0, 1, 3)), RNG.normal(0, 5, 3))
    T_AC = T_AB @ T_BC
    assert np.allclose(inv_T(T_AB) @ T_AC, T_BC, atol=1e-9)


def test_yaw_and_xy_extraction():
    for yaw in np.linspace(-3.0, 3.0, 25):
        T = make_T(rot_z(yaw), [1.5, -2.5, 4.0])
        assert abs(yaw_of(T) - yaw) < 1e-9
        assert np.allclose(xy_of(T), [1.5, -2.5])


def test_downward_camera_extrinsic():
    # rpy=[180,0,0] must send body +Z (up) to camera -Z, i.e. the camera looks
    # down: a point 5 m below the body sits at +Z (in front) in the camera.
    R = rpy_to_R(180, 0, 0)
    assert np.allclose(R @ np.array([0, 0, -1.0]), [0, 0, 1.0], atol=1e-9)


def test_rvec_tvec_builder():
    rvec = RNG.normal(0, 1, 3)
    tvec = RNG.normal(0, 5, 3)
    T = T_from_rvec_tvec(rvec, tvec)
    assert np.allclose(T[:3, :3], rot_from_rvec(rvec), atol=1e-12)
    assert np.allclose(T[:3, 3], tvec, atol=1e-12)


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
