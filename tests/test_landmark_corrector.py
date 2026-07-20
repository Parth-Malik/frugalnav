"""
Unit tests for the landmark corrector -- Siddharth's Week 3 core.
    python tests/test_landmark_corrector.py

The key property proven here: fed a NOISE-FREE sighting synthesised from a known
drone pose, the corrector recovers that world position to floating-point
precision. That isolates the corrector math from detector/sensor noise.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.geometry import inv_T, make_T, rot_z, rpy_to_R, rvec_from_rot
from core.landmark_map import LandmarkMap, MarkerEntry
from core.landmark_corrector import LandmarkCorrector, reanchor
from core.types import LandmarkFix, MarkerSighting, NavState


def _build_map():
    """One marker at world (10, 5), a downward camera, target B at (20, 20)."""
    T_WM = make_T(rot_z(0.0), [10.0, 5.0, 0.0])
    entry = MarkerEntry(marker_id=42, T_WM=T_WM, size_m=0.3)
    K = [[800, 0, 640], [0, 800, 360], [0, 0, 1]]
    T_BC = make_T(rpy_to_R(180, 0, 0), [0, 0, 0])
    return LandmarkMap([entry], target_B=[20.0, 20.0], K=K, dist=[0]*5, T_BC=T_BC)


def _sighting_from_pose(lmap, entry, T_WB_true):
    """The exact (noise-free) sighting a camera at T_WB_true would produce."""
    T_CM = inv_T(T_WB_true @ lmap.T_BC) @ entry.T_WM
    return MarkerSighting(entry.marker_id, rvec_from_rot(T_CM[:3, :3]), T_CM[:3, 3])


def test_corrector_recovers_true_position():
    lmap = _build_map()
    corr = LandmarkCorrector(lmap)
    entry = lmap.get(42)
    for xy in [(9.5, 5.2), (10.0, 5.0), (11.3, 4.1), (10.0, 8.0)]:
        for yaw in (0.0, 0.5, -1.2):
            T_WB = make_T(rot_z(yaw), [xy[0], xy[1], 3.0])
            fix = corr.correct(_sighting_from_pose(lmap, entry, T_WB))
            assert fix is not None
            assert np.allclose(fix.xy, xy, atol=1e-9), (xy, yaw, fix.xy)
            assert abs(((fix.yaw - yaw + np.pi) % (2*np.pi)) - np.pi) < 1e-9


def test_unknown_marker_returns_none():
    lmap = _build_map()
    corr = LandmarkCorrector(lmap)
    bogus = MarkerSighting(999, np.zeros(3), np.array([0, 0, 3.0]))
    assert corr.correct(bogus) is None


def test_reanchor_gain_extremes():
    state = NavState(xy=np.array([5.0, 5.0]))          # drifted estimate
    fix = LandmarkFix(xy=np.array([7.0, 8.0]), yaw=0.3,
                      covariance=np.eye(2) * 1e-4, marker_id=1)
    # gain=1 snaps fully to the fix; gain=0 ignores it
    assert np.allclose(reanchor(state, fix, gain=1.0).xy, [7.0, 8.0])
    assert np.allclose(reanchor(state, fix, gain=0.0).xy, [5.0, 5.0])


def test_reanchor_reduces_error():
    # optimal (Kalman) gain must move the estimate toward the fix and shrink cov
    truth = np.array([7.0, 8.0])
    state = NavState(xy=np.array([5.0, 5.0]), covariance=np.eye(2) * 1.0)
    fix = LandmarkFix(xy=truth.copy(), yaw=0.0,
                      covariance=np.eye(2) * 1e-3, marker_id=1)
    out = reanchor(state, fix)
    assert np.linalg.norm(out.xy - truth) < np.linalg.norm(state.xy - truth)
    assert np.trace(out.covariance) < np.trace(state.covariance)


def test_target_B_is_constant():
    # correcting the drone must never move B: B lives in the map, untouched.
    lmap = _build_map()
    B_before = lmap.target_B.copy()
    corr = LandmarkCorrector(lmap)
    entry = lmap.get(42)
    T_WB = make_T(rot_z(0.4), [9.0, 6.0, 3.0])
    _ = corr.correct(_sighting_from_pose(lmap, entry, T_WB))
    assert np.allclose(lmap.target_B, B_before)


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
