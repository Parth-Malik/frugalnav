"""
Unit tests for core/vector_to_target.py.
    python tests/test_vector_to_target.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vector_to_target import (
    direction_error_deg, heading_to_target, position_error, vector_to_target,
)


def test_vector_and_heading():
    assert np.allclose(vector_to_target([1, 2], [4, 6]), [3, 4])
    # B due north of the drone -> heading +90 deg
    assert abs(heading_to_target([0, 0], [0, 5]) - np.pi / 2) < 1e-9
    # B due east -> heading 0
    assert abs(heading_to_target([0, 0], [5, 0]) - 0.0) < 1e-9


def test_position_error():
    assert abs(position_error([0, 0], [3, 4]) - 5.0) < 1e-12


def test_direction_error_zero_when_estimate_correct():
    # arccos loses precision near 1, so ~0 (< 1e-4 deg) is the right "is zero" check
    for B in ([0, 10], [5, -3], [12, 7]):
        assert direction_error_deg([1, 1], [1, 1], B) < 1e-4


def test_direction_error_ninety_degrees():
    # true is west of B, est is south of B -> vectors to B are perpendicular
    B = [0.0, 0.0]
    true_pos = [10.0, 0.0]    # (B - true) = (-10, 0)
    est_pos = [0.0, 10.0]     # (B - est)  = (0, -10)
    assert abs(direction_error_deg(est_pos, true_pos, B) - 90.0) < 1e-6


def test_direction_error_symmetric_and_bounded():
    B = [0.0, 20.0]
    a = direction_error_deg([3, 5], [0, 5], B)
    b = direction_error_deg([0, 5], [3, 5], B)
    assert abs(a - b) < 1e-9          # symmetric in the two positions
    assert 0.0 <= a <= 180.0


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
