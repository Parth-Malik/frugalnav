"""
Unit tests for core/metrics.py.
    python tests/test_metrics.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import metrics


def test_peak_and_final_drift():
    true = np.zeros((5, 2))
    est = np.array([[0, 0], [0, 1], [0, 3], [0, 2], [0, 0.5]], dtype=float)
    assert abs(metrics.peak_drift(est, true) - 3.0) < 1e-12
    assert abs(metrics.final_drift(est, true) - 0.5) < 1e-12
    assert abs(metrics.arrival_error(est, true) - 0.5) < 1e-12   # == final drift


def test_rmse():
    true = np.zeros((3, 2))
    est = np.array([[3, 4], [0, 0], [0, 0]], dtype=float)   # errors: 5, 0, 0
    assert abs(metrics.rmse(est, true) - np.sqrt(25.0 / 3)) < 1e-12


def test_correction_count():
    assert metrics.correction_count([]) == 0
    assert metrics.correction_count([2, 7, 9]) == 3


def test_marker_success_rate():
    assert metrics.marker_success_rate(8, 10) == 0.8
    assert metrics.marker_success_rate(0, 0) == 0.0        # no divide-by-zero


def test_summarize_keys():
    true = np.zeros((4, 2))
    run = {
        "policy_name": "test",
        "est_xy": np.array([[0, 0], [0, 0.5], [0, 1.0], [0, 0.2]], dtype=float),
        "correction_steps": [3],
        "n_invocations": 4,
        "n_detected": 3,
    }
    s = metrics.summarize(run, true)
    for key in ("peak_drift_m", "arrival_error_m", "rmse_m",
                "avl_invocations", "corrections", "marker_success_rate"):
        assert key in s
    assert s["corrections"] == 1
    assert abs(s["peak_drift_m"] - 1.0) < 1e-12
    assert abs(s["marker_success_rate"] - 0.75) < 1e-12


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
