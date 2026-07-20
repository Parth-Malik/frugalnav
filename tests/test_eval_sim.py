"""
Verification tests for the Week 6 evaluation.
    python tests/test_eval_sim.py

Encodes the evaluation's claims as executable checks over the aggregated matrix.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import eval_sim

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "config", "scenarios.json")


def _summary():
    config = json.load(open(CONFIG, encoding="utf-8"))
    _, summary = eval_sim.evaluate(config)
    return {(r["scenario"], r["policy"]): r for r in summary}


S = _summary()


def test_scenario_A_has_no_corrections():
    # no markers -> no policy can correct
    for pol in ("none", "fixed", "uncertainty"):
        assert S[("A", pol)]["avl_invocations_mean"] == 0.0


def test_scenario_A_policies_are_identical():
    base = S[("A", "none")]["arrival_error_m_mean"]
    for pol in ("fixed", "uncertainty"):
        assert abs(S[("A", pol)]["arrival_error_m_mean"] - base) < 1e-9


def test_correction_beats_none():
    # with markers, both correcting policies must beat the no-correction baseline
    for sid in ("B", "C"):
        none_err = S[(sid, "none")]["peak_drift_m_mean"]
        for pol in ("fixed", "uncertainty"):
            assert S[(sid, pol)]["peak_drift_m_mean"] < none_err


def test_uncertainty_is_more_frugal():
    # the headline: uncertainty-aware uses fewer AVL calls than fixed-period
    for sid in ("B", "C"):
        assert (S[(sid, "uncertainty")]["avl_invocations_mean"]
                < S[(sid, "fixed")]["avl_invocations_mean"])


def test_uncertainty_accuracy_is_comparable():
    # "similar accuracy": uncertainty peak drift within 2x of fixed (and both small)
    for sid in ("B", "C"):
        f = S[(sid, "fixed")]["peak_drift_m_mean"]
        u = S[(sid, "uncertainty")]["peak_drift_m_mean"]
        assert u < 2.0 * f
        assert u < 1.0            # sub-metre


def test_marker_success_rate_in_range():
    for (sid, pol), r in S.items():
        assert 0.0 <= r["marker_success_rate_mean"] <= 1.0


def test_determinism():
    config = json.load(open(CONFIG, encoding="utf-8"))
    _, s1 = eval_sim.evaluate(config)
    _, s2 = eval_sim.evaluate(config)
    a = {(r["scenario"], r["policy"]): r["peak_drift_m_mean"] for r in s1}
    b = {(r["scenario"], r["policy"]): r["peak_drift_m_mean"] for r in s2}
    assert a == b


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
