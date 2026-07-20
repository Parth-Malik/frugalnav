"""
Unit tests for core/policies.py.
    python tests/test_policies.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.policies import (
    FixedPeriodPolicy, NonePolicy, PolicyContext, UncertaintyPolicy,
)


def _ctx(steps_since=0, U=0.0, marker=True, step=0):
    return PolicyContext(step=step, t=float(step), steps_since_last=steps_since,
                         U=U, marker_available=marker)


def test_none_never_corrects():
    p = NonePolicy()
    assert not p.should_correct(_ctx(steps_since=999, U=999, marker=True))


def test_fixed_period_respects_period():
    p = FixedPeriodPolicy(5)
    assert not p.should_correct(_ctx(steps_since=4, marker=True))
    assert p.should_correct(_ctx(steps_since=5, marker=True))
    assert p.should_correct(_ctx(steps_since=9, marker=True))


def test_fixed_period_requires_marker():
    p = FixedPeriodPolicy(3)
    assert not p.should_correct(_ctx(steps_since=10, marker=False))


def test_uncertainty_respects_threshold():
    p = UncertaintyPolicy(0.7)
    assert not p.should_correct(_ctx(U=0.69, marker=True))
    assert p.should_correct(_ctx(U=0.71, marker=True))


def test_uncertainty_requires_marker():
    p = UncertaintyPolicy(0.5)
    assert not p.should_correct(_ctx(U=5.0, marker=False))


def test_names_are_descriptive():
    assert NonePolicy().name == "none"
    assert "5" in FixedPeriodPolicy(5).name
    assert "0.7" in UncertaintyPolicy(0.7).name


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
