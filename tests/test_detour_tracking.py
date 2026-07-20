"""
Verification tests for the Week 5 deliverable (integration-level).
    python tests/test_detour_tracking.py

These encode the plan's Week-5 claim as executable checks:
  * obstacle avoidance is independent of localization (both runs clear it),
  * with the landmark corrector, VIO tracks through the detour (features stay
    above the observability floor, vector-to-B stays small) and the drone ARRIVES,
  * without it, the drone still avoids the obstacle but MISSES B.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.landmark_corrector import LandmarkCorrector
from harness import detour_sim as ds

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "config", "detour_scenario.json")


def _runs(seed=0):
    lmap, obstacle, sim = ds.load_scenario(CONFIG)
    corrector = LandmarkCorrector(lmap)
    corr = ds.run_detour(lmap, obstacle, sim, corrector, correct=True, seed=seed)
    none = ds.run_detour(lmap, obstacle, sim, corrector, correct=False, seed=seed)
    return corr, none, float(sim["observability_floor"])


def _peak_dir_pre_approach(run, guard_m=5.0):
    d = np.linalg.norm(run["true_xy"] - run["B"], axis=1)
    mask = d > guard_m
    return float(run["dir_err"][mask].max()) if mask.any() else 0.0


def test_both_clear_the_obstacle():
    # avoidance is reactive on TRUE sensing -> drift must not cause a collision
    for seed in range(3):
        corr, none, _ = _runs(seed)
        assert not corr["obstacle_hit"], f"corrected hit obstacle (seed {seed})"
        assert not none["obstacle_hit"], f"uncorrected hit obstacle (seed {seed})"


def test_correction_arrives_at_B():
    for seed in range(3):
        corr, _, _ = _runs(seed)
        assert corr["arrival_miss_m"] < 2.0, corr["arrival_miss_m"]


def test_no_correction_misses_B():
    for seed in range(3):
        corr, none, _ = _runs(seed)
        assert none["arrival_miss_m"] > 3.0                 # meaningfully misses
        assert none["arrival_miss_m"] > corr["arrival_miss_m"]   # correction helps


def test_tracking_survives_detour():
    for seed in range(3):
        corr, _, floor = _runs(seed)
        assert corr["feat"].min() > floor                   # never lost tracking
        assert _peak_dir_pre_approach(corr) < 15.0          # vector-to-B stayed correct


def test_deterministic_given_seed():
    a, _, _ = _runs(7)
    b, _, _ = _runs(7)
    assert np.allclose(a["true_xy"], b["true_xy"])


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
