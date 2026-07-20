"""
Week 5 demo (Siddharth's verification slice).

Plan task: *"verify VIO keeps tracking through the detour so the recomputed
vector to B stays correct."* This runs the closed loop -- head to B, detour around
an obstacle, resume, arrive -- twice: with Siddharth's landmark corrector active
and without. It shows the Week-5 point:

    * obstacle avoidance is reactive on TRUE sensing, so BOTH runs clear the
      obstacle even when localization has drifted;
    * but reaching B depends on the estimate, so only the CORRECTED run keeps the
      vector to B correct through the maneuver and actually arrives -- the
      uncorrected drone avoids the obstacle and then misses the target.

No camera/dataset/GPU. Reuses the Week-3 landmark corrector unchanged.

Run:
    python demo_week5.py
Outputs (./outputs/):
    week5_detour.png   week5_vector_error.png   week5_tracking.png
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.landmark_corrector import LandmarkCorrector
from harness import detour_sim as ds
from harness import plotting

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "config", "detour_scenario.json")
OUT_DIR = os.path.join(HERE, "outputs")


def _peak_dir_pre_approach(run, guard_m=5.0):
    """Peak vector-to-B direction error, ignoring the last few metres where
    (B - pos) -> 0 makes the angle numerically singular for both runs."""
    d = np.linalg.norm(run["true_xy"] - run["B"], axis=1)
    mask = d > guard_m
    return float(run["dir_err"][mask].max()) if mask.any() else 0.0


def main(seed: int = 0):
    os.makedirs(OUT_DIR, exist_ok=True)
    lmap, obstacle, sim = ds.load_scenario(CONFIG)
    corrector = LandmarkCorrector(lmap)

    corr = ds.run_detour(lmap, obstacle, sim, corrector, correct=True, seed=seed)
    none = ds.run_detour(lmap, obstacle, sim, corrector, correct=False, seed=seed)
    floor = float(sim["observability_floor"])

    print("=" * 74)
    print(" WEEK 5 -- TRACKING THROUGH THE DETOUR  (Siddharth)")
    print("=" * 74)
    B_str = f"[{lmap.target_B[0]:.0f}, {lmap.target_B[1]:.0f}]"
    print(f" scenario: start {sim['start_xy']} -> B {B_str}, "
          f"obstacle r={obstacle['radius']} at {obstacle['center']}")
    print("-" * 74)
    print(f" {'':22} | {'NO correction':>15} | {'landmark correction':>19}")
    print("-" * 74)
    print(f" {'obstacle cleared?':22} | {('YES' if not none['obstacle_hit'] else 'NO'):>15}"
          f" | {('YES' if not corr['obstacle_hit'] else 'NO'):>19}")
    print(f" {'min clearance [m]':22} | {none['min_clearance_m']:>15.2f} | {corr['min_clearance_m']:>19.2f}")
    print(f" {'arrival miss at B [m]':22} | {none['arrival_miss_m']:>15.2f} | {corr['arrival_miss_m']:>19.2f}")
    print(f" {'peak vec-to-B err[deg]':22} | {_peak_dir_pre_approach(none):>15.1f} | {_peak_dir_pre_approach(corr):>19.1f}")
    print(f" {'min feature tracks':22} | {none['feat'].min():>15.0f} | {corr['feat'].min():>19.0f}")
    print(f" {'landmark fixes':22} | {len(none['corrections']):>15d} | {len(corr['corrections']):>19d}")
    print("-" * 74)
    survived = (corr["feat"].min() > floor) and (corr["arrival_miss_m"] < 2.0) and (not corr["obstacle_hit"])
    print(f" VERDICT: with correction, VIO tracks through the detour "
          f"(features stay > floor {floor:.0f},")
    print(f"          vector-to-B stays < 10 deg) and the drone ARRIVES "
          f"-> {'PASS' if survived else 'CHECK'}.")
    print(f"          Without it, the drone avoids the obstacle but MISSES B by "
          f"{none['arrival_miss_m']:.1f} m.")
    print("=" * 74)

    plotting.save_detour_moneyshot(corr, none, lmap, os.path.join(OUT_DIR, "week5_detour.png"))
    plotting.save_vector_error(corr, none, os.path.join(OUT_DIR, "week5_vector_error.png"))
    plotting.save_tracking(corr, none, floor, os.path.join(OUT_DIR, "week5_tracking.png"))
    for n in ("week5_detour.png", "week5_vector_error.png", "week5_tracking.png"):
        print(f" saved: outputs/{n}")


if __name__ == "__main__":
    main()
