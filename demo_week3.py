"""
Week 3 money-shot demo (Siddharth).

Runs the drift-injection scenario twice on the SAME flight -- once with the
landmark corrector active, once without -- and shows the headline result:

    drift accumulates, then SNAPS BACK to truth at every marker, with target B
    fixed in the world frame.

No webcam, no dataset, no GPU needed. It exercises the real landmark corrector
on physically-consistent ArUco sightings generated from ground truth.

Run:
    python demo_week3.py
Outputs (saved next to this script, in ./outputs/):
    week3_moneyshot.png   -- XY trajectories
    week3_error.png       -- position error vs time
"""
from __future__ import annotations

import os
import sys

import numpy as np

# make `core` / `harness` importable no matter the working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.landmark_map import LandmarkMap
from core.landmark_corrector import LandmarkCorrector
from harness import drift_sim
from harness import plotting


HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "config", "landmark_map.json")
OUT_DIR = os.path.join(HERE, "outputs")


def main(seed: int = 7):
    os.makedirs(OUT_DIR, exist_ok=True)

    lmap = LandmarkMap.from_json(CONFIG)
    corrector = LandmarkCorrector(lmap)
    scenario = drift_sim.build_scenario()

    corrected = drift_sim.run_estimator(scenario, lmap, corrector,
                                        correct=True, gain=1.0, seed=seed)
    uncorrected = drift_sim.run_estimator(scenario, lmap, corrector,
                                          correct=False, seed=seed)

    true_xy = scenario["true_xy"]
    err_c = np.linalg.norm(corrected["est_xy"] - true_xy, axis=1)
    err_u = np.linalg.norm(uncorrected["est_xy"] - true_xy, axis=1)

    n_corr = len(corrected["correction_steps"])
    succ = corrected["n_detected"] / max(1, corrected["n_marker_frames"])

    print("=" * 66)
    print(" WEEK 3 -- LANDMARK CORRECTOR  (Siddharth)")
    print("=" * 66)
    print(f" map            : {len(lmap)} markers, target B at {lmap.target_B}")
    print(f" flight         : {len(true_xy)} steps, L-shaped path over the markers")
    print(f" corrections    : {n_corr} landmark fixes fired")
    print(f" marker success : {succ*100:5.1f}%  "
          f"({corrected['n_detected']}/{corrected['n_marker_frames']} in-range frames)")
    print("-" * 66)
    print(f" peak drift  -- no correction : {err_u.max():6.2f} m")
    print(f" peak drift  -- corrected     : {err_c.max():6.2f} m")
    print(f" final drift -- no correction : {err_u[-1]:6.2f} m")
    print(f" final drift -- corrected     : {err_c[-1]:6.2f} m")
    print("-" * 66)
    # target B is a world-frame constant -> the recomputed vector to B is only
    # as good as the estimate; show how much correction improves it.
    v_true = lmap.target_B - true_xy[-1]
    v_c = lmap.target_B - corrected["est_xy"][-1]
    v_u = lmap.target_B - uncorrected["est_xy"][-1]
    print(f" vector-to-B error -- no correction : {np.linalg.norm(v_u - v_true):6.2f} m")
    print(f" vector-to-B error -- corrected     : {np.linalg.norm(v_c - v_true):6.2f} m")
    print("=" * 66)

    mshot = os.path.join(OUT_DIR, "week3_moneyshot.png")
    eplot = os.path.join(OUT_DIR, "week3_error.png")
    plotting.save_moneyshot(scenario, corrected, uncorrected, lmap, mshot)
    plotting.save_error_plot(scenario, corrected, uncorrected, eplot)
    print(f" saved: {os.path.relpath(mshot, HERE)}")
    print(f" saved: {os.path.relpath(eplot, HERE)}")


if __name__ == "__main__":
    main()
