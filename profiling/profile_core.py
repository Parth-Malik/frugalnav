"""
profiling/profile_core.py
-------------------------
Per-module timing of the portable core, to feed the RISC-V feasibility study
(profiling/riscv_feasibility.md). Reports mean per-call latency for each hot-loop
module on THIS host (Python + NumPy). Python is ~50-100x slower than the C++ port,
so these numbers are used for the RELATIVE cost breakdown (which module dominates);
the ABSOLUTE per-step latency comes from the compiled C++ benchmark (cpp/main.cpp).

    python profiling/profile_core.py

Everything here is CPU-only and dataset-free -- it profiles the decision hot loop,
not the VIO/ArUco front-ends (those are characterised separately in the writeup,
since they are the compute-bound modules, not the scheduler).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.controller import TargetCentricController
from core.landmark_corrector import LandmarkCorrector
from core.navigator import SensorInput, build_navigator
from core.obstacle_avoidance import ObstacleAvoidance, ttc_from_range
from core.state_fusion import StateFusion
from core.types import MarkerSighting
from core.uncertainty_scheduler import UncertaintyScheduler
from harness import integrated_sim as sim


def _bench(fn, n=200000):
    fn()  # warm up
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6   # microseconds/call


def main():
    print("=" * 66)
    print(" FrugalNav portable-core per-module profile (host: Python+NumPy)")
    print("=" * 66)

    # scheduler
    sch = UncertaintyScheduler()
    cues = dict(sigma_pos=0.8, sigma_head=0.1, feature_loss=10.0, blur=220.0,
                imu_bias=0.03, active_features=90)
    us_sched = _bench(lambda: sch.compute(cues))

    # state fusion (predict + update)
    fus = StateFusion(init_xy=(10, 0))
    from core.types import LandmarkFix
    fix = LandmarkFix(xy=np.array([10.0, 0.0]), yaw=0.0,
                      covariance=np.eye(2) * 0.0004, marker_id=1)
    def fuse():
        fus.predict((0.06, 0.0))
        fus.update(fix)
    us_fuse = _bench(fuse, n=100000)

    # controller
    ctrl = TargetCentricController(target_B=(0, 0))
    us_ctrl = _bench(lambda: ctrl.command((30.0, 12.0), (0.0, 0.5)))

    # obstacle avoidance
    av = ObstacleAvoidance()
    us_av = _bench(lambda: av.update(np.array([-1.0, 0.0]),
                                     ttc_from_range(3.0, 2.0), 0.2))

    # landmark corrector (one sighting -> fix)
    world = sim.World()
    lmap = sim.build_landmark_map(world)
    corr = LandmarkCorrector(lmap)
    entry = lmap.get(2)
    rng = np.random.default_rng(0)
    sight = sim.synth_sighting(np.array([25.0, 10.0]), -2.6, entry, lmap, rng, 0, 0.1)
    us_corr = _bench(lambda: corr.correct(sight), n=100000)

    # whole navigator step (predict+schedule+correct-attempt+avoid+command)
    nav = build_navigator(world.B, landmark_map=lmap, start_xy=world.start)
    si = SensorInput(t=0.0, vio_delta=np.array([-0.9, -0.4]),
                     cues=dict(feature_loss=8.0, blur=210.0, imu_bias=0.02,
                               sigma_head=0.01, active_features=90),
                     sighting=None, obstacle=(1.5, 0.3))
    us_nav = _bench(lambda: nav.step(si), n=100000)

    rows = [
        ("uncertainty scheduler", us_sched, "~40 scalar float ops; the contribution"),
        ("state fusion (2x2 EKF)", us_fuse, "predict + Kalman update, 2x2 inverse"),
        ("target-centric controller", us_ctrl, "vector + clamp"),
        ("obstacle avoidance (TTC)", us_av, "hysteresis + perpendicular evasion"),
        ("landmark corrector", us_corr, "one 4x4 chain + 2x2 (per sighting only)"),
        ("full navigator.step()", us_nav, "the whole decision hot loop"),
    ]
    print(f" {'module':<28}{'us/call':>10}   notes")
    print("-" * 66)
    for name, us, note in rows:
        print(f" {name:<28}{us:>9.3f}   {note}")
    print("-" * 66)
    print(" Note: NumPy per-call overhead dominates these tiny ops in Python; the")
    print(" C++ port (cpp/main.cpp) runs the same decision hot loop in ~90 ns/step.")
    print("=" * 66)


if __name__ == "__main__":
    main()
