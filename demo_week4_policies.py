"""
Week 4 mid-term demo (Siddharth's evaluation slice).

Produces the mid-term comparison the plan asks for -- *uncertainty-triggered vs
fixed-period vs none* -- and the headline result of the whole project:

    similar accuracy, far fewer corrections  =  the frugality win.

Runs three policies on the same flight (which crosses a feature-poor "hard
zone"), evaluates Siddharth's metrics, and writes a table + CSV + four figures.
No camera, dataset, or GPU needed.

Run:
    python demo_week4.py
Outputs (./outputs/):
    week4_trajectories.png   week4_error.png   week4_uncertainty.png
    week4_pareto.png         week4_metrics.csv
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import metrics
from core.policies import FixedPeriodPolicy, NonePolicy, UncertaintyPolicy
from harness import nav_sim, plotting

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")

SEEDS = range(25)          # average metrics/Pareto over many runs (plan: xN for variance)
PLOT_SEED = 1              # one representative run for the trajectory/error/U figures

# headline operating points (verified to give matched accuracy)
FIXED_PERIOD = 5
U_THRESHOLD = 0.7


def avg_metrics(scenario, policy_factory):
    """Mean of Siddharth's metrics for a policy across SEEDS."""
    rows = []
    for s in SEEDS:
        run = nav_sim.run_policy(scenario, policy_factory(), seed=s)
        rows.append(metrics.summarize(run, scenario["true_xy"]))
    keys = ["peak_drift_m", "arrival_error_m", "rmse_m",
            "avl_invocations", "corrections", "marker_success_rate"]
    out = {"policy": rows[0]["policy"]}
    for k in keys:
        out[k] = float(np.mean([r[k] for r in rows]))
    return out


def avg_pareto(scenario, factories):
    pts = []
    for f in factories:
        inv = np.mean([nav_sim.run_policy(scenario, f(), seed=s)["n_invocations"] for s in SEEDS])
        peak = np.mean([metrics.peak_drift(nav_sim.run_policy(scenario, f(), seed=s)["est_xy"],
                                           scenario["true_xy"]) for s in SEEDS])
        pts.append({"invocations": float(inv), "peak_drift_m": float(peak)})
    return pts


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sc = nav_sim.build_scenario()

    policies = {
        "none": lambda: NonePolicy(),
        "fixed": lambda: FixedPeriodPolicy(FIXED_PERIOD),
        "uncertainty": lambda: UncertaintyPolicy(U_THRESHOLD),
    }

    # --- averaged metrics table -------------------------------------------
    table = [avg_metrics(sc, policies[k]) for k in ("none", "fixed", "uncertainty")]

    print("=" * 92)
    print(" WEEK 4 -- SCHEDULING COMPARISON  (mid-term)   [mean over "
          f"{len(list(SEEDS))} seeds]")
    print("=" * 92)
    hdr = f"{'policy':>26} | {'peak':>6} | {'arrival':>7} | {'rmse':>5} | {'AVL calls':>9} | {'fixes':>5} | {'succ%':>5}"
    print(hdr)
    print("-" * 92)
    for r in table:
        print(f"{r['policy']:>26} | {r['peak_drift_m']:6.2f} | {r['arrival_error_m']:7.2f} |"
              f" {r['rmse_m']:5.2f} | {r['avl_invocations']:9.1f} | {r['corrections']:5.1f} |"
              f" {r['marker_success_rate']*100:5.1f}")
    print("-" * 92)

    fixed_row = table[1]
    unc_row = table[2]
    saved = (1.0 - unc_row["avl_invocations"] / max(1e-9, fixed_row["avl_invocations"])) * 100
    print(f" HEADLINE: uncertainty-aware matches fixed-period accuracy "
          f"(peak {unc_row['peak_drift_m']:.2f} m vs {fixed_row['peak_drift_m']:.2f} m)")
    print(f"           using {unc_row['avl_invocations']:.1f} AVL calls vs "
          f"{fixed_row['avl_invocations']:.1f}  ->  {saved:.0f}% fewer corrections (compute/power).")
    print("=" * 92)

    # --- CSV ---------------------------------------------------------------
    csv_path = os.path.join(OUT_DIR, "week4_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    # --- figures from one representative run --------------------------------
    runs = [nav_sim.run_policy(sc, policies[k](), seed=PLOT_SEED)
            for k in ("none", "fixed", "uncertainty")]
    unc_run = runs[2]
    plotting.save_trajectories(sc, runs, os.path.join(OUT_DIR, "week4_trajectories.png"))
    plotting.save_error_time(sc, runs, os.path.join(OUT_DIR, "week4_error.png"))
    plotting.save_uncertainty_signal(sc, unc_run, U_THRESHOLD,
                                     os.path.join(OUT_DIR, "week4_uncertainty.png"))

    # --- Pareto (averaged sweeps) ------------------------------------------
    fixed_curve = avg_pareto(sc, [lambda P=P: FixedPeriodPolicy(P)
                                  for P in (2, 3, 4, 5, 6, 8, 10, 14)])
    unc_curve = avg_pareto(sc, [lambda t=t: UncertaintyPolicy(t)
                                for t in (0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.4)])
    none_pt = avg_pareto(sc, [lambda: NonePolicy()])[0]
    highlight = ({"invocations": fixed_row["avl_invocations"], "peak_drift_m": fixed_row["peak_drift_m"]},
                 {"invocations": unc_row["avl_invocations"], "peak_drift_m": unc_row["peak_drift_m"]})
    plotting.save_pareto(fixed_curve, unc_curve, none_pt,
                         os.path.join(OUT_DIR, "week4_pareto.png"), highlight=highlight)

    for name in ("week4_trajectories.png", "week4_error.png", "week4_uncertainty.png",
                 "week4_pareto.png", "week4_metrics.csv"):
        print(f" saved: outputs/{name}")


if __name__ == "__main__":
    main()
