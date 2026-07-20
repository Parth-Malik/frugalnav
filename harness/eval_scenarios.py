"""
harness/eval_scenarios.py  (throwaway scaffolding)
--------------------------------------------------
Week-6 evaluation driven by the REAL portable core (`core/navigator.py`), so the
'uncertainty' policy is Rohan's actual UncertaintyScheduler (two-tier trigger +
hysteresis, physically-scaled cues) -- not a toy threshold. The same three
scenarios the plan calls for, each run for none / fixed-period / uncertainty over
several seeds:

    A  open, NO markers        -> no fix is ever possible: the drift floor
    B  markers along the path  -> benign; corrections keep error small
    C  markers + obstacle       -> a detour injects bursty drift mid-flight

Because every scenario runs the identical core, the numbers here are directly
comparable to the end-to-end money-shot (run_demo.py). Siddharth's analytic-U
evaluation (harness/eval_sim.py, demo_week6.py) is the lightweight cross-check
that reaches the same conclusion with a simpler drift model.

Headline: uncertainty-aware matches fixed-period accuracy at roughly HALF the
corrections wherever drift is heterogeneous (B and C).
"""
from __future__ import annotations

import csv

import numpy as np

from harness import integrated_sim as sim

POLICIES = ("none", "fixed", "uncertainty")
POLICY_LABEL = {"none": "none", "fixed": "fixed-period", "uncertainty": "uncertainty-aware"}

SCENARIOS = {
    "A": ("A: open, no markers",
          dict(has_markers=False, has_obstacle=False)),
    "B": ("B: markers along path",
          dict(has_markers=True, has_obstacle=False)),
    "C": ("C: markers + obstacle (detour)",
          dict(has_markers=True, has_obstacle=True)),
}

METRIC_KEYS = ["arrival_error_m", "peak_drift_m", "rmse_m",
               "avl_invocations", "corrections", "marker_success_rate"]


def _metrics(run):
    err = run["err"]
    inv = run["invocations"]
    fixes = len(run["corrections"])
    return {
        "arrival_error_m": float(run["arrival_miss_m"]),
        "peak_drift_m": float(np.max(err)),
        "rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "avl_invocations": int(inv),
        "corrections": int(fixes),
        "marker_success_rate": float(fixes / inv) if inv > 0 else float("nan"),
        "min_clearance_m": float(run["min_clearance_m"]),
    }


def evaluate(seeds=5, fixed_period=45):
    """Run the full (scenario x policy x seed) matrix on the real core."""
    raw_rows, summary_rows = [], []
    for sid, (label, flags) in SCENARIOS.items():
        for pol in POLICIES:
            per_seed = []
            for s in range(seeds):
                world = sim.World(**flags)
                run = sim.run(world, policy=pol, seed=s, fixed_period=fixed_period)
                m = _metrics(run)
                m.update(scenario=sid, scenario_label=label, policy=pol, seed=s,
                         has_markers=flags["has_markers"])
                raw_rows.append(m)
                per_seed.append(m)
            row = {"scenario": sid, "scenario_label": label, "policy": pol,
                   "has_markers": flags["has_markers"], "n_seeds": len(per_seed)}
            for k in METRIC_KEYS + ["min_clearance_m"]:
                vals = np.array([r[k] for r in per_seed], dtype=float)
                finite = vals[np.isfinite(vals)]
                row[f"{k}_mean"] = float(finite.mean()) if finite.size else float("nan")
                row[f"{k}_std"] = float(finite.std()) if finite.size else float("nan")
            summary_rows.append(row)
    return raw_rows, summary_rows


def write_csv(rows, path, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


if __name__ == "__main__":
    raw, summ = evaluate(seeds=5)
    print(f"{'scen':<5}{'policy':<14}{'arrival':>9}{'peak':>8}{'rmse':>8}"
          f"{'AVL':>6}{'fixes':>7}{'succ%':>7}")
    for sid in SCENARIOS:
        for pol in POLICIES:
            r = next(x for x in summ if x["scenario"] == sid and x["policy"] == pol)
            sr = r["marker_success_rate_mean"]
            print(f"{sid:<5}{POLICY_LABEL[pol]:<14}"
                  f"{r['arrival_error_m_mean']:>7.2f}m{r['peak_drift_m_mean']:>7.2f}"
                  f"{r['rmse_m_mean']:>8.2f}{r['avl_invocations_mean']:>6.1f}"
                  f"{r['corrections_mean']:>7.1f}"
                  f"{(100*sr if sr==sr else 0):>6.0f} ")
        print("-" * 64)
