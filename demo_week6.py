"""
Week 6 evaluation demo (Siddharth's slice).

Plan task: *"metrics -- final arrival error, peak drift, AVL correction count
(frugality), marker success rate. Compare none / fixed-period / uncertainty-aware"*
across scenarios A (open, no markers), B (markers), C (obstacles + markers), each
x5 seeds for variance.

This runs the full matrix, prints the results table, writes the CSVs, and saves
the figures that back the chapter's headline:

    similar accuracy, far fewer corrections = the frugality win.

Run:
    python demo_week6.py
Outputs (./outputs/):
    week6_results.csv   week6_runs.csv
    week6_evaluation.png   week6_frugality.png
"""
from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import eval_sim, plotting

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "config", "scenarios.json")
OUT_DIR = os.path.join(HERE, "outputs")

_PLABEL = {"none": "none", "fixed": "fixed-period", "uncertainty": "uncertainty"}


def _fmt(mean, std):
    return f"{mean:.2f}+/-{std:.2f}"


def _print_table(summary):
    print("=" * 96)
    print(" WEEK 6 -- EVALUATION MATRIX  (mean +/- std over seeds)   [Siddharth: metrics]")
    print("=" * 96)
    print(f" {'scenario':32} {'policy':13} {'arrival[m]':>12} {'peak[m]':>11} "
          f"{'AVL':>10} {'succ%':>6}")
    print("-" * 96)
    last = None
    for r in summary:
        scen = r["scenario_label"] if r["scenario"] != last else ""
        last = r["scenario"]
        if r["avl_invocations_mean"] > 0:
            succ = f"{r['marker_success_rate_mean']*100:5.1f}"
        else:
            succ = "   - "
        print(f" {scen:32} {_PLABEL[r['policy']]:13} "
              f"{_fmt(r['arrival_error_m_mean'], r['arrival_error_m_std']):>12} "
              f"{_fmt(r['peak_drift_m_mean'], r['peak_drift_m_std']):>11} "
              f"{_fmt(r['avl_invocations_mean'], r['avl_invocations_std']):>10} {succ:>6}")
        if r["policy"] == "uncertainty":
            print("-" * 96)


def _headline(summary):
    def get(sid, pol):
        return next(r for r in summary if r["scenario"] == sid and r["policy"] == pol)
    print(" HEADLINE")
    a = get("A", "none")
    print(f"  - Scenario A (no markers): every policy lands at "
          f"{a['arrival_error_m_mean']:.2f} m -- AVL is essential; scheduling is moot without markers.")
    for sid in ("B", "C"):
        f, u = get(sid, "fixed"), get(sid, "uncertainty")
        saved = (1 - u["avl_invocations_mean"] / max(1e-9, f["avl_invocations_mean"])) * 100
        print(f"  - Scenario {sid}: uncertainty-aware peak {u['peak_drift_m_mean']:.2f} m "
              f"vs fixed {f['peak_drift_m_mean']:.2f} m, using {u['avl_invocations_mean']:.1f} "
              f"vs {f['avl_invocations_mean']:.1f} AVL calls  ->  {saved:.0f}% fewer corrections.")
    print("=" * 96)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    config = json.load(open(CONFIG, encoding="utf-8"))
    raw, summary = eval_sim.evaluate(config)

    _print_table(summary)
    _headline(summary)

    # summary CSV
    with open(os.path.join(OUT_DIR, "week6_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    # raw per-run CSV (the x5 variance)
    with open(os.path.join(OUT_DIR, "week6_runs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0].keys()))
        w.writeheader()
        w.writerows(raw)

    plotting.save_evaluation_bars(summary, os.path.join(OUT_DIR, "week6_evaluation.png"))
    plotting.save_frugality_scatter(summary, os.path.join(OUT_DIR, "week6_frugality.png"))

    for n in ("week6_results.csv", "week6_runs.csv", "week6_evaluation.png", "week6_frugality.png"):
        print(f" saved: outputs/{n}")


if __name__ == "__main__":
    main()
