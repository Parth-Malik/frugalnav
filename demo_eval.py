"""
demo_eval.py  --  Week 6 evaluation on the REAL core (none / fixed / uncertainty)
=================================================================================
Runs the scenario x policy x seed matrix through the actual portable core (so the
'uncertainty' policy is Rohan's real UncertaintyScheduler), prints the headline
table, writes the raw + summary CSVs, and saves the comparison figures.

    python demo_eval.py [--seeds 5]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import eval_scenarios as E
from harness import plotting

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    raw, summ = E.evaluate(seeds=args.seeds)

    # ---- headline table ----
    print("\n" + "=" * 82)
    print(f" Week 6 evaluation on the real core  --  {args.seeds} seeds  "
          f"(mean over seeds)")
    print("=" * 82)
    print(f" {'scen':<5}{'policy':<18}{'arrival[m]':>11}{'peak[m]':>9}"
          f"{'rmse[m]':>9}{'AVL fixes':>11}{'succ%':>7}")
    print("-" * 82)
    for sid in E.SCENARIOS:
        for pol in E.POLICIES:
            r = next(x for x in summ if x["scenario"] == sid and x["policy"] == pol)
            sr = r["marker_success_rate_mean"]
            print(f" {sid:<5}{E.POLICY_LABEL[pol]:<18}"
                  f"{r['arrival_error_m_mean']:>10.2f} {r['peak_drift_m_mean']:>8.2f} "
                  f"{r['rmse_m_mean']:>8.2f} {r['corrections_mean']:>10.1f} "
                  f"{(100 * sr if sr == sr else 0):>6.0f}")
        print("-" * 82)

    # ---- headline sentences ----
    def pick(sid, pol):
        return next(x for x in summ if x["scenario"] == sid and x["policy"] == pol)
    for sid in ("B", "C"):
        f, u = pick(sid, "fixed"), pick(sid, "uncertainty")
        saved = 100 * (1 - u["corrections_mean"] / max(f["corrections_mean"], 1e-9))
        print(f" {sid}: uncertainty-aware {u['peak_drift_m_mean']:.2f} m peak vs fixed "
              f"{f['peak_drift_m_mean']:.2f} m, {u['corrections_mean']:.0f} vs "
              f"{f['corrections_mean']:.0f} fixes  ->  {saved:.0f}% fewer corrections.")
    a = pick("A", "uncertainty")
    print(f" A: no markers -> every policy misses by {a['arrival_error_m_mean']:.2f} m; "
          f"AVL is essential, scheduling is moot without landmarks.")
    print("=" * 82)

    # ---- CSVs ----
    raw_fields = (["scenario", "scenario_label", "policy", "seed", "has_markers"]
                  + E.METRIC_KEYS + ["min_clearance_m"])
    summ_fields = (["scenario", "scenario_label", "policy", "has_markers", "n_seeds"]
                   + [f"{k}_{s}" for k in E.METRIC_KEYS + ["min_clearance_m"]
                      for s in ("mean", "std")])
    E.write_csv(raw, os.path.join(OUT, "eval_real_runs.csv"), raw_fields)
    E.write_csv(summ, os.path.join(OUT, "eval_real_summary.csv"), summ_fields)

    # ---- figures (reuse the Week-6 grouped-bar + frugality-scatter plots) ----
    bars = os.path.join(OUT, "eval_real_bars.png")
    scat = os.path.join(OUT, "eval_real_frugality.png")
    plotting.save_evaluation_bars(summ, bars)
    plotting.save_frugality_scatter(summ, scat)

    for f in ("eval_real_runs.csv", "eval_real_summary.csv",
              "eval_real_bars.png", "eval_real_frugality.png"):
        print(f" saved: outputs/{f}")


if __name__ == "__main__":
    main()
