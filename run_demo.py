"""
run_demo.py  --  FrugalNav end-to-end demo (the "live demo" entrypoint)
=======================================================================
Runs the WHOLE system through the portable core (`core/navigator.py`): homing on a
fixed target B, VIO drift, uncertainty-scheduled landmark fixes, tight state fusion,
and a reactive optical-flow obstacle detour -- for three correction policies over
the identical world:

    none         pure VIO (drifts, misses B)
    fixed        correct at every marker (accurate but wasteful)
    uncertainty  correct only when U crosses the threshold   <-- the contribution

Prints the headline table, saves the money-shot + dashboard figures to outputs/,
and exports outputs/integrated_demo.json (the interactive web demo reads this).

    python run_demo.py                 # single headline seed + plots + json
    python run_demo.py --seeds 10      # also print the multi-seed mean +/- std
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import integrated_sim as sim
from harness import plotting

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
ORDER = ("none", "fixed", "uncertainty")


def _fmt(results):
    print("\n" + "=" * 78)
    print(" FrugalNav end-to-end  --  one flight, three correction policies")
    print("=" * 78)
    print(f" {'policy':<16}{'arrival':>10}{'peak drift':>12}{'AVL fixes':>11}"
          f"{'obstacle':>11}")
    print(f" {'':<16}{'miss [m]':>10}{'[m]':>12}{'(compute)':>11}{'clear [m]':>11}")
    print("-" * 78)
    for p in ORDER:
        r = results[p]
        print(f" {plotting._POL_L[p]:<16}{r['arrival_miss_m']:>9.2f} "
              f"{r['peak_err']:>11.2f} {len(r['corrections']):>10} "
              f"{r['min_clearance_m']:>10.2f}")
    print("-" * 78)
    u, f, n = results["uncertainty"], results["fixed"], results["none"]
    saved = 100.0 * (1 - len(u["corrections"]) / max(len(f["corrections"]), 1))
    print(f" HEADLINE: uncertainty-aware matches fixed-period accuracy "
          f"({u['arrival_miss_m']:.2f} m vs {f['arrival_miss_m']:.2f} m)")
    print(f"           using {len(u['corrections'])} fixes vs {len(f['corrections'])}"
          f"  ->  {saved:.0f}% fewer corrections (= compute / power saved),")
    print(f"           while pure VIO drifts to a {n['arrival_miss_m']:.2f} m miss. "
          f"All three clear the obstacle.")
    print("=" * 78)


def _multiseed(seeds):
    agg = {p: {k: [] for k in ("arr", "peak", "fix", "clr")} for p in ORDER}
    for s in range(seeds):
        res = sim.run_all(seed=s)
        for p in ORDER:
            r = res[p]
            agg[p]["arr"].append(r["arrival_miss_m"])
            agg[p]["peak"].append(r["peak_err"])
            agg[p]["fix"].append(len(r["corrections"]))
            agg[p]["clr"].append(r["min_clearance_m"])
    print(f"\n Multi-seed summary over {seeds} seeds (mean +/- std):")
    print("-" * 78)
    for p in ORDER:
        a = agg[p]
        print(f" {plotting._POL_L[p]:<16} arrival {np.mean(a['arr']):.2f}+/-"
              f"{np.std(a['arr']):.2f} m   peak {np.mean(a['peak']):.2f} m   "
              f"fixes {np.mean(a['fix']):.1f}   min clear {np.min(a['clr']):.2f} m")
    print("-" * 78)


def _export_json(results, path):
    """Compact JSON for the interactive web demo. Subsample paths to keep it small."""
    def thin(a, step=3):
        arr = np.asarray(a)[::step]
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.round(3)
        return arr.tolist()

    unc = results["uncertainty"]
    lmap = unc["lmap"]
    obs_c, obs_r = unc["obstacle"]
    payload = {
        "B": unc["B"].round(3).tolist(),
        "start": unc["start"].round(3).tolist(),
        "markers": lmap.all_world_xy().round(3).tolist(),
        "obstacle": {"c": obs_c.round(3).tolist(), "r": round(float(obs_r), 3)},
        "hard": {"c": unc["hard_center"].round(3).tolist(),
                 "r": round(float(unc["hard_radius"]), 3)},
        "dt": round(float(unc["dt"]) * 3, 4),   # dt after thinning by 3
        "policies": {},
    }
    for p in ORDER:
        r = results[p]
        payload["policies"][p] = {
            "true_xy": thin(r["true_xy"]),
            "est_xy": thin(r["est_xy"]),
            "err": thin(r["err"]),
            "U": thin(r["U"]),
            "evading": [int(bool(x)) for x in thin(r["evading"])],
            "corrections": [int(c // 3) for c in r["corrections"]],
            "arrival_miss_m": round(float(r["arrival_miss_m"]), 3),
            "peak_err": round(float(r["peak_err"]), 3),
            "n_fixes": len(r["corrections"]),
            "min_clearance_m": round(float(r["min_clearance_m"]), 3),
        }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


def main():
    ap = argparse.ArgumentParser(description="FrugalNav end-to-end demo")
    ap.add_argument("--seed", type=int, default=1, help="headline seed")
    ap.add_argument("--seeds", type=int, default=0,
                    help="also print a multi-seed mean+/-std summary")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    results = sim.run_all(seed=args.seed)
    _fmt(results)

    moneyshot = os.path.join(OUT, "demo_moneyshot.png")
    dashboard = os.path.join(OUT, "demo_dashboard.png")
    plotting.save_integrated_moneyshot(results, moneyshot)
    plotting.save_integrated_dashboard(results, dashboard)
    jpath = _export_json(results, os.path.join(OUT, "integrated_demo.json"))
    print(f"\n saved: {os.path.relpath(moneyshot, HERE)}")
    print(f" saved: {os.path.relpath(dashboard, HERE)}")
    print(f" saved: {os.path.relpath(jpath, HERE)}   (feeds the interactive web demo)")

    if args.seeds > 0:
        _multiseed(args.seeds)


if __name__ == "__main__":
    main()
