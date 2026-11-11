"""
tune_repaired.py -- calibrates FrugalNav-R on HELD-OUT seeds 100-119 only
(the evaluation uses seeds 0-29, as for every other policy).

  step 1: process-noise coefficients (Q_QUAD, Q_QUAL) so that a never-correcting
          filter is consistent (mean NEES ~ 2 for a 2-D state);
  step 2: the uncertainty budget whose mean RMSE matches the 5 s fixed timer.
"""
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import frugalnav_hf as hf

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SEEDS = range(100, 120)
_cam = None


def _init():
    global _cam
    hf.DETECT_MS = json.load(open(os.path.join(RES, "detector_timing.json")))["median_ms"]
    _cam = hf.CameraModel()


def _run(job):
    qq, ql, kind, param, w, c, s = job
    hf.Q_QUAD, hf.Q_QUAL = qq, ql
    world = [x for x in hf.make_worlds() if x.name == w][0]
    pol = {"none": lambda: hf.NoCorrection(), "rep": lambda: hf.Repaired(param),
           "fixed": lambda: hf.FixedTimer(5.0)}[kind]()
    r = hf.fly(world, c, pol, s, cam=_cam)
    r.pop("trace")
    r.update(qq=qq, ql=ql, kind=kind)
    return r


def main():
    worlds = [w.name for w in hf.make_worlds()]
    conds = list(hf.CONDITIONS)
    with ProcessPoolExecutor(initializer=_init) as ex:
        grid = list(itertools.product([2e-4, 4e-4, 8e-4, 1.6e-3], [0.02, 0.08]))
        jobs = [(qq, ql, "none", None, w, c, s) for qq, ql in grid
                for w in worlds for c in conds for s in SEEDS[:10]]
        d = pd.DataFrame(ex.map(_run, jobs, chunksize=8))
        cal = d.groupby(["qq", "ql"]).nees.agg(["mean", "median"]).reset_index()
        print(cal.to_string())
        # choose the pair whose mean NEES is closest to 2 (log distance)
        best = cal.iloc[(np.log(cal["mean"] / 2.0)).abs().argmin()]
        qq, ql = float(best.qq), float(best.ql)
        print("chosen Q_QUAD, Q_QUAL =", qq, ql)

        budgets = [0.3, 0.4, 0.5, 0.6, 0.8]
        jobs = [(qq, ql, "rep", b, w, c, s) for b in budgets for w in worlds for c in conds for s in SEEDS]
        jobs += [(qq, ql, "fixed", None, w, c, s) for w in worlds for c in conds for s in SEEDS]
        d2 = pd.DataFrame(ex.map(_run, jobs, chunksize=8))
    tab = d2.groupby(["kind", "budget"], dropna=False)[["rmse", "attempts", "nees"]].mean()
    print(tab.round(3).to_string())
    fixed_rmse = d2[d2.kind == "fixed"].rmse.mean()
    reps = d2[d2.kind == "rep"].groupby("budget")[["rmse", "attempts"]].mean().reset_index()
    ok = reps[reps.rmse <= fixed_rmse * 1.10]           # "almost as accurate": within 10%
    chosen = float(ok.budget.max()) if len(ok) else float(reps.loc[reps.rmse.idxmin(), "budget"])
    out = dict(Q_QUAD=qq, Q_QUAL=ql, budget=chosen, fixed_rmse_tuning=fixed_rmse,
               calibration=cal.to_dict("records"), budget_table=reps.to_dict("records"))
    json.dump(out, open(os.path.join(RES, "repaired_tuning.json"), "w"), indent=2, default=float)
    print("chosen budget:", chosen)


if __name__ == "__main__":
    main()
