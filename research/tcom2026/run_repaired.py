"""
run_repaired.py -- evaluates FrugalNav-R on the SAME test seeds (0-29) as the
other policies, with coefficients and budget fixed beforehand by tune_repaired.py.

  main  : all worlds x all conditions at the tuned budget
  sweep : budgets 0.3-0.8 m, Ideal and Combined, for the cost-accuracy curve
Writes results/runs_repaired.csv and results/repaired_tests.csv.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import frugalnav_hf as hf

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TUNE = json.load(open(os.path.join(RES, "repaired_tuning.json")))
CFG = TUNE["configs"]
SWEEP_BUDGETS = [0.6, 0.8, 1.0, 1.6]          # R-frugal family (tau 0.70, 3 s cooldown)
_cam = None


def _init():
    global _cam
    hf.DETECT_MS = json.load(open(os.path.join(RES, "detector_timing.json")))["median_ms"]
    hf.Q_QUAD, hf.Q_QUAL = TUNE["Q_QUAD"], TUNE["Q_QUAL"]
    _cam = hf.CameraModel()


def _run(job):
    w, c, name, b, tau, refr, s, block = job
    world = [x for x in hf.make_worlds() if x.name == w][0]
    pol = hf.Repaired(b, tau, refr)
    pol.name = name
    r = hf.fly(world, c, pol, s, cam=_cam)
    r.pop("trace")
    r["block"] = block
    return r


def main():
    worlds = [w.name for w in hf.make_worlds()]
    jobs = [(w, c, n, k["budget"], k["tau"], k["refractory"], s, "main")
            for n, k in CFG.items() for w in worlds for c in hf.CONDITIONS for s in range(30)]
    jobs += [(w, c, "R-frugal", b, 0.70, 30, s, "sweep") for w in worlds for c in ("Ideal", "Combined")
             for b in SWEEP_BUDGETS for s in range(30)]
    with ProcessPoolExecutor(initializer=_init) as ex:
        d = pd.DataFrame(ex.map(_run, jobs, chunksize=6))
    d.to_csv(os.path.join(RES, "runs_repaired.csv"), index=False)

    base = pd.read_csv(os.path.join(RES, "runs.csv"))
    base = base[base.block == "main"]
    allrep = d[d.block == "main"]
    rows = []
    for name in CFG:
      rep = allrep[allrep.policy == name]
      for w in worlds:
        for c in hf.CONDITIONS:
            r = rep[(rep.world == w) & (rep.cond == c)].set_index("seed").sort_index()
            for other in ("Fixed timer", "Adaptive (U)", "Always-correct"):
                o = base[(base.world == w) & (base.cond == c) & (base.policy == other)].set_index("seed").sort_index()
                rows.append(dict(config=name, world=w, cond=c, vs=other,
                                 rmse_rep=r.rmse.mean(), rmse_other=o.rmse.mean(),
                                 med_rep=r.rmse.median(), med_other=o.rmse.median(),
                                 att_rep=r.attempts.mean(), att_other=o.attempts.mean(),
                                 p_rmse=wilcoxon(r.rmse, o.rmse).pvalue,
                                 p_att=wilcoxon(r.attempts, o.attempts).pvalue))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, "repaired_tests.csv"), index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string())
    print(allrep.groupby(["policy", "cond"])[["rmse", "attempts", "nees", "bits_per_fix", "arrived"]].mean().round(3))
    print(allrep.groupby("policy")[["rmse", "attempts", "nees", "bits_per_fix", "compute_share", "trig_soft", "trig_sigma", "trig_obs"]].mean().round(3))


if __name__ == "__main__":
    main()
