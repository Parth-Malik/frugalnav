"""
reasons_and_tests.py -- (1) re-flies the adaptive main-block runs (same seeds, so
identical flights) to log WHICH trigger tier fired; (2) paired matched-cost tests
between the adaptive policy and the fixed timer from runs.csv.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import run_experiments as rx

RES = rx.RES


def main():
    js = [j for j in rx.jobs() if j[5] == "main" and j[2] == "adaptive"]
    with ProcessPoolExecutor(initializer=rx._init) as ex:
        rows = list(ex.map(rx._run, js, chunksize=6))
    d = pd.DataFrame(rows)
    g = d.groupby(["world", "cond"])[["attempts", "trig_soft", "trig_sigma", "trig_obs", "rmse"]].mean()
    g.to_csv(os.path.join(RES, "trigger_reasons.csv"))
    print(g.round(2).to_string())

    df = pd.read_csv(os.path.join(RES, "runs.csv"))
    tests = []
    # matched-cost pairs chosen from the Combined sweep (closest mean invocations)
    pairs = [("Open arena", 0.30, 5.0), ("City block", 0.30, 5.0), ("City block", 0.35, 8.0),
             ("Rocky canyon", 0.45, 1.0)]
    for w, tau, P in pairs:
        c = df[(df.world == w) & (df.cond == "Combined")]
        a = c[(c.policy == "Adaptive (U)") & (c.tau == tau)].set_index("seed").sort_index()
        f = c[(c.policy == "Fixed timer") & (c.period == P)].set_index("seed").sort_index()
        diff = a.rmse - f.rmse
        boot = [np.mean(np.random.default_rng(i).choice(diff.values, len(diff))) for i in range(2000)]
        tests.append(dict(world=w, tau=tau, period=P, att_adaptive=a.attempts.mean(),
                          att_fixed=f.attempts.mean(), rmse_adaptive=a.rmse.mean(),
                          rmse_fixed=f.rmse.mean(), diff=diff.mean(),
                          ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5),
                          p=wilcoxon(a.rmse, f.rmse).pvalue))
    # always-correct vs best fixed timer in Combined
    for w, P in (("City block", 2.0), ("Rocky canyon", 3.0), ("Open arena", 1.0)):
        c = df[(df.world == w) & (df.cond == "Combined")]
        a = c[c.policy == "Always-correct"].set_index("seed").sort_index()
        f = c[(c.policy == "Fixed timer") & (c.period == P)].set_index("seed").sort_index()
        diff = a.rmse - f.rmse
        tests.append(dict(world=w, tau="always", period=P, att_adaptive=a.attempts.mean(),
                          att_fixed=f.attempts.mean(), rmse_adaptive=a.rmse.mean(),
                          rmse_fixed=f.rmse.mean(), diff=diff.mean(),
                          ci_lo=np.nan, ci_hi=np.nan, p=wilcoxon(a.rmse, f.rmse).pvalue))
    # median RMSE too (robust to the wind outliers)
    med = df[df.block == "main"].groupby(["world", "cond", "policy"]).rmse.median().unstack()
    med.to_csv(os.path.join(RES, "median_rmse.csv"))
    print(med.round(3).to_string())
    t = pd.DataFrame(tests)
    t.to_csv(os.path.join(RES, "matched_tests.csv"), index=False)
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()
