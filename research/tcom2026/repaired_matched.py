"""repaired_matched.py -- paired tests: repaired (R-frugal family) vs the fixed timer
at the nearest cost, Combined condition, test seeds 0-29."""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
a = pd.read_csv(os.path.join(RES, "runs.csv"))
r = pd.read_csv(os.path.join(RES, "runs_repaired.csv"))
pairs = [("Open arena", 0.8, 8.0), ("Open arena", 0.6, 3.0), ("City block", 0.8, 5.0),
         ("City block", 1.0, 12.0), ("Rocky canyon", 0.8, 5.0), ("Rocky canyon", 1.0, 8.0)]
rows = []
for w, b, P in pairs:
    x = r[(r.world == w) & (r.cond == "Combined") & (r.policy == "R-frugal") & (r.budget == b)].set_index("seed").sort_index()
    f = a[(a.world == w) & (a.cond == "Combined") & (a.policy == "Fixed timer") & (a.period == P)].set_index("seed").sort_index()
    d = (x.rmse - f.rmse).values
    boot = [np.mean(np.random.default_rng(i).choice(d, len(d))) for i in range(2000)]
    rows.append(dict(world=w, budget=b, period=P, calls_r=x.attempts.mean(), calls_f=f.attempts.mean(),
                     rmse_r=x.rmse.mean(), rmse_f=f.rmse.mean(), diff=d.mean(),
                     ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5),
                     p=wilcoxon(x.rmse, f.rmse).pvalue))
t = pd.DataFrame(rows)
t.to_csv(os.path.join(RES, "repaired_matched.csv"), index=False)
print(t.round(3).to_string())
