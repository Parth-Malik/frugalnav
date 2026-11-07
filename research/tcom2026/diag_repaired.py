"""diag_repaired.py -- trigger-tier breakdown of FrugalNav-R on held-out seeds."""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import frugalnav_hf as hf
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
_cam = None
def _init():
    global _cam
    hf.DETECT_MS = 36.4; hf.Q_QUAD, hf.Q_QUAL = 8e-4, 0.02; _cam = hf.CameraModel()
def _run(j):
    w, c, b, s, refr, tau = j
    world = [x for x in hf.make_worlds() if x.name == w][0]
    pol = hf.FixedTimer(refr) if b == 0 else hf.Repaired(b, tau, refr)
    r = hf.fly(world, c, pol, s, cam=_cam); r.pop("trace")
    r.update(refr=refr, tau_=tau); return r
if __name__ == "__main__":
    W = [w.name for w in hf.make_worlds()]
    # budget 0 = fixed timer with period `refr` seconds (reference curve)
    grid = [(0, 3.0, 0), (0, 5.0, 0), (0, 8.0, 0), (0, 12.0, 0),
            (0.8, 15, 0.45), (0.8, 15, 0.60), (0.8, 30, 0.60), (1.2, 15, 0.45),
            (1.0, 30, 0.45), (1.0, 20, 0.55), (1.2, 30, 0.70), (0.6, 30, 0.70)]
    jobs = [(w, c, b, s, r, t) for (b, r, t) in grid for w in W for c in hf.CONDITIONS for s in range(100, 110)]
    with ProcessPoolExecutor(initializer=_init) as ex:
        d = pd.DataFrame(ex.map(_run, jobs, chunksize=6))
    pd.set_option("display.width", 200)
    d.to_csv(os.path.join(RES, "repaired_diag.csv"), index=False)
    print(d.groupby(["budget", "refr", "tau_", "world"])[["rmse", "attempts", "trig_soft", "trig_sigma", "trig_obs"]].mean().round(2).to_string())
    print(d.groupby(["budget", "refr", "tau_"])[["rmse", "attempts"]].mean().round(3).to_string())
