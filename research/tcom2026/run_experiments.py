"""
run_experiments.py
------------------
Runs the full study on FrugalNav-HF and writes results/*.csv.

  main  : 3 policies x 3 worlds x 4 realism conditions x N seeds  (RQ1, RQ2, RQ4)
  sweep : adaptive threshold tau and fixed-timer period, all worlds, in the
          Ideal and Combined conditions                            (RQ3, matched cost)
"""
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import frugalnav_hf as hf

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
N_SEEDS = 30
TAUS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
PERIODS = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]

_cam = None
_worlds = None


def _init():
    global _cam, _worlds
    with open(os.path.join(RES, "detector_timing.json")) as f:
        hf.DETECT_MS = json.load(f)["median_ms"]
    _cam = hf.CameraModel()
    _worlds = {w.name: w for w in hf.make_worlds()}


def _run(job):
    world, cond, kind, param, seed, block = job
    if kind == "always":
        pol, tau = hf.AlwaysCorrect(), 0.45
    elif kind == "fixed":
        pol, tau = hf.FixedTimer(param), 0.45
    else:
        pol, tau = hf.Adaptive(), param
    r = hf.fly(_worlds[world], cond, pol, seed, tau=tau, cam=_cam)
    r.pop("trace")
    r["block"] = block
    return r


def jobs():
    names = [w.name for w in hf.make_worlds()]
    out = []
    for w in names:
        for c in hf.CONDITIONS:
            for s in range(N_SEEDS):
                out += [(w, c, "always", None, s, "main"),
                        (w, c, "fixed", 5.0, s, "main"),
                        (w, c, "adaptive", 0.45, s, "main")]
        for c in ("Ideal", "Combined"):
            for s in range(N_SEEDS):
                out += [(w, c, "adaptive", tau, s, "sweep") for tau in TAUS if tau != 0.45]
                out += [(w, c, "fixed", P, s, "sweep") for P in PERIODS if P != 5.0]
    return out


def main():
    js = jobs()
    print(f"{len(js)} flights")
    with ProcessPoolExecutor(initializer=_init) as ex:
        rows = list(ex.map(_run, js, chunksize=8))
    keys = list(rows[0].keys())
    with open(os.path.join(RES, "runs.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print("wrote results/runs.csv")


if __name__ == "__main__":
    main()
