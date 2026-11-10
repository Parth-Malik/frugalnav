"""
poster_figs.py -- large-type versions of two figures for the A-size poster:
  poster_tradeoff.png : cost vs accuracy, Combined condition (from results/)
  poster_flight.png   : true error in one canyon flight (seed 3, as in the paper)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import frugalnav_hf as hf

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})
C_T, C_V2, C_V1, C_A = "#6B7280", "#E8590C", "#A78BFA", "#2563EB"

# ---- cost vs accuracy
p = pd.read_csv(os.path.join(RES, "pareto_combined.csv"))
fig, axes = plt.subplots(1, 3, figsize=(10, 4.2), sharey=True)
for ax, w in zip(axes, ["Open arena", "City block", "Rocky canyon"]):
    d = p[p.world == w]
    f = d[d.policy == "fixed"].sort_values("attempts")
    v2 = d[d.policy == "R-frugal"].sort_values("attempts")
    ax.plot(f.attempts, f.rmse, "o-", color=C_T, lw=2.4, ms=6, label="Fixed timer")
    ax.plot(v2.attempts, v2.rmse, "^-", color=C_V2, lw=2.8, ms=7, label="FrugalNav v2")
    ax.set_title(w)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 2.0)
    ax.set_xlabel("Marker reads per flight")
axes[0].set_ylabel("Position error, RMSE (m)")
axes[0].legend(frameon=False, fontsize=12, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "poster_tradeoff.png"), dpi=220)
plt.close(fig)

# ---- one canyon flight
cam = hf.CameraModel()
hf.DETECT_MS = json.load(open(os.path.join(RES, "detector_timing.json")))["median_ms"]
canyon = [w for w in hf.make_worlds() if w.name == "Rocky canyon"][0]
v2 = hf.Repaired(1.0, 0.55, 20)
fig, ax = plt.subplots(figsize=(7, 4.1))
haze = None
for pol, col, lab, lw in ((hf.FixedTimer(5.0), C_T, "Fixed timer", 2.2),
                          (hf.Adaptive(), C_V1, "First release (v1)", 2.2),
                          (v2, C_V2, "FrugalNav v2", 2.8)):
    r = hf.fly(canyon, "Combined", pol, seed=3, cam=cam, log=True)
    tr = r["trace"]
    ax.plot(tr[:, 0], tr[:, 10], color=col, lw=lw, label=f"{lab}: {r['attempts']} reads")
    haze = tr[:, 8] < 4.5 if haze is None else haze
    t = tr[:, 0]
ax.fill_between(t, 0, 1, where=haze[:len(t)], transform=ax.get_xaxis_transform(),
                color="#9CA3AF", alpha=0.25, lw=0)
ax.text(t[haze[:len(t)]].mean(), 1.36, "haze patch", ha="center", color="#4B5563", fontsize=12)
ax.set_ylim(0, 1.45)
ax.set_xlabel("Time (s)")
ax.set_ylabel("True position error (m)")
ax.legend(frameon=False, fontsize=12, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
          handlelength=1.4, columnspacing=1.0)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "poster_flight.png"), dpi=220, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)
print("ok")
