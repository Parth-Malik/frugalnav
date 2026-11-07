"""
analyze.py -- tables, statistics and figures for the paper, from results/runs.csv.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import frugalnav_hf as hf

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})
# display names: the updated scheduler is FrugalNav v2; the first release is v1
LABEL = {"Always-correct": "Always-correct", "Fixed timer": "Fixed timer",
         "Adaptive (U)": "FrugalNav v1 (earlier)", "R-match": "FrugalNav v2", "R-frugal": "FrugalNav v2"}
COL = {"Always-correct": "#4C6EF5", "Fixed timer": "#868E96", "Adaptive (U)": "#B197FC",
       "R-match": "#E8590C", "R-frugal": "#E8590C"}
WORLDS = ["Open arena", "City block", "Rocky canyon"]
CONDS = ["Ideal", "Wind", "Degraded", "Combined"]
POLS = ["Always-correct", "Fixed timer", "Adaptive (U)"]

df = pd.read_csv(os.path.join(RES, "runs.csv"))
REP = os.path.join(RES, "runs_repaired.csv")
if os.path.exists(REP):
    df = pd.concat([df, pd.read_csv(REP)], ignore_index=True)
main = df[df.block == "main"]
POLS4 = ["Always-correct", "Fixed timer", "R-match", "Adaptive (U)"] if (main.policy == "R-match").any() else POLS
out = {}

# ---------------------------------------------------------------- Table: main
agg = (main.groupby(["world", "cond", "policy"])
       .agg(rmse=("rmse", "mean"), rmse_sd=("rmse", "std"), peak=("peak", "mean"),
            arr=("arrival_err", "mean"), att=("attempts", "mean"), fixes=("fixes", "mean"),
            bits=("bits_per_fix", "mean"), comp=("compute_share", "mean"),
            succ=("arrived", "mean"), nees=("nees", "mean"), energy=("energy_Wh", "mean"),
            time=("time_s", "mean"))
       .reset_index())
agg.to_csv(os.path.join(RES, "summary_main.csv"), index=False)

# paired tests adaptive vs fixed, and adaptive vs always, per cell
tests = []
for w in WORLDS:
    for c in CONDS:
        cell = main[(main.world == w) & (main.cond == c)].pivot(index="seed", columns="policy")
        for metric in ("rmse", "attempts"):
            a = cell[metric]["Adaptive (U)"]
            for other in ("Fixed timer", "Always-correct"):
                b = cell[metric][other]
                d = (a - b).values
                try:
                    pval = wilcoxon(a, b).pvalue
                except ValueError:
                    pval = 1.0
                tests.append(dict(world=w, cond=c, metric=metric, vs=other,
                                  mean_adaptive=a.mean(), mean_other=b.mean(),
                                  mean_diff=d.mean(), pct=100 * d.mean() / b.mean(), p=pval))
pd.DataFrame(tests).to_csv(os.path.join(RES, "paired_tests.csv"), index=False)

# pooled headline numbers
def pooled(pol, m, conds=CONDS):
    return main[(main.policy == pol) & main.cond.isin(conds)][m].mean()

for pol in POLS4 + (["R-frugal"] if (main.policy == "R-frugal").any() else []):
    out[pol] = {m: pooled(pol, m) for m in
                ("rmse", "peak", "arrival_err", "attempts", "fixes", "bits_per_fix",
                 "compute_share", "arrived", "nees", "energy_Wh")}
    out[pol]["rmse_ideal"] = pooled(pol, "rmse", ["Ideal"])
    out[pol]["rmse_comb"] = pooled(pol, "rmse", ["Combined"])
    out[pol]["att_ideal"] = pooled(pol, "attempts", ["Ideal"])
    out[pol]["att_comb"] = pooled(pol, "attempts", ["Combined"])
    out[pol]["bits_ideal"] = pooled(pol, "bits_per_fix", ["Ideal"])
    out[pol]["bits_comb"] = pooled(pol, "bits_per_fix", ["Combined"])

# ------------------------------------------------------------- Figure 2: bars
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharey="row")
x = np.arange(len(CONDS))
bw = 0.8 / len(POLS4)
for j, w in enumerate(WORLDS):
    for k, pol in enumerate(POLS4):
        s = agg[(agg.world == w) & (agg.policy == pol)].set_index("cond").loc[CONDS]
        off = (k - (len(POLS4) - 1) / 2) * bw
        axes[0, j].bar(x + off, s.rmse, bw, color=COL[pol], label=LABEL[pol])
        axes[1, j].bar(x + off, s.att, bw, color=COL[pol])
    axes[0, j].set_title(w, fontsize=9, fontweight="bold")
    axes[1, j].set_xticks(x, CONDS, rotation=30, fontsize=8)
    axes[0, j].set_xticks(x, [""] * 4)
axes[0, 0].set_ylabel("Position RMSE (m)")
axes[1, 0].set_ylabel("Detector invocations")
axes[1, 0].set_yscale("log")
axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_main.png"))
plt.close(fig)

# ----------------------------------------------------------- Figure 3: Pareto
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
pareto = []
for j, w in enumerate(WORLDS):
    d = df[(df.world == w) & (df.cond == "Combined")]
    fx = d[d.policy == "Fixed timer"].groupby("period")[["attempts", "rmse"]].mean().reset_index()
    ad = d[d.policy == "Adaptive (U)"].groupby("tau")[["attempts", "rmse"]].mean().reset_index()
    al = d[d.policy == "Always-correct"][["attempts", "rmse"]].mean()
    ax = axes[j]
    ax.plot(fx.attempts, fx.rmse, "o-", color=COL["Fixed timer"], ms=3.5, label="Fixed timer (T = 1-12 s)")
    ax.plot(ad.attempts, ad.rmse, "s-", color=COL["Adaptive (U)"], ms=3.5, label="FrugalNav v1, earlier (tau 0.30-0.60)")
    ax.plot([al.attempts], [al.rmse], "*", color=COL["Always-correct"], ms=9, label="Always-correct")
    rf = d[d.policy == "R-frugal"].groupby("budget")[["attempts", "rmse"]].mean().reset_index()
    if len(rf):
        ax.plot(rf.attempts, rf.rmse, "^-", color=COL["R-frugal"], ms=3.5, label="FrugalNav v2 (budget 0.6-1.6 m)")
        rm = d[d.policy == "R-match"][["attempts", "rmse"]].mean()
        ax.plot([rm.attempts], [rm.rmse], "D", color="#A61E4D", ms=5, label="FrugalNav v2, main setting")
        for _, r in rf.iterrows():
            pareto.append(dict(world=w, policy="R-frugal", param=r.budget, attempts=r.attempts, rmse=r.rmse))
        pareto.append(dict(world=w, policy="R-match", param=1.0, attempts=rm.attempts, rmse=rm.rmse))
    for _, r in ad[ad.tau.isin([0.30, 0.60] if w == "Rocky canyon" else [0.30, 0.45, 0.60])].iterrows():
        ax.annotate(f"tau {r.tau:.2f}", (r.attempts, r.rmse), fontsize=6, xytext=(4, -3),
                    textcoords="offset points", color=COL["Adaptive (U)"])
    ax.set_xscale("log")
    ax.set_title(w, fontsize=9, fontweight="bold")
    ax.set_xlabel("Detector invocations per flight")
    for _, r in ad.iterrows():
        pareto.append(dict(world=w, policy="adaptive", param=r.tau, attempts=r.attempts, rmse=r.rmse))
    for _, r in fx.iterrows():
        pareto.append(dict(world=w, policy="fixed", param=r.period, attempts=r.attempts, rmse=r.rmse))
    pareto.append(dict(world=w, policy="always", param=np.nan, attempts=al.attempts, rmse=al.rmse))
axes[0].set_ylabel("Position RMSE (m)")
axes[0].legend(frameon=False, fontsize=6.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_pareto.png"))
plt.close(fig)
pd.DataFrame(pareto).to_csv(os.path.join(RES, "pareto_combined.csv"), index=False)

# Ideal-condition sweep, too (for text)
pid = []
for w in WORLDS:
    d = df[(df.world == w) & (df.cond == "Ideal")]
    for pol, key in (("Fixed timer", "period"), ("Adaptive (U)", "tau")):
        g = d[d.policy == pol].groupby(key)[["attempts", "rmse"]].mean().reset_index()
        for _, r in g.iterrows():
            pid.append(dict(world=w, policy=pol, param=r[key], attempts=r.attempts, rmse=r.rmse))
pd.DataFrame(pid).to_csv(os.path.join(RES, "pareto_ideal.csv"), index=False)

# ------------------------------------------------------------- Figure 4: bits
fig, ax = plt.subplots(figsize=(4.6, 2.5))
for k, pol in enumerate(POLS4):
    m = [main[(main.policy == pol) & (main.cond == c)].bits_per_fix.mean() for c in CONDS]
    ax.bar(x + (k - (len(POLS4) - 1) / 2) * (0.8 / len(POLS4)), m, 0.8 / len(POLS4), color=COL[pol],
           label=LABEL[pol])
ax.set_xticks(x, CONDS)
ax.set_ylabel("Information per fix (bits)")
ax.set_ylim(0, 8.5)
ax.legend(frameon=False, fontsize=6.5, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_bits.png"))
plt.close(fig)

# ----------------------------------------------- Figure 1: example time series
cam = hf.CameraModel()
with open(os.path.join(RES, "detector_timing.json")) as f:
    timing = json.load(f)
hf.DETECT_MS = timing["median_ms"]
canyon = [w for w in hf.make_worlds() if w.name == "Rocky canyon"][0]
fig, axes = plt.subplots(3, 1, figsize=(6.6, 4.8), sharex=True)
v2 = hf.Repaired(1.0, 0.55, 20)
v2.name = "R-match"
for pol in (hf.AlwaysCorrect(), hf.FixedTimer(5.0), v2, hf.Adaptive()):
    r = hf.fly(canyon, "Combined", pol, seed=3, cam=cam, log=True)
    tr = r["trace"]
    axes[0].plot(tr[:, 0], tr[:, 10], color=COL[pol.name], lw=1.1, label=f"{LABEL[pol.name]} ({r['attempts']} invocations)")
    axes[1].plot(tr[:, 0], tr[:, 5], color=COL[pol.name], lw=1.1)
    if pol.name == "Adaptive (U)":
        axes[2].plot(tr[:, 0], tr[:, 6], color=COL[pol.name], lw=1.1, label="U (v1 run)")
        axes[2].plot(tr[:, 0], tr[:, 7], color="#2F9E44", lw=1.0, ls="--", label="sharpness")
        haze = tr[:, 8] < 4.5
        for ax in axes:
            ax.fill_between(tr[:, 0], 0, 1, where=haze, transform=ax.get_xaxis_transform(),
                            color="#ADB5BD", alpha=0.25, lw=0)
axes[2].axhline(0.45, color="k", lw=0.6, ls=":")
axes[0].set_ylabel("True error (m)")
axes[1].set_ylabel("Filter sigma (m)")
axes[2].set_ylabel("Cue value")
axes[2].set_xlabel("Time (s)   [grey band = haze patch]")
axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
axes[2].legend(frameon=False, fontsize=6.5, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_timeseries.png"))
plt.close(fig)

# ------------------------------------------ Figure A1: detection calibration
rows = np.load(os.path.join(RES, "camera_lut.npz"))["rows"]
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
for ax, alt in zip(axes, (3.0, 5.0)):
    for g, ls in ((1.0, "-"), (0.6, "--"), (0.3, ":")):
        m = (rows[:, 0] == alt) & (rows[:, 1] == 1.0) & (rows[:, 3] == 0.03) & (rows[:, 4] == g)
        r = rows[m]
        r = r[np.argsort(r[:, 2])]
        ax.plot(r[:, 2], r[:, 5], ls, color="#212529", label=f"light gain {g}")
    ax.set_title(f"Altitude {alt:.0f} m (lens blur 1 px, noise 0.03)", fontsize=8)
    ax.set_xlabel("Motion blur (px)")
axes[0].set_ylabel("ArUco detection rate")
axes[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "figA1_detection.png"))
plt.close(fig)

# raw-vs-compensated Laplacian (the noise confound)
m5 = (rows[:, 0] == 5.0) & (rows[:, 1] == 0) & (rows[:, 2] == 0) & (rows[:, 4] == 1.0)
out["lapvar_noise"] = {f"{r[3]:.2f}": dict(raw=r[7], compensated=r[8]) for r in rows[m5]}
mb = (rows[:, 0] == 5.0) & (rows[:, 1] == 1.0) & (rows[:, 3] == 0.0) & (rows[:, 4] == 1.0)
out["lapvar_motion"] = {f"{r[2]:.0f}": dict(raw=r[7], compensated=r[8], p_det=r[5]) for r in rows[mb]}
out["timing"] = timing
out["n_flights"] = int(len(df))
out["n_main"] = int(len(main))
with open(os.path.join(RES, "headline.json"), "w") as f:
    json.dump(out, f, indent=2, default=float)

pd.set_option("display.width", 200)
print(agg.round(3).to_string())
print(pd.DataFrame(tests).round(4).to_string())
print(json.dumps(out, indent=1, default=float))
