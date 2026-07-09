"""
demo_week4.py
-------------
Rohan's Week-4 deliverable + MID-TERM demo.

This graduates the uncertainty metric from the Week-1 starter (sigma_pos + feature_loss)
to the FULL 5-cue U:
    U = w1*n(sigma_pos) + w2*n(sigma_head) + w3*n(blur) + w4*n(feature_loss) + w5*n(imu_bias)
with the cue bounds and weights tuned to the REAL EuRoC MH_01 signal ranges.

It produces three things the mid-term needs:
  1. The policy comparison the plan asks for:  none / fixed-period / uncertainty-aware.
  2. An ABLATION that justifies the contribution: a covariance-ONLY scheduler vs the
     full MULTI-CUE U, at the same threshold -- does adding leading-indicator cues help?
  3. A GLASS-BOX figure: U(t) broken into its per-cue contributions, so you can see
     WHICH signal drives each correction (this is the interpretability story).

Run on the built-in synthetic trajectory (no download needed):
    python demo_week4.py
Run on real EuRoC (what the team uses):
    python demo_week4.py --euroc "D:/drone/datasets/MH_01_easy"
"""
import sys, os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from core.uncertainty_scheduler import UncertaintyScheduler, SchedulerConfig
from harness.drift_scaffold import EurocDriftSource, calibrate_sim_noise

TAU, REFRACTORY = 0.5, 60
CUES = ("sigma_pos", "sigma_head", "blur", "feature_loss", "imu_bias")
CUE_COLORS = {"sigma_pos": "#1f77b4", "sigma_head": "#ff7f0e", "blur": "#2ca02c",
              "feature_loss": "#9467bd", "imu_bias": "#8c564b"}

# The full 5-cue weighting (the Week-4 contribution) and the naive covariance-only baseline.
MULTI_CUE = {"sigma_pos": 0.45, "sigma_head": 0.20, "blur": 0.20,
             "feature_loss": 0.10, "imu_bias": 0.05}
COV_ONLY = {"sigma_pos": 1.0, "sigma_head": 0.0, "blur": 0.0,
            "feature_loss": 0.0, "imu_bias": 0.0}


def synthetic_euroc():
    n = 2000
    t = np.linspace(0, 30, n)
    ang = np.linspace(0, 1.4, n)
    x, y = 45 * np.cos(ang), 45 * np.sin(ang)
    gt = dict(t=t, x=x, y=y, z=np.zeros(n),
              vx=np.gradient(x, t), vy=np.gradient(y, t), vz=np.zeros(n),
              bgx=np.full(n, 0.003), bgy=np.full(n, 0.001), bgz=np.full(n, 0.002))
    imu = dict(t=t, wx=0.3*np.sin(2*t), wy=np.zeros(n), wz=0.4*np.cos(1.5*t),
               ax=0.5*np.sin(t), ay=np.zeros(n), az=np.full(n, 9.81))
    return gt, imu


def run(policy, gt, imu, weights, seed=1, fixed_period_s=2.5):
    """Run one policy with one weighting. Logs U and its per-cue contributions."""
    src = EurocDriftSource(gt, imu, seed=seed, fix_every_m=6.0)
    cfg = SchedulerConfig(tau=TAU, refractory_ticks=REFRACTORY, weights=dict(weights))
    sch = UncertaintyScheduler(cfg)
    log = dict(t=[], err=[], U=[], comp={k: [] for k in CUES},
               trig_t=[], gt=[], est=[], corrections=0)
    last_fix_t = -1e9
    while True:
        sig = src.update()
        if sig is None:
            break
        U, trig, why, comp = sch.compute(sig.cues)
        want = (policy == "fixed" and (sig.t - last_fix_t) >= fixed_period_s) \
            or (policy == "adaptive" and trig)
        if want and src.apply_fix():
            sch.reset_after_fix(); last_fix_t = sig.t
            log["corrections"] += 1; log["trig_t"].append(sig.t)
        log["t"].append(sig.t); log["err"].append(sig.error()); log["U"].append(U)
        for k in CUES:                                   # weighted contribution of each cue
            log["comp"][k].append(cfg.weights[k] * comp[k])
        log["gt"].append(sig.gt); log["est"].append(sig.est)
    log["gt"] = np.array(log["gt"]); log["est"] = np.array(log["est"])
    log["final_err"] = log["err"][-1]; log["peak_err"] = max(log["err"])
    log["mean_err"] = float(np.mean(log["err"]))
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--euroc", help="path to a real EuRoC sequence folder")
    a = ap.parse_args()

    if a.euroc:
        from harness.euroc_reader import EurocReader
        r = EurocReader(a.euroc)
        gt, imu = r.ground_truth(), r.imu()
        src_label = f"real EuRoC: {os.path.basename(a.euroc.rstrip('/\\'))}"
    else:
        gt, imu = synthetic_euroc()
        src_label = "built-in synthetic EuRoC-style trajectory"

    print(f"\nData source: {src_label}")
    print("Calibration (uncorrected drift on this sequence):")
    for k, v in calibrate_sim_noise(gt, imu, seed=1).items():
        print(f"   {k:28s}: {v}")

    # ---- 1. policy comparison (none / fixed / adaptive) on the full 5-cue U ----
    results = {p: run(p, gt, imu, MULTI_CUE) for p in ("none", "fixed", "adaptive")}
    print(f"\n[Policy comparison]  full 5-cue U")
    print(f"{'policy':<10}{'corrections':>12}{'final_err(m)':>14}"
          f"{'peak_err(m)':>13}{'mean_err(m)':>13}")
    print("-" * 62)
    for p, lg in results.items():
        print(f"{p:<10}{lg['corrections']:>12}{lg['final_err']:>14.2f}"
              f"{lg['peak_err']:>13.2f}{lg['mean_err']:>13.2f}")

    # ---- 2. ablation: covariance-only U vs multi-cue U (adaptive policy) ----
    abl = {"covariance-only": run("adaptive", gt, imu, COV_ONLY),
           "multi-cue (5)": run("adaptive", gt, imu, MULTI_CUE)}
    print(f"\n[Ablation]  adaptive policy, same tau={TAU}: does adding cues help?")
    print(f"{'U variant':<18}{'corrections':>12}{'final_err(m)':>14}{'peak_err(m)':>13}")
    print("-" * 57)
    for name, lg in abl.items():
        print(f"{name:<18}{lg['corrections']:>12}{lg['final_err']:>14.2f}{lg['peak_err']:>13.2f}")

    # ---- 3. figure ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    colors = {"none": "#d1495b", "fixed": "#edae49", "adaptive": "#2e7d32"}

    lg = results["adaptive"]
    ax[0].plot(lg["gt"][:, 0], lg["gt"][:, 1], "-", color="#1f77b4", lw=2, label="ground truth")
    ax[0].plot(lg["est"][:, 0], lg["est"][:, 1], "--", color=colors["adaptive"], lw=1.5,
               label="estimate (adaptive, 5-cue)")
    ax[0].plot(0, 0, "g*", ms=18, label="target B")
    ax[0].set_title(f"Trajectory\n{src_label}"); ax[0].axis("equal")
    ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)"); ax[0].legend(fontsize=8)

    for p, lg in results.items():
        ax[1].plot(lg["t"], lg["err"], color=colors[p], lw=1.6,
                   label=f"{p} ({lg['corrections']} fixes)")
    ax[1].set_title("Estimate error vs time"); ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("‖true − est‖ (m)"); ax[1].legend(fontsize=8)

    # glass-box: stacked per-cue contribution to U, with tau and triggers
    lg = results["adaptive"]
    t = np.array(lg["t"])
    stack = np.vstack([np.array(lg["comp"][k]) for k in CUES])
    ax[2].stackplot(t, stack, labels=[k for k in CUES],
                    colors=[CUE_COLORS[k] for k in CUES], alpha=0.85)
    ax[2].axhline(TAU, ls=":", color="k", lw=1.2, label=f"τ={TAU}")
    for tt in lg["trig_t"]:
        ax[2].axvline(tt, color="k", alpha=0.25, lw=0.8)
    ax[2].set_title("Glass-box U(t): per-cue contribution (adaptive)")
    ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("U  [0,1]"); ax[2].set_ylim(0, 1)
    ax[2].legend(fontsize=7, loc="upper right", ncol=2)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "demo_week4.png")
    fig.savefig(out, dpi=130)
    print(f"\nFigure written to {out}")


if __name__ == "__main__":
    main()
