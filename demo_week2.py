"""
demo_week2.py
-------------
Rohan's Week-2 deliverable: the scheduler running on REAL-DATA-DERIVED drift (the
EuRoC drift scaffold), through the VioSource adapter, under three correction policies.

Produces:
  - a calibration report (real drift as % of distance -- grounds the sim in reality)
  - the policy comparison table (none / fixed / adaptive)
  - demo_week2.png : trajectory (GT vs drifting estimate), error-over-time, U(t)+triggers

Run on the built-in synthetic EuRoC-style trajectory (works anywhere, no download):
    python3 demo_week2.py
Run on a REAL EuRoC sequence once you've downloaded one (e.g. MH_01_easy):
    python3 demo_week2.py --euroc "D:/drone/datasets/MH_01_easy"
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


def synthetic_euroc():
    """A curved 30 s trajectory in EuRoC dict format, so the demo runs with no download."""
    n = 2000
    t = np.linspace(0, 30, n)
    ang = np.linspace(0, 1.4, n)
    x, y = 45 * np.cos(ang), 45 * np.sin(ang)
    gt = dict(t=t, x=x, y=y, z=np.zeros(n),
              vx=np.gradient(x, t), vy=np.gradient(y, t), vz=np.zeros(n),
              bgx=np.full(n, 0.003), bgy=np.full(n, 0.001), bgz=np.full(n, 0.002))
    # IMU with motion-correlated rotation (drives the blur/feature proxies)
    imu = dict(t=t, wx=0.3*np.sin(2*t), wy=np.zeros(n), wz=0.4*np.cos(1.5*t),
               ax=0.5*np.sin(t), ay=np.zeros(n), az=np.full(n, 9.81))
    return gt, imu


def run(policy, gt, imu, seed=1, fixed_period_s=2.5):
    src = EurocDriftSource(gt, imu, seed=seed, fix_every_m=6.0)
    sch = UncertaintyScheduler(SchedulerConfig(
        tau=TAU, refractory_ticks=REFRACTORY,
        weights={"sigma_pos": 0.6, "feature_loss": 0.2, "blur": 0.2,
                 "sigma_head": 0.0, "imu_bias": 0.0}))
    log = dict(t=[], err=[], U=[], trig_t=[], gt=[], est=[], corrections=0)
    last_fix_t = -1e9
    while True:
        sig = src.update()
        if sig is None:
            break
        U, trig, why, _ = sch.compute(sig.cues)
        want = (policy == "fixed" and (sig.t - last_fix_t) >= fixed_period_s) \
            or (policy == "adaptive" and trig)
        if want and src.apply_fix():
            sch.reset_after_fix(); last_fix_t = sig.t
            log["corrections"] += 1; log["trig_t"].append(sig.t)
        log["t"].append(sig.t); log["err"].append(sig.error()); log["U"].append(U)
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
        src_label = f"real EuRoC: {os.path.basename(a.euroc.rstrip('/'))}"
    else:
        gt, imu = synthetic_euroc()
        src_label = "built-in synthetic EuRoC-style trajectory"

    print(f"\nData source: {src_label}")
    print("Calibration (uncorrected drift on this sequence):")
    for k, v in calibrate_sim_noise(gt, imu, seed=1).items():
        print(f"   {k:28s}: {v}")

    results = {p: run(p, gt, imu) for p in ("none", "fixed", "adaptive")}
    print(f"\n{'policy':<10}{'corrections':>12}{'final_err(m)':>14}"
          f"{'peak_err(m)':>13}{'mean_err(m)':>13}")
    print("-" * 62)
    for p, lg in results.items():
        print(f"{p:<10}{lg['corrections']:>12}{lg['final_err']:>14.2f}"
              f"{lg['peak_err']:>13.2f}{lg['mean_err']:>13.2f}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    colors = {"none": "#d1495b", "fixed": "#edae49", "adaptive": "#2e7d32"}

    lg = results["adaptive"]
    ax[0].plot(lg["gt"][:, 0], lg["gt"][:, 1], "-", color="#1f77b4", lw=2, label="ground truth")
    ax[0].plot(lg["est"][:, 0], lg["est"][:, 1], "--", color=colors["adaptive"], lw=1.5,
               label="drifting estimate (adaptive)")
    ax[0].plot(0, 0, "g*", ms=18, label="target B")
    ax[0].set_title(f"Trajectory\n{src_label}"); ax[0].axis("equal")
    ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)"); ax[0].legend(fontsize=8)

    for p, lg in results.items():
        ax[1].plot(lg["t"], lg["err"], color=colors[p], lw=1.6,
                   label=f"{p} ({lg['corrections']} fixes)")
    ax[1].set_title("Estimate error vs time (VIO drift)")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("‖true − est‖ (m)"); ax[1].legend(fontsize=8)

    lg = results["adaptive"]
    ax[2].plot(lg["t"], lg["U"], color=colors["adaptive"], lw=1.2, label="U(t)")
    ax[2].axhline(TAU, ls=":", color="k", label=f"τ={TAU}")
    for tt in lg["trig_t"]:
        ax[2].axvline(tt, color="#d1495b", alpha=0.4, lw=1)
    ax[2].set_title("Uncertainty U(t) and triggers (adaptive)")
    ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("U  [0,1]"); ax[2].set_ylim(0, 1)
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "demo_week2.png")
    fig.savefig(out, dpi=130)
    print(f"\nFigure written to {out}")


if __name__ == "__main__":
    main()
