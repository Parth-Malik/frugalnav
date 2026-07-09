"""
demo.py  --  FrugalNav live demonstration.

One end-to-end run of the whole navigation system, staged for a live walkthrough:

    Stage 1  THE PROBLEM      GPS-denied VIO drifts without bound.
    Stage 2  THE CORRECTION   Landmark fixes fused into the estimate snap it back to truth.
    Stage 3  THE CONTRIBUTION Uncertainty-aware scheduling: correct only when needed.
    Stage 4  THE GLASS BOX    Watch the confidence signal rise, trigger, and reset.

Every policy runs on the SAME trajectory and the SAME drift, so the only thing that
changes between them is WHEN a correction is spent. Produces one dashboard figure.

    python demo.py               # built-in circular trajectory
    python demo.py --pause       # wait for <Enter> between stages (for presenting)
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.se3 import make_se3
from core.interfaces import LandmarkFix
from core.state_fusion import StateFusion
from core.uncertainty_scheduler import UncertaintyScheduler
from core.scheduler_bridge import cues_from_pipeline, pipeline_scheduler_config
from harness.drift_injection import PoseDriftAdapter

BLUE, RED, GREEN, GREY = "#1f77b4", "#d1495b", "#2e7d32", "#888888"


def banner(txt):
    print("\n" + "=" * 68 + f"\n  {txt}\n" + "=" * 68)


def make_trajectory(n=600, duration=60.0):
    t = np.linspace(0, duration, n)
    gt = [make_se3(np.eye(3), [10 * np.cos(ti * 0.1), 10 * np.sin(ti * 0.1), 0.0]) for ti in t]
    return t, gt


def _fix(gt_pose, ts, rng):
    pose = gt_pose.copy()
    pose[:3, 3] += rng.normal(0, 0.02, 3)
    return LandmarkFix(valid=True, timestamp=ts, marker_id=1, pose_world=pose, pos_std_m=0.05)


def run(policy, t, gt, period_s=8.0, seed=1):
    """policy in {'none','fixed','uncertainty'}. Returns a log dict."""
    adapter = PoseDriftAdapter(pos_bias=(0.05, 0.02, 0.0), random_walk_std=0.02, seed=seed)
    fusion = StateFusion(); fusion.reset(initial_pose=gt[0])
    sched = UncertaintyScheduler(pipeline_scheduler_config()) if policy == "uncertainty" else None
    rng = np.random.default_rng(123)

    L = dict(t=[], est=[], err=[], U=[], fix_t=[], fix_xy=[], reasons=[], count=0)
    last_fixed = -1e9
    for ti, gp in zip(t, gt):
        vio = adapter.update(ti, gp)
        fusion.predict(vio)

        U, trig, reason = 0.0, False, "none"
        if policy == "fixed":
            trig = (ti - last_fixed) >= period_s
        elif policy == "uncertainty":
            U, trig, reason, _ = sched.compute(cues_from_pipeline(vio, fusion.estimate.pos_std_m))

        if trig:
            fusion.correct(_fix(gp, ti, rng))
            if sched is not None:
                sched.reset_after_fix()
            last_fixed = ti
            L["count"] += 1
            L["fix_t"].append(ti); L["fix_xy"].append(fusion.estimate.pose_world[:3, 3].copy())
            L["reasons"].append(reason)

        xy = fusion.estimate.pose_world[:3, 3]
        L["t"].append(ti); L["est"].append(xy.copy())
        L["err"].append(float(np.linalg.norm(xy - gp[:3, 3]))); L["U"].append(U)
    for k in ("t", "est", "err", "U"):
        L[k] = np.array(L[k])
    L["fix_xy"] = np.array(L["fix_xy"]) if L["fix_xy"] else np.empty((0, 3))
    L["mean_err"], L["final_err"] = float(L["err"].mean()), float(L["err"][-1])
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause", action="store_true", help="wait for Enter between stages")
    a = ap.parse_args()
    step = (lambda: input("\n   [Enter] to continue...")) if a.pause else (lambda: None)

    print("\n" + "*" * 68)
    print("*  FrugalNav  --  Frugal GPS-Denied UAV Navigation".ljust(67) + "*")
    print("*  Uncertainty-aware landmark scheduling for ultra-low-power RISC-V".ljust(67) + "*")
    print("*" * 68)

    t, gt = make_trajectory()
    gt_xy = np.array([p[:3, 3] for p in gt])
    tau = pipeline_scheduler_config().tau

    banner("STAGE 1  --  THE PROBLEM:  GPS-denied VIO drifts")
    none = run("none", t, gt)
    print(f"   With NO correction, the estimate drifts away from ground truth.")
    print(f"   -> final drift: {none['final_err']:.2f} m   (mean {none['mean_err']:.2f} m)")
    step()

    banner("STAGE 2  --  THE CORRECTION:  landmark fixes snap it back")
    fixed = run("fixed", t, gt, period_s=8.0)
    print(f"   Fusing an absolute landmark fix on a fixed timer bounds the drift.")
    print(f"   -> corrections: {fixed['count']:2d}   final err: {fixed['final_err']:.2f} m"
          f"   mean err: {fixed['mean_err']:.2f} m")
    step()

    banner("STAGE 3  --  THE CONTRIBUTION:  correct only when uncertain")
    unc = run("uncertainty", t, gt)
    saved = fixed["count"] - unc["count"]
    pct = 100.0 * saved / max(fixed["count"], 1)
    print(f"   The scheduler spends a fix ONLY when confidence U crosses tau={tau:.2f}.")
    print(f"   -> corrections: {unc['count']:2d}   final err: {unc['final_err']:.2f} m"
          f"   mean err: {unc['mean_err']:.2f} m")
    print(f"\n   RESULT:  {unc['count']} corrections vs {fixed['count']} "
          f"({pct:.0f}% fewer) at comparable accuracy.")
    print(f"            Every correction saved is compute and power saved.")
    step()

    banner("STAGE 4  --  THE GLASS BOX:  why each correction fired")
    from collections import Counter
    reasons = Counter(unc["reasons"])
    for r, c in reasons.items():
        print(f"   {c:2d} x  {r}")
    print("   -> the decision is interpretable: each fix is traced to a named cue.")

    # ---------------- dashboard ----------------
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("FrugalNav  --  Integrated System Demonstration", fontsize=15, fontweight="bold")

    a0 = ax[0, 0]
    a0.plot(gt_xy[:, 0], gt_xy[:, 1], "k--", lw=1.5, label="Ground truth")
    a0.plot(none["est"][:, 0], none["est"][:, 1], color=RED, alpha=0.6, label="No correction (drifts)")
    a0.plot(unc["est"][:, 0], unc["est"][:, 1], color=GREEN, lw=2, label="Uncertainty-aware (fused)")
    if len(unc["fix_xy"]):
        a0.scatter(unc["fix_xy"][:, 0], unc["fix_xy"][:, 1], c=GREEN, marker="x", s=70, zorder=5, label="AVL fix")
    a0.set_title("1. Trajectory: drift vs corrected"); a0.set_xlabel("X (m)"); a0.set_ylabel("Y (m)")
    a0.axis("equal"); a0.grid(alpha=0.3); a0.legend(fontsize=8)

    a1 = ax[0, 1]
    a1.plot(none["t"], none["err"], color=RED, label=f"none ({none['count']} fixes)")
    a1.plot(fixed["t"], fixed["err"], color=BLUE, label=f"fixed ({fixed['count']} fixes)")
    a1.plot(unc["t"], unc["err"], color=GREEN, lw=2, label=f"uncertainty ({unc['count']} fixes)")
    a1.set_title("2. Position error vs time"); a1.set_xlabel("time (s)"); a1.set_ylabel("error (m)")
    a1.grid(alpha=0.3); a1.legend(fontsize=8)

    a2 = ax[1, 0]
    names = ["fixed", "uncertainty"]
    counts = [fixed["count"], unc["count"]]
    errs = [fixed["mean_err"], unc["mean_err"]]
    x = np.arange(2); a2b = a2.twinx()
    b1 = a2.bar(x - 0.2, counts, 0.4, color=BLUE, label="corrections")
    b2 = a2b.bar(x + 0.2, errs, 0.4, color=GREEN, label="mean error (m)")
    a2.set_xticks(x); a2.set_xticklabels(["Fixed-period", "Uncertainty-aware"])
    a2.set_ylabel("corrections spent"); a2b.set_ylabel("mean error (m)")
    a2.set_title("3. Frugality: corrections vs accuracy")
    a2.bar_label(b1); a2b.bar_label(b2, fmt="%.2f")

    a3 = ax[1, 1]
    a3.plot(unc["t"], unc["U"], color="#6a3d9a", lw=1.5, label="confidence signal U(t)")
    a3.axhline(tau, ls=":", color="k", label=f"threshold tau={tau:.2f}")
    for ft in unc["fix_t"]:
        a3.axvline(ft, color=GREEN, alpha=0.35, lw=1)
    a3.set_title("4. Glass box: U rises, triggers a fix, resets"); a3.set_xlabel("time (s)")
    a3.set_ylabel("U  [0, 1]"); a3.set_ylim(0, 1); a3.grid(alpha=0.3); a3.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("demo_dashboard.png", dpi=140)
    banner("DONE  --  dashboard written to  demo_dashboard.png")
    print("   Open it to show all four panels together.\n")


if __name__ == "__main__":
    main()
