"""Week 4 mid-term deliverable: none vs fixed-period vs uncertainty-aware."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.se3 import make_se3
from core.interfaces import LandmarkFix
from core.state_fusion import StateFusion
from core.uncertainty_scheduler import UncertaintyScheduler, SchedulerConfig
from harness.drift_injection import PoseDriftAdapter


def make_trajectory(n=600, duration=60.0):
    t = np.linspace(0, duration, n)
    gt = [make_se3(np.eye(3), [10 * np.cos(ti * 0.1), 10 * np.sin(ti * 0.1), 0.0]) for ti in t]
    return t, gt


def _fix_at(gt_pose, timestamp, rng, noise_m=0.02, pos_std_m=0.05):
    pose = gt_pose.copy()
    pose[:3, 3] += rng.normal(0, noise_m, 3)
    return LandmarkFix(valid=True, timestamp=timestamp, marker_id=1,
                       pose_world=pose, pos_std_m=pos_std_m)


def run_policy(t, gt, policy, period_s=8.0, cfg=None, seed=1):
    adapter = PoseDriftAdapter(pos_bias=(0.05, 0.02, 0.0), random_walk_std=0.02, seed=seed)
    fusion = StateFusion()
    fusion.reset(initial_pose=gt[0])
    scheduler = UncertaintyScheduler(cfg) if policy == "uncertainty" else None
    fix_rng = np.random.default_rng(123)

    traj, errors, fix_pts = [], [], []
    last_fixed_time = -1e9
    count = 0

    for ti, gt_p in zip(t, gt):
        vio_out = adapter.update(ti, gt_p)
        fusion.predict(vio_out)

        do_fix = False
        if policy == "fixed":
            if ti - last_fixed_time >= period_s:
                do_fix = True
                last_fixed_time = ti
        elif policy == "uncertainty":
            if scheduler.should_correct(fusion.estimate.pos_std_m, vio_out):
                do_fix = True

        if do_fix:
            fusion.correct(_fix_at(gt_p, ti, fix_rng))
            count += 1
            fix_pts.append(fusion.estimate.pose_world[:3, 3].copy())

        est_xy = fusion.estimate.pose_world[:3, 3]
        traj.append(est_xy.copy())
        errors.append(float(np.linalg.norm(est_xy - gt_p[:3, 3])))

    return np.array(traj), np.array(errors), count, np.array(fix_pts) if fix_pts else np.empty((0, 3))


def run_comparison():
    t, gt = make_trajectory()
    gt_xy = np.array([p[:3, 3] for p in gt])
    cfg = SchedulerConfig(a_sigma=1.0, a_feat=0.5, a_blur=0.2, a_bias=0.1, threshold=0.35)

    results = {
        "none": run_policy(t, gt, "none"),
        "fixed": run_policy(t, gt, "fixed", period_s=8.0),
        "uncertainty": run_policy(t, gt, "uncertainty", cfg=cfg),
    }

    print(f"{'policy':<14}{'corrections':>12}{'mean err (m)':>15}{'final err (m)':>15}")
    for name in ("none", "fixed", "uncertainty"):
        _, errs, count, _ = results[name]
        print(f"{name:<14}{count:>12}{errs.mean():>15.2f}{errs[-1]:>15.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(gt_xy[:, 0], gt_xy[:, 1], "k--", label="Ground Truth")
    ax1.plot(results["none"][0][:, 0], results["none"][0][:, 1], "r-", alpha=0.5, label="No correction")
    ax1.plot(results["fixed"][0][:, 0], results["fixed"][0][:, 1], "b-", alpha=0.7, label="Fixed-period")
    ax1.plot(results["uncertainty"][0][:, 0], results["uncertainty"][0][:, 1], "g-", linewidth=2, label="Uncertainty-aware")
    fp = results["uncertainty"][3]
    if len(fp):
        ax1.scatter(fp[:, 0], fp[:, 1], c="green", marker="x", s=70, zorder=5, label="AVL fix (uncertainty)")
    ax1.set_title("Trajectories under three correction policies")
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)"); ax1.axis("equal"); ax1.legend(); ax1.grid(True)

    names = ["fixed", "uncertainty"]
    counts = [results[n][2] for n in names]
    mean_errs = [results[n][1].mean() for n in names]
    x = np.arange(len(names))
    ax2b = ax2.twinx()
    b1 = ax2.bar(x - 0.2, counts, width=0.4, color="steelblue", label="Corrections")
    b2 = ax2b.bar(x + 0.2, mean_errs, width=0.4, color="seagreen", label="Mean error (m)")
    ax2.set_xticks(x); ax2.set_xticklabels(["Fixed-period", "Uncertainty-aware"])
    ax2.set_ylabel("Corrections spent"); ax2b.set_ylabel("Mean error (m)")
    ax2.set_title("Frugality: corrections vs accuracy")
    ax2.bar_label(b1); ax2b.bar_label(b2, fmt="%.2f")

    fig.tight_layout()
    fig.savefig("week4_scheduler_comparison.png", dpi=200, bbox_inches="tight")
    print("saved week4_scheduler_comparison.png")


if __name__ == "__main__":
    run_comparison()
