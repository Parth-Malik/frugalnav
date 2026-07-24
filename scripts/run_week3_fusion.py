"""Week 3 deliverable: state fusion with periodic corrections (snap-back demo)."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.se3 import make_se3
from core.state_fusion import StateFusion
from harness.drift_injection import PoseDriftAdapter
from harness.synthetic_landmarks import SyntheticLandmarkSource


def run_demo():
    t = np.linspace(0, 60, 600)
    gt_poses = [make_se3(np.eye(3), [10 * np.cos(ti * 0.1), 10 * np.sin(ti * 0.1), 0.0]) for ti in t]
    adapter = PoseDriftAdapter(pos_bias=(0.05, 0.02, 0.0), random_walk_std=0.02)
    fusion_no_fix = StateFusion(); fusion_fixed = StateFusion()
    fusion_no_fix.reset(initial_pose=gt_poses[0]); fusion_fixed.reset(initial_pose=gt_poses[0])
    landmarks = SyntheticLandmarkSource(t, gt_poses, period_s=8.0, pos_std_m=0.05, noise_m=0.02)
    traj_gt, traj_drift, traj_fused, fix_points = [], [], [], []
    for ti, gt_p in zip(t, gt_poses):
        vio_out = adapter.update(ti, gt_p)
        fusion_no_fix.predict(vio_out); fusion_fixed.predict(vio_out)
        fix = landmarks.try_fix(ti)
        if fix.valid:
            fusion_fixed.correct(fix)
            fix_points.append(fusion_fixed.estimate.pose_world[:3, 3].copy())
        traj_gt.append(gt_p[:3, 3].copy())
        traj_drift.append(fusion_no_fix.estimate.pose_world[:3, 3].copy())
        traj_fused.append(fusion_fixed.estimate.pose_world[:3, 3].copy())
    traj_gt = np.array(traj_gt); traj_drift = np.array(traj_drift); traj_fused = np.array(traj_fused)
    fix_points = np.array(fix_points) if fix_points else np.empty((0, 3))
    plt.figure(figsize=(10, 6))
    plt.plot(traj_gt[:, 0], traj_gt[:, 1], "k--", label="Ground Truth")
    plt.plot(traj_drift[:, 0], traj_drift[:, 1], "r-", alpha=0.6, label="VIO Drift Only")
    plt.plot(traj_fused[:, 0], traj_fused[:, 1], "g-", linewidth=2, label="Fused")
    if len(fix_points) > 0:
        plt.scatter(fix_points[:, 0], fix_points[:, 1], c="blue", marker="x", s=80,
                    zorder=5, label="AVL Fix Event")
    plt.title("Week 3: State Fusion with Periodic Corrections")
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.axis("equal"); plt.legend(); plt.grid(True)
    plt.savefig("week3_fusion_snapback.png", dpi=200, bbox_inches="tight")
    print(f"corrections applied : {fusion_fixed.correction_count}")
    print(f"final drift-only err: {np.linalg.norm(traj_drift[-1] - traj_gt[-1]):.2f} m")
    print(f"final fused err     : {np.linalg.norm(traj_fused[-1] - traj_gt[-1]):.2f} m")
    print("saved week3_fusion_snapback.png")


if __name__ == "__main__":
    run_demo()
