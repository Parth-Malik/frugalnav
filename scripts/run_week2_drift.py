import sys
import os
import matplotlib.pyplot as plt
import numpy as np

from harness.dataset_reader import read_euroc_stream
from harness.groundtruth import read_euroc_groundtruth
from harness.drift_injection import DriftInjectionAdapter
from core.pipeline import FrugalPipeline

def main():
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EUROC_PATH", "data/MH_01")

    if not os.path.exists(os.path.join(dataset_path, "mav0")):
        print(f"Error: Dataset not found at '{dataset_path}'.")
        print("Usage: python scripts/run_week2_drift.py <path/to/MH_01>")
        sys.exit(1)

    print(f"Loading Ground Truth from {dataset_path}...")
    ts, gt_poses = read_euroc_groundtruth(dataset_path)

    print("Initializing Drift Adapter and Pipeline...")
    vio = DriftInjectionAdapter(ts, gt_poses, drift_rate_m_per_s=0.05)
    pipe = FrugalPipeline(vio)

    print("Streaming Sensor Data...")
    stream = read_euroc_stream(dataset_path, require_image_exists=False)
    traj = pipe.replay(stream, initial_pose=gt_poses[0])

    print("Extracting metrics...")
    est_ts = np.array([p.timestamp for p in traj])
    est_pos = np.array([p.pose_world[:3, 3] for p in traj])
    est_std = np.array([p.pos_std_m for p in traj])

    gt_pos = np.array([p[:3, 3] for p in gt_poses])

    # Interpolate GT positions to estimate timestamps to compute exact error
    gt_interp_x = np.interp(est_ts, ts, gt_pos[:, 0])
    gt_interp_y = np.interp(est_ts, ts, gt_pos[:, 1])
    gt_interp_z = np.interp(est_ts, ts, gt_pos[:, 2])

    error_m = np.sqrt((est_pos[:, 0] - gt_interp_x)**2 +
                      (est_pos[:, 1] - gt_interp_y)**2 +
                      (est_pos[:, 2] - gt_interp_z)**2)

    final_drift = error_m[-1] if len(error_m) > 0 else 0.0

    print("Generating week2_drift.png...")
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Panel 1: XY Path
    axs[0].plot(gt_pos[:, 0], gt_pos[:, 1], 'k--', label="Ground Truth")
    if len(est_pos) > 0:
        axs[0].plot(est_pos[:, 0], est_pos[:, 1], 'r-', label="FrugalNav VIO")
        axs[0].plot(est_pos[0, 0], est_pos[0, 1], 'go', label="Start")
    axs[0].set_title("VIO Estimate vs Ground Truth (XY Plane)")
    axs[0].set_xlabel("X (m)")
    axs[0].set_ylabel("Y (m)")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].axis('equal')

    # Panel 2: Position Error
    axs[1].plot(est_ts - est_ts[0], error_m, 'b-')
    axs[1].set_title("Position Error Magnitude")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Error (m)")
    axs[1].grid(True)

    # Panel 3: Position Std
    axs[2].plot(est_ts - est_ts[0], est_std, 'm-')
    axs[2].set_title("Reported pos_std_m")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Std Dev (m)")
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig("week2_drift.png", dpi=300)
    print(f"Done. Final drift: {final_drift:.3f} m")

if __name__ == "__main__":
    main()
