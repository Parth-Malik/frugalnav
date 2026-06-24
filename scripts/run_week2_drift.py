import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from harness.dataset_reader import read_euroc_stream
from harness.groundtruth import read_euroc_groundtruth
from harness.drift_injection import DriftInjectionAdapter
from core.pipeline import FrugalPipeline

def main():
    dataset_path = "data/MH_01"
    
    # Check if real dataset exists
    if not os.path.exists(os.path.join(dataset_path, "mav0")):
        print(f"Error: Could not find real EuRoC dataset at {dataset_path}")
        print("Please download MH_01_easy and extract it into data/MH_01 to generate the final plot.")
        sys.exit(1)

    print("Loading Real EuRoC Ground Truth...")
    gt_ts, gt_poses, gt_vels, gt_ang_vels = read_euroc_groundtruth(dataset_path)
    
    print("Initializing Drift Adapter and Pipeline...")
    adapter = DriftInjectionAdapter(gt_ts, gt_poses, gt_vels, gt_ang_vels, drift_rate_m_per_s=0.08)
    pipe = FrugalPipeline(adapter)
    
    print("Streaming MH_01 Sensor Data (This simulates live execution)...")
    stream = read_euroc_stream(dataset_path, require_image_exists=False)
    
    # Replay trajectory seeded at the exact real-world starting position
    estimates = pipe.replay(stream, initial_pose=gt_poses[0])
    
    print("Generating Deliverable Plot...")
    est_positions = np.array([est.pose_world[:3, 3] for est in estimates])
    gt_positions = np.array([p[:3, 3] for p in gt_poses])
    
    plt.figure(figsize=(10, 6))
    plt.plot(gt_positions[:, 0], gt_positions[:, 1], 'k-', linewidth=2, label="EuRoC Ground Truth")
    if len(est_positions) > 0:
        plt.plot(est_positions[:, 0], est_positions[:, 1], 'r--', linewidth=1.5, label="FrugalNav VIO (Simulated Drift)")
    
    plt.xlabel("World X (m)")
    plt.ylabel("World Y (m)")
    plt.title("FrugalNav Week 2: VIO Estimate vs Real Ground Truth")
    plt.legend()
    plt.grid(True, linestyle=':')
    
    out_file = "drift_plot.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Success! Deliverable saved to {out_file}")

if __name__ == "__main__":
    main()
