"""EuRoC Ground Truth Parser. Ensures strict units and quaternion ordering."""

import csv
import os
import numpy as np

from core.interfaces import DTYPE
from core.se3 import q_to_R, make_se3

def read_euroc_groundtruth(dataset_path: str):
    """Return (timestamps_sec: np.ndarray (N,), poses: list[np.ndarray (4,4)])."""
    gt_csv_path = os.path.join(dataset_path, "mav0", "state_groundtruth_estimate0", "data.csv")
    
    if not os.path.exists(gt_csv_path):
        raise FileNotFoundError(f"Ground truth CSV not found at: {gt_csv_path}")

    timestamps = []
    poses = []

    with open(gt_csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].lstrip().startswith("#"):
                continue
            try:
                # 0: timestamp[ns] -> seconds
                ts_sec = float(row[0]) * 1e-9
                
                # 1,2,3: position
                px, py, pz = float(row[1]), float(row[2]), float(row[3])
                t = np.array([px, py, pz], dtype=DTYPE)
                
                # 4,5,6,7: quaternion (w, x, y, z)
                qw, qx, qy, qz = float(row[4]), float(row[5]), float(row[6]), float(row[7])
                q = np.array([qw, qx, qy, qz], dtype=DTYPE)
                
                timestamps.append(ts_sec)
                poses.append(make_se3(q_to_R(q), t))
                
            except ValueError:
                continue

    return np.array(timestamps, dtype=DTYPE), poses
