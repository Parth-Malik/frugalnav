"""EuRoC Ground Truth Parser. Ensures strict units and quaternion ordering."""

import csv
import os
import numpy as np
from core.interfaces import DTYPE
from core.se3 import q_to_R, make_se3

def read_euroc_groundtruth(dataset_path: str):
    """
    Parses EuRoC state_groundtruth_estimate0.
    Returns: timestamps (sec), poses (4x4 SE3), velocities (m/s), angular_vels (rad/s)
    Quaternions in EuRoC are w, x, y, z format.
    """
    gt_csv_path = os.path.join(dataset_path, "mav0", "state_groundtruth_estimate0", "data.csv")
    
    timestamps = []
    poses = []
    velocities = []
    angular_vels = []
    
    if not os.path.exists(gt_csv_path):
        return np.array(timestamps), poses, np.array(velocities), np.array(angular_vels)

    with open(gt_csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].lstrip().startswith("#"):
                continue
            try:
                # 0: timestamp(ns)
                ts = float(row[0]) * 1e-9
                
                # 1,2,3: p_RS_R_x, p_RS_R_y, p_RS_R_z
                t = np.array([float(row[1]), float(row[2]), float(row[3])], dtype=DTYPE)
                
                # 4,5,6,7: q_RS_w, q_RS_x, q_RS_y, q_RS_z (EuRoC default is w, x, y, z)
                q = np.array([float(row[4]), float(row[5]), float(row[6]), float(row[7])], dtype=DTYPE)
                
                # 8,9,10: v_RS_R_x, v_RS_R_y, v_RS_R_z
                v = np.array([float(row[8]), float(row[9]), float(row[10])], dtype=DTYPE)
                
                # 11,12,13: b_w_RS_S_x, b_w_RS_S_y, b_w_RS_S_z
                w = np.array([float(row[11]), float(row[12]), float(row[13])], dtype=DTYPE)
                
                timestamps.append(ts)
                poses.append(make_se3(q_to_R(q), t))
                velocities.append(v)
                angular_vels.append(w)
            except ValueError:
                continue
                
    return np.array(timestamps, dtype=DTYPE), poses, np.array(velocities, dtype=DTYPE), np.array(angular_vels, dtype=DTYPE)
