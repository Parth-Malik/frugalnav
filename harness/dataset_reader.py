"""EuRoC reader. Merges IMU + camera events into a SensorInput stream."""

import csv
import os
import numpy as np
from core.interfaces import SensorInput

def _rows(csv_path):
    with open(csv_path, "r") as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#"):
                continue
            try:
                float(row[0])
            except ValueError:
                continue
            yield row

def read_euroc_stream(dataset_path: str, require_image_exists: bool = True):
    imu_csv = os.path.join(dataset_path, "mav0", "imu0", "data.csv")
    cam_csv = os.path.join(dataset_path, "mav0", "cam0", "data.csv")
    cam_dir = os.path.join(dataset_path, "mav0", "cam0", "data")

    events = []
    if os.path.exists(imu_csv):
        for row in _rows(imu_csv):
            events.append({
                "timestamp": float(row[0]) * 1e-9,
                "type": "imu",
                "angular_vel": np.array([float(row[1]), float(row[2]), float(row[3])]),
                "linear_accel": np.array([float(row[4]), float(row[5]), float(row[6])]),
            })
            
    if os.path.exists(cam_csv):
        for row in _rows(cam_csv):
            events.append({
                "timestamp": float(row[0]) * 1e-9,
                "type": "cam",
                "img_path": os.path.join(cam_dir, row[1].strip()),
            })

    events.sort(key=lambda e: e["timestamp"])

    last_accel, last_gyro = np.zeros(3), np.zeros(3)
    have_imu = False
    
    for e in events:
        if e["type"] == "imu":
            last_accel, last_gyro = e["linear_accel"], e["angular_vel"]
            have_imu = True
            yield SensorInput(e["timestamp"], last_accel, last_gyro, False, None)
        else:
            if not have_imu:
                continue
            if require_image_exists and not os.path.exists(e["img_path"]):
                continue
            yield SensorInput(e["timestamp"], last_accel, last_gyro, True, e["img_path"])
