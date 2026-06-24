import csv
import os
import numpy as np
from core.interfaces import SensorInput

def read_euroc_stream(dataset_path: str):
    imu_csv_path = os.path.join(dataset_path, "mav0", "imu0", "data.csv")
    cam_csv_path = os.path.join(dataset_path, "mav0", "cam0", "data.csv")
    events = []

    if os.path.exists(imu_csv_path):
        with open(imu_csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                ts = float(row[0]) * 1e-9
                events.append({
                    'timestamp': ts,
                    'type': 'imu',
                    'angular_vel': np.array([float(row[1]), float(row[2]), float(row[3])]),
                    'linear_accel': np.array([float(row[4]), float(row[5]), float(row[6])])
                })

    if os.path.exists(cam_csv_path):
        with open(cam_csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                ts = float(row[0]) * 1e-9
                events.append({
                    'timestamp': ts,
                    'type': 'cam',
                    'img_path': os.path.join(dataset_path, "mav0", "cam0", "data", row[1])
                })

    events.sort(key=lambda x: x['timestamp'])
    last_accel, last_gyro = np.zeros(3), np.zeros(3)

    for event in events:
        if event['type'] == 'imu':
            last_accel = event['linear_accel']
            last_gyro = event['angular_vel']
            yield SensorInput(event['timestamp'], last_accel, last_gyro, False, None)
        elif event['type'] == 'cam':
            yield SensorInput(event['timestamp'], last_accel, last_gyro, True, event['img_path'])
