"""
harness/euroc_reader.py
-----------------------
Reads the EuRoC MAV dataset (the standard free VIO benchmark: real camera + IMU +
Vicon/Leica ground truth from an actual drone). You only need the small CSVs for the
drift scaffold -- you do NOT need to load the ~1.4 GB of images for Rohan's Week-2 work.

EuRoC layout (e.g. after unzipping MH_01_easy):
  <root>/mav0/imu0/data.csv                       # IMU @ ~200 Hz
  <root>/mav0/state_groundtruth_estimate0/data.csv# pose + velocity + biases @ ~200 Hz
  <root>/mav0/cam0/data.csv                        # frame timestamps -> filenames

CSV headers start with '#'. Timestamps are in nanoseconds.

Usage:
  from harness.euroc_reader import EurocReader
  r = EurocReader("path/to/MH_01_easy")
  gt = r.ground_truth()    # dict of numpy arrays: t, x, y, z, vx, vy, vz, bgx..., bax...
  imu = r.imu()            # dict: t, wx, wy, wz, ax, ay, az
"""
import os
import csv
import numpy as np


def _read_csv_numeric(path):
    """Read an EuRoC CSV (header line starts with '#') into a list of float rows."""
    rows = []
    with open(path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                # a non-numeric field (e.g. an image filename) -> keep only leading numbers
                nums = []
                for p in parts:
                    try:
                        nums.append(float(p))
                    except ValueError:
                        break
                if nums:
                    rows.append(nums)
    return np.array(rows, dtype=float)


class EurocReader:
    def __init__(self, root):
        self.root = root
        self.mav0 = os.path.join(root, "mav0")
        if not os.path.isdir(self.mav0):
            # tolerate being pointed straight at mav0
            self.mav0 = root if os.path.isdir(os.path.join(root, "imu0")) else self.mav0

    def _path(self, *p):
        return os.path.join(self.mav0, *p)

    def imu(self):
        a = _read_csv_numeric(self._path("imu0", "data.csv"))
        return dict(t=a[:, 0] * 1e-9,                       # ns -> s
                    wx=a[:, 1], wy=a[:, 2], wz=a[:, 3],
                    ax=a[:, 4], ay=a[:, 5], az=a[:, 6])

    def ground_truth(self):
        a = _read_csv_numeric(self._path("state_groundtruth_estimate0", "data.csv"))
        d = dict(t=a[:, 0] * 1e-9, x=a[:, 1], y=a[:, 2], z=a[:, 3])
        if a.shape[1] >= 11:                                # velocities present
            d.update(vx=a[:, 8], vy=a[:, 9], vz=a[:, 10])
        if a.shape[1] >= 17:                                # gyro + accel biases present
            d.update(bgx=a[:, 11], bgy=a[:, 12], bgz=a[:, 13],
                     bax=a[:, 14], bay=a[:, 15], baz=a[:, 16])
        return d

    def cam_timestamps(self):
        a = _read_csv_numeric(self._path("cam0", "data.csv"))
        return a[:, 0] * 1e-9


# ----------------------------- self-test ------------------------------------
# Builds a tiny fake dataset in EuRoC format and reads it back, so this verifies
# without needing the 1.4 GB download.
if __name__ == "__main__":
    import tempfile
    root = tempfile.mkdtemp()
    mav0 = os.path.join(root, "mav0")
    for sub in ("imu0", "state_groundtruth_estimate0", "cam0"):
        os.makedirs(os.path.join(mav0, sub), exist_ok=True)

    t0 = 1_403_636_579_758_555_000                       # a real EuRoC-style ns epoch
    # IMU: 200 Hz for 2 s
    with open(os.path.join(mav0, "imu0", "data.csv"), "w") as f:
        f.write("#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n")
        for i in range(400):
            f.write(f"{t0+int(i*5e6)},0.01,0.0,0.02,0.1,0.0,9.81\n")
    # ground truth: 200 Hz, moving in +x at 1 m/s, with small constant biases
    with open(os.path.join(mav0, "state_groundtruth_estimate0", "data.csv"), "w") as f:
        f.write("#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z,bg_x,bg_y,bg_z,ba_x,ba_y,ba_z\n")
        for i in range(400):
            t = i * 5e-3
            f.write(f"{t0+int(i*5e6)},{t*1.0},{0.0},{1.0},1,0,0,0,1.0,0,0,"
                    f"0.002,0.001,0.0015,0.01,0.0,0.0\n")
    # cam: 20 Hz frame list
    with open(os.path.join(mav0, "cam0", "data.csv"), "w") as f:
        f.write("#timestamp [ns],filename\n")
        for i in range(40):
            ts = t0 + int(i * 50e6)
            f.write(f"{ts},{ts}.png\n")

    r = EurocReader(root)
    gt, imu, cam = r.ground_truth(), r.imu(), r.cam_timestamps()
    print(f"ground truth: {len(gt['t'])} poses, x goes {gt['x'][0]:.2f} -> {gt['x'][-1]:.2f} m")
    print(f"imu: {len(imu['t'])} samples @ ~{1/np.mean(np.diff(imu['t'])):.0f} Hz")
    print(f"cam: {len(cam)} frames @ ~{1/np.mean(np.diff(cam)):.0f} fps")
    print(f"gyro bias present: {'bgx' in gt}  (bias_x ~ {gt.get('bgx',[0])[0]} rad/s)")
    assert abs(gt['x'][-1] - 1.995) < 0.01 and 'bgx' in gt
    print("\neuroc_reader self-test passed.")
