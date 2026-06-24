import numpy as np
from core.interfaces import VioOutput, DTYPE
from core.se3 import make_se3, relative_se3

class DriftInjectionAdapter:
    def __init__(self, gt_timestamps, gt_poses, gt_velocities, gt_angular_vels, 
                 drift_rate_m_per_s=0.05, drift_dir=(1.0, 1.0, 1.0), 
                 active_features=35, imu_bias_norm=0.05):
                 
        if len(gt_timestamps) == 0:
            raise ValueError("DriftInjectionAdapter needs a non-empty ground-truth trajectory.")
            
        self.gt_timestamps = np.asarray(gt_timestamps, dtype=DTYPE)
        self.gt_poses = gt_poses
        self.gt_velocities = gt_velocities
        self.gt_angular_vels = gt_angular_vels
        self.drift_rate = float(drift_rate_m_per_s)
        
        d = np.asarray(drift_dir, dtype=DTYPE)
        self.drift_dir = d / (np.linalg.norm(d) + 1e-12)
        
        self.active_features = int(active_features)
        self.imu_bias_norm = float(imu_bias_norm)
        self.reset()

    def reset(self):
        self.last_idx = None
        self.last_ts = None
        self.accumulated_drift_m = 0.0
        self._j = 0 

    def update(self, sensor_input):
        t = sensor_input.timestamp
        
        while self._j + 1 < len(self.gt_timestamps) and \
              abs(self.gt_timestamps[self._j+1] - t) <= abs(self.gt_timestamps[self._j] - t):
            self._j += 1
            
        idx = self._j
        
        if self.last_idx is None:
            self.last_idx = idx

        delta = relative_se3(self.gt_poses[self.last_idx], self.gt_poses[idx])
        self.last_idx = idx

        dt = (t - self.last_ts) if (self.last_ts is not None and t > self.last_ts) else 0.0
        self.last_ts = t
        
        step = self.drift_dir * self.drift_rate * dt
        self.accumulated_drift_m += float(np.linalg.norm(step))
        modified_delta = delta @ make_se3(np.eye(3, dtype=DTYPE), step)

        # Inject some slight noise into the velocity so the controller has to do real work later
        vel = self.gt_velocities[idx] + np.random.normal(0, 0.01, 3)
        ang_vel = self.gt_angular_vels[idx]

        return VioOutput(
            timestamp=t,
            delta_pose=modified_delta,
            velocity_world=vel,
            angular_vel_world=ang_vel,
            pos_std_m=self.accumulated_drift_m,
            active_features=self.active_features,
            imu_bias_norm=self.imu_bias_norm,
        )
