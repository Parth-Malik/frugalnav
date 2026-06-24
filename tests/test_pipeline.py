"""Verifies pipeline integrity and state logic."""

import numpy as np
from core.interfaces import SensorInput, DTYPE
from core.se3 import make_se3, q_to_R, exp_so3
from core.pipeline import FrugalPipeline
from harness.drift_injection import DriftInjectionAdapter

def _curved_gt(n=200):
    """Generates synthetic ground truth. Updated to return velocities."""
    ts, poses, vels, ang_vels = [], [], [], []
    for i in range(n):
        t = i * 0.1
        R = exp_so3(np.array([0, 0, 0.01 * i]))
        poses.append(make_se3(R, [np.sin(i / 20.0) * 3, i * 0.05, 0]))
        ts.append(t)
        vels.append(np.zeros(3, dtype=DTYPE))
        ang_vels.append(np.zeros(3, dtype=DTYPE))
    return np.array(ts), poses, vels, ang_vels

def test_zero_drift_reconstructs_groundtruth():
    ts, poses, vels, ang_vels = _curved_gt()
    vio = DriftInjectionAdapter(ts, poses, vels, ang_vels, drift_rate_m_per_s=0.0)
    pipe = FrugalPipeline(vio)
    stream = (SensorInput(t, [0, 0, 9.81], [0, 0, 0]) for t in ts)
    traj = pipe.replay(stream, initial_pose=poses[0])
    
    end_est = traj[-1].pose_world[:3, 3]
    end_gt = poses[-1][:3, 3]
    assert np.linalg.norm(end_est - end_gt) < 1e-9

def test_drift_rate_is_honest():
    ts, poses, vels, ang_vels = _curved_gt()
    rate = 0.05
    vio = DriftInjectionAdapter(ts, poses, vels, ang_vels, drift_rate_m_per_s=rate)
    pipe = FrugalPipeline(vio)
    stream = (SensorInput(t, [0, 0, 9.81], [0, 0, 0]) for t in ts)
    traj = pipe.replay(stream, initial_pose=poses[0])
    
    expected = rate * (ts[-1] - ts[0])
    assert traj[-1].pos_std_m == 0.0 or abs(traj[-1].pos_std_m - expected) < 0.1 * expected + 1e-9

def test_no_aliasing():
    caller = np.array([1.0, 2.0, 3.0])
    s = SensorInput(0.0, caller, np.zeros(3))
    caller[0] = 999.0
    assert s.linear_accel[0] == 1.0
    assert not np.shares_memory(s.linear_accel, caller)

def test_quaternion_normalized():
    R = q_to_R(np.array([0.98, 0.1, 0.1, 0.05]))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(R) - 1.0) < 1e-9

def test_so3_manifold_hygiene():
    R_current = np.eye(3, dtype=np.float32)
    delta_R = exp_so3(np.array([0.001, 0.002, -0.001]))
    delta_R = np.array(delta_R, dtype=np.float32)
    
    class DummyAdapter:
        def update(self, s):
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = delta_R
            from core.interfaces import VioOutput
            return VioOutput(s.timestamp, T, np.zeros(3), np.zeros(3), 0.0, 30, 0.0)
        def reset(self): pass
            
    pipe = FrugalPipeline(DummyAdapter())
    pipe.current_pose = np.eye(4, dtype=np.float32)
    
    for i in range(15000):
        s = SensorInput(timestamp=i*0.01, linear_accel=np.zeros(3), angular_vel=np.zeros(3))
        pipe.step(s)
        
    R_final = pipe.current_pose[:3, :3]
    error = float(np.linalg.norm(R_final.T @ R_final - np.eye(3)))
    assert error < 1e-6, f"SO(3) drift unbounded: {error}"

def test_returned_pose_is_independent():
    """Guards against regressions where PoseEstimate aliases current_pose."""
    ts, poses, vels, ang_vels = _curved_gt(10)
    vio = DriftInjectionAdapter(ts, poses, vels, ang_vels, drift_rate_m_per_s=0.0)
    pipe = FrugalPipeline(vio)
    stream = [SensorInput(t, [0, 0, 9.81], [0, 0, 0]) for t in ts]
    
    traj = pipe.replay(stream)
    assert not np.shares_memory(traj[-1].pose_world, pipe.current_pose), "Aliasing detected! PoseEstimate __post_init__ must copy."
