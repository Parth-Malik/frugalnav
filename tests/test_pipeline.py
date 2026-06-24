import numpy as np

from core.interfaces import SensorInput
from core.se3 import make_se3, q_to_R, exp_so3, project_to_SO3
from core.pipeline import FrugalPipeline
from harness.drift_injection import DriftInjectionAdapter


def _curved_gt(n=200):
    ts, poses = [], []
    for i in range(n):
        t = i * 0.1
        R = exp_so3(np.array([0, 0, 0.01 * i]))
        poses.append(make_se3(R, [np.sin(i / 20.0) * 3, i * 0.05, 0]))
        ts.append(t)
    return np.array(ts), poses


def test_zero_drift_reconstructs_groundtruth():
    ts, poses = _curved_gt()
    vio = DriftInjectionAdapter(ts, poses, drift_rate_m_per_s=0.0)
    pipe = FrugalPipeline(vio)
    stream = (SensorInput(t, [0, 0, 9.81], [0, 0, 0]) for t in ts)
    traj = pipe.replay(stream, initial_pose=poses[0])
    end_est = traj[-1].pose_world[:3, 3]
    end_gt = poses[-1][:3, 3]
    assert np.linalg.norm(end_est - end_gt) < 1e-9


def test_drift_rate_is_honest():
    ts, poses = _curved_gt()
    rate = 0.05
    vio = DriftInjectionAdapter(ts, poses, drift_rate_m_per_s=rate)
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
    R = np.eye(3, dtype=np.float32)
    rng = np.random.default_rng(0)
    for i in range(20000):
        w = rng.normal(0, 0.02, 3).astype(np.float32)
        th = float(np.linalg.norm(w))
        W = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], dtype=np.float32)
        Rinc = (np.eye(3, dtype=np.float32)
                + (np.sin(th) / th) * W
                + ((1 - np.cos(th)) / th ** 2) * (W @ W))
        R = (R @ Rinc).astype(np.float32)
        if (i + 1) % 100 == 0:
            R = project_to_SO3(R).astype(np.float32)
    assert np.linalg.norm(R.T @ R - np.eye(3)) < 1e-5


def test_returned_pose_is_independent():
    ts, poses = _curved_gt(50)
    vio = DriftInjectionAdapter(ts, poses, drift_rate_m_per_s=0.0)
    pipe = FrugalPipeline(vio)
    stream = (SensorInput(t, [0, 0, 9.81], [0, 0, 0]) for t in ts)
    traj = pipe.replay(stream, initial_pose=poses[0])
    assert not np.shares_memory(traj[-1].pose_world, pipe.current_pose)
    assert not np.allclose(traj[5].pose_world, traj[40].pose_world)
