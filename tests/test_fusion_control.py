import numpy as np

from core.interfaces import VioOutput, LandmarkFix, PoseEstimate
from core.state_fusion import StateFusion
from core.controller import TargetCentricController


def test_fusion_bounds_error():
    fusion_no_fix = StateFusion()
    fusion_fixed = StateFusion()
    for i in range(100):
        delta = np.eye(4); delta[0, 3] = 0.1
        delta_noisy = delta.copy(); delta_noisy[0, 3] += 0.05
        vio = VioOutput(timestamp=i * 0.1, delta_pose=delta_noisy, pos_std_m=0.05,
                        velocity_body=np.array([1.0, 0, 0]))
        fusion_no_fix.predict(vio)
        fusion_fixed.predict(vio)
        if i % 10 == 0 and i > 0:
            gt_pose = np.eye(4); gt_pose[0, 3] = (i + 1) * 0.1
            fix = LandmarkFix(valid=True, timestamp=i * 0.1, marker_id=1,
                              pose_world=gt_pose, pos_std_m=0.01)
            fusion_fixed.correct(fix)
    gt_final_x = 100 * 0.1
    err_no_fix = abs(fusion_no_fix.estimate.pose_world[0, 3] - gt_final_x)
    err_fixed = abs(fusion_fixed.estimate.pose_world[0, 3] - gt_final_x)
    assert err_fixed < err_no_fix * 0.2


def test_velocity_no_spike_on_correction():
    fusion = StateFusion()
    vio = VioOutput(timestamp=1.0, delta_pose=np.eye(4), pos_std_m=0.1,
                    velocity_body=np.array([1.5, 0.0, 0.0]))
    fusion.predict(vio)
    snap_pose = np.eye(4); snap_pose[0, 3] = 1.0
    fix = LandmarkFix(valid=True, timestamp=1.0, marker_id=1,
                      pose_world=snap_pose, pos_std_m=0.01)
    fusion.correct(fix)
    np.testing.assert_allclose(fusion.estimate.velocity_world, [1.5, 0.0, 0.0], atol=1e-5)


def test_controller_points_to_target():
    target = np.array([10.0, 0.0, 0.0])
    controller = TargetCentricController(target_world=target, kp=1.0, v_max=2.0)
    pose = np.eye(4); pose[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    est = PoseEstimate(timestamp=0.0, pose_world=pose, velocity_world=np.zeros(3), pos_std_m=0.0)
    cmd = controller.command(est)
    v_world = pose[:3, :3] @ cmd.linear_vel
    e_world = target - pose[:3, 3]
    assert np.dot(v_world, e_world) > 0
    assert np.linalg.norm(cmd.linear_vel) <= 2.0 + 1e-9


def test_controller_zero_at_target():
    target = np.array([5.0, 5.0, 2.0])
    controller = TargetCentricController(target_world=target)
    pose = np.eye(4); pose[:3, 3] = target
    est = PoseEstimate(timestamp=0.0, pose_world=pose, velocity_world=np.zeros(3), pos_std_m=0.0)
    cmd = controller.command(est)
    assert np.linalg.norm(cmd.linear_vel) < 1e-5
