import numpy as np

from core.interfaces import VioOutput
from core.uncertainty_scheduler import UncertaintyScheduler, SchedulerConfig


def _vio(pos_std=0.0, features=150, blur=0.0, bias=0.0):
    return VioOutput(timestamp=0.0, delta_pose=np.eye(4), pos_std_m=pos_std,
                     active_features=features, imu_bias_norm=bias, blur=blur)


def test_U_increases_with_uncertainty():
    sched = UncertaintyScheduler()
    assert sched.compute_U(0.50, _vio(features=150)) > sched.compute_U(0.05, _vio(features=150))
    assert sched.compute_U(0.1, _vio(features=40)) > sched.compute_U(0.1, _vio(features=150))


def test_threshold_triggers():
    cfg = SchedulerConfig(threshold=0.3, a_sigma=1.0, a_feat=0.0)
    sched = UncertaintyScheduler(cfg)
    assert sched.should_correct(0.1, _vio(features=150)) is False
    assert sched.should_correct(0.5, _vio(features=150)) is True


def test_observability_floor_forces_fix():
    cfg = SchedulerConfig(threshold=1e9, feature_floor=20)
    sched = UncertaintyScheduler(cfg)
    assert sched.should_correct(0.0, _vio(features=10)) is True
    assert sched.should_correct(0.0, _vio(features=150)) is False


def test_uncertainty_aware_is_frugal():
    from scripts.run_week4_comparison import make_trajectory, run_policy
    t, gt = make_trajectory()
    _, fixed_err, fixed_count, _ = run_policy(t, gt, "fixed", period_s=8.0)
    cfg = SchedulerConfig(threshold=0.35)
    _, unc_err, unc_count, _ = run_policy(t, gt, "uncertainty", cfg=cfg)
    assert unc_count < fixed_count
    assert unc_err.mean() < fixed_err.mean() * 1.6
