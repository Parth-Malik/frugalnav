"""Integration tests: Rohan's scheduler driving Parth's pipeline via the bridge."""

import numpy as np

from core.interfaces import VioOutput
from core.scheduler_bridge import PipelineScheduler, cues_from_pipeline
from scripts.run_week4_comparison import make_trajectory, run_policy


def _vio(features=150, bias=0.0):
    return VioOutput(timestamp=0.0, delta_pose=np.eye(4), pos_std_m=0.0,
                     active_features=features, imu_bias_norm=bias)


def test_bridge_cue_mapping():
    cues = cues_from_pipeline(_vio(features=90, bias=0.03), fused_sigma=0.4)
    assert cues["sigma_pos"] == 0.4
    assert cues["active_features"] == 90
    assert cues["imu_bias"] == 0.03


def test_low_uncertainty_does_not_trigger():
    s = PipelineScheduler()
    assert s.should_correct(_vio(features=150), fused_sigma=0.05) is False


def test_high_uncertainty_triggers():
    s = PipelineScheduler()
    assert s.should_correct(_vio(features=150), fused_sigma=0.9) is True


def test_observability_floor_triggers():
    s = PipelineScheduler()
    assert s.should_correct(_vio(features=10), fused_sigma=0.05) is True


def test_uncertainty_aware_is_frugal():
    """Headline: fewer corrections than fixed-period at comparable accuracy."""
    t, gt = make_trajectory()
    _, fixed_err, fixed_count, _ = run_policy(t, gt, "fixed", period_s=8.0)
    _, unc_err, unc_count, _ = run_policy(t, gt, "uncertainty")
    assert unc_count < fixed_count
    assert unc_err.mean() < fixed_err.mean() * 1.6
