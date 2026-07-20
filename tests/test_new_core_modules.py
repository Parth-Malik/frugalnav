"""
Unit tests for the three core modules added in the merge (controller,
state_fusion, obstacle_avoidance) -- the pieces the plan calls for but that were
not in either teammate's drop. Pure-core, no OpenCV / no dataset needed.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.controller import ControllerConfig, TargetCentricController
from core.obstacle_avoidance import (
    AvoidanceConfig,
    ObstacleAvoidance,
    ttc_from_flow_divergence,
    ttc_from_looming,
    ttc_from_range,
)
from core.state_fusion import FusionConfig, StateFusion
from core.types import LandmarkFix


# ------------------------------ controller ---------------------------------
def test_controller_homes_toward_origin_and_clamps():
    ctrl = TargetCentricController(target_B=(0.0, 0.0), cfg=ControllerConfig(v_max=2.0))
    cmd = ctrl.command((60.0, 25.0))
    v = np.array([cmd.vx, cmd.vy])
    assert np.linalg.norm(v) <= 2.0 + 1e-9                 # speed-clamped
    assert np.dot(v, np.array([60.0, 25.0])) < 0          # points back to B


def test_controller_offset_is_estimate_minus_B():
    ctrl = TargetCentricController(target_B=(30.0, 42.0))
    assert np.allclose(ctrl.offset((25.0, 40.0)), [-5.0, -2.0])


def test_controller_evasion_merges_in():
    ctrl = TargetCentricController(target_B=(0.0, 0.0))
    base = ctrl.command((60.0, 25.0))
    bent = ctrl.command((60.0, 25.0), evasion=(0.0, 3.0))
    assert bent.vy > base.vy


def test_controller_arrival_is_B_relative():
    ctrl = TargetCentricController(target_B=(30.0, 42.0), cfg=ControllerConfig(arrive_tol=1.0))
    assert not ctrl.arrived((0.0, 0.0))
    assert ctrl.arrived((30.3, 42.2))


# ------------------------------ state fusion -------------------------------
def test_fusion_predict_grows_update_shrinks():
    fus = StateFusion(init_xy=(0.0, 0.0), init_std=0.05)
    s0 = fus.sigma_pos()
    for _ in range(50):
        fus.predict((0.2, 0.0))
    s1 = fus.sigma_pos()
    assert s1 > s0
    fus.update(LandmarkFix(xy=np.array([10.0, 0.0]), yaw=0.0,
                           covariance=np.eye(2) * (0.02 ** 2), marker_id=1))
    s2 = fus.sigma_pos()
    assert s2 < s1
    assert abs(fus.state.xy[0] - 10.0) < 0.5               # pulled onto the fix


def test_fusion_forced_gain_one_snaps():
    fus = StateFusion(init_xy=(5.0, 5.0), init_std=1.0)
    fus.update(LandmarkFix(xy=np.array([0.0, 0.0]), yaw=0.0,
                           covariance=np.eye(2) * 0.01, marker_id=0), gain=1.0)
    assert np.allclose(fus.state.xy, [0.0, 0.0], atol=1e-6)


def test_fusion_covariance_stays_symmetric_psd():
    fus = StateFusion(init_std=0.5)
    fus.predict((1.0, 0.5))
    fus.update(LandmarkFix(xy=np.array([1.1, 0.4]), yaw=0.1,
                           covariance=np.eye(2) * 0.03, marker_id=2))
    P = fus.state.covariance
    assert np.allclose(P, P.T, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(P) > 0)


# --------------------------- obstacle avoidance ----------------------------
def test_ttc_cue_math():
    assert ttc_from_range(6.0, 2.0) == 3.0
    assert ttc_from_range(5.0, 0.0) == float("inf")
    assert ttc_from_looming(10.0, 10.0, 0.1) == float("inf")   # not expanding
    assert abs(ttc_from_looming(10.0, 11.0, 0.1) - 1.1) < 1e-6
    assert abs(ttc_from_flow_divergence(0.5) - 2.0) < 1e-9


def test_avoidance_no_evasion_when_clear():
    av = ObstacleAvoidance()
    assert np.allclose(av.update(np.array([1.0, 0.0]), ttc=5.0), [0, 0])
    assert not av.evading


def test_avoidance_dodges_away_from_obstacle_side():
    av = ObstacleAvoidance(AvoidanceConfig(ttc_trigger=2.0, gain=2.0))
    seek = np.array([1.0, 0.0])
    e_left_obstacle = av.update(seek, ttc=1.0, bearing=0.3)     # obstacle left
    assert e_left_obstacle[1] < 0                              # -> dodge right
    av.reset()
    e_right_obstacle = av.update(seek, ttc=1.0, bearing=-0.3)   # obstacle right
    assert e_right_obstacle[1] > 0                             # -> dodge left


def test_avoidance_hysteresis_holds_between_trigger_and_release():
    av = ObstacleAvoidance(AvoidanceConfig(ttc_trigger=2.0, ttc_release=3.0))
    seek = np.array([1.0, 0.0])
    av.update(seek, ttc=1.0, bearing=0.3)                      # engage
    assert av.evading
    av.update(seek, ttc=2.5, bearing=0.3)                      # in the band
    assert av.evading                                          # still evading
    av.update(seek, ttc=4.0, bearing=0.3)                      # clear
    assert not av.evading


def test_avoidance_urgency_scales_with_closeness():
    av = ObstacleAvoidance(AvoidanceConfig(ttc_trigger=2.0, ttc_min=0.4, gain=2.0))
    seek = np.array([1.0, 0.0])
    far = np.linalg.norm(av.update(seek, ttc=1.9, bearing=-0.3))
    near = np.linalg.norm(av.update(seek, ttc=0.5, bearing=-0.3))
    assert near > far
