"""Tests for FrugalNav v2: honest process noise, look-ahead, cooldown-respecting
floors, and the noise-compensated sharpness cue. v1 behaviour must be unchanged
when the v2 options are off."""
import os

import numpy as np
import pytest

from core.landmark_map import LandmarkMap
from core.navigator import SensorInput, build_navigator
from core.state_fusion import FusionConfig, StateFusion
from core.types import LandmarkFix
from core.uncertainty_scheduler import SchedulerConfig, UncertaintyScheduler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP = os.path.join(ROOT, "config", "landmark_map.json")


# ------------------------------------------------------------------ fusion
def test_v1_fusion_growth_unchanged():
    f = StateFusion(init_std=0.1)
    for _ in range(50):
        f.predict((0.2, 0.0))
    # v1: variance grows by exactly q_per_metre per metre (10 m flown)
    assert f.state.covariance[0, 0] == pytest.approx(0.01 + 0.02 * 10.0)


def test_v2_growth_is_quadratic_in_distance_since_fix():
    f = StateFusion(init_std=0.1, cfg=FusionConfig.v2())
    for _ in range(50):
        f.predict((0.2, 0.0))
    expected = 0.01 + 0.02 * 10.0 + 8.0e-4 * 10.0 ** 2
    assert f.state.covariance[0, 0] == pytest.approx(expected)
    assert f.dist_since_fix == pytest.approx(10.0)


def test_v2_poor_image_quality_adds_variance():
    good, bad = StateFusion(cfg=FusionConfig.v2()), StateFusion(cfg=FusionConfig.v2())
    for _ in range(20):
        good.predict((0.5, 0.0), quality=1.0)
        bad.predict((0.5, 0.0), quality=0.2)
    assert bad.sigma_pos() > good.sigma_pos()


def test_fix_resets_distance_and_predicted_sigma_looks_ahead():
    f = StateFusion(cfg=FusionConfig.v2())
    for _ in range(40):
        f.predict((0.25, 0.0))
    assert f.predicted_sigma(10.0) > f.sigma_pos()
    assert f.predicted_sigma(0.0) == pytest.approx(f.sigma_pos())
    f.update(LandmarkFix(xy=np.array([10.0, 0.0]), yaw=0.0,
                         covariance=np.eye(2) * 0.05 ** 2, marker_id=0))
    assert f.dist_since_fix == 0.0


# ---------------------------------------------------------------- scheduler
def test_v1_floors_still_override_cooldown():
    s = UncertaintyScheduler()
    s.reset_after_fix()
    _, trig, why, _ = s.compute(dict(sigma_pos=0.3, active_features=10))
    assert trig and why == "observability_floor"


def test_v2_floors_wait_for_cooldown():
    s = UncertaintyScheduler(SchedulerConfig.v2())
    s.reset_after_fix()
    _, trig, _, _ = s.compute(dict(sigma_pos=0.3, active_features=10))
    assert not trig
    for _ in range(25):                                   # let the cooldown expire
        s.compute(dict(sigma_pos=0.1, active_features=120))
    _, trig, why, _ = s.compute(dict(sigma_pos=0.3, active_features=10))
    assert trig and why == "observability_floor"


def test_v2_emergency_overrides_cooldown():
    s = UncertaintyScheduler(SchedulerConfig.v2(budget_m=1.0))
    s.reset_after_fix()
    _, trig, why, _ = s.compute(dict(sigma_pos=2.5, active_features=120))
    assert trig and why == "hard_floor_sigma"


def test_v2_position_alone_can_trigger_at_budget():
    # v1's flaw: sigma_pos alone could never exceed tau. v2 fires at the budget.
    s = UncertaintyScheduler(SchedulerConfig.v2(budget_m=1.0))
    _, trig, why, _ = s.compute(dict(sigma_pos=1.05, active_features=120))
    assert trig and why == "hard_floor_sigma"


# ------------------------------------------------------------- map + navigator
def test_distance_to_next_marker():
    lm = LandmarkMap.from_json(MAP)
    d = lm.distance_to_next_marker((0.0, 0.0), (0.0, 38.0), exclude_radius=3.0)
    assert d == pytest.approx(6.0)               # marker 0 sits at (0, 6)
    d2 = lm.distance_to_next_marker((0.0, 5.0), (0.0, 38.0), exclude_radius=3.0)
    assert d2 == pytest.approx(9.0)              # marker 0 is in view -> next is (0, 14)


def test_v2_navigator_runs_and_uses_lookahead():
    lm = LandmarkMap.from_json(MAP)
    nav = build_navigator(lm.target_B, landmark_map=lm)          # v2 by default
    assert nav.scheduler.cfg.lookahead
    out = None
    for k in range(30):
        out = nav.step(SensorInput(t=0.1 * k, vio_delta=np.array([0.0, 0.2]),
                                   cues={"image_quality": 0.8, "active_features": 100}))
    assert out.sigma_pos >= nav.fusion.sigma_pos()               # predicted >= current
    nav1 = build_navigator(lm.target_B, landmark_map=lm, version="v1")
    assert not nav1.scheduler.cfg.lookahead


# ---------------------------------------------------------------- blur cue
def test_noise_compensated_sharpness():
    cv2 = pytest.importorskip("cv2")
    from core.blur_metric import laplacian_sharpness, laplacian_sharpness_nc, sharpness_score
    rng = np.random.default_rng(0)
    base = cv2.resize(rng.normal(128, 25, (60, 80)), (320, 240), interpolation=cv2.INTER_CUBIC)
    clean = np.clip(base, 0, 255).astype(np.uint8)
    sigma = 0.03 * 255
    noisy = np.clip(base + rng.normal(0, sigma, base.shape), 0, 255).astype(np.uint8)
    raw_c, raw_n = laplacian_sharpness(clean), laplacian_sharpness(noisy)
    assert raw_n > 2 * raw_c                                     # raw cue fooled by noise
    assert laplacian_sharpness_nc(noisy, sigma) == pytest.approx(raw_c, rel=0.25)
    assert sharpness_score(clean, raw_c) == pytest.approx(1.0, abs=1e-6)
