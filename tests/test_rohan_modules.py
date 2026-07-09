"""
tests/test_rohan_modules.py
---------------------------
Unit tests for the two modules Rohan owns. Run with:  pytest -q
(or directly:  python3 tests/test_rohan_modules.py)

These pin behaviour so that when you (or an AI assistant) refactor later, you
immediately know if something broke. Add a test every time you add a U cue.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.uncertainty_scheduler import UncertaintyScheduler, SchedulerConfig, normalize
from harness.kinematic_sim import KinematicSim, SimConfig


# ---------- uncertainty scheduler ----------
def test_normalize_clamps_and_inverts():
    assert normalize(0.5, 0.0, 1.0) == 0.5
    assert normalize(-5, 0.0, 1.0) == 0.0          # clamps low
    assert normalize(99, 0.0, 1.0) == 1.0          # clamps high
    assert normalize(0.0, 0.0, 1.0, invert=True) == 1.0   # inverted: low raw -> high unc

def test_weights_must_sum_to_one():
    bad = SchedulerConfig(weights={"sigma_pos": 0.5, "feature_loss": 0.2,
                                   "sigma_head": 0.0, "blur": 0.0, "imu_bias": 0.0})
    try:
        UncertaintyScheduler(bad)
        assert False, "should have rejected weights that don't sum to 1"
    except AssertionError as e:
        assert "sum to 1" in str(e)

def test_calm_state_does_not_trigger():
    sch = UncertaintyScheduler()
    U, trig, why, _ = sch.compute(dict(sigma_pos=0.1, feature_loss=2.0, active_features=120))
    assert U < 0.2 and not trig

def test_hard_floor_overrides_refractory():
    sch = UncertaintyScheduler()
    sch.reset_after_fix()                          # enter refractory
    _, trig, why, _ = sch.compute(dict(sigma_pos=2.5, feature_loss=0, active_features=120))
    assert trig and why == "hard_floor_sigma"

def test_observability_floor():
    sch = UncertaintyScheduler()
    _, trig, why, _ = sch.compute(dict(sigma_pos=0.2, feature_loss=0, active_features=10))
    assert trig and why == "observability_floor"


# ---------- week 4: the full 5-cue U (one test per newly-activated cue) ----------
_CALM = dict(sigma_pos=0.1, sigma_head=0.01, feature_loss=0.0,
             blur=300.0, imu_bias=0.0, active_features=120)

def test_default_is_full_five_cue():
    """Week 4 graduated the default scheduler from 2 cues to all 5, weights sum to 1."""
    cfg = SchedulerConfig()
    for k in ("sigma_pos", "sigma_head", "blur", "feature_loss", "imu_bias"):
        assert cfg.weights[k] > 0, f"{k} should carry weight in the full U"
    assert abs(sum(cfg.weights.values()) - 1.0) < 1e-9

def test_sigma_head_raises_U():
    sch = UncertaintyScheduler()
    U0, *_ = sch.compute(dict(_CALM))
    U1, *_ = sch.compute({**_CALM, "sigma_head": 0.30})
    assert U1 > U0

def test_blur_raises_U_when_image_blurry():
    sch = UncertaintyScheduler()
    U_sharp, *_ = sch.compute({**_CALM, "blur": 300.0})   # high Laplacian = sharp
    U_blurry, *_ = sch.compute({**_CALM, "blur": 150.0})  # low Laplacian = blurry
    assert U_blurry > U_sharp                              # blur cue is inverted

def test_imu_bias_raises_U():
    sch = UncertaintyScheduler()
    U0, *_ = sch.compute(dict(_CALM))
    U1, *_ = sch.compute({**_CALM, "imu_bias": 0.15})
    assert U1 > U0


# ---------- kinematic sim ----------
def test_no_correction_drifts():
    sim = KinematicSim(cfg=SimConfig(seed=1))
    peak = 0.0
    while not sim.arrived and sim.steps < sim.cfg.max_steps:
        sim.step(); peak = max(peak, sim.error)
    assert peak > 0.3                              # meaningful drift accrues

def test_marker_fix_reduces_error():
    sim = KinematicSim(cfg=SimConfig(seed=1))
    # drift for a while, then force the drone over a marker and fix
    for _ in range(300):
        sim.step()
    before = sim.error
    sim.true = sim.true * 0 + [sim.world.markers[0].x, sim.world.markers[0].y]
    fixed = sim.apply_fix()
    assert fixed and sim.error <= before + 1e-6

def test_sim_is_reproducible():
    a = KinematicSim(cfg=SimConfig(seed=7)); a.step(); a.step()
    b = KinematicSim(cfg=SimConfig(seed=7)); b.step(); b.step()
    assert abs(a.error - b.error) < 1e-12


# ---------- week 2: adapter + drift scaffold ----------
def _toy_euroc():
    import numpy as np
    n = 800; t = np.linspace(0, 12, n)
    x = 30 * np.cos(np.linspace(0, 1.0, n)); y = 30 * np.sin(np.linspace(0, 1.0, n))
    gt = dict(t=t, x=x, y=y, z=np.zeros(n),
              vx=np.gradient(x, t), vy=np.gradient(y, t), vz=np.zeros(n),
              bgx=np.full(n, 0.003), bgy=np.full(n, 0.001), bgz=np.full(n, 0.002))
    imu = dict(t=t, wx=0.2*np.sin(t), wy=np.zeros(n), wz=0.3*np.cos(t),
               ax=np.zeros(n), ay=np.zeros(n), az=np.full(n, 9.81))
    return gt, imu

def test_vio_signals_error():
    from core.vio_adapter import VioSignals
    s = VioSignals(t=0, est=(3.0, 4.0), cues={}, gt=(0.0, 0.0))
    assert abs(s.error() - 5.0) < 1e-9            # 3-4-5 triangle
    assert VioSignals(t=0, est=(1, 1), cues={}, gt=None).error() is None

def test_drift_scaffold_drifts_and_exposes_cues():
    from harness.drift_scaffold import EurocDriftSource
    gt, imu = _toy_euroc()
    src = EurocDriftSource(gt, imu, seed=1)
    peak = 0.0; last = None
    while True:
        sig = src.update()
        if sig is None:
            break
        last = sig; peak = max(peak, sig.error())
    assert peak > 0.3                              # real-data-derived drift accrues
    for key in ("sigma_pos", "feature_loss", "blur", "imu_bias", "active_features"):
        assert key in last.cues                    # all glass-box signals present

def test_calibration_reports_drift_pct():
    from harness.drift_scaffold import calibrate_sim_noise
    gt, imu = _toy_euroc()
    rep = calibrate_sim_noise(gt, imu, seed=1)
    assert rep["drift_pct_of_distance"] > 0 and rep["path_length_m"] > 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
