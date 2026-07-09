"""
core/uncertainty_scheduler.py
------------------------------
Rohan's headline contribution: the Uncertainty-Aware Landmark Scheduler.

It decides *when* to ask for an absolute fix (AVL). The whole frugality argument
of the project lives here: correct only when a fused confidence metric U says the
estimate is about to fail, instead of correcting on a fixed timer.

This module is PURE LOGIC. It takes a bundle of "cues" (numbers the rest of the
system measures) and returns U plus a trigger decision. It imports nothing heavy,
has no global state beyond a small refractory counter, and is the easiest module
to port to C++/RISC-V later. That is deliberate -- it lives in core/.

U = a1*n(sigma_pos) + a2*n(sigma_head) + a3*n(feature_loss) + a4*n(blur) + a5*n(imu_bias)
with the weights summing to 1 so U is always in [0, 1].

Two-tier trigger (this structure is what makes it defensible in review):
  1. HARD FLOOR: if sigma_pos exceeds a metres budget, force a fix no matter what.
                 This bounds the worst case regardless of how the soft weights are tuned.
  2. SOFT TRIGGER: otherwise fire when U > tau. The non-covariance cues (blur,
                 feature loss, bias) are *leading indicators* -- they let us correct
                 just before drift blows up, which is the actual value over plain
                 covariance gating.
Plus HYSTERESIS: after a fix we suppress re-triggering for a few ticks so we do not
chatter while one correction settles.
"""
from dataclasses import dataclass, field


def normalize(value, lo, hi, invert=False):
    """Map a raw cue into [0, 1] using calibrated bounds.

    lo  -> the value we consider 'totally fine' (maps to 0 uncertainty)
    hi  -> the value we consider 'as bad as it gets' (maps to 1 uncertainty)
    invert=True for cues where a HIGHER raw number means LESS uncertainty.
       (Blur is the classic case: a high Laplacian variance = a SHARP image =
        low uncertainty, so the raw sharpness must be inverted into 'blur uncertainty'.)
    """
    if hi == lo:
        return 0.0
    x = (value - lo) / (hi - lo)
    x = max(0.0, min(1.0, x))          # clamp to [0, 1]
    return (1.0 - x) if invert else x


@dataclass
class CueBounds:
    """Calibrated [lo, hi] range for each cue, plus whether to invert it.
    TUNED in Week 4 against the REAL EuRoC MH_01 signal ranges (p5..p95 measured
    from the drift scaffold over the whole sequence), not guessed:
        sigma_pos   p5..p95 ~ 2.1 .. 8.9   sigma_head p5..p95 ~ 0.018 .. 0.264
        blur        p5..p95 ~ 213 .. 289   imu_bias   ~ 0.103 (constant on MH_01)
    """
    sigma_pos:    tuple = (0.05, 2.0)    # metres of std-dev. 5 cm fine, 2 m alarming.
    sigma_head:   tuple = (0.01, 0.30)   # radians (~0.6 deg fine, ~17 deg alarming). matches MH_01.
    feature_loss: tuple = (0.0, 30.0)    # features lost per second (spiky leading indicator).
    blur:         tuple = (150.0, 300.0) # raw Laplacian variance; INVERTED. lo=150 (MH_01 floor ~141).
    imu_bias:     tuple = (0.0, 0.15)    # rad/s of gyro-bias magnitude. ~0.10 on MH_01 (near-constant).


@dataclass
class SchedulerConfig:
    # Weights. Week 1 STARTED with only sigma_pos + feature_loss (per the plan).
    # Week 4 graduates to the full 5-cue U. Weights reflect how DISCRIMINATIVE each
    # cue measured on real MH_01: sigma_pos/sigma_head/blur vary usefully, feature_loss
    # is a sparse spike, imu_bias is near-constant on MH_01 so it gets a small weight
    # (it earns its keep on sequences that actually exhibit gyro-bias drift). Sum = 1.0.
    weights: dict = field(default_factory=lambda: {
        "sigma_pos":    0.45,
        "sigma_head":   0.20,
        "blur":         0.20,
        "feature_loss": 0.10,
        "imu_bias":     0.05,
    })
    tau: float = 0.45              # soft threshold on U in [0,1]
    sigma_pos_floor: float = 1.5   # HARD floor: force a fix if sigma_pos exceeds this (m)
    feature_floor: int = 20        # observability floor: force a fix if tracks < this
    refractory_ticks: int = 15     # suppress re-trigger for N ticks after a fix
    bounds: CueBounds = field(default_factory=CueBounds)

    def validate(self):
        s = sum(self.weights.values())
        assert abs(s - 1.0) < 1e-6, f"weights must sum to 1.0, got {s}"


class UncertaintyScheduler:
    def __init__(self, config: SchedulerConfig | None = None):
        self.cfg = config or SchedulerConfig()
        self.cfg.validate()
        self._refractory = 0           # ticks remaining in the no-trigger window
        self.last_components = {}      # exposed for logging / plotting

    def reset_after_fix(self):
        """Call this right after a correction is applied."""
        self._refractory = self.cfg.refractory_ticks

    def compute(self, cues: dict):
        """cues keys: sigma_pos, sigma_head, feature_loss, blur (raw Laplacian var),
                      imu_bias, active_features
        Returns (U, trigger: bool, reason: str, components: dict)."""
        b = self.cfg.bounds
        comp = {
            "sigma_pos":    normalize(cues["sigma_pos"],    *b.sigma_pos),
            "sigma_head":   normalize(cues.get("sigma_head", 0.0), *b.sigma_head),
            "feature_loss": normalize(cues.get("feature_loss", 0.0), *b.feature_loss),
            "blur":         normalize(cues.get("blur", b.blur[1]), *b.blur, invert=True),
            "imu_bias":     normalize(cues.get("imu_bias", 0.0), *b.imu_bias),
        }
        U = sum(self.cfg.weights[k] * comp[k] for k in comp)
        self.last_components = comp

        # tick down the refractory window
        if self._refractory > 0:
            self._refractory -= 1

        # --- trigger logic ---
        reason = "none"
        trigger = False
        active = cues.get("active_features", 9999)
        if cues["sigma_pos"] > self.cfg.sigma_pos_floor:
            trigger, reason = True, "hard_floor_sigma"
        elif active < self.cfg.feature_floor:
            trigger, reason = True, "observability_floor"
        elif self._refractory == 0 and U > self.cfg.tau:
            trigger, reason = True, "soft_U"

        # Note: hard floors override the refractory window on purpose -- safety first.
        return U, trigger, reason, comp


# ----------------------------- self-test ------------------------------------
# Run `python3 core/uncertainty_scheduler.py` for a quick sanity check without pytest.
if __name__ == "__main__":
    sch = UncertaintyScheduler()

    # 1) A calm, confident state -> low U, no trigger.
    U, trig, why, _ = sch.compute(dict(sigma_pos=0.1, feature_loss=2.0, active_features=120))
    assert U < 0.2 and not trig, (U, trig)
    print(f"calm:      U={U:.3f} trigger={trig} ({why})")

    # 2) Drift has grown but below the hard floor; soft trigger should fire.
    U, trig, why, _ = sch.compute(dict(sigma_pos=1.2, feature_loss=25.0, active_features=80))
    assert trig and why == "soft_U", (U, trig, why)
    print(f"drifting:  U={U:.3f} trigger={trig} ({why})")
    sch.reset_after_fix()

    # 3) Right after a fix, refractory suppresses the soft trigger.
    U, trig, why, _ = sch.compute(dict(sigma_pos=1.2, feature_loss=25.0, active_features=80))
    assert not trig, (U, trig, why)
    print(f"refractory:U={U:.3f} trigger={trig} ({why})  <- correctly suppressed")

    # 4) Hard floor overrides everything, even refractory.
    U, trig, why, _ = sch.compute(dict(sigma_pos=2.5, feature_loss=25.0, active_features=80))
    assert trig and why == "hard_floor_sigma", (U, trig, why)
    print(f"hardfloor: U={U:.3f} trigger={trig} ({why})")

    # 5) Observability floor: too few features -> force a fix.
    U, trig, why, _ = sch.compute(dict(sigma_pos=0.3, feature_loss=5.0, active_features=12))
    assert trig and why == "observability_floor", (U, trig, why)
    print(f"few-feat:  U={U:.3f} trigger={trig} ({why})")

    print("\nAll uncertainty_scheduler self-tests passed.")
