"""
core_ext.py -- the repaired scheduler (FrugalNav-R). The vendored FrugalNav code in
./core is NOT modified; this subclass changes only what Section 4 diagnosed:

  1. the position term reads the PREDICTED uncertainty at the next mapped marker,
     normalised against a metres budget, so the covariance alone can trigger
     (via the hard floor) and the image cues fire it earlier when imagery is poor;
  2. the safety floors respect the cooldown instead of firing on every frame.

The honest process-noise model it relies on lives in frugalnav_hf.fly (variant
"repaired").
"""
from core.uncertainty_scheduler import CueBounds, SchedulerConfig, UncertaintyScheduler


class RepairedScheduler(UncertaintyScheduler):
    def __init__(self, budget_m, tau=0.45, refractory_ticks=15):
        cfg = SchedulerConfig(tau=tau, sigma_pos_floor=budget_m, refractory_ticks=refractory_ticks,
                              bounds=CueBounds(sigma_pos=(0.05, budget_m), blur=(90.0, 270.0)))
        super().__init__(cfg)

    def compute(self, cues):
        U, trig, reason, comp = super().compute(cues)
        # every tier now respects the cooldown: one fix resets the predicted
        # uncertainty, so re-firing on the next frames only re-buys the same fix
        if trig and reason != "soft_U" and self._refractory > 0:
            trig, reason = False, "none"
        return U, trig, reason, comp
