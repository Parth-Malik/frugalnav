"""Uncertainty-aware scheduler (portable core/) — the project's core contribution.

Fires an absolute correction only when a fused confidence metric U exceeds a
threshold, bounding drift while minimizing corrections (and thus compute/power):

    U = a_sigma*fused_sigma + a_feat*FeatureLoss + a_blur*blur + a_bias*imu_bias_norm

An observability floor overrides U: if active features drop below a minimum, a fix
is forced regardless of U. Weights live in a plain dataclass (no yaml) so the module
stays portable to the RISC-V target.
"""

from dataclasses import dataclass

from core.interfaces import VioOutput


@dataclass
class SchedulerConfig:
    a_sigma: float = 1.0
    a_feat: float = 0.5
    a_blur: float = 0.2
    a_bias: float = 0.1
    threshold: float = 0.35
    feature_floor: int = 20
    nominal_features: int = 150


class UncertaintyScheduler:
    def __init__(self, cfg: SchedulerConfig = None):
        self.cfg = cfg or SchedulerConfig()
        self.last_U = 0.0

    def compute_U(self, fused_sigma: float, vio_out: VioOutput) -> float:
        c = self.cfg
        feature_loss = max(0.0, 1.0 - vio_out.active_features / c.nominal_features)
        self.last_U = float(
            c.a_sigma * fused_sigma
            + c.a_feat * feature_loss
            + c.a_blur * vio_out.blur
            + c.a_bias * vio_out.imu_bias_norm
        )
        return self.last_U

    def should_correct(self, fused_sigma: float, vio_out: VioOutput) -> bool:
        if vio_out.active_features < self.cfg.feature_floor:
            self.compute_U(fused_sigma, vio_out)
            return True
        return self.compute_U(fused_sigma, vio_out) > self.cfg.threshold
