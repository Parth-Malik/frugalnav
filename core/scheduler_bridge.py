"""core/scheduler_bridge.py — the integration seam.

Rohan's UncertaintyScheduler is the canonical decision engine; it consumes a dict of
named cues. Parth's fusion pipeline produces a VioOutput plus a fused position sigma.
This bridge maps one to the other, so the SAME scheduler that runs Rohan's multi-cue
EuRoC demo also decides when Parth's fusion pipeline requests a correction.

Only semantically-matching cues are forwarded. Cues the synthetic pipeline does not
model (image blur, heading sigma, feature-loss rate) are omitted; the scheduler's
own .get() defaults treat them as 'calm', so they never spuriously raise U. On the
real-VIO path those cues become live and the full multi-cue U takes over unchanged.
"""

from core.uncertainty_scheduler import UncertaintyScheduler, SchedulerConfig, CueBounds


def cues_from_pipeline(vio_out, fused_sigma):
    """Build Rohan's cue dict from Parth's VioOutput + the fused position sigma."""
    return {
        "sigma_pos": float(fused_sigma),
        "active_features": int(vio_out.active_features),
        "imu_bias": float(vio_out.imu_bias_norm),
        # sigma_head / feature_loss / blur intentionally omitted -> scheduler defaults = calm
    }


def pipeline_scheduler_config():
    """A SchedulerConfig calibrated to the synthetic-pipeline signal scale.

    The synthetic drift source only varies position uncertainty meaningfully, so U is
    driven by sigma_pos here; the two-tier trigger, observability floor and hysteresis
    all come from Rohan's engine unchanged."""
    return SchedulerConfig(
        weights={"sigma_pos": 1.0, "sigma_head": 0.0, "blur": 0.0,
                 "feature_loss": 0.0, "imu_bias": 0.0},
        bounds=CueBounds(sigma_pos=(0.05, 0.55)),   # tuned to the pipeline's fused-sigma range
        tau=0.60,
        sigma_pos_floor=0.9,                        # hard bound well above normal operation
        feature_floor=20,
        refractory_ticks=8,
    )


class PipelineScheduler:
    """Convenience wrapper: Parth's pipeline calls should_correct(vio_out, fused_sigma)."""

    def __init__(self, config=None):
        self.sched = UncertaintyScheduler(config or pipeline_scheduler_config())
        self.last_U = 0.0

    def should_correct(self, vio_out, fused_sigma) -> bool:
        U, trigger, reason, _ = self.sched.compute(cues_from_pipeline(vio_out, fused_sigma))
        self.last_U = U
        return trigger

    def reset_after_fix(self):
        self.sched.reset_after_fix()
