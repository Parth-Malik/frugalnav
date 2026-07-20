"""
Reference uncertainty metric U (plan, Week 4).

    U = α₁·σ_pos + α₂·σ_head + α₃·FeatureLoss + α₄·Blur + α₅·IMUbias

Ownership: **Rohan** owns the full metric and the α tuning. **Siddharth**
contributes the Blur term (`blur_metric.py`, his vision domain) and *consumes* U
to compare scheduling policies in the evaluation. This module is the shared
assembler both sides build on. Following the plan, the defaults start from the
two terms it says to begin with (σ_pos + FeatureLoss) and layer the rest on.

Keeping U a plain weighted sum of glass-box signals is deliberate: every term is
inspectable, and the scheduler's threshold has an interpretable meaning.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UncertaintyWeights:
    a_sigma_pos: float = 1.0        # σ_pos: position-covariance growth since last fix
    a_sigma_head: float = 0.0       # σ_head: heading covariance (added later)
    a_feature_loss: float = 0.5     # FeatureLoss: fraction of lost tracks
    a_blur: float = 0.3             # Blur: Siddharth's image-sharpness term
    a_imu_bias: float = 0.0         # IMUbias: bias instability (added later)


def uncertainty(sigma_pos: float, feature_loss: float = 0.0, blur: float = 0.0,
                sigma_head: float = 0.0, imu_bias: float = 0.0,
                w: "UncertaintyWeights | None" = None) -> float:
    """Fuse the glass-box signals into a single localization-confidence scalar."""
    if w is None:
        w = UncertaintyWeights()
    return (w.a_sigma_pos * sigma_pos
            + w.a_sigma_head * sigma_head
            + w.a_feature_loss * feature_loss
            + w.a_blur * blur
            + w.a_imu_bias * imu_bias)
