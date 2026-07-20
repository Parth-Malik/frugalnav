"""
Correction policies -- the three the mid-term compares (plan, Week 4):

    none          : never request an AVL fix (pure VIO drift, the lower bound)
    fixed-period  : request a fix on a fixed schedule, blind to conditions
    uncertainty   : request a fix only when U exceeds a threshold  ← the contribution

Each policy is a pure decision function of a small context struct, so the same
objects drive the sim here and (unchanged) the portable core later. A policy can
only *request* a fix; a fix actually happens when a marker is in view AND
detection succeeds -- that gating lives in the sim.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PolicyContext:
    step: int
    t: float
    steps_since_last: int          # steps since the last successful fix
    U: float                       # current uncertainty metric
    marker_available: bool         # is a mapped marker in view this step?


class CorrectionPolicy:
    name = "base"

    def should_correct(self, ctx: PolicyContext) -> bool:
        raise NotImplementedError


class NonePolicy(CorrectionPolicy):
    """Never correct. The drift lower bound / baseline."""
    name = "none"

    def should_correct(self, ctx: PolicyContext) -> bool:
        return False


class FixedPeriodPolicy(CorrectionPolicy):
    """Correct on a fixed schedule -- the naive frugal baseline. Cannot adapt to
    where drift actually grows, so it must be tuned to the WORST segment and
    therefore over-corrects everywhere else."""

    def __init__(self, period_steps: int):
        self.period = int(period_steps)
        self.name = f"fixed(P={period_steps})"

    def should_correct(self, ctx: PolicyContext) -> bool:
        return ctx.marker_available and ctx.steps_since_last >= self.period


class UncertaintyPolicy(CorrectionPolicy):
    """Correct only when the fused confidence metric U crosses a threshold. Spends
    fixes where drift is actually accumulating; skips easy stretches."""

    def __init__(self, threshold: float):
        self.threshold = float(threshold)
        self.name = f"uncertainty(U>{threshold:g})"

    def should_correct(self, ctx: PolicyContext) -> bool:
        return ctx.marker_available and ctx.U > self.threshold
