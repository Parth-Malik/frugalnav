"""
core/navigator.py
-----------------
The integrated portable hot loop -- the one place every module meets. This is the
"portable navigation core" the plan promises (section 6): sensor struct in, command
struct out, fixed-size state, no heavy dependencies. It is exactly what ports to
C++/Eigen (Week 7) and, ultimately, the RISC-V SoC.

One step wires the whole architecture together:

    VIO delta ─► StateFusion.predict ──────────────► fused NavState (P grows)
                                    │
    glass-box cues + fused sigma_pos ─► UncertaintyScheduler.compute ─► (U, trigger)
                                    │
       if trigger AND a marker is in view:
           LandmarkCorrector.correct(sighting) ─► LandmarkFix
           StateFusion.update(fix) ─────────────► fused NavState (P shrinks)   [tight]
                                    │
    obstacle (ttc,bearing) ─► ObstacleAvoidance.update ─► evasion vector
                                    │
    TargetCentricController.command(fused_xy, evasion) ─► VelocityCmd  (out)

The scheduler reads sigma_pos FROM the fused state, so the trigger signal and the
estimate are the same object -- the "decoupled modules, one fused estimate" design
(plan clarification 1). Only the drone state ever moves on a fix; target B is a
world constant, so the vector to B self-corrects (clarification 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.controller import TargetCentricController
from core.landmark_corrector import LandmarkCorrector
from core.obstacle_avoidance import ObstacleAvoidance
from core.state_fusion import StateFusion
from core.types import MarkerSighting, VelocityCmd
from core.uncertainty_scheduler import UncertaintyScheduler


@dataclass
class SensorInput:
    """Everything the core learns about one instant. Populated by a front-end (the
    sim, a dataset replay, or a real phone/VIO rig) -- the core never sees pixels."""
    t: float
    vio_delta: np.ndarray                    # (2,) relative-motion increment from VIO
    cues: dict = field(default_factory=dict) # feature_loss, blur, imu_bias,
                                             # active_features, sigma_head, extra_var
    sighting: "MarkerSighting | None" = None # a mapped marker seen this frame, or None
    obstacle: "tuple | None" = None          # (ttc_s, bearing_rad) or None if clear


@dataclass
class NavOutput:
    """Command struct out, plus the introspection the demos/plots and the report
    need (all glass-box)."""
    cmd: VelocityCmd
    est_xy: np.ndarray
    sigma_pos: float
    U: float
    trigger: bool
    reason: str
    corrected: bool
    evading: bool
    components: dict


class Navigator:
    def __init__(self, controller, scheduler, fusion, corrector=None, avoider=None,
                 fuse_gain=None):
        self.controller = controller          # TargetCentricController
        self.scheduler = scheduler            # UncertaintyScheduler
        self.fusion = fusion                  # StateFusion (owns the fused NavState)
        self.corrector = corrector            # LandmarkCorrector or None
        self.avoider = avoider                # ObstacleAvoidance or None
        self.fuse_gain = fuse_gain            # None = optimal Kalman blend; 1.0 = snap

    @property
    def est_xy(self) -> np.ndarray:
        return self.fusion.state.xy

    def step(self, si: SensorInput) -> NavOutput:
        # 1. PREDICT: dead-reckon the fused state on the VIO motion; P grows.
        self.fusion.predict(si.vio_delta, extra_var=si.cues.get("extra_var"), t=si.t)

        # 2. CUES: sigma_pos comes from the fused covariance (trigger == estimate).
        cues = dict(si.cues)
        cues["sigma_pos"] = self.fusion.sigma_pos()
        cues.setdefault("active_features", 9999)

        # 3. SCHEDULE: should we spend an AVL fix now?
        U, trigger, reason, comp = self.scheduler.compute(cues)

        # 4. CORRECT: only if the scheduler fired AND a marker is actually in view.
        corrected = False
        if trigger and si.sighting is not None and self.corrector is not None:
            fix = self.corrector.correct(si.sighting)
            if fix is not None:
                self.fusion.update(fix, gain=self.fuse_gain)   # tight fusion
                self.scheduler.reset_after_fix()               # start refractory window
                corrected = True

        est = self.fusion.state.xy

        # 5. AVOID: reactive evasion from time-to-contact (real sensing, not estimate).
        # Always tick the avoider so its hysteresis releases when the way clears;
        # a missing obstacle cue means "no contact", i.e. TTC = +inf.
        evasion = np.zeros(2)
        seek_dir = self.controller.B - est                     # world vector toward B
        if self.avoider is not None:
            ttc, bearing = si.obstacle if si.obstacle is not None else (float("inf"), 0.0)
            evasion = self.avoider.update(seek_dir, ttc, bearing)

        # 6. COMMAND: target-centric homing with evasion merged in.
        cmd = self.controller.command(est, evasion)

        return NavOutput(
            cmd=cmd, est_xy=est.copy(), sigma_pos=cues["sigma_pos"],
            U=U, trigger=trigger, reason=reason, corrected=corrected,
            evading=bool(self.avoider.evading) if self.avoider is not None else False,
            components=comp,
        )

    def arrived(self) -> bool:
        return self.controller.arrived(self.fusion.state.xy)


def build_navigator(target_B, landmark_map=None, start_xy=(0.0, 0.0),
                    scheduler=None, controller=None, fusion=None,
                    avoider=None, fuse_gain=None):
    """Assemble a Navigator from the core modules with sensible defaults. Pass a
    LandmarkMap to enable landmark correction; omit it for a pure-VIO baseline."""
    controller = controller or TargetCentricController(target_B=target_B)
    scheduler = scheduler or UncertaintyScheduler()
    fusion = fusion or StateFusion(init_xy=start_xy)
    corrector = LandmarkCorrector(landmark_map) if landmark_map is not None else None
    avoider = avoider if avoider is not None else ObstacleAvoidance()
    return Navigator(controller, scheduler, fusion, corrector, avoider, fuse_gain)
