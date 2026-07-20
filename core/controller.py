"""
core/controller.py
------------------
The target-centric controller -- Parth's module (plan section 1, "Target-centric
controller"). Until now this logic lived inline inside `harness/kinematic_sim.py`;
it belongs in the portable core, because it is part of the hot loop that ports to
C++/RISC-V and it must be independently testable.

Target-centric formulation
---------------------------
The drone does not track a path; it tracks a POINT. Target B is a fixed point in
the world frame (defined by the landmark map -- plan clarification 2). The
controller publishes the target-centric OFFSET  o = est - B  and commands a
velocity that drives that offset to zero:

    v_cmd = -Kp * (est - B)  +  evasion         (world frame)

With B at the world origin (the sim's convention) this is exactly the plan's
"command vector (-x, -y)". The evasion vector from the obstacle module is merged
in additively, then the whole command is clamped to v_max.

The feedback loop the plan flags (section 2, constraint 4)
----------------------------------------------------------
The command is computed from the ESTIMATE, not truth. So estimate error feeds
straight into a command error, which drives more true drift -- a loop unique to
the target-centric formulation. This module is where that loop closes, which is
exactly why the scheduler's job is to keep `est` honest before the loop compounds.

Pure and portable: NumPy only, fixed-size state, no allocation in `command()`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.types import VelocityCmd


@dataclass
class ControllerConfig:
    kp: float = 0.6            # homing proportional gain (1/s)
    v_max: float = 2.0        # speed clamp [m/s] -- handheld-ish, keeps KLT comfortable
    arrive_tol: float = 1.0   # [m]: 'arrived' when the estimated offset is within this
    deadband: float = 0.0     # [m]: below this offset, stop commanding (anti-jitter)


class TargetCentricController:
    """Homes the drone onto target B from the fused estimate.

    B is a world-frame constant; passing it explicitly (default: origin) keeps the
    controller usable both in the sim (B at 0,0) and on a surveyed map where B is a
    real coordinate. Correcting drift moves `est`, never B, so the commanded vector
    self-corrects the moment a landmark fix lands (plan clarification 2)."""

    def __init__(self, target_B=(0.0, 0.0), cfg: ControllerConfig | None = None):
        self.B = np.asarray(target_B, dtype=float).reshape(2)
        self.cfg = cfg or ControllerConfig()

    # -- the published target-centric quantity ------------------------------
    def offset(self, est_xy) -> np.ndarray:
        """o = est - B, the target-centric offset the controller regulates to 0."""
        return np.asarray(est_xy, dtype=float).reshape(2) - self.B

    def range_to_target(self, est_xy) -> float:
        return float(np.linalg.norm(self.offset(est_xy)))

    # -- the command --------------------------------------------------------
    def command(self, est_xy, evasion=(0.0, 0.0)) -> VelocityCmd:
        """Return the velocity command driving `est` toward B, with an evasion
        vector merged in. Deterministic; no internal state mutated."""
        o = self.offset(est_xy)
        ev = np.asarray(evasion, dtype=float).reshape(2)

        if np.linalg.norm(o) <= self.cfg.deadband:
            seek = np.zeros(2)
        else:
            seek = -self.cfg.kp * o                 # drive the offset to zero

        v = seek + ev
        speed = float(np.linalg.norm(v))
        if speed > self.cfg.v_max:
            v = v / speed * self.cfg.v_max
        return VelocityCmd(vx=float(v[0]), vy=float(v[1]))

    def arrived(self, est_xy) -> bool:
        """The drone BELIEVES it has arrived when its estimated offset is small.
        (Its TRUE miss distance then equals the residual localization error -- the
        arrival-error metric. That coupling is the whole point of section 2.4.)"""
        return self.range_to_target(est_xy) < self.cfg.arrive_tol


# ----------------------------- self-test ------------------------------------
if __name__ == "__main__":
    # A drone offset from B should be commanded back toward B, speed-clamped.
    ctrl = TargetCentricController(target_B=(0.0, 0.0))
    cmd = ctrl.command((60.0, 25.0))
    v = np.array([cmd.vx, cmd.vy])
    assert np.linalg.norm(v) <= ctrl.cfg.v_max + 1e-9
    # command must point back toward the origin (opposite the offset)
    assert float(np.dot(v, np.array([60.0, 25.0]))) < 0
    print(f"homing cmd from (60,25): v=({cmd.vx:.3f},{cmd.vy:.3f})  |v|={np.linalg.norm(v):.3f}")

    # Evasion merges in and bends the command sideways.
    cmd2 = ctrl.command((60.0, 25.0), evasion=(0.0, 2.0))
    assert cmd2.vy > cmd.vy
    print(f"with evasion (0,2): v=({cmd2.vx:.3f},{cmd2.vy:.3f})")

    # A non-origin target: command drives toward B, arrival is B-relative.
    ctrl2 = TargetCentricController(target_B=(30.0, 42.0))
    assert not ctrl2.arrived((0.0, 0.0))
    assert ctrl2.arrived((30.2, 42.1))
    c3 = ctrl2.command((25.0, 40.0))
    assert c3.vx > 0 and c3.vy > 0           # B is up-and-right of the estimate
    print(f"toward B=(30,42) from (25,40): v=({c3.vx:.3f},{c3.vy:.3f})")

    print("\ncontroller self-test passed.")
