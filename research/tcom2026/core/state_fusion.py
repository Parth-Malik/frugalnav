"""
core/state_fusion.py
--------------------
Tight state fusion -- Parth's module (plan section 1, "State fusion"). This is the
half of the architecture that makes the plan's split honest:

    "Decoupled modules, fused estimate." The software modules are decoupled and
    independently testable; the ESTIMATE is one tightly-fused state. An AVL fix is
    merged INTO the state, not kept as a parallel track. (Plan clarification 1.)

So there is exactly ONE `NavState` the rest of the system reads, and it is a
proper little estimator:

  * predict(dt): between fixes, VIO supplies relative motion. Position integrates
    that motion and the covariance GROWS (process noise ~ how much we distrust the
    dead-reckoning). This growing covariance is what the scheduler reads as
    sigma_pos -- the fused estimate and the trigger signal are the same object.

  * update(fix): a landmark fix is an absolute position measurement. A Kalman
    gain K = P (P + R)^-1 optimally blends prior covariance P with the fix
    covariance R, pulls the position onto the fix, and SHRINKS P. Because only the
    drone's state moves (target B is a world constant), the vector to B is
    corrected for free -- correcting drift never moves B.

Siddharth's `landmark_corrector.reanchor()` is the decoupled REFERENCE version of
the measurement update (one fix, no prediction bookkeeping); this module is the
canonical stateful estimator that owns the fused state across the whole flight.

Pure NumPy, 2x2 fixed state. Maps directly onto the C++/Eigen port.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.types import LandmarkFix, NavState


@dataclass
class FusionConfig:
    # process-noise growth per unit distance travelled (variance, m^2 per m).
    # Higher = we trust dead-reckoning less, so P grows faster between fixes.
    q_per_metre: float = 0.02
    # a small floor so P never collapses to exactly zero after a fix.
    p_floor: float = 1e-4


class StateFusion:
    """Owns the single fused NavState. Predict with VIO motion; update with fixes."""

    def __init__(self, init_xy=(0.0, 0.0), init_yaw=0.0, init_std=0.1,
                 cfg: FusionConfig | None = None):
        self.cfg = cfg or FusionConfig()
        self.state = NavState(
            xy=np.asarray(init_xy, dtype=float).reshape(2).copy(),
            yaw=float(init_yaw),
            covariance=np.eye(2) * (float(init_std) ** 2),
        )

    # -- prediction (dead-reckon on VIO relative motion) --------------------
    def predict(self, delta_xy, yaw=None, extra_var=None, t=None) -> NavState:
        """Advance the state by a VIO relative-motion increment `delta_xy` and grow
        the covariance. `extra_var` optionally injects additional process variance
        for a hard/low-texture stretch (the scheduler will see the larger sigma_pos)."""
        d = np.asarray(delta_xy, dtype=float).reshape(2)
        self.state.xy = self.state.xy + d
        if yaw is not None:
            self.state.yaw = float(yaw)

        dist = float(np.linalg.norm(d))
        q = self.cfg.q_per_metre * dist
        if extra_var:
            q += float(extra_var)
        self.state.covariance = self.state.covariance + np.eye(2) * q
        if t is not None:
            self.state.timestamp = float(t)
        return self.state

    # -- measurement update (absolute landmark fix) ------------------------
    def update(self, fix: LandmarkFix, gain=None) -> NavState:
        """Merge an absolute LandmarkFix into the state (tight fusion).

        gain=None  -> optimal Kalman blend K = P (P+R)^-1
        gain=float -> forced fixed gain in [0,1] (gain=1.0 snaps fully onto the
                      fix; used for the crisp 'money-shot' demo behaviour).
        Only the drone state moves; target B (elsewhere) is untouched."""
        P = np.asarray(self.state.covariance, dtype=float).reshape(2, 2)
        R = np.asarray(fix.covariance, dtype=float).reshape(2, 2)

        if gain is None:
            K = P @ np.linalg.inv(P + R)
        else:
            K = np.eye(2) * float(gain)

        innovation = np.asarray(fix.xy, dtype=float).reshape(2) - self.state.xy
        self.state.xy = self.state.xy + K @ innovation
        newP = (np.eye(2) - K) @ P
        # keep symmetric + floored (numerical hygiene for the long C++ port too)
        newP = 0.5 * (newP + newP.T) + np.eye(2) * self.cfg.p_floor
        self.state.covariance = newP
        self.state.yaw = float(fix.yaw)
        self.state.timestamp = float(fix.timestamp)
        return self.state

    # -- what the scheduler reads ------------------------------------------
    def sigma_pos(self) -> float:
        """Position std-dev [m] = sqrt of the larger covariance eigenvalue. This is
        the fused estimate's own uncertainty -- the primary cue the scheduler gates
        on, so the trigger signal and the estimate are literally the same state."""
        P = np.asarray(self.state.covariance, dtype=float).reshape(2, 2)
        evals = np.linalg.eigvalsh(0.5 * (P + P.T))
        return float(np.sqrt(max(evals[-1], 0.0)))


# ----------------------------- self-test ------------------------------------
if __name__ == "__main__":
    # 1) prediction grows covariance; a fix shrinks it and pulls onto the fix.
    fus = StateFusion(init_xy=(0.0, 0.0), init_std=0.05)
    s0 = fus.sigma_pos()
    for _ in range(50):
        fus.predict((0.2, 0.0))                       # travel +10 m in +x
    s1 = fus.sigma_pos()
    assert s1 > s0, (s0, s1)
    print(f"sigma_pos grew under dead-reckoning: {s0:.3f} -> {s1:.3f} m")

    fix = LandmarkFix(xy=np.array([10.0, 0.0]), yaw=0.0,
                      covariance=np.eye(2) * (0.02 ** 2), marker_id=3, timestamp=1.0)
    fus.update(fix)                                   # optimal blend
    s2 = fus.sigma_pos()
    assert s2 < s1, (s1, s2)
    assert abs(fus.state.xy[0] - 10.0) < 0.5          # pulled onto the fix
    print(f"sigma_pos shrank after fix: {s1:.3f} -> {s2:.3f} m; xy={fus.state.xy}")

    # 2) forced gain=1.0 snaps fully onto the fix (money-shot behaviour).
    fus2 = StateFusion(init_xy=(5.0, 5.0), init_std=1.0)
    fus2.update(LandmarkFix(xy=np.array([0.0, 0.0]), yaw=0.0,
                            covariance=np.eye(2) * 0.01, marker_id=0), gain=1.0)
    assert np.allclose(fus2.state.xy, [0.0, 0.0], atol=1e-6)
    print(f"gain=1.0 snap: xy={fus2.state.xy}")

    print("\nstate_fusion self-test passed.")
