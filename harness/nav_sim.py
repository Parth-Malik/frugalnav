"""
Navigation simulator for the scheduling comparison (harness -- throwaway).

A flight from start to target B along +X, crossing a HARD ZONE in the middle --
a feature-poor, high-drift stretch (think: over water, low texture, motion blur).
Markers recur along the whole path. Given a correction policy, it:

    1. integrates drifted odometry (drift rate is higher in the hard zone),
    2. tracks the glass-box signals and assembles the uncertainty metric U,
    3. at each marker opportunity, lets the policy decide whether to spend a fix,
    4. records the trajectory + bookkeeping for the metrics.

The heterogeneous drift is the whole point: it is what separates the policies.
A fixed-period schedule is blind to it (it must be tuned to the worst segment and
so over-corrects the easy ones); the uncertainty policy spends fixes exactly
where drift is actually growing. Nothing here is rigged -- U is driven by the
same local drift rate that drives the real error, which is precisely why
thresholding U bounds the error efficiently.
"""
from __future__ import annotations

import numpy as np

from core.policies import PolicyContext
from core.uncertainty import UncertaintyWeights, uncertainty


def build_scenario(length=60.0, step_len=1.0, hard_zone=(24.0, 40.0),
                   marker_spacing=2):
    """Ground truth: straight flight 0 -> length along X; B at the far end."""
    xs = np.arange(0.0, length + 1e-9, step_len)
    true_xy = np.column_stack([xs, np.zeros_like(xs)])
    n = len(xs)

    in_hard = (xs >= hard_zone[0]) & (xs <= hard_zone[1])
    drift_rate = np.where(in_hard, 4.0, 1.0)          # 4x drift in the hard zone

    marker_available = np.zeros(n, dtype=bool)
    marker_available[::marker_spacing] = True

    return {
        "xs": xs,
        "true_xy": true_xy,
        "B": true_xy[-1].copy(),
        "n": n,
        "in_hard": in_hard,
        "hard_zone": hard_zone,
        "drift_rate": drift_rate,
        "marker_available": marker_available,
        "marker_spacing": marker_spacing,
    }


def run_policy(scenario, policy, seed=0, weights=None,
               base_drift=0.03, lateral_bias=0.6, fix_noise=0.03, detect_prob=0.97):
    """Run one policy over the scenario. Returns estimate + U trace + bookkeeping.

    Drift model: each step adds white noise (std ∝ local drift rate) plus a small
    systematic lateral bias (also ∝ rate) -- an integrated relative-motion error.
    A successful fix snaps the estimate back to truth (± fix noise) and resets the
    uncertainty accumulators.
    """
    rng = np.random.default_rng(seed)
    if weights is None:
        weights = UncertaintyWeights()

    true_xy = scenario["true_xy"]
    n = scenario["n"]
    rate_mult = scenario["drift_rate"]
    in_hard = scenario["in_hard"]
    marker_available = scenario["marker_available"]

    est = np.zeros((n, 2))
    est[0] = true_xy[0].copy()
    U_trace = np.zeros(n)

    sigma_pos = 0.0          # grows since last fix, at the local drift rate
    steps_since = 0
    correction_steps = []
    n_invocations = 0        # detector/corrector runs = compute spent
    n_detected = 0           # successful fixes

    for k in range(1, n):
        rate = base_drift * rate_mult[k]

        # 1. propagate with drifted relative motion
        true_delta = true_xy[k] - true_xy[k - 1]
        drift_vec = rng.normal(0.0, rate, 2) + np.array([0.0, lateral_bias * rate])
        est[k] = est[k - 1] + true_delta + drift_vec

        # 2. glass-box signals -> uncertainty metric U
        sigma_pos += rate
        steps_since += 1
        feature_loss = (0.8 if in_hard[k] else 0.1) + rng.normal(0.0, 0.03)
        feature_loss = max(0.0, feature_loss)
        blur = max(0.0, rng.normal(0.10, 0.05))     # modeled here; real blur_metric
                                                     # is exercised in demo_blur_metric.py
        U = uncertainty(sigma_pos, feature_loss=feature_loss, blur=blur, w=weights)
        U_trace[k] = U

        # 3. policy decision at marker opportunities
        ctx = PolicyContext(step=k, t=float(k), steps_since_last=steps_since,
                            U=U, marker_available=bool(marker_available[k]))
        if policy.should_correct(ctx):
            n_invocations += 1                       # we run the detector...
            if rng.random() < detect_prob:           # ...and it may or may not succeed
                n_detected += 1
                est[k] = true_xy[k] + rng.normal(0.0, fix_noise, 2)
                sigma_pos = 0.0
                steps_since = 0
                correction_steps.append(k)

    return {
        "policy_name": policy.name,
        "est_xy": est,
        "U": U_trace,
        "correction_steps": correction_steps,
        "n_invocations": n_invocations,
        "n_detected": n_detected,
    }


def fixed_period_sweep(scenario, periods, seed=0, **kw):
    """Run a range of fixed periods -> the (invocations, peak-drift) Pareto curve
    the uncertainty policy is measured against."""
    from core.policies import FixedPeriodPolicy
    from core.metrics import peak_drift
    out = []
    for p in periods:
        run = run_policy(scenario, FixedPeriodPolicy(p), seed=seed, **kw)
        out.append({
            "period": p,
            "invocations": run["n_invocations"],
            "corrections": len(run["correction_steps"]),
            "peak_drift_m": peak_drift(run["est_xy"], scenario["true_xy"]),
        })
    return out
