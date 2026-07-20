"""
Evaluation simulator for Week 6 (harness -- throwaway).

One parameterised model, three terrains, so the three correction policies are
compared on equal footing (the whole point of an evaluation chapter):

    A  open, NO markers          -> no fix is ever possible: the drift floor
    B  markers along the path    -> benign; corrections keep error small
    C  obstacles + markers        -> longer route with a high-drift "maneuver"
                                     zone (the detour) -> bursty drift

The drift model, uncertainty signal and correction mechanics are exactly those of
Week 4 (`nav_sim`), generalised so each scenario is just a drift-rate profile +
marker layout. This module runs the full matrix (scenario x policy x seed) and
aggregates Siddharth's metrics.
"""
from __future__ import annotations

import numpy as np

from core.metrics import summarize
from core.policies import FixedPeriodPolicy, NonePolicy, UncertaintyPolicy
from core.uncertainty import UncertaintyWeights, uncertainty


# ----------------------------------------------------------------------------
def build_scenario(spec, common):
    """spec: one scenario dict from the config. Returns arrays the sim needs."""
    step = float(common["step_len"])
    xs = np.arange(0.0, float(spec["length"]) + 1e-9, step)
    n = len(xs)
    true_xy = np.column_stack([xs, np.zeros_like(xs)])       # straight reference path

    lo, hi = spec["hard_zone"]
    in_hard = (xs >= lo) & (xs <= hi)
    drift_rate = np.where(in_hard, float(spec["hard_rate"]), 1.0)

    marker_available = np.zeros(n, dtype=bool)
    if spec["markers"]:
        marker_available[:: int(common["marker_spacing"])] = True

    return {
        "label": spec["label"],
        "xs": xs, "true_xy": true_xy, "B": true_xy[-1].copy(), "n": n,
        "in_hard": in_hard, "drift_rate": drift_rate,
        "marker_available": marker_available,
    }


def run_policy(scenario, policy, common, seed=0, weights=None):
    """Identical mechanics to Week-4 nav_sim.run_policy, driven by the scenario's
    per-step drift-rate and marker-availability arrays."""
    from core.policies import PolicyContext
    rng = np.random.default_rng(seed)
    if weights is None:
        weights = UncertaintyWeights()

    true_xy = scenario["true_xy"]
    n = scenario["n"]
    rate_mult = scenario["drift_rate"]
    in_hard = scenario["in_hard"]
    marker_available = scenario["marker_available"]

    base_drift = float(common["base_drift"])
    lateral_bias = float(common["lateral_bias"])
    fix_noise = float(common["fix_noise"])
    detect_prob = float(common["detect_prob"])

    est = np.zeros((n, 2))
    est[0] = true_xy[0].copy()
    U_trace = np.zeros(n)

    sigma_pos = 0.0
    steps_since = 0
    correction_steps = []
    n_invocations = 0
    n_detected = 0

    for k in range(1, n):
        rate = base_drift * rate_mult[k]

        true_delta = true_xy[k] - true_xy[k - 1]
        drift_vec = rng.normal(0.0, rate, 2) + np.array([0.0, lateral_bias * rate])
        est[k] = est[k - 1] + true_delta + drift_vec

        sigma_pos += rate
        steps_since += 1
        feature_loss = max(0.0, (0.8 if in_hard[k] else 0.1) + rng.normal(0.0, 0.03))
        blur = max(0.0, rng.normal(0.10, 0.05))
        U = uncertainty(sigma_pos, feature_loss=feature_loss, blur=blur, w=weights)
        U_trace[k] = U

        ctx = PolicyContext(step=k, t=float(k), steps_since_last=steps_since,
                            U=U, marker_available=bool(marker_available[k]))
        if policy.should_correct(ctx):
            n_invocations += 1
            if rng.random() < detect_prob:
                n_detected += 1
                est[k] = true_xy[k] + rng.normal(0.0, fix_noise, 2)
                sigma_pos = 0.0
                steps_since = 0
                correction_steps.append(k)

    return {
        "policy_name": policy.name,
        "est_xy": est, "U": U_trace,
        "correction_steps": correction_steps,
        "n_invocations": n_invocations, "n_detected": n_detected,
    }


# ----------------------------------------------------------------------------
def make_policies(cfg_policies):
    """The three policies compared, in a fixed display order."""
    return [
        ("none", lambda: NonePolicy()),
        ("fixed", lambda: FixedPeriodPolicy(int(cfg_policies["fixed_period"]))),
        ("uncertainty", lambda: UncertaintyPolicy(float(cfg_policies["uncertainty_threshold"]))),
    ]


def evaluate(config):
    """Run the full matrix (scenario x policy x seed) and aggregate.

    Returns (raw_rows, summary_rows):
      raw_rows    -- one dict per individual run (for the CSV / variance)
      summary_rows-- one dict per (scenario, policy) with mean & std of each metric
    """
    common = config["common"]
    seeds = range(int(config["seeds"]))
    policies = make_policies(config["policies"])

    metric_keys = ["arrival_error_m", "peak_drift_m", "rmse_m",
                   "avl_invocations", "corrections", "marker_success_rate"]

    raw_rows, summary_rows = [], []
    for sid, spec in config["scenarios"].items():
        scen = build_scenario(spec, common)
        for pkey, factory in policies:
            per_seed = []
            for s in seeds:
                run = run_policy(scen, factory(), common, seed=s)
                m = summarize(run, scen["true_xy"])
                m.update(scenario=sid, scenario_label=spec["label"],
                         policy=pkey, seed=s, has_markers=bool(spec["markers"]))
                raw_rows.append(m)
                per_seed.append(m)

            row = {"scenario": sid, "scenario_label": spec["label"], "policy": pkey,
                   "has_markers": bool(spec["markers"]), "n_seeds": len(per_seed)}
            for k in metric_keys:
                vals = np.array([r[k] for r in per_seed], dtype=float)
                row[f"{k}_mean"] = float(vals.mean())
                row[f"{k}_std"] = float(vals.std())
            summary_rows.append(row)

    return raw_rows, summary_rows
