"""
Evaluation metrics -- Siddharth owns these (plan, Weeks 4 & 6).

The headline comparison of the project: *similar accuracy, far fewer
corrections.* These functions turn a run (estimate vs ground truth + the list of
corrections) into the numbers that make that case:

    peak drift            worst-case localization error over the flight  [m]
    arrival error         miss distance at target B                      [m]
    RMSE                  overall track accuracy                         [m]
    AVL invocations       times the detector/corrector was run (compute)
    correction count      successful fixes applied (frugality)
    marker success rate   successful fixes / invocations
"""
from __future__ import annotations

import numpy as np


def _err(est_xy, true_xy):
    return np.linalg.norm(np.asarray(est_xy) - np.asarray(true_xy), axis=1)


def peak_drift(est_xy, true_xy) -> float:
    return float(_err(est_xy, true_xy).max())


def final_drift(est_xy, true_xy) -> float:
    return float(_err(est_xy, true_xy)[-1])


def rmse(est_xy, true_xy) -> float:
    e = _err(est_xy, true_xy)
    return float(np.sqrt(np.mean(e ** 2)))


def arrival_error(est_xy, true_xy) -> float:
    """Miss distance at the target. Under estimate-based guidance the drone stops
    when its ESTIMATE reaches B, so its true distance to B equals the final
    localization error -- i.e. how far it actually is from B when it 'arrives'."""
    return final_drift(est_xy, true_xy)


def correction_count(correction_steps) -> int:
    return int(len(correction_steps))


def marker_success_rate(n_detected: int, n_invocations: int) -> float:
    return float(n_detected / max(1, n_invocations))


def summarize(run: dict, true_xy) -> dict:
    """Roll a run dict (from nav_sim.run_policy) into a flat metrics dict."""
    est = run["est_xy"]
    return {
        "policy": run.get("policy_name", "?"),
        "peak_drift_m": peak_drift(est, true_xy),
        "arrival_error_m": arrival_error(est, true_xy),
        "rmse_m": rmse(est, true_xy),
        "avl_invocations": int(run["n_invocations"]),
        "corrections": correction_count(run["correction_steps"]),
        "marker_success_rate": marker_success_rate(run["n_detected"], run["n_invocations"]),
    }
