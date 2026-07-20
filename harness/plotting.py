"""
harness/plotting.py  (MERGED -- throwaway scaffolding, plan section 6)
---------------------------------------------------------------------
All figure generators for the demos, merged from Siddharth's per-week plotting
modules (weeks 3-6). Non-interactive Agg backend so every demo saves PNGs
headlessly (no display needed, works in CI / over SSH). Function names are
disjoint across weeks, so this is a straight union:

  Week 3 (landmark correction) : save_moneyshot, save_error_plot
  Week 4 (scheduling mid-term)  : save_trajectories, save_error_time,
                                  save_uncertainty_signal, save_pareto
  Week 5 (obstacle detour)      : save_detour_moneyshot, save_vector_error,
                                  save_tracking
  Week 6 (evaluation)           : save_evaluation_bars, save_frugality_scatter
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


# ============================ from week3 ============================


def _drift(est_xy, true_xy):
    return np.linalg.norm(est_xy - true_xy, axis=1)


def save_moneyshot(scenario, corrected, uncorrected, lmap, out_path, title=None):
    """The Week 3 money-shot: drift accumulates, then snaps back at each marker,
    with target B fixed in the world frame."""
    true_xy = scenario["true_xy"]
    markers = lmap.all_world_xy()
    B = lmap.target_B

    fig, ax = plt.subplots(figsize=(7.5, 8.0))
    ax.plot(true_xy[:, 0], true_xy[:, 1], "-", color="#2ca02c", lw=2.5,
            label="ground truth", zorder=3)
    ax.plot(uncorrected["est_xy"][:, 0], uncorrected["est_xy"][:, 1], "--",
            color="#d62728", lw=2.0, label="VIO estimate, NO correction", zorder=4)
    ax.plot(corrected["est_xy"][:, 0], corrected["est_xy"][:, 1], "-",
            color="#1f77b4", lw=2.0, label="VIO + landmark correction", zorder=5)

    ax.scatter(markers[:, 0], markers[:, 1], marker="s", s=70,
               facecolor="white", edgecolor="black", linewidths=1.5,
               label="ArUco markers (world frame)", zorder=6)

    # mark where corrections fired
    cs = corrected["correction_steps"]
    if cs:
        pts = corrected["est_xy"][cs]
        ax.scatter(pts[:, 0], pts[:, 1], marker="o", s=45, color="#1f77b4",
                   edgecolor="white", linewidths=0.8, zorder=7,
                   label="correction fired")

    ax.scatter([B[0]], [B[1]], marker="*", s=340, color="#ff7f0e",
               edgecolor="black", linewidths=1.0, label="target B (fixed)", zorder=8)

    ax.set_xlabel("world X [m]")
    ax.set_ylabel("world Y [m]")
    ax.set_title(title or "Week 3 -- landmark correction snaps drift back to truth")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_error_plot(scenario, corrected, uncorrected, out_path):
    """Localization error vs time: monotonic growth without correction, the
    classic sawtooth (grow, then snap to ~0 at a marker) with correction."""
    true_xy = scenario["true_xy"]
    t = scenario["times"]
    err_u = _drift(uncorrected["est_xy"], true_xy)
    err_c = _drift(corrected["est_xy"], true_xy)

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.plot(t, err_u, "--", color="#d62728", lw=2.0, label="NO correction")
    ax.plot(t, err_c, "-", color="#1f77b4", lw=2.0, label="landmark correction")
    for k in corrected["correction_steps"]:
        ax.axvline(t[k], color="#1f77b4", alpha=0.18, lw=1.0)

    ax.set_xlabel("step")
    ax.set_ylabel("position error  ||est - truth||  [m]")
    ax.set_title("Week 3 -- drift error over time (vertical lines = corrections)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

# ============================ from week4 ============================

_COLORS = {"none": "#7f7f7f"}
_FIXED_C = "#d62728"
_UNC_C = "#1f77b4"


def _policy_color(name):
    if name.startswith("none"):
        return "#7f7f7f"
    if name.startswith("fixed"):
        return _FIXED_C
    return _UNC_C


def _shade_hard(ax, scenario, vertical=True):
    lo, hi = scenario["hard_zone"]
    if vertical:
        ax.axvspan(lo, hi, color="#ffe08a", alpha=0.35, label="hard zone (feature-poor)")
    else:
        ax.axvspan(lo, hi, color="#ffe08a", alpha=0.35)


def save_trajectories(scenario, runs, out_path):
    true_xy = scenario["true_xy"]
    B = scenario["B"]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    _shade_hard(ax, scenario)
    ax.plot(true_xy[:, 0], true_xy[:, 1], "-", color="#2ca02c", lw=2.5,
            label="ground truth", zorder=3)
    for run in runs:
        c = _policy_color(run["policy_name"])
        ax.plot(run["est_xy"][:, 0], run["est_xy"][:, 1], "-", lw=1.8,
                color=c, label=run["policy_name"], zorder=4)
        cs = run["correction_steps"]
        if cs:
            pts = run["est_xy"][cs]
            ax.scatter(pts[:, 0], pts[:, 1], s=28, color=c, edgecolor="white",
                       linewidths=0.6, zorder=5)
    ax.scatter([B[0]], [B[1]], marker="*", s=320, color="#ff7f0e",
               edgecolor="black", zorder=6, label="target B")
    ax.set_xlabel("world X [m]")
    ax.set_ylabel("world Y [m]  (lateral drift)")
    ax.set_title("Week 4 -- trajectories under each correction policy")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_error_time(scenario, runs, out_path):
    true_xy = scenario["true_xy"]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    _shade_hard(ax, scenario)
    for run in runs:
        err = np.linalg.norm(run["est_xy"] - true_xy, axis=1)
        ax.plot(scenario["xs"], err, "-", lw=1.9,
                color=_policy_color(run["policy_name"]), label=run["policy_name"])
    ax.set_xlabel("world X [m]  (~ time)")
    ax.set_ylabel("position error  ||est - truth||  [m]")
    ax.set_title("Week 4 -- localization error over the flight")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_uncertainty_signal(scenario, unc_run, threshold, out_path):
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    _shade_hard(ax, scenario)
    ax.plot(scenario["xs"], unc_run["U"], "-", color=_UNC_C, lw=1.8,
            label="uncertainty U")
    ax.axhline(threshold, color="black", ls="--", lw=1.3,
               label=f"threshold ({threshold:g})")
    for k in unc_run["correction_steps"]:
        ax.axvline(scenario["xs"][k], color=_UNC_C, alpha=0.25, lw=1.2)
    ax.scatter(scenario["xs"][unc_run["correction_steps"]],
               unc_run["U"][unc_run["correction_steps"]], s=40, color=_UNC_C,
               edgecolor="white", zorder=5, label="fix fired")
    ax.set_xlabel("world X [m]  (~ time)")
    ax.set_ylabel("uncertainty  U")
    ax.set_title("Week 4 -- U rises with drift; the scheduler fires only above threshold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_pareto(fixed_curve, unc_curve, none_point, out_path, highlight=None):
    """Frugality Pareto: AVL invocations (compute) vs peak drift (accuracy).
    Lower-left is better. The uncertainty curve sitting below/left of the fixed
    curve is the whole result: same accuracy for fewer corrections."""
    fig, ax = plt.subplots(figsize=(7.8, 5.6))

    fx = [r["invocations"] for r in fixed_curve]
    fy = [r["peak_drift_m"] for r in fixed_curve]
    ax.plot(fx, fy, "-o", color=_FIXED_C, lw=2, ms=6, label="fixed-period (swept P)")

    ux = [r["invocations"] for r in unc_curve]
    uy = [r["peak_drift_m"] for r in unc_curve]
    ax.plot(ux, uy, "-s", color=_UNC_C, lw=2, ms=6, label="uncertainty-aware (swept U*)")

    ax.scatter([none_point["invocations"]], [none_point["peak_drift_m"]],
               marker="X", s=120, color="#7f7f7f", zorder=5, label="none")

    if highlight:
        f, u = highlight
        ax.annotate("", xy=(u["invocations"], u["peak_drift_m"]),
                    xytext=(f["invocations"], f["peak_drift_m"]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        ax.annotate("same accuracy,\nfewer AVL calls",
                    xy=((f["invocations"]+u["invocations"])/2,
                        (f["peak_drift_m"]+u["peak_drift_m"])/2 + 0.05),
                    fontsize=9, ha="center")

    ax.set_xlabel("AVL invocations over the flight  (compute / power)")
    ax.set_ylabel("peak drift  [m]  (accuracy)")
    ax.set_title("Week 4 -- frugality Pareto: uncertainty-aware dominates fixed-period")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

# ============================ from week5 ============================

_CORR = "#1f77b4"      # with landmark correction
_NONE = "#d62728"      # no correction


def _evade_span(run):
    """(first, last) step index where the run was evading, for shading."""
    idx = [i for i, m in enumerate(run["mode"]) if m == "evade"]
    return (idx[0], idx[-1]) if idx else (None, None)


def save_detour_moneyshot(corr, none, lmap, out_path):
    obs_c, obs_r, standoff = corr["obstacle"]
    B = corr["B"]
    start = corr["start"]
    markers = lmap.all_world_xy()

    fig, ax = plt.subplots(figsize=(7.2, 8.4))

    # obstacle + standoff ring
    ax.add_patch(Circle(obs_c, obs_r, color="#555555", alpha=0.85, zorder=2))
    ax.add_patch(Circle(obs_c, standoff, fill=False, ls="--", ec="#999999", zorder=2))
    ax.text(obs_c[0], obs_c[1], "obstacle", color="white", ha="center", va="center",
            fontsize=8, zorder=3)

    ax.plot(none["true_xy"][:, 0], none["true_xy"][:, 1], "-", color=_NONE, lw=2.2,
            label=f"NO correction (miss {none['arrival_miss_m']:.1f} m)", zorder=4)
    ax.plot(corr["true_xy"][:, 0], corr["true_xy"][:, 1], "-", color=_CORR, lw=2.2,
            label=f"landmark correction (miss {corr['arrival_miss_m']:.1f} m)", zorder=5)

    ax.scatter(markers[:, 0], markers[:, 1], marker="s", s=55, facecolor="white",
               edgecolor="black", linewidths=1.3, label="ArUco markers", zorder=6)
    cs = corr["corrections"]
    if cs:
        pts = corr["true_xy"][cs]
        ax.scatter(pts[:, 0], pts[:, 1], s=30, color=_CORR, edgecolor="white",
                   linewidths=0.6, zorder=7)
    ax.scatter([start[0]], [start[1]], marker="o", s=90, color="black",
               label="start", zorder=6)
    ax.scatter([B[0]], [B[1]], marker="*", s=340, color="#ff7f0e", edgecolor="black",
               label="target B", zorder=8)

    ax.set_xlabel("world X [m]")
    ax.set_ylabel("world Y [m]")
    ax.set_title("Week 5 -- both avoid the obstacle; only the corrected drone reaches B")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_vector_error(corr, none, out_path):
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    e0, e1 = _evade_span(corr)
    if e0 is not None:
        ax.axvspan(e0, e1, color="#ffe08a", alpha=0.4, label="detour (evade)")
    ax.plot(np.arange(len(none["dir_err"])), none["dir_err"], "-", color=_NONE, lw=2.0,
            label="NO correction")
    ax.plot(np.arange(len(corr["dir_err"])), corr["dir_err"], "-", color=_CORR, lw=2.0,
            label="landmark correction")
    ax.axhline(10.0, color="gray", ls=":", lw=1, label="10 deg")
    ax.set_xlabel("step")
    ax.set_ylabel("vector-to-B direction error [deg]")
    ax.set_title("Week 5 -- does the recomputed heading to B stay correct through the detour?")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_tracking(corr, none, floor, out_path):
    """Localization error (top) and feature-track count vs the observability floor
    (bottom) -- the evidence that VIO keeps tracking through the maneuver."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True)
    e0, e1 = _evade_span(corr)

    for ax in (ax1, ax2):
        if e0 is not None:
            ax.axvspan(e0, e1, color="#ffe08a", alpha=0.4)

    ax1.plot(np.arange(len(none["pos_err"])), none["pos_err"], "-", color=_NONE, lw=2.0,
             label="NO correction")
    ax1.plot(np.arange(len(corr["pos_err"])), corr["pos_err"], "-", color=_CORR, lw=2.0,
             label="landmark correction")
    ax1.set_ylabel("localization error [m]")
    ax1.set_title("Week 5 -- tracking continuity through the detour (yellow = evade)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(np.arange(len(corr["feat"])), corr["feat"], "-", color="#2ca02c", lw=2.0,
             label="active feature tracks")
    ax2.axhline(floor, color=_NONE, ls="--", lw=1.5,
                label=f"observability floor ({floor:.0f})")
    ax2.fill_between(np.arange(len(corr["feat"])), 0, floor, color="#ffcccc", alpha=0.4)
    ax2.set_xlabel("step")
    ax2.set_ylabel("feature tracks")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

# ============================ from week6 ============================

_POLICY_ORDER = ["none", "fixed", "uncertainty"]
_POLICY_COLOR = {"none": "#7f7f7f", "fixed": "#d62728", "uncertainty": "#1f77b4"}
_POLICY_LABEL = {"none": "none", "fixed": "fixed-period", "uncertainty": "uncertainty-aware"}


def _grid(summary_rows, mean_key, std_key):
    """-> (scenario_ids, scenario_labels, {policy: (means, stds)})."""
    scen_ids = sorted({r["scenario"] for r in summary_rows})
    labels = {r["scenario"]: r["scenario_label"] for r in summary_rows}
    data = {p: ([], []) for p in _POLICY_ORDER}
    for sid in scen_ids:
        for p in _POLICY_ORDER:
            row = next(r for r in summary_rows if r["scenario"] == sid and r["policy"] == p)
            data[p][0].append(row[mean_key])
            data[p][1].append(row[std_key])
    return scen_ids, [labels[s] for s in scen_ids], data


def _bars(ax, summary_rows, mean_key, std_key, ylabel, title):
    scen_ids, scen_labels, data = _grid(summary_rows, mean_key, std_key)
    x = np.arange(len(scen_ids))
    w = 0.26
    for i, p in enumerate(_POLICY_ORDER):
        means, stds = data[p]
        ax.bar(x + (i - 1) * w, means, w, yerr=stds, capsize=3,
               color=_POLICY_COLOR[p], label=_POLICY_LABEL[p])
    ax.set_xticks(x)
    ax.set_xticklabels(scen_ids)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)


def save_evaluation_bars(summary_rows, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    _bars(axes[0], summary_rows, "arrival_error_m_mean", "arrival_error_m_std",
          "arrival error at B [m]", "Accuracy: arrival error")
    _bars(axes[1], summary_rows, "peak_drift_m_mean", "peak_drift_m_std",
          "peak drift [m]", "Accuracy: peak drift")
    _bars(axes[2], summary_rows, "avl_invocations_mean", "avl_invocations_std",
          "AVL invocations", "Frugality: correction count")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Week 6 -- evaluation across scenarios A/B/C  (mean +/- std over seeds)",
                 fontsize=13)
    # scenario legend
    scen_ids = sorted({r["scenario"] for r in summary_rows})
    labels = [next(r["scenario_label"] for r in summary_rows if r["scenario"] == s)
              for s in scen_ids]
    fig.text(0.5, 0.005, "   |   ".join(labels), ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_frugality_scatter(summary_rows, out_path):
    """Accuracy vs corrections for the marker scenarios (B, C): the closer to the
    bottom-left, the better. uncertainty-aware sits left of fixed at ~equal height."""
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for r in summary_rows:
        if not r["has_markers"]:
            continue
        p = r["policy"]
        ax.scatter(r["avl_invocations_mean"], r["peak_drift_m_mean"], s=140,
                   color=_POLICY_COLOR[p], edgecolor="black", zorder=4)
        ax.annotate(f"{r['scenario']}:{_POLICY_LABEL[p].split('-')[0]}",
                    (r["avl_invocations_mean"], r["peak_drift_m_mean"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=8)
    # connect fixed->uncertainty per scenario to show the shift
    for sid in sorted({r["scenario"] for r in summary_rows if r["has_markers"]}):
        f = next(r for r in summary_rows if r["scenario"] == sid and r["policy"] == "fixed")
        u = next(r for r in summary_rows if r["scenario"] == sid and r["policy"] == "uncertainty")
        ax.annotate("", xy=(u["avl_invocations_mean"], u["peak_drift_m_mean"]),
                    xytext=(f["avl_invocations_mean"], f["peak_drift_m_mean"]),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.3))
    ax.set_xlabel("AVL invocations (compute / power)")
    ax.set_ylabel("peak drift [m] (accuracy)")
    ax.set_title("Week 6 -- fewer corrections at ~equal accuracy (arrow: fixed -> uncertainty)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

# ==================== integrated end-to-end demo (run_demo) ==================
_POL_C = {"none": "#7f7f7f", "fixed": "#d62728", "uncertainty": "#1f77b4"}
_POL_L = {"none": "none (pure VIO)", "fixed": "fixed-period",
          "uncertainty": "uncertainty-aware"}


def save_integrated_moneyshot(results, out_path):
    """The whole system in one frame: true flight path (with the obstacle detour),
    markers, hard patch, target B, and where the uncertainty scheduler chose to
    correct -- vs the pure-VIO path that misses B."""
    unc = results["uncertainty"]
    none = results["none"]
    lmap = unc["lmap"]
    obs_c, obs_r = unc["obstacle"]
    B, start = unc["B"], unc["start"]
    markers = lmap.all_world_xy()

    fig, ax = plt.subplots(figsize=(8.2, 8.6))

    # hard patch (feature-poor zone)
    ax.add_patch(Circle(unc["hard_center"], unc["hard_radius"], color="#ffe08a",
                        alpha=0.35, zorder=1))
    ax.text(unc["hard_center"][0], unc["hard_center"][1], "feature-poor\nhard patch",
            ha="center", va="center", fontsize=8, color="#8a6d00", zorder=2)

    # obstacle
    ax.add_patch(Circle(obs_c, obs_r, color="#555555", alpha=0.9, zorder=3))
    ax.add_patch(Circle(obs_c, obs_r + 0.6, fill=False, ls="--", ec="#999999", zorder=3))
    ax.text(obs_c[0], obs_c[1], "obstacle", color="white", ha="center", va="center",
            fontsize=7, zorder=4)

    # true flight paths
    ax.plot(none["true_xy"][:, 0], none["true_xy"][:, 1], "--", color=_POL_C["none"],
            lw=2.0, label=f"none: flies blind, misses B by {none['arrival_miss_m']:.1f} m",
            zorder=5)
    ax.plot(unc["true_xy"][:, 0], unc["true_xy"][:, 1], "-", color=_POL_C["uncertainty"],
            lw=2.4, label=f"uncertainty-aware: arrives ({unc['arrival_miss_m']:.1f} m), "
                          f"{len(unc['corrections'])} fixes", zorder=6)

    # markers + corrections + start + B
    ax.scatter(markers[:, 0], markers[:, 1], marker="s", s=70, facecolor="white",
               edgecolor="black", linewidths=1.4, label="ArUco markers (world frame)",
               zorder=7)
    cs = unc["corrections"]
    if cs:
        pts = unc["true_xy"][cs]
        ax.scatter(pts[:, 0], pts[:, 1], marker="o", s=90, color=_POL_C["uncertainty"],
                   edgecolor="white", linewidths=1.2, zorder=8,
                   label="correction fired (U > threshold)")
    ax.scatter([start[0]], [start[1]], marker="o", s=110, color="black",
               label="start", zorder=8)
    ax.scatter([B[0]], [B[1]], marker="*", s=420, color="#ff7f0e", edgecolor="black",
               linewidths=1.0, label="target B (fixed in world frame)", zorder=9)

    ax.set_xlabel("world X [m]")
    ax.set_ylabel("world Y [m]")
    ax.set_title("FrugalNav end-to-end: drift, uncertainty-scheduled fixes, "
                 "detour, arrival")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_integrated_dashboard(results, out_path):
    """Three stacked panels: localization error, the uncertainty signal U with its
    threshold and firing points, and the running correction count (frugality)."""
    unc = results["uncertainty"]
    dt = unc["dt"]
    order = ["none", "fixed", "uncertainty"]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11.0, 9.0), sharex=True)

    # -- panel 1: localization error over time --
    for p in order:
        r = results[p]
        t = np.arange(len(r["err"])) * dt
        ax1.plot(t, r["err"], "-", color=_POL_C[p], lw=1.9,
                 label=f"{_POL_L[p]}  (peak {r['peak_err']:.2f} m, "
                       f"{len(r['corrections'])} fixes)")
    ax1.set_ylabel("localization error [m]")
    ax1.set_title("Localization error: uncertainty-aware tracks fixed-period at a "
                  "fraction of the fixes")
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(True, alpha=0.3)

    # -- panel 2: the uncertainty signal U --
    t = np.arange(len(unc["U"])) * dt
    ax2.plot(t, unc["U"], "-", color=_POL_C["uncertainty"], lw=1.8, label="uncertainty U")
    ax2.axhline(0.45, color="black", ls="--", lw=1.2, label="threshold tau = 0.45")
    cs = unc["corrections"]
    if cs:
        ax2.scatter(np.array(cs) * dt, unc["U"][cs], s=70, color=_POL_C["uncertainty"],
                    edgecolor="white", zorder=5, label="fix fired")
    # shade evasion span
    ev = unc["evading"]
    if ev.any():
        idx = np.where(ev)[0]
        ax2.axvspan(idx[0] * dt, idx[-1] * dt, color="#ffd8b0", alpha=0.5,
                    label="obstacle detour")
    ax2.set_ylabel("uncertainty  U")
    ax2.set_title("U rises in the hard patch and the detour; the scheduler fires only "
                  "above threshold")
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(True, alpha=0.3)

    # -- panel 3: cumulative corrections (frugality) --
    for p in ("fixed", "uncertainty"):
        r = results[p]
        t = np.arange(len(r["err"])) * dt
        cum = np.zeros(len(r["err"]))
        for c in r["corrections"]:
            cum[c:] += 1
        ax3.plot(t, cum, "-", color=_POL_C[p], lw=2.0, drawstyle="steps-post",
                 label=f"{_POL_L[p]}: {len(r['corrections'])} total")
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("corrections used")
    ax3.set_title("Frugality: fewer AVL fixes = less compute / power on the RISC-V core")
    ax3.legend(loc="upper left", fontsize=8.5)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
