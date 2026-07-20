"""
demo_week1.py
-------------
Rohan's day-one win: run the SAME flight under three correction policies and show
that the uncertainty-aware policy matches the accuracy of fixed-interval while using
far fewer corrections. This is a miniature of the Week-6 headline result, built from
your two modules (kinematic_sim + uncertainty_scheduler) before the team's VIO exists.

Run:  python3 demo_week1.py
Out:  demo_week1.png  (+ a printed summary table)
"""
import sys, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless: just save a file
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from harness.kinematic_sim import KinematicSim, SimConfig, World, Marker, Obstacle
from core.uncertainty_scheduler import UncertaintyScheduler, SchedulerConfig

TAU = 0.5                 # soft threshold used everywhere in this demo
REFRACTORY = 60


def dense_world():
    """A long, mostly-easy route with two SHORT hard bursts. Markers are spread along
    it (many correction CHANCES). A frugal scheduler should stay quiet through the long
    easy stretches and only spend fixes in/after the bursts, while fixed-interval keeps
    correcting the whole way."""
    start = (90.0, 40.0)
    markers = []
    for i, f in enumerate(np.linspace(0.05, 0.97, 12)):     # 12 markers along the route
        markers.append(Marker(i, start[0] * (1 - f), start[1] * (1 - f), radius=3.0))
    # guarantee a marker sits inside each hard burst so adaptive CAN fix there
    markers += [Marker(90, 70, 31, radius=4.0), Marker(91, 30, 13, radius=4.0)]
    return World(start=start, markers=markers,
                 obstacles=[Obstacle(50, 23, 2.0)],
                 hard_patches=[(70, 31, 4.0, 0.9), (30, 13, 4.0, 0.85)])


def run(policy, seed=2, fixed_period_s=2.5):
    """policy in {'none','fixed','adaptive'}. Returns a log dict."""
    # Bias-driven drift (the real VIO killer) + longer refractory so the adaptive
    # policy holds off during the long easy stretches and only spends fixes in/after
    # the hard patches.
    cfg = SimConfig(seed=seed, vel_noise_base=0.02, vel_noise_hard=0.35,
                    gyro_bias_walk=0.004, max_steps=3000)
    sim = KinematicSim(world=dense_world(), cfg=cfg)
    sch = UncertaintyScheduler(SchedulerConfig(
        tau=TAU, refractory_ticks=REFRACTORY,
        weights={"sigma_pos": 0.7, "feature_loss": 0.3,
                 "sigma_head": 0.0, "blur": 0.0, "imu_bias": 0.0}))
    log = dict(t=[], err=[], U=[], trig_t=[], true=[], est=[], corrections=0)
    last_fix_t = -1e9

    while not sim.arrived and sim.steps < cfg.max_steps:
        sim.step()
        cues = sim.cues()
        U, trig, why, _ = sch.compute(cues)

        want_fix = False
        if policy == "fixed":
            want_fix = (sim.t - last_fix_t) >= fixed_period_s
        elif policy == "adaptive":
            want_fix = trig

        if want_fix and sim.apply_fix():               # only succeeds if a marker is in view
            sch.reset_after_fix()
            last_fix_t = sim.t
            log["corrections"] += 1
            log["trig_t"].append(sim.t)

        log["t"].append(sim.t)
        log["err"].append(sim.error)
        log["U"].append(U)
        log["true"].append(sim.true.copy())
        log["est"].append(sim.est.copy())

    log["true"] = np.array(log["true"])
    log["est"] = np.array(log["est"])
    log["final_err"] = log["err"][-1]
    log["peak_err"] = max(log["err"])
    log["mean_err"] = float(np.mean(log["err"]))
    log["target_miss"] = float(np.linalg.norm(sim.true))   # how far from B we ACTUALLY ended
    return sim, log


def main():
    results = {p: run(p) for p in ("none", "fixed", "adaptive")}

    # ---- summary table ----
    print(f"\n{'policy':<10}{'corrections':>12}{'target_miss(m)':>16}"
          f"{'peak_err(m)':>13}{'mean_err(m)':>13}")
    print("-" * 64)
    for p, (_, log) in results.items():
        print(f"{p:<10}{log['corrections']:>12}{log['target_miss']:>16.2f}"
              f"{log['peak_err']:>13.2f}{log['mean_err']:>13.2f}")
    print("\ntarget_miss = how far from B the drone ACTUALLY ended up (the mission error).")
    print("The story: 'none' misses B because it homes on a drifting estimate; 'adaptive'")
    print("matches 'fixed' accuracy with fewer corrections = the frugality win.")

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    colors = {"none": "#d1495b", "fixed": "#edae49", "adaptive": "#2e7d32"}

    # panel 1: trajectories (adaptive vs ground truth) + world
    sim, log = results["adaptive"]
    w = sim.world
    for (hx, hy, hr, _) in w.hard_patches:
        ax[0].add_patch(plt.Circle((hx, hy), hr, color="#b0b0b0", alpha=0.25, lw=0))
    for m in w.markers:
        ax[0].plot(m.x, m.y, "ks", ms=4)
    for ob in w.obstacles:
        ax[0].add_patch(plt.Circle((ob.x, ob.y), ob.radius, color="#444", alpha=0.6))
    ax[0].plot(log["true"][:, 0], log["true"][:, 1], "-", color="#1f77b4", lw=2, label="true path")
    ax[0].plot(log["est"][:, 0], log["est"][:, 1], "--", color=colors["adaptive"], lw=1.5, label="adaptive estimate")
    ax[0].plot(0, 0, "g*", ms=18, label="target B")
    ax[0].set_title("Trajectory (adaptive)\ngrey=hard patch, ■=marker, ●=obstacle")
    ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)"); ax[0].legend(fontsize=8); ax[0].axis("equal")

    # panel 2: estimate error over time, all three policies
    for p, (_, lg) in results.items():
        ax[1].plot(lg["t"], lg["err"], color=colors[p], lw=1.6,
                   label=f"{p} ({lg['corrections']} fixes)")
    ax[1].set_title("Estimate error vs time"); ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("‖true − est‖ (m)"); ax[1].legend(fontsize=8)

    # panel 3: U(t) with the adaptive trigger points and the threshold
    _, lg = results["adaptive"]
    ax[2].plot(lg["t"], lg["U"], color=colors["adaptive"], lw=1.4, label="U(t)")
    ax[2].axhline(TAU, ls=":", color="k", label=f"τ={TAU}")
    for tt in lg["trig_t"]:
        ax[2].axvline(tt, color="#d1495b", alpha=0.4, lw=1)
    ax[2].set_title("Uncertainty U(t) and triggers (adaptive)")
    ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("U  [0,1]"); ax[2].set_ylim(0, 1)
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "demo_week1.png")
    fig.savefig(out, dpi=130)
    print(f"\nFigure written to {out}")


if __name__ == "__main__":
    main()
