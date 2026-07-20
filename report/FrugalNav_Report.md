# FrugalNav — An Uncertainty-Scheduled VIO + AVL Navigator for Ultra-Low-Power RISC-V

**Parth Malik (2024406) · Siddharth Bhardwaj (2024553) · Rohan Yadav (2024478)**
B.Tech, 3rd Year · Independent Project

---

## Abstract

A UAV that loses GPS must navigate by dead-reckoning its motion (Visual-Inertial
Odometry, VIO), which drifts without bound, and periodically cancel that drift with
an absolute fix from a known landmark (Absolute Visual Localization, AVL). Each
absolute fix, however, costs camera capture, marker detection and pose estimation —
i.e. compute and power, the scarcest resources on a small drone. We present
**FrugalNav**, whose contribution is not a new VIO or a new detector but the layer
that decides *when* to correct: an **uncertainty-aware landmark scheduler** that
fires an AVL fix only when a fused localization-confidence metric **U** indicates
drift is about to compound. On an end-to-end navigator built from a real portable
core, uncertainty-aware scheduling **matches the accuracy of correcting at every
marker while spending half the corrections**, and where no markers exist it
correctly degrades to the same result as any other policy. The entire decision hot
loop is ported to header-only C++ and shown, by measurement and a cycle model, to
fit an ultra-low-power RISC-V SoC (GAP9) with orders of magnitude of margin — the
binding constraint is the VIO/detector front-ends, exactly the compute the
scheduler removes when it skips a fix.

---

## 1. Introduction

### 1.1 Problem

GPS-denied navigation (indoors, urban canyons, under jamming) forces a UAV onto
relative sensing. VIO fuses a camera and an IMU into a smooth relative pose, but its
error grows with distance travelled — heading bias and scale error integrate into
metres of drift over a short flight. The standard remedy is to periodically observe
a landmark whose world position is known and snap the estimate back to truth. In our
setting the landmarks are **ArUco markers** — the tractable, near-zero-cost cousin
of satellite/scene matching — surveyed once into a world-frame map.

The overlooked question is **scheduling**. Correcting on a fixed timer is simple but
blind: it must be tuned to the *worst* stretch of the flight and therefore
over-corrects everywhere else, burning detector compute (and battery) on fixes that
buy almost nothing. Correcting never is the drift floor. The right policy corrects
*exactly where drift is actually growing* and skips the easy stretches.

### 1.2 Contribution

> An **uncertainty-aware landmark scheduler** that invokes AVL only when a fused
> localization-confidence metric **U** exceeds a threshold — bounding VIO drift while
> minimizing correction frequency, and therefore compute and power, on a portable
> navigation core targetable to an ultra-low-power RISC-V SoC.

We explicitly do **not** contribute a VIO (we consume the signals a standard
MSCKF/graph VIO already exposes) or a scene-matching front-end (we use ArUco). The
novelty is one layer up — the frugal decision of *when to correct* — together with a
dependency-light portable core that realises it and a feasibility argument for
ultra-low-power hardware.

### 1.3 Scope

Validated on public datasets, phone/webcam recordings, and a closed-loop kinematic
simulator driven by the real core; profiled and partially ported to C++; assessed
for RISC-V. Out of scope (future work): physical flight, neural scene matching,
custom silicon, multi-UAV. The lack of a drone removes only the flying demo — always
future work — not the algorithm.

---

## 2. Related Work

**VIO.** OpenVINS (MSCKF) and ORB-SLAM3 are the standard open, CPU-capable VIO
front-ends and the ones our adapter targets; EuRoC MAV, TUM-VI and ADVIO are the
benchmarks they are evaluated on. We treat VIO as a *glass box* — a source of a pose
plus covariance, feature count, and IMU bias — rather than reimplementing it.

**Absolute visual localization.** Fixing drift against known landmarks ranges from
fiducial markers (ArUco/AprilTag) with `solvePnP`, to learned descriptors
(SuperPoint/SuperGlue) matched against a geo-referenced map. We deliberately take the
frugal end: markers store only *ID + pose*, so a map of ~800 markers is ~20 KB rather
than the hundreds of KB of image descriptors.

**When-to-sense / active perception.** Deciding when to spend an expensive
measurement is the subject of active-perception and event-triggered estimation. Our
scheduler is an event-triggered AVL gate specialised to VIO drift, with an
interpretable weighted-sum confidence metric and safety floors — chosen for
defensibility and portability over a learned trigger.

**Frugal / TinyML navigation.** Parallel-Ultra-Low-Power platforms such as
GreenWaves GAP9 (a multi-core RISC-V cluster in a sub-200 mW envelope) motivate
navigation stacks whose hot loop is scalar, allocation-free, and small. Our core is
written to that constraint from the start.

---

## 3. System Architecture

When GPS is lost the system runs continuous relative localization corrected by
intermittent absolute fixes, all fused into one estimate that drives a
target-centric controller, with reactive obstacle avoidance layered on top.

| Module | Role | Signal | Source file |
|---|---|---|---|
| VIO front-end / drift scaffold | continuous relative pose + uncertainty | camera + IMU | `harness/drift_scaffold.py`, `core/vio_adapter.py` |
| **Uncertainty scheduler (U)** | decide *when* to request AVL | VIO internals + blur | `core/uncertainty_scheduler.py` |
| Landmark corrector (AVL) | absolute position fix | down camera + ArUco | `core/aruco_detector.py`, `landmark_corrector.py` |
| State fusion | merge AVL fix into the estimate | — | `core/state_fusion.py` |
| Target-centric controller | drive toward B | fused (x,y) | `core/controller.py` |
| Obstacle avoidance | reactive evasion | optical-flow TTC | `core/obstacle_avoidance.py` |
| Integration | the hot loop tying it together | struct in → cmd out | `core/navigator.py` |

Two design decisions resolve ambiguities from earlier drafts:

**(1) Decoupled modules, one fused estimate.** The *software* modules are decoupled
— each independently testable and independently mappable to a core. The *estimate*
is a single tightly-fused `NavState`: an AVL fix is merged **into** it (a Kalman
update in `state_fusion.py`), not kept as a parallel track. Crucially the scheduler
reads its primary cue, `sigma_pos`, *from that same fused covariance* — so the
trigger signal and the estimate are literally the same object.

**(2) The landmark map defines the world frame.** Target **B** is a fixed point in
the landmark/world frame — not "100 m ahead" in the drifting VIO frame. VIO provides
relative motion between fixes; markers re-anchor both the drone and B to the world
frame. Correcting drift moves only the drone's estimate, never B, so the recomputed
vector (B − estimate) is corrected for free.

---

## 4. Uncertainty-Aware Scheduling (the contribution)

### 4.1 The metric U

U fuses five glass-box cues into one confidence scalar in [0, 1], each normalized to
calibrated bounds (tuned against the real EuRoC MH_01 signal ranges):

```
U = w1·n(σ_pos) + w2·n(σ_head) + w3·n(blur) + w4·n(feature_loss) + w5·n(imu_bias)
    weights (0.45, 0.20, 0.20, 0.10, 0.05) sum to 1  ⇒  U ∈ [0, 1]
```

Keeping U a plain weighted sum of inspectable signals is deliberate: every term has
a physical meaning and the threshold has an interpretable one. `σ_pos` and `σ_head`
are covariance growth; `blur` (inverted Laplacian variance) and `feature_loss` are
*leading* indicators that spike *before* covariance blows up; `imu_bias` earns its
small weight on sequences that actually exhibit gyro-bias drift.

### 4.2 Two-tier trigger + hysteresis

The structure is what makes it defensible in review:

1. **Hard floor** — if `σ_pos` exceeds a metres budget, force a fix regardless of U.
   Bounds the worst case no matter how the soft weights are tuned.
2. **Observability floor** — if active feature tracks drop below ~20, force a fix.
3. **Soft trigger** — otherwise fire when `U > τ`. The leading indicators let us
   correct *just before* drift compounds, which is the value over plain covariance
   gating.
4. **Hysteresis** — after a fix, suppress re-triggering for a refractory window so a
   single correction settles without chatter. Hard floors override the window
   (safety first).

### 4.3 Constraints analyzed as report content

The scheduler exists because of concrete physical constraints, which we model
directly rather than merely list:

* **KLT displacement limit** `δ = v·Δt·f/Z` degrades past ~15–20 px; comfortable at
  our handheld ~1.5 m/s, and modelled analytically for the 5–8 m/s flight regime that
  motivates a 60 Hz camera.
* **Landmark density** `D ≤ vT` couples speed, spacing, and map memory.
* **Map memory** — ID + pose only, ~24 B/marker → ~20 KB for ~800 markers.
* **Target-centric error propagation** — drift → wrong (B−x) vector → control error
  → more drift. This feedback loop is unique to the target-centric formulation; the
  scheduler's job is to break it before it compounds. Our closed-loop sim exhibits it
  directly (the controller commands off the *estimate*).
* **Observability floor** — enforced as a hard trigger.
* **Compute-aware scheduling** — fewer fires ⇒ the detector/PnP path runs less ⇒
  compute and power scale with need, not with a fixed timer.

---

## 5. Portable Core & Implementation

**The seam.** `core/vio_adapter.py` defines a tiny `VioSource` interface: `update()`
yields a `VioSignals` bundle (estimate + cues + optional ground truth); `apply_fix()`
applies an absolute correction. The kinematic sim, the EuRoC drift scaffold, and (in
future) a real OpenVINS process all implement it, so the scheduler is agnostic to
where its signals come from — a real VIO drops in by writing one subclass.

**Struct in, command out.** `core/navigator.py` is the integrated hot loop:
`SensorInput` (VIO delta + cues + optional sighting + optional TTC) in, `VelocityCmd`
out, with fixed-size state and no allocation. This is the module that ports to C++
and RISC-V.

**Data, for free.** Real VIO with ground truth from EuRoC; a drift scaffold that
turns real trajectories into honest drifting estimates when a full VIO build is
unavailable; ArUco markers printed on paper and detected by a webcam. Everything is
CPU-only — no GPU, ROS, or Gazebo on the critical path.

**Tests.** 72 unit tests over the core (`pytest -q`), plus per-module `__main__`
self-tests. Building the C++ port cross-checked the scheduler against the Python
reference and, in doing so, **caught a latent bug**: an unrun `__main__` self-test
case asserted a soft trigger for a state whose U was actually 0.35 < τ. The scheduler
logic was correct; the *test case* was unrealistically sparse (real drift raises
several cues together). Both self-tests were fixed and a regression test added — a
concrete example of the port paying for itself.

---

## 6. Evaluation

### 6.1 End-to-end money-shot

`run_demo.py` flies the full system to a fixed target through a feature-poor "hard
patch" and an obstacle, under three policies over the identical world. Figures:
`outputs/demo_moneyshot.png` (trajectory) and `outputs/demo_dashboard.png` (error, U,
and cumulative corrections over time).

The dashboard tells the story in three panels: (a) pure-VIO error ramps to several
metres while both correcting policies stay near zero; (b) U rises in the hard patch
and the detour and the scheduler fires only above threshold; (c) the correction count
diverges — fixed-period climbs to 6, uncertainty-aware stops at 3.

### 6.2 Controlled A/B/C evaluation on the real core

`demo_eval.py` runs the scenario × policy × seed matrix through the *actual portable
core* (so "uncertainty" is the real scheduler, not a toy threshold). Representative
means over 5 seeds:

| Scenario | Policy | Arrival [m] | Peak [m] | RMSE [m] | AVL fixes |
|---|---|---:|---:|---:|---:|
| **A** open, no markers | none / fixed / uncertainty | 2.24 | 2.13 | 1.05 | 0 |
| **B** markers along path | none | 2.59 | 2.57 | 1.24 | 0 |
| | fixed-period | 0.96 | 0.31 | 0.12 | 6 |
| | **uncertainty-aware** | **0.97** | **0.30** | **0.13** | **3** |
| **C** markers + obstacle | none | 3.29 | 3.34 | 1.60 | 0 |
| | fixed-period | 0.96 | 0.50 | 0.17 | 6 |
| | **uncertainty-aware** | **1.03** | **0.60** | **0.22** | **3** |

**Reading.**
* **A** — with no markers, all three policies are identical: absolute fixes are
  essential and scheduling is moot without landmarks. This is the control that shows
  the win in B/C is real and not a modelling artifact.
* **B & C** — uncertainty-aware **matches fixed-period accuracy (0.97 vs 0.96 m
  arrival; 0.30 vs 0.31 m peak) using half the fixes (3 vs 6)** → 50 % fewer detector
  invocations at equal accuracy. This is the frugality result: *similar accuracy, far
  fewer corrections.*

The advantage requires **heterogeneous** drift (concentrated in the hard patch and
the maneuver); under perfectly uniform drift every policy converges, which is the
honest boundary of the claim and is why the sim earns its drift in specific zones.

### 6.3 Cross-check

Siddharth's analytic-U evaluation (`demo_week6.py`, a simpler drift model and a
threshold policy) reaches the same conclusion (~40–60 % fewer corrections at
comparable accuracy in the marker scenarios), giving two independent routes to the
headline. Figures: `outputs/week6_evaluation.png`, `outputs/eval_real_bars.png`,
`outputs/eval_real_frugality.png`.

### 6.4 Obstacle avoidance & tracking continuity

The reactive optical-flow (time-to-contact) evader keeps a positive standoff margin
across all seeds and policies while the drone continues to VIO-track through the
maneuver, so the recomputed vector to B stays valid after the detour (Week-5
verification, `demo_week5.py`, `outputs/week5_*.png`).

---

## 7. RISC-V Feasibility (summary)

Full study: [`../profiling/riscv_feasibility.md`](../profiling/riscv_feasibility.md).
Target reference: GreenWaves **GAP9** (multi-core RISC-V, ~370 MHz, ~1.6 MB L2,
sub-200 mW).

**Measured** (compiled C++ port, `cpp/main.cpp`, `-O2`, x86 host): scheduler
≈ **77 ns/step**, full decision hot loop ≈ **93 ns/step**, with the C++ U matching the
Python reference to 4 decimals.

**Modeled** (op-count + clock/IPC derating to GAP9): scheduler ≈ **0.15 µs/call**,
full hot loop ≈ **1.2 µs/step** — under 0.01 % of a 30 Hz frame. Memory: landmark map
~20 KB, decision-core state < 1 KB — well inside 1.6 MB.

**Per-module verdict:** the scheduler and the whole decision hot loop fit *trivially*
(µs against a multi-ms frame; kB against MB). Feasibility is **gated by the VIO and
ArUco front-ends** (~1–8 ms/frame, literature) — which is precisely why the plan does
not reinvent them and why the scheduler adds value: by event-gating the AVL path it
removes the largest avoidable slice of front-end compute, extending flight time on
the same battery. All GAP9 numbers are marked modeled; on-silicon bring-up is future
work.

---

## 8. Limitations & Threats to Validity

* **No physical flight.** Results are on datasets, recordings, and a closed-loop
  kinematic simulator driven by the real core. The simulator's drift is *calibrated*
  against real EuRoC drift-rate, but it is still a model.
* **The frugality win needs heterogeneous drift.** Under uniform drift, adaptive and
  fixed scheduling converge; we state this boundary explicitly and design the sim to
  earn its drift in physically-motivated zones (feature-poor patch, aggressive
  maneuver) rather than everywhere.
* **RISC-V numbers are modeled**, not run on GAP9 silicon; front-end costs are from
  the literature, not re-measured. Derating is kept conservative and every number is
  tagged measured/modeled.
* **ArUco, not scene matching.** Markers must be surveyed and present; learned
  landmark matching is deliberately out of scope (future work).
* **U weights are hand-tuned** against MH_01 ranges, not learned; this is a
  defensibility/portability choice, and a learned trigger is a natural extension.

---

## 9. Conclusion & Future Work

FrugalNav keeps the original idea's heart — continuous frugal VIO grounded by
intermittent absolute markers on ultra-low-power RISC-V — and sharpens the
contribution to the one genuinely novel, frugal decision: **uncertainty-aware
scheduling of when to correct.** On an end-to-end navigator built from a real,
dependency-light portable core, it delivers the accuracy of correcting at every
marker for half the corrections, degrades gracefully when no markers exist, and ports
to a header-only C++ hot loop that fits an ultra-low-power RISC-V SoC with enormous
margin. The deliverable is a validated, profiled navigation core ready for a
follow-up hardware project.

**Future work:** physical flight and on-silicon GAP9 bring-up; a learned or adaptive
threshold; neural/satellite scene matching (SuperPoint) as a second AVL back-end
behind the same scheduler; multi-UAV shared-map correction.

---

## Appendix A — Reproduction

```bash
python -m venv .venv && . .venv/Scripts/activate && pip install -r requirements.txt
python -m pytest -q            # 72 passing
python run_demo.py --seeds 8   # money-shot + dashboard + demo JSON
python demo_eval.py --seeds 5  # A/B/C matrix + CSVs + bar/frugality plots
bash cpp/build_and_run.sh      # C++ port: self-test + hot-loop benchmark
python profiling/profile_core.py
```

## Appendix B — Ownership

Parth — portable core, state fusion, controller, navigator integration, C++ port.
Siddharth — ArUco detection, landmark corrector + world map, evaluation metrics.
Rohan — uncertainty metric U + scheduler, kinematic sim, optical-flow avoidance,
RISC-V feasibility study.
