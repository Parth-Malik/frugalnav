# Independent Project — Final Plan
**Project:** Frugal GPS-Denied UAV Navigation via Uncertainty-Aware Landmark Scheduling
**Working title:** *FrugalNav — an uncertainty-scheduled VIO+AVL navigator for ultra-low-power RISC-V*
**Team:** Parth Malik (2024406), Siddharth Bhardwaj (2024553), Rohan Yadav (2024478)
**Year:** 3rd Year, B.Tech
**Plan date:** June 2026 — supersedes all earlier drafts (proposal, refined architecture, weekly plan v3, v4)

---

## 0. The one-sentence contribution

> An **uncertainty-aware landmark scheduler** that invokes absolute visual localization (AVL) only when a fused localization-confidence metric *U* exceeds a threshold — bounding VIO drift while minimizing correction frequency, and therefore compute and power, on a portable navigation core targetable to an ultra-low-power RISC-V SoC.

We are **not** inventing a VIO. We use an existing one (OpenVINS / ORB-SLAM3). The novelty lives one layer up: *deciding when to correct.*

---

## 1. Locked architecture (single coherent story)

When GPS is lost, the system runs continuous relative localization corrected by intermittent absolute fixes, all fused into one estimate, driving a target-centric controller.

| Module | Role | Source signal | Notes |
|---|---|---|---|
| **VIO front-end** | Continuous relative pose + its own uncertainty | Camera + IMU | OpenVINS (MSCKF) or ORB-SLAM3. Glass-box: exposes covariance, feature count, IMU bias. |
| **Uncertainty scheduler (U)** | Decides *when* to request an AVL fix | VIO internals + image blur | **The contribution.** Fires AVL only when U > threshold. |
| **Landmark corrector (AVL)** | Absolute position fix | Downward camera + ArUco | Marker ID → known world pose from the landmark map. |
| **State fusion** | Merge AVL fix into the estimate | — | Tightly fused (correction enters the VIO/EKF state). |
| **Target-centric controller** | Drive toward target | Fused (x, y) | Command vector (−x, −y); merges evasion vector. |
| **Obstacle avoidance** | Reactive evasion | Monocular optical-flow expansion (time-to-contact) | Phone camera; iPhone-Pro LiDAR depth as fallback. |

### Two clarifications that resolve inconsistencies in the earlier drafts
1. **Decoupled modules, fused estimate.** The *software* modules are decoupled — each independently testable and independently mappable to a RISC-V core. The *estimate* is tightly fused — an AVL fix is merged into the VIO state, not maintained as a separate parallel track. (Reconciles "tightly coupled" in the proposal with "decoupled threads" in the weekly plan.)
2. **The landmark map defines the world frame.** Target B is a fixed point expressed in the landmark/world frame — **not** "100 m north of start" in the drifting VIO frame. VIO provides relative motion between fixes; markers re-anchor both the drone and B to the world frame. Correcting drift never moves B. (Fixes the frame mismatch in v3.)

---

## 2. Constraints we analyze explicitly (from the refined architecture)

These are not just risks — they are **report content** (the Evaluation and Analysis chapters).

1. **KLT displacement limit.** δ_pixel = (v·Δt·f_px)/Z. Degrades past ~15–20 px. *Applies to the deployment flight regime (5–8 m/s)* and motivates a 60 Hz camera. Our handheld data collection is slow (~1.5 m/s) so KLT is comfortable — we model the fast-flight limit analytically.
2. **Landmark density.** D ≤ vT couples speed, marker spacing, and map memory.
3. **Map memory (optimized).** ArUco markers store only ID + pose (~24 B), not satellite descriptors. ~800 markers ≈ **~20 KB** — trivially inside GAP9's ~1.6 MB. (Down from the ~400 KB descriptor estimate.)
4. **Target-centric error propagation.** Drift → wrong (−x,−y) vector → control error → more drift. This feedback loop is unique to the target-centric formulation and is analyzed directly; the scheduler's job is to break it before it compounds.
5. **Observability floor.** When active feature tracks drop below ~20, U must spike and force a landmark acquisition.
6. **Compute-aware scheduling.** AVL frequency adapts to processor load (more fixes when idle, VIO-propagate under load).

---

## 3. Hardware reality — laptops + phones only, zero purchases

| Need | How we meet it, free |
|---|---|
| Camera + IMU sensor | A phone. ARKit/ARCore are production VI-SLAM; phones are what academic VIO benchmarks (e.g. ADVIO) are built on. |
| Real VIO with ground truth | Public datasets: EuRoC MAV, TUM-VI, ADVIO. |
| Our own sequences | Free IMU+camera logger app on a phone; walk a path; feed raw stream to OpenVINS/ORB-SLAM3 on a laptop. |
| Reference / pseudo-ground-truth | ARKit/ARCore pose output, compared against our pipeline. |
| Landmarks (AVL) | ArUco markers printed on paper, detected by phone/webcam — real images. |
| Depth (obstacle) | Monocular optical-flow expansion from phone video; iPhone-Pro LiDAR if a teammate has one. |
| Compute | Laptop CPUs. **Nothing needs a GPU** — OpenVINS/ORB-SLAM3, ArUco, and the scheduler are all CPU; ARKit/ARCore run on the phone. (Rohan's no-GPU laptop is fine.) |

**In scope:** the full algorithm — VIO + uncertainty scheduler + ArUco correction + target-centric control + optical-flow obstacle avoidance — validated on datasets, own phone recordings, and a kinematic sim; profiled and partially ported to C++; RISC-V feasibility study.

**Out of scope (Future Work):** physical flight, neural/satellite scene-matching (SuperPoint), custom silicon, multi-UAV. (Flight was already future work; the budget removes only the flying demo.)

---

## 4. Tools (lean stack — no ROS/Gazebo on the critical path)

Python + NumPy · OpenCV (ArUco) · OpenVINS *or* ORB-SLAM3 (standalone, no ROS) · EuRoC/TUM-VI/ADVIO datasets · a phone IMU+camera logger · matplotlib · evo (trajectory error) · Eigen + C++ (Week 7 port) · GitHub. *Optional side branch only:* ROS 2 Humble + Gazebo **Fortress** (the one supported pair) for a flying demo — never load-bearing.

---

## 5. Team roles

| Member | Owns |
|---|---|
| **Parth** | Portable core architecture, state fusion, target-centric controller, integration |
| **Siddharth** | ArUco detection + landmark corrector + evaluation metrics + landmark/world-frame map |
| **Rohan** (no GPU — fine) | Uncertainty metric *U*, kinematic sim, optical-flow obstacle module, report, RISC-V study |

---

## 6. Repository layout (this is what makes it "industrial-grade")

```
core/      # portable, zero ROS/Gazebo/OS deps. NumPy now -> C++/Eigen later. This is what ports to RISC-V.
           #   vio_adapter, uncertainty_scheduler, landmark_corrector, state_fusion, controller
harness/   # throwaway: dataset reader, phone-recording reader, kinematic sim, plotting
tests/     # unit tests on the core
config/    # landmark world-frame map (ID -> pose), U thresholds, alphas
```
Sensor structs in, command struct out. Fixed-size buffers, no dynamic allocation in the hot loop.

---

## 7. Week-by-week (8 weeks)

### Week 1 — Skeleton + data, no simulator
- All: Python env (`numpy opencv-python matplotlib evo`). No ROS, no Gazebo.
- Parth: `core/` + `harness/` with stub interfaces (`SensorInput`, `LandmarkFix`, `VelocityCmd`).
- Siddharth: download EuRoC `MH_01`; ArUco detection working on a phone/webcam (print markers, read ID + pose).
- Rohan: install a phone IMU+camera logger; capture a first raw clip; confirm timestamps line up.
- **Deliverable:** repo replays a dataset trajectory; ArUco demo video. `setup.md` of exact steps.

### Week 2 — VIO running + drift scaffold
- Parth: ORB-SLAM3 `mono_inertial_euroc` (or OpenVINS serial) on EuRoC → real trajectory + covariance. *Fallback:* drift-injection model (bias + random walk) so the scheduler can proceed regardless.
- Siddharth: build the landmark **world-frame** map config; define target B in that frame.
- Rohan: feed a phone recording through the VIO; first own-data trajectory.
- **Deliverable:** drift plot (VIO vs ground truth diverging). Glass-box signals (σ_pos, feature count, IMU bias) exposed to the core.

### Week 3 — Controller + landmark correction
- Parth: target-centric controller (publish (x,y) offset, command (−x,−y)); state fusion that merges an AVL fix into the estimate.
- Siddharth: landmark corrector — on marker sighting, look up world pose, emit `LandmarkFix`, re-anchor (x,y) and B.
- **Deliverable:** money-shot v1 — drift accumulates, then snaps back to truth at each marker, with B fixed.

### Week 4 — Uncertainty scheduler + MID-TERM
- Rohan: implement U = α₁σ_pos + α₂σ_head + α₃·FeatureLoss + α₄·Blur + α₅·IMUbias (start with σ_pos + FeatureLoss, add terms).
- Parth: scheduler fires AVL only when U > threshold.
- **MID-TERM demo:** drift plot; drift-with-correction; **uncertainty-triggered vs fixed-period vs none**; architecture diagram; Weeks 5–8 plan.

### Week 5 — Obstacle avoidance (real, frugal)
- Rohan: monocular optical-flow expansion / time-to-contact from phone video → evasion vector perpendicular to (−x,−y); release when clear. (iPhone-Pro LiDAR depth as fallback.)
- Rohan: ~100-line kinematic sim (drone, target, obstacles) to test the control merge.
- Siddharth: verify VIO keeps tracking *through* the detour so the recomputed vector to B stays correct.
- **Deliverable:** sim + recorded-video run: head to target, detour, resume, arrive.

### Week 6 — Evaluation
- Parth: scenarios A (open, no markers), B (markers along path), C (obstacles + markers). Each ×5 for variance.
- Siddharth: metrics — final arrival error, peak drift, **AVL correction count** (frugality), marker success rate. Compare **none / fixed-period / uncertainty-aware**.
- Rohan: Evaluation chapter (4–5 pp). Headline: *similar accuracy, far fewer corrections* = the frugality win.

### Week 7 — C++ port + RISC-V study
- Parth: port the core hot loop to C++/Eigen (identical interfaces; fixed buffers).
- Rohan: profile each module; compare to GAP9 (~1.6 MB, sub-200 mW, ~1.5 ms/frame baseline). Memory: VIO state + ArUco map (~20 KB) + tracker. Estimate cycle/memory in GVSOC if time allows.
- **Deliverable:** RISC-V feasibility chapter with a per-module fits/doesn't-fit verdict; profiled (real) vs estimated (model) numbers clearly marked.

### Week 8 — Report, demo, defense
- Report (20–25 pp): Intro · Related Work · Uncertainty-Aware Scheduling · Portable Core Architecture · Evaluation · RISC-V Feasibility · Limitations · Conclusion.
- Demo video: recorded-data + sim flythrough (drift correction at markers, detour, arrival). *Optional:* Gazebo-Fortress eye-candy if the side branch worked.
- Repo cleanup, README, tag `v1.0`.

---

## 8. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| VIO won't compile | Medium | Not a hard dependency — drift-injection scaffold gives identical scheduler input. |
| Phone camera–IMU sync drift | Medium | Calibrate time offset (Week 1–2); datasets as the synced baseline. |
| ArUco unreliable | Low | Large printed markers, good lighting; dataset-frame fixes as backup. |
| Optical-flow obstacle module hard | Medium | iPhone-Pro LiDAR depth fallback; keep module isolated. |
| ROS/Gazebo dependency hell | **Removed** | Off the critical path; optional Fortress-only side branch. |
| Scope creep (SuperPoint/satellite) | High | Out of scope — ArUco only; neural matching is Future Work. |
| C++ port runs long | Medium | Port only the hot loop; Python core is the fallback deliverable. |

---

## 9. Final deliverables

- [ ] Report PDF (20–25 pp)
- [ ] Slides (~20)
- [ ] Demo video (recorded-data + sim)
- [ ] GitHub repo, tagged `v1.0`, reproducible from datasets
- [ ] Evaluation CSVs + trajectory/uncertainty plots
- [ ] Portable `core/` (Python) + C++ hot-loop port
- [ ] RISC-V feasibility chapter with per-module verdict

---

## 10. Note to the advisor

This plan keeps the original idea's heart — continuous frugal VIO grounded by intermittent absolute "geographical markings" on ultra-low-power RISC-V — and sharpens the contribution to the one genuinely novel, frugal decision: *uncertainty-aware scheduling of when to correct.* We use existing VIO (we don't reinvent it) and ArUco markers (the tractable, far cheaper cousin of satellite scene-matching). We develop on public datasets and our own phone recordings, so the lack of a drone or cameras removes only the flying demo — which was always future work. The deliverable is a portable, dependency-light navigation core, validated and profiled, ready for a follow-up hardware project to build on.