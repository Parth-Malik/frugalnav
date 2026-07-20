# FrugalNav

**An uncertainty-scheduled VIO + landmark navigator for GPS-denied UAVs, built for ultra-low-power RISC-V.**

When GPS is lost, a drone dead-reckons with Visual-Inertial Odometry (VIO) and
drifts. It can cancel that drift with an absolute fix from a known landmark
(ArUco marker) — but every fix costs camera + detection + PnP compute, i.e. power.
FrugalNav's contribution is the layer that decides ***when* to correct**:

> An **uncertainty-aware landmark scheduler** invokes absolute visual localization
> only when a fused localization-confidence metric **U** crosses a threshold —
> bounding drift while minimizing correction frequency, and therefore compute and
> power, on a portable core targetable to an ultra-low-power RISC-V SoC.

We do **not** reinvent VIO (we consume OpenVINS/ORB-SLAM3 signals) or scene
matching (we use ArUco markers). The novelty lives one layer up: *scheduling the
correction.*

---

## Headline result

Full end-to-end flight (homing to a target through a feature-poor patch and an
obstacle detour), three policies for deciding when to spend a landmark fix,
evaluated on the **real portable core** over scenarios A/B/C × seeds:

| Policy | Arrival miss | Peak drift | AVL fixes |
|---|---:|---:|---:|
| none (pure VIO) | ~3–5 m | ~3–5 m | 0 |
| fixed-period | **~1.0 m** | ~0.5 m | 6 |
| **uncertainty-aware** | **~1.0 m** | ~0.6 m | **3** |

**Same accuracy as correcting at every marker, for half the corrections** — each
skipped fix is ArUco+PnP compute (and power) saved. Where there are no markers
(scenario A) every policy ties: absolute fixes are essential, scheduling is moot.

▶ **[Live interactive flight console](web/index.html)** — open in any browser
(or the published artifact) to fly it yourself and toggle policies.

---

## Quickstart

```bash
# 1. Python 3.10+ with a fresh venv (the committed .venv, if present, is stale)
python -m venv .venv
. .venv/Scripts/activate          # Windows;  on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the end-to-end demo (prints the headline, saves figures + JSON)
python run_demo.py --seed 1 --seeds 8

# 3. Run the Week-6 evaluation on the real core (A/B/C × policies × seeds)
python demo_eval.py --seeds 5

# 4. Run the tests
python -m pytest -q                # 72 passing

# 5. Build + run the C++ / RISC-V core port
bash cpp/build_and_run.sh          # or: g++ -O2 -std=c++14 -Icpp cpp/main.cpp -o d && ./d
```

The live web demo needs no server — open `web/index.html` directly, or serve the
folder (`python -m http.server` inside `web/`).

### Gazebo + RViz + EuRoC (ROS 2 Humble)

A ROS 2 package (`ros2_ws/src/frugalnav_ros`) runs the **same C++ core** as live
nodes — see [`docs/GAZEBO_DEMO.md`](docs/GAZEBO_DEMO.md). In WSL Ubuntu 22.04:

```bash
source /opt/ros/humble/setup.bash
cd ros2_ws && colcon build --symlink-install && source install/setup.bash

ros2 launch frugalnav_ros gazebo_demo.launch.py   # drone flies to B in Gazebo, viz in RViz
ros2 launch frugalnav_ros euroc_demo.launch.py    # scheduler over real EuRoC MH_01, in RViz
```

The Gazebo node drives a real simulated drone with the scheduler+controller
(verified: homes to the target with 3 uncertainty-scheduled fixes); the EuRoC node
runs the scheduler over the real MH_01 trajectory (truth vs pure-VIO drift vs
uncertainty-aware, in RViz).

## What each command shows

| Command | Output |
|---|---|
| `python run_demo.py` | `outputs/demo_moneyshot.png`, `demo_dashboard.png`, `integrated_demo.json` |
| `python demo_eval.py` | `outputs/eval_real_*.csv`, `eval_real_bars.png`, `eval_real_frugality.png` |
| `python demo_week3.py` | Week-3 money-shot: drift snaps back at each marker |
| `python demo_week4_policies.py` | Week-4 scheduling comparison + frugality Pareto |
| `python demo_week5.py` | Week-5 obstacle detour: both avoid, only the corrected drone reaches B |
| `python demo_week6.py` | Siddharth's analytic-U evaluation (cross-check) |
| `python profiling/profile_core.py` | per-module latency breakdown |

---

## Architecture

When GPS is lost: continuous relative localization (VIO), corrected by
intermittent absolute fixes (ArUco), all fused into one estimate, driving a
target-centric controller, with reactive optical-flow obstacle avoidance.

```
 VIO delta ─► state_fusion.predict ───────────────► fused NavState (P grows)
                          │
 glass-box cues + fused σ ─► uncertainty_scheduler ─► (U, trigger?)   ← the contribution
                          │
   if trigger AND marker in view:
       aruco_detector ─► landmark_corrector ─► LandmarkFix
       state_fusion.update(fix) ───────────────► fused NavState (P shrinks)   [tight]
                          │
 optical flow (TTC) ─► obstacle_avoidance ─► evasion vector
                          │
 controller.command(fused xy, evasion) ─────────► VelocityCmd  (out)
```

**Decoupled modules, one fused estimate.** Each module is independently testable
and independently portable; the *estimate* is a single tightly-fused state that a
landmark fix is merged *into*. The **landmark map defines the world frame** — the
target B is a fixed world-frame point, so correcting drift never moves B.

### Repo layout

```
core/       # portable, zero ROS/OS deps — this is what ports to RISC-V
            #   vio_adapter, uncertainty_scheduler, state_fusion, controller,
            #   obstacle_avoidance, landmark_corrector, landmark_map, aruco_detector,
            #   geometry, vector_to_target, metrics, policies, navigator (integration)
harness/    # throwaway scaffolding: dataset/phone readers, drift + kinematic sims,
            #   integrated_sim (money-shot), eval_scenarios (A/B/C), plotting
cpp/        # C++14 header-only port of the decision hot loop (+ CMake, benchmark)
profiling/  # per-module profiler + RISC-V feasibility study
config/     # landmark world-frame map, scenario configs, thresholds
tests/      # 72 unit tests over the core
tools/      # ArUco marker-sheet generator (print + fly)
web/        # the live interactive flight console (self-contained HTML)
report/     # the technical report
outputs/    # generated figures, CSVs, demo JSON
docs/team/  # per-member weekly notes
```

Only `aruco_detector`, `blur_metric`, and the optical-flow front-end touch OpenCV
(image work is inherently pixel work). Everything downstream of a `MarkerSighting`
/ `Cues` struct is pure scalar/NumPy → allocation-free C++ → RISC-V.

---

## Data & hardware

Zero purchases. Developed on **laptops + phones only**:

* **VIO with ground truth** — public datasets (EuRoC MAV `MH_01`; `harness/euroc_reader.py`).
  A drift scaffold (`harness/drift_scaffold.py`) turns real trajectories into honest
  drifting estimates when a full VIO build isn't available.
* **Landmarks** — ArUco markers printed on paper, detected by a webcam/phone
  (`core/aruco_detector.py`, `tools/generate_marker_sheet.py`).
* **Everything is CPU-only** — no GPU, no CUDA, no ROS, no Gazebo.

Datasets are not committed (see `.gitignore`); point the readers at your local copy.

---

## Team

| Member | Owns |
|---|---|
| **Parth Malik** (2024406) | Portable core architecture, state fusion, target-centric controller, integration |
| **Siddharth Bhardwaj** (2024553) | ArUco detection, landmark corrector, landmark/world map, evaluation metrics |
| **Rohan Yadav** (2024478) | Uncertainty metric U + scheduler, kinematic sim, optical-flow avoidance, RISC-V study |

## Status

**In scope & done:** VIO drift modelling · uncertainty scheduler · ArUco correction
· tight state fusion · target-centric control · optical-flow obstacle avoidance ·
integrated end-to-end navigator · A/B/C evaluation · C++ port · RISC-V feasibility ·
live demo. See [`report/FrugalNav_Report.md`](report/FrugalNav_Report.md).

**Future work:** physical flight; neural/satellite scene matching (SuperPoint);
on-silicon GAP9 bring-up; multi-UAV.
