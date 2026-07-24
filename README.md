# FrugalNav

**An uncertainty-scheduled VIO + AVL navigator for GPS-denied UAVs, built for ultra-low-power RISC-V.**

---

## 1. What this project is

When a UAV loses GPS it has to fall back on **Visual-Inertial Odometry (VIO)** — estimating
its own motion by watching how the world moves across its camera. VIO is *relative*: it
tells you how far you moved since the last frame, never where you actually are. Small
per-frame errors accumulate, so the position estimate **drifts** — metres of error after a
minute of flight.

The standard cure is **Absolute Visual Localization (AVL)**: occasionally recognise a
landmark whose world position you already know (here, an ArUco marker) and snap the
estimate back onto it. This works, but it is not free. Every fix costs a camera frame, a
marker detection, a `solvePnP` pose solve, and a fusion update. On a battery-powered drone
running a low-power RISC-V class processor, that compute *is* flight time.

Almost all published work optimises **how** to correct. FrugalNav asks a different
question:

> ### Given that each correction costs power, *when* is a correction actually worth spending?

That is the contribution — one layer above VIO and above scene matching:

> An **uncertainty-aware landmark scheduler** fires an absolute fix only when a fused
> localization-confidence metric **U** crosses a threshold. It bounds drift while
> minimising correction frequency, and therefore compute and power, on a portable core
> that ports cleanly to an ultra-low-power RISC-V SoC.

We deliberately do **not** reinvent VIO (we consume its signals) or scene matching (we use
ArUco). The novelty is the *scheduling decision*, and the whole repository exists to build
that decision, measure it honestly, and demonstrate it flying.

### The result in one line

**The same accuracy as correcting at every single marker, for roughly half the corrections.**

| Policy | Arrival miss | Peak drift | AVL fixes |
|---|---:|---:|---:|
| none (pure VIO) | ~3–5 m | ~3–5 m | 0 |
| fixed-period | ~1.0 m | ~0.5 m | 6 |
| **uncertainty-aware** | **~1.0 m** | ~0.6 m | **3** |

Every skipped fix is an ArUco detection plus a PnP solve not performed. On the EuRoC
MH\_01 sequence the same scheduler needed **68 fixes against 259** for equal accuracy —
a **74 % reduction**. Where no markers exist at all, every policy ties: absolute fixes are
essential and scheduling is moot. We report that too.

---

## 2. How it works

### 2.1 The estimation loop

```
 VIO delta ─► state_fusion.predict ─────────────────► fused NavState (covariance P grows)
                       │
 image cues + fused σ ─► uncertainty_scheduler ─────► (U, trigger?)      ◄── the contribution
                       │
   if trigger AND a marker is in view:
       aruco_detector ─► landmark_corrector ─► LandmarkFix
       state_fusion.update(fix) ───────────────────► fused NavState (P shrinks)
                       │
 obstacle sensing ─► obstacle_avoidance ────────────► evasion vector
                       │
 controller.command(fused xy, evasion) ─────────────► VelocityCmd  (out to the drone)
```

Each module is independently testable and independently portable. The *estimate* is a
single tightly-fused state that a landmark fix is merged **into** — not a separate
"corrected" copy. Crucially, **the landmark map defines the world frame**, so the target
B is a fixed world-frame point and correcting drift never moves the goal.

### 2.2 The scheduler — what U actually is

`U` is a normalised confidence score fused from *glass-box* cues, each of which is cheap to
compute and physically meaningful:

| Cue | Meaning | Source |
|---|---|---|
| `sigma_pos` | positional uncertainty from the fusion covariance | `state_fusion` |
| `sigma_head` | heading uncertainty | `state_fusion` |
| `blur` | Laplacian-variance sharpness — a blurry frame makes detection unreliable | `blur_metric` |
| `feature_loss` | drop in trackable features frame-to-frame | perception |
| `active_features` | how much texture is currently trackable | perception |
| `imu_bias` | drift-rate proxy | VIO adapter |

Each cue is normalised against bounds (`CueBounds`), weighted, and combined into `U`.
When `U` crosses `tau`, the scheduler returns `trigger = True` and the navigator spends a
fix **if** one is available. After a fix, `reset_after_fix()` collapses the accumulated
uncertainty. The whole decision is a handful of scalar operations — no allocation, no
matrix inversion — which is exactly why it ports to a microcontroller-class core.

### 2.3 Why this is cheap enough for RISC-V

Two deliberate engineering choices carry the "frugal" claim beyond the scheduler itself:

* **Motion → sparse optical flow.** Demo 3's VIO tracks ~90 features with Lucas-Kanade.
  This is the same class of computation as the PX4Flow sensor, which runs on a small MCU.
  No dense flow, no stereo matching, no learned model.
* **Obstacles → a laser rangefinder, not camera vision.** A range sensor returns distances
  directly, so detection is a handful of comparisons per cycle — the sort of thing a
  sub-$5 time-of-flight sensor does on an 8-bit micro. Camera-based obstacle detection
  (stereo depth, dense optical-flow time-to-contact, or a CNN) is precisely the CPU-heavy
  work that would destroy the power budget, so it is avoided on purpose.

`cpp/` contains a header-only C++14 port of the decision hot loop, and `profiling/`
measures per-module latency to back the feasibility argument.

### 2.4 What is measured and what is given — honest scope

This matters more than any performance number, so it is stated plainly.

| Signal | Demo 1 (sandbox) | Demo 2 (real camera) | Demo 3 (full realism) |
|---|---|---|---|
| Motion (VIO) | injected | simulated drift model | **real optical flow from images** |
| Position (AVL) | injected | **real ArUco + solvePnP** | **real ArUco + solvePnP** |
| Obstacles | known map | known map | **real laser rangefinder** |
| Wind | injected (you set it) | **inferred** | **inferred** |

In **Demo 3 no navigation decision uses ground truth.** Seeking, avoidance, fusion and
prediction all run on the estimate, the laser scan, the optical flow and the ArUco fixes.
Ground truth is touched only to draw the error in RViz, to teleport on reset, and for two
calibration steps:

1. **Altitude** is read from the simulator as a stand-in for a cheap altimeter — optical
   flow needs a metric scale, and every real flow deck pairs with one.
2. **Two one-time mounting calibrations** (the camera→world rotation for VIO, the
   extrinsic offset for AVL) are fitted against truth once at startup, equivalent to
   bench-calibrating a sensor mount at the factory. Runtime uses the calibrated result.

Also honest: the arenas and sensors are simulated in Gazebo, and the RISC-V target is a
feasibility study (C++ port plus profiling), not silicon. The laser avoider is a reactive
potential field, so it takes a conservative, sometimes curved path rather than an optimal
one.

---

## 3. The three demo families

| Family | Maps | What it demonstrates | Best viewed in |
|---|---|---|---|
| **1 — Sandbox** | `demo`, `canopy` | Scheduler + full weather simulator + manual flight | Gazebo 3D |
| **2 — Real camera** | `real`, `real_dense` | Genuine ArUco perception, wind inferred | RViz + camera feed |
| **3 — Full realism** | `real`, `real_dense` | Real optical-flow VIO + real laser obstacles | RViz + both feeds |

---

## 4. Setup

### 4.1 Python side (works on plain Windows/Linux/macOS)

```bash
python -m venv .venv
. .venv/Scripts/activate          # Windows;  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### 4.2 ROS 2 side (for the Gazebo demos)

Requires **ROS 2 Humble + Gazebo Classic 11 + RViz2**, developed under WSL Ubuntu 22.04.

```bash
source /opt/ros/humble/setup.bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

> **NumPy note:** the ROS OpenCV/cv_bridge stack breaks under NumPy 2. If perception fails
> to import, run `pip3 install --user "numpy<2"`.

---

## 5. Running everything

### 5.1 Python demos (no ROS needed)

```bash
python run_demo.py --seed 1 --seeds 8     # end-to-end flight, saves figures + JSON
python demo_eval.py --seeds 5             # A/B/C scenarios x policies x seeds
python -m pytest -q                       # 79 tests
bash cpp/build_and_run.sh                 # build + benchmark the C++ core
python profiling/profile_core.py          # per-module latency breakdown
```

| Command | Output |
|---|---|
| `python run_demo.py` | `outputs/demo_moneyshot.png`, `demo_dashboard.png`, `integrated_demo.json` |
| `python demo_eval.py` | `outputs/eval_real_*.csv`, `eval_real_bars.png`, `eval_real_frugality.png` |
| `python demo_week3.py` | drift snapping back at each marker |
| `python demo_week4_policies.py` | the scheduling comparison + frugality Pareto |
| `python demo_week5.py` | obstacle detour: both avoid, only the corrected drone reaches B |
| `python demo_week6.py` | analytic-U evaluation cross-check |

The browser console needs no server — open `web/index.html` directly.

### 5.2 Before every Gazebo run

Run **one** launch at a time; leftovers fight over `/frugalnav/cmd_vel`.

```bash
pkill -9 -f gzserver; pkill -9 -f gzclient; pkill -9 -f rviz2; pkill -9 -f frugalnav; sleep 3
```

Start **every** terminal with these three lines:

```bash
source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.sh
source /mnt/c/Users/parth/Downloads/drone/ros2_ws/install/setup.bash
```

### 5.3 Demo 1 — Sandbox

**Terminal 1** (use `map:=canopy` for the dense forest):

```bash
ros2 launch frugalnav_ros interactive_demo.launch.py map:=demo
```

**Terminal 2** — fly it:

```bash
ros2 run frugalnav_ros frugalnav_teleop.py
```

Controls: **W A S D** move · **1/2/3** auto/manual/EuRoC · **U/N/M** altitude up/down/auto ·
**] [** wind · **- =** fog · **T** rain · **G** weather master · **R** reset · **P** pause.

### 5.4 Demo 2 — Real camera perception

**Terminal 1** (`start_paused:=true` holds it until you press PLAY):

```bash
ros2 launch frugalnav_ros real_demo.launch.py map:=real start_paused:=true gui:=true
```

**Terminal 2** — mission control:

```bash
ros2 run frugalnav_ros frugalnav_mission_control.py
```

**Terminal 3** — the drone's camera feed:

```bash
ros2 run rqt_image_view rqt_image_view /frugalnav/down_cam/annotated
```

Use `map:=real_dense` for the big cluttered arena (14 obstacles, 20 markers).

### 5.5 Demo 3 — Full realism

**Terminal 1** — Demo 3 is the heaviest (two cameras + laser + VIO), so prefer
`gui:=false` and watch in RViz; the Gazebo 3D window can starve the camera frames the
optical-flow VIO depends on:

```bash
ros2 launch frugalnav_ros real3_demo.launch.py map:=real gui:=false start_paused:=true
```

**Terminal 2** — mission control:

```bash
ros2 run frugalnav_ros frugalnav_mission_control.py
```

**Terminal 3** — downward camera (ArUco / VIO view):

```bash
ros2 run rqt_image_view rqt_image_view /frugalnav/down_cam/annotated
```

**Terminal 4** — forward camera (obstacle view):

```bash
ros2 run rqt_image_view rqt_image_view /frugalnav/front_cam/annotated
```

Then press **SPACE** in Terminal 2 to fly.

**Mission-control keys:** **SPACE** play/pause (a true freeze — the wind stops too) ·
**R** reset (teleports home and holds) · **I/J/K/L** place the drone before starting ·
**1/2** auto/manual · **W A S D** fly, **X** stop · **] [** wind · **T** rain/gust ·
**G** weather · **Q** quit.

**What you should see in RViz:** faint grey pillars (reference only — the navigator does
not read them), a yellow target, a blue drone cube, a green true path, a cyan estimated
path, **red dots** where the laser actually detects obstacles, yellow dots at each frugal
fix, and a cyan wind arrow.

### 5.6 The original ROS demos

```bash
ros2 launch frugalnav_ros gazebo_demo.launch.py   # scheduler flying a simple Gazebo drone
ros2 launch frugalnav_ros euroc_demo.launch.py    # scheduler over real EuRoC MH_01 in RViz
```

### 5.7 One-click Windows launchers

`run_interactive_demo.bat` · `run_canopy_map.bat` · `run_real_demo.bat` ·
`run_gazebo_demo.bat` · `run_euroc_demo.bat`

---

## 6. Full file reference

### 6.1 `core/` — the portable algorithm

Zero ROS and zero OS dependencies. **This is the product**; everything else builds, tests
or demonstrates it. Only `aruco_detector` and `blur_metric` touch OpenCV, because image
work is inherently pixel work; everything downstream of a `MarkerSighting` / `Cues` struct
is pure scalar/NumPy and therefore portable to allocation-free C++.

**`uncertainty_scheduler.py`** — The contribution. Defines `CueBounds` (per-cue
normalisation ranges), `SchedulerConfig` (weights, the trigger threshold `tau`, and
`sigma_pos_floor`), and `UncertaintyScheduler`. `compute(cues)` normalises every cue,
fuses them into the scalar `U`, and returns `(U, trigger, reason, detail)` — `reason`
naming which cue dominated, which is what makes the decision auditable rather than a black
box. `reset_after_fix()` collapses accumulated uncertainty once a correction is spent.

**`state_fusion.py`** — The single fused estimate. `FusionConfig` and `StateFusion` keep
position plus a growing covariance. `predict(delta)` dead-reckons on a VIO increment and
inflates the covariance by `q_per_metre` per metre travelled — this is what makes
uncertainty grow with distance rather than with wall-clock time. `update(LandmarkFix)`
merges an absolute fix and shrinks the covariance. `sigma_pos()` exposes the positional
uncertainty the scheduler consumes, closing the loop between estimation and scheduling.

**`controller.py`** — `ControllerConfig` (`kp`, `v_max`, `arrive_tol`) and
`TargetCentricController`. Produces a velocity command toward the world-frame target and
answers `arrived()`. Target-centric by design: because the goal is a fixed world point,
a drift correction changes where the drone thinks *it* is, never where the goal is.

**`obstacle_avoidance.py`** — Reactive avoidance with several time-to-contact estimators:
`ttc_from_range`, `ttc_from_looming`, `ttc_from_flow_divergence` and `estimate_ttc_lk`
(Lucas-Kanade based). `AvoidanceConfig` sets trigger/release thresholds and gain;
`ObstacleAvoidance.update()` returns an evasion vector. Used by Demos 1 and 2; Demo 3
replaces it with a laser potential field built in the ROS node.

**`landmark_corrector.py`** — Siddharth's AVL core. Converts a `MarkerSighting` plus the
known marker world pose into a `LandmarkFix`, including the `reanchor` operation that
re-expresses the estimate in the landmark's frame.

**`landmark_map.py`** — `MarkerEntry` and `LandmarkMap`: the id → world-pose table that
**defines the world frame**. Everything downstream is expressed relative to this map.

**`aruco_detector.py`** — `ArucoDetector` and `marker_object_points`. Wraps `cv2.aruco`
detection plus pose solving and emits a `MarkerSighting`. The only marker-facing OpenCV
dependency in the core.

**`blur_metric.py`** — Image-quality input to `U`. `laplacian_sharpness` measures focus
via Laplacian variance, `blur_badness` maps it into the cue range, `gaussian_blur` is used
to synthesise degraded frames for testing.

**`geometry.py`** — Pure-NumPy SE(3) helpers: `rot_from_rvec`, `rvec_from_rot`, `rot_z`,
`rpy_to_R`, `make_T`, `T_from_rvec_tvec`, `inv_T`, `translation_of`. `rvec_from_rot` uses
a quaternion (Shepperd) formulation because the naive axis-angle inverse is numerically
unstable near 180°, which previously produced metre-level position errors. There is a
regression test for exactly that case.

**`se3.py`** — Lower-level manifold utilities: `skew`, `exp_so3`, `make_se3`, `inv_se3`,
`relative_se3`, `q_to_R`, and `project_to_SO3` (which re-orthonormalises a rotation that
has drifted off the manifold through accumulated float error).

**`uncertainty.py`** — `UncertaintyWeights` and the reference `uncertainty()` function:
the analytic definition of U, kept separate from the scheduler so the metric can be
validated independently of the triggering policy.

**`policies.py`** — The three correction strategies the evaluation compares:
`NonePolicy` (pure VIO), `FixedPeriodPolicy` (correct every N seconds) and
`UncertaintyPolicy` (FrugalNav). All share the `CorrectionPolicy` interface and a
`PolicyContext`, so the experiment swaps one object and changes nothing else — that is
what makes the headline comparison fair.

**`metrics.py`** — Siddharth's evaluation metrics: `peak_drift`, `final_drift`, `rmse`,
`arrival_error`, `correction_count`, `marker_success_rate`, and `summarize`. These define
what "same accuracy for fewer fixes" quantitatively means.

**`navigator.py`** — Integration layer. `Navigator` and `build_navigator` wire fusion,
scheduler, corrector, avoidance and controller into one object with a single step
function, consuming `SensorInput` and emitting `NavOutput`.

**`pipeline.py`** — `FrugalPipeline`, an alternative end-to-end assembly used by the
harness and the tests.

**`scheduler_bridge.py`** — The integration seam between the pipeline and the scheduler:
`cues_from_pipeline` adapts pipeline state into the scheduler's cue dictionary,
`pipeline_scheduler_config` maps configuration, and `PipelineScheduler` binds them. This
exists so Rohan's scheduler could be merged without either side rewriting the other.

**`vio_adapter.py`** — `VioSignals` and `VioSource`: the contract for whatever produces
relative motion, so a real VIO, a dataset replay or a drift model are interchangeable.

**`vector_to_target.py`** — `vector_to_target`, `heading_to_target`, `position_error`,
`direction_error_deg`: the target-centric error definitions that stay meaningful even when
the drone is mid-detour and not pointing at the goal.

**`types.py` / `interfaces.py`** — The fixed-shape structs that cross module boundaries
(`MarkerSighting`, `LandmarkFix`, `NavState`, `VelocityCmd`, `SensorInput`, `VioOutput`,
`PoseEstimate`). Deliberately fixed-shape so they map onto C++/Eigen structs.

### 6.2 `harness/` — measurement scaffolding (throwaway, does not ship)

**`euroc_reader.py` / `dataset_reader.py` / `groundtruth.py`** — Read the EuRoC MAV
dataset (MH\_01) and its ground truth, enforcing strict units and quaternion ordering
(a silent source of error if got wrong).

**`drift_sim.py` / `drift_injection.py` / `drift_scaffold.py`** — Turn a true trajectory
into an honestly-drifting estimate when a full VIO build is not available. This is how the
project studies drift without needing OpenVINS on a laptop.

**`synthetic_landmarks.py`** — A stand-in landmark source for testing the corrector
without a camera.

**`kinematic_sim.py` / `nav_sim.py` / `detour_sim.py` / `integrated_sim.py`** — Closed-loop
simulators of increasing completeness, from bare kinematics to the full money-shot run.

**`eval_scenarios.py` / `eval_sim.py`** — The A/B/C scenario definitions and the evaluation
driver that produces the headline table.

**`plotting.py`** — All figure generation into `outputs/`.

**`check_sync.py`** — Sanity checks that timestamps line up across sources.

### 6.3 `ros2_ws/src/frugalnav_ros/` — the live demos

**Python nodes (`scripts/`)**

* **`frugalnav_perception.py`** — Turns the downward camera image into navigation signals.
  Runs `cv2.aruco.detectMarkers` with sub-pixel corner refinement plus `solvePnP`
  (`SOLVEPNP_IPPE_SQUARE`), looks each detected id up in the marker map and recovers the
  drone's absolute world position → `/frugalnav/fix`. Also measures Laplacian sharpness
  and trackable-feature count → `/frugalnav/cues`. Publishes an annotated feed on
  `/frugalnav/down_cam/annotated`. Contains a one-time mounting calibration that fits an
  image→world offset over 40 samples, which reduced fix error from about 4 m to 0.05 m.
* **`frugalnav_vio.py`** — Demo 3's real VIO front end. Tracks ~90 sparse features with
  Lucas-Kanade between frames, converts median pixel flow into metric velocity using
  altitude and focal length, and publishes `/frugalnav/vio`. The camera→world mapping is
  fitted with an **orthogonal (Procrustes)** calibration rather than plain least squares —
  a general 2×2 fit overfits the mostly one-directional calibration motion and diverges
  once the drone turns. It stays silent until calibrated, so an unscaled velocity can
  never enter the wind loop.
* **`frugalnav_real_node.py`** — Demo 2 navigator. Runs the real core on camera fixes and
  image cues, with a simulated VIO drift model, and infers wind from commanded-versus-
  achieved velocity.
* **`frugalnav_real3_node.py`** — Demo 3 navigator. Predicts on real optical-flow velocity,
  avoids obstacles from `/frugalnav/scan`, spends ArUco fixes frugally, and infers wind.
  The laser potential field bins the scan into angular sectors and uses only the nearest
  return per sector (one push per obstacle, not one per ray, otherwise a wide pillar
  out-pushes the seek velocity), cancels the velocity component heading into an obstacle so
  the drone slides around it, and adds a goal-side tangential swirl so seek and repulsion
  cannot balance into a standstill.
* **`frugalnav_wind.py`** — The disturbance the navigator cannot see. Sits between the
  navigator and the drone: `nav_cmd + wind → cmd_vel`. Publishes the true wind on
  `/frugalnav/wind_true` for RViz comparison only — nothing feeds it back to the navigator.
* **`frugalnav_mission_control.py`** — Control panel for the camera demos: play/pause,
  reset, repositioning the start point, manual flight, wind and weather.
* **`frugalnav_teleop.py`** — The sandbox keyboard panel and weather simulator.
* **`frugalnav_front_view.py`** — Draws laser proximity onto the forward camera image
  (`OBSTACLE AHEAD`, `AVOIDING`, `PATH CLEAR`). Display only.

**C++ nodes (`src/`)** — `frugalnav_gazebo_node.cpp` (scheduler flying a Gazebo drone),
`frugalnav_euroc_node.cpp` (scheduler over the EuRoC trajectory), and
`frugalnav_interactive_node.cpp` (the sandbox brain: auto/manual/EuRoC modes, weather,
altitude, reset-by-teleport). All three call the vendored header-only core in
`include/frugalnav/` unchanged, which is the point: the same code that ports to RISC-V is
the code flying the demo.

**Assets** — `worlds/*.world` (`demo`, `canopy`, `real`, `real_dense`),
`config/*_scene.txt` (target/start/pillars/markers, loaded at runtime so one binary flies
any map), `models/` (drone variants: plain, `_cam` with a downward camera, `_real3` adding
a forward camera and a 360° laser mounted above the body so it cannot range itself),
`media/` (48 generated ArUco textures plus the Gazebo material script), `launch/` (one per
demo), `rviz/frugalnav.rviz`.

**`tools/`** — `gen_maps.py` generates every world plus its scene file
(`python tools/gen_maps.py real` rebuilds only the real maps and leaves the sandbox
untouched); `gen_aruco_media.py` renders the marker textures.

### 6.4 `cpp/` — the RISC-V feasibility port

`frugalnav/uncertainty_scheduler.hpp` and `frugalnav/nav_core.hpp` are a header-only C++14
port of the decision hot loop; `main.cpp` benchmarks it; `build_and_run.sh` compiles and
runs in one step (Windows Application Control can block a freshly-compiled unsigned binary
launched separately, so the script does both together).

### 6.5 Everything else

* **`profiling/`** — per-module latency breakdown backing the RISC-V argument.
* **`tests/`** — 79 unit tests over the core, including the 180° rotation regression.
* **`web/index.html`** — self-contained browser flight console; toggle policies live.
* **`report/`** — the technical report.
* **`docs/`** — `RUN_COMMANDS.md` (every command), `FrugalNav_Guide.pdf` (illustrated
  guide), `GAZEBO_DEMO.md`, `architecture.md`.
* **`outputs/`** — generated figures, CSVs and demo JSON.
* **`demo_week1..6.py`**, **`run_demo.py`**, **`demo_eval.py`** — the weekly deliverables
  and the two main entry points.

---

## 7. Team

| Member | Owns |
|---|---|
| **Parth Malik** (2024406) | Portable core architecture, state fusion, target-centric controller, EuRoC dataset reader, repository integration |
| **Siddharth Bhardwaj** (2024553) | ArUco detection, landmark corrector, landmark/world map, evaluation metrics |
| **Rohan Yadav** (2024478) | Uncertainty metric U and the scheduler, kinematic simulation, optical-flow avoidance, RISC-V study |

Development ran as weekly branches (`w2/parth-euroc-reader`, `w3/fusion-and-control`,
`w4/uncertainty-scheduler`, `integrate/rohan-scheduler`) merged through pull requests.
The later phases — the C++/RISC-V port, the ROS 2 + Gazebo demo suite, the real
camera-perception pipeline and the Demo 3 realism work — were integrated on top of that
foundation.

---

## 8. Troubleshooting

**A window only appears as a taskbar icon (WSLg).** Click the taskbar icon; if it is
off-screen, hover it, right-click the thumbnail, choose *Move*, press an arrow key and
move the mouse. Failing that, `wsl --shutdown` in PowerShell and relaunch.

**The drone will not move.** You launched with `start_paused:=true` — start mission control
and press **SPACE**, or relaunch without that argument.

**RViz shows no arena.** The scene is published latched (transient-local); make sure you
rebuilt after pulling.

**Demo 3's estimate diverges.** The sim is running too slowly and starving the optical-flow
VIO of frames. Run with `gui:=false` and close spare image viewers.

**The drone jerks or fights itself.** Two launches are running. Kill everything with the
cleanup line in §5.2 and start exactly one.

**`pytest` fails collecting `ros2_ws`.** Already handled by `pytest.ini`; run plain
`pytest` from the repository root.

---

## 9. Status

**Done:** VIO drift modelling · uncertainty scheduler · ArUco correction · tight state
fusion · target-centric control · obstacle avoidance · integrated navigator · A/B/C
evaluation · EuRoC validation · C++ port · RISC-V feasibility study · browser demo ·
three ROS 2 + Gazebo demo families with real camera perception, real optical-flow VIO and
real laser obstacle sensing.

**Future work:** physical flight; a real VIO front end (OpenVINS/ORB-SLAM3) in place of the
optical-flow stand-in; learned scene matching (SuperPoint) instead of fiducials;
on-silicon GAP9 bring-up; multi-UAV coordination.
