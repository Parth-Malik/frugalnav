# FrugalNav — Gazebo + RViz + EuRoC demos (ROS 2 Humble)

Two ROS 2 demos that run the **real FrugalNav C++ core** (`cpp/frugalnav/*.hpp` —
the uncertainty scheduler, state fusion, controller, obstacle avoidance) as live
ROS nodes:

1. **Gazebo demo** — a holonomic drone flies to a target through a feature-poor
   patch and an obstacle, in a Gazebo world, driven frame-by-frame by the
   scheduler; visualized in Gazebo **and** RViz.
2. **EuRoC demo** — the scheduler runs over the real EuRoC MH_01 ground-truth
   trajectory, shown in RViz as truth (green) vs pure-VIO drift (red) vs
   uncertainty-aware (cyan).

Everything runs in **WSL Ubuntu 22.04** with ROS 2 Humble + Gazebo Classic 11.

---

## One-time setup

ROS 2 Humble + Gazebo + RViz are already installed. Build the workspace:

```bash
# in WSL Ubuntu 22.04
source /opt/ros/humble/setup.bash
cd /mnt/c/Users/parth/Downloads/drone/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

> The workspace lives inside the repo at `ros2_ws/`. `build/`, `install/`, `log/`
> are git-ignored. Building on the Windows mount (`/mnt/c`) works but is slower than
> a native path; to speed it up, copy `ros2_ws/` to `~` and build there.

Add sourcing to each new shell (or your `~/.bashrc`):

```bash
source /opt/ros/humble/setup.bash
source /mnt/c/Users/parth/Downloads/drone/ros2_ws/install/setup.bash
```

---

## 1. Gazebo + RViz demo

```bash
ros2 launch frugalnav_ros gazebo_demo.launch.py
```

Gazebo opens with the world + drone; RViz opens with the navigation view. The drone
lifts off the green start pad and homes on the gold target pole, snapping its
estimate back at scheduler-chosen marker tiles and detouring around the obstacle.

**Verified headless result:** the drone homes (58,24)→(0,0) and **arrives 1.14 m
from the target using 3 uncertainty-scheduled fixes** (est-error 0.37 m) — the same
outcome as the Python `run_demo.py` money-shot, now driven by the C++ core in Gazebo.

**Headless** (no GUI — for verification / low-resource machines):

```bash
ros2 launch frugalnav_ros gazebo_demo.launch.py gui:=false rviz:=false
```

Watch it work from the terminal:

```bash
ros2 topic echo /frugalnav/U            # the uncertainty signal
ros2 topic echo /frugalnav/cmd_vel      # the scheduler-driven velocity command
ros2 node info /frugalnav_gazebo_node
```

**Topics**

| Topic | Type | Meaning |
|---|---|---|
| `/frugalnav/truth` | `nav_msgs/Odometry` | drone ground-truth pose (Gazebo p3d) → node input |
| `/frugalnav/cmd_vel` | `geometry_msgs/Twist` | scheduler+controller output → drives the drone |
| `/frugalnav/U` | `std_msgs/Float32` | live uncertainty metric |
| `/frugalnav/scene` | `MarkerArray` | static world (markers, obstacle, target, hard patch) |
| `/frugalnav/viz` | `MarkerArray` | live truth/estimate paths, drone, fixes, HUD text |

---

## 2. EuRoC MH_01 demo (RViz)

```bash
ros2 launch frugalnav_ros euroc_demo.launch.py
```

Point it at a different EuRoC sequence or change playback speed:

```bash
ros2 launch frugalnav_ros euroc_demo.launch.py \
  gt_csv:=/path/to/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv \
  rate_hz:=90.0
```

RViz shows four trajectories over the real flight: **green** ground truth, **red**
pure VIO (never corrected, drifts away), **amber** fixed-period (correct at every
marker pass), **cyan** uncertainty-aware (snaps back only at the markers the
scheduler decides are worth spending). The node prints final/peak error and fix
counts for all policies when the sequence ends.

**Verified headless result (MH_01):**

| Policy | Final error | Peak error | Fixes |
|---|---:|---:|---:|
| none (pure VIO) | 1.79 m | 1.79 m | 0 |
| fixed-period | 0.02 m | 0.06 m | 259 |
| **uncertainty-aware** | **0.04 m** | **0.06 m** | **68** |

→ same accuracy as fixed-period, **74% fewer corrections** on real data.

---

## 3. Interactive arena — fly it yourself

A richer world (`frugalnav_arena.world`: a pillar slalom, buildings, perimeter
walls, a dense marker field, hard patch, target) with a **multi-mode** node you
drive from the keyboard. Two terminals:

```bash
# terminal 1 — the sim (Gazebo + RViz)
ros2 launch frugalnav_ros interactive_demo.launch.py

# terminal 2 — keyboard mission control
ros2 run frugalnav_ros frugalnav_teleop.py
```

…or just double-click **`run_interactive_demo.bat`** on Windows (it opens the sim
in one window and puts keyboard control in the other).

**Controls (in the teleop terminal):**

| Key | Action |
|---|---|
| `1` | **AUTO** — the scheduler flies the drone to the target, weaving the pillars |
| `2` | **MANUAL** — you fly with **W A S D** (W=north, S=south, A=west, D=east) |
| `3` | **EUROC** — the drone flies the real EuRoC MH_01 trajectory through the arena |
| `W`/`A`/`S`/`D` | fly (in MANUAL); `SPACE`/`K` = stop |
| `R` | **RESET** — teleport the drone back to start & clear the estimate (the "rewind") |
| `P` | **PAUSE / RESUME** |
| `Q` | quit teleop |

There is no physics "rewind" in Gazebo, so **R restarts the run** (teleport to
start + fresh estimator) and **P pauses** — together with manual flight that gives
you full control. Gazebo's own toolbar also has play / pause / step buttons at the
bottom, and `Ctrl+R` resets the world.

In **MANUAL** mode the estimator keeps running as you fly, so you can *watch* VIO
drift accumulate (green truth vs cyan estimate diverging) and see the scheduler fire
a correction when you pass a marker — a hands-on feel for what the scheduler does.

## How it maps to the C++ core

Both nodes `#include "frugalnav/uncertainty_scheduler.hpp"` and
`"frugalnav/nav_core.hpp"` — the same headers benchmarked in `cpp/` and targeted at
RISC-V — and call them unchanged each frame:

```
StateFusion.predict(drifted VIO delta)         // covariance grows
  → UncertaintyScheduler.compute(cues)          // U, trigger?      ← the contribution
  → (if fired & marker in view) StateFusion.update(fix)   // snap back
  → ObstacleAvoidance.update(ttc, bearing)      // evasion vector
  → TargetCentricController.command(est, evade) // velocity out
```

Gazebo provides physics/ground-truth motion; the EuRoC node provides real recorded
motion. In both, only the **ArUco detection** is stubbed (as a perfect absolute
fix) — that geometry is validated separately by the Python landmark corrector and
its unit tests. The scheduling decision, the fusion, the control, and the avoidance
are the real portable core.

## Troubleshooting

- **`ros2: command not found`** → `source /opt/ros/humble/setup.bash` first.
- **`package 'frugalnav_ros' not found`** → `source ros2_ws/install/setup.bash`.
- **Gazebo GUI is black / slow in WSL** → WSLg renders GUI; give it a few seconds,
  or run headless (`gui:=false`) and use RViz, which is lighter.
- **Drone doesn't move** → check `ros2 topic echo /frugalnav/truth` is publishing
  (p3d plugin) and `/frugalnav/cmd_vel` is non-zero (the node).
