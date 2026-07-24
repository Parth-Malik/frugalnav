# FrugalNav — How to run every demo

Complete, copy-paste terminal commands for all five demos. Everything runs in
**WSL Ubuntu 22.04**. Open a new Ubuntu terminal for each "Terminal N" below.

There are **two families** of demo:

| Family | Maps | Drone flies on | Best window to watch |
|--------|------|----------------|----------------------|
| **Sandbox** (interactive) | `demo`, `canopy` | injected/simulated signals + full weather sim | **Gazebo 3D** (the world is the point) |
| **Real** (camera perception) | `real`, `real_dense` | a genuine downward **camera** (real ArUco detection) | **RViz** + the **camera feed** |

---

## 0. Do this before EVERY run

Two rules save you every time:

1. **Run only ONE launch at a time.** All demos share `/frugalnav/cmd_vel`; leftovers
   from a previous run will fight over the drone.
2. **Clean up first.** In any Ubuntu terminal:

```bash
pkill -9 -f gzserver; pkill -9 -f gzclient; pkill -9 -f rviz2; pkill -9 -f frugalnav; sleep 3
```

Every launch/teleop terminal begins with the **same three source lines** — just paste
them at the top of each terminal:

```bash
source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.sh
source /mnt/c/Users/parth/Downloads/drone/ros2_ws/install/setup.bash
```

---

## 1. DEMO map (sandbox)

**Terminal 1 — launch the world (watch this one):**
```bash
source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.sh
source /mnt/c/Users/parth/Downloads/drone/ros2_ws/install/setup.bash
ros2 launch frugalnav_ros interactive_demo.launch.py map:=demo
```

**Terminal 2 — fly it / weather (keep this window focused when pressing keys):**
```bash
source /opt/ros/humble/setup.bash
source /mnt/c/Users/parth/Downloads/drone/ros2_ws/install/setup.bash
ros2 run frugalnav_ros frugalnav_teleop.py
```

One-click alternative (Windows Explorer): double-click `Downloads\drone\run_interactive_demo.bat`.

---

## 2. CANOPY map (sandbox, dense forest)

Exactly the same as the demo map — only the last line changes.

**Terminal 1 — launch the world:**
```bash
source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.sh
source /mnt/c/Users/parth/Downloads/drone/ros2_ws/install/setup.bash
ros2 launch frugalnav_ros interactive_demo.launch.py map:=canopy
```

**Terminal 2 — fly it / weather:**
```bash
source /opt/ros/humble/setup.bash
source /mnt/c/Users/parth/Downloads/drone/ros2_ws/install/setup.bash
ros2 run frugalnav_ros frugalnav_teleop.py
```

One-click alternative: double-click `Downloads\drone\run_canopy_map.bat`.

### Keyboard controls (Terminal 2, both sandbox maps)

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| **W A S D** | move | **1 / 2 / 3** | AUTO / MANUAL / EuRoC |
| **SPACE** | stop | **U / N / M** | altitude up / down / auto |
| **] / [** | wind + / − | **- / =** | fog + / clearer |
| **T** | rain on/off | **G** | weather master on/off |
| **R** | reset (rewind to start) | **P** | pause / resume |

> Press **2** to take manual control, then fly with **WASD**. Watch the HUD at the top of RViz.

---

## 3. SHOW THE DRONE FEED (what the camera sees)

The camera only runs during the **real** demos (Section 4 / 5). While a real demo is
running, open one more terminal:

**Terminal 3 — live camera view with ArUco detection drawn on it:**
```bash
source /opt/ros/humble/setup.bash
source /mnt/c/Users/parth/Downloads/drone/ros2_ws/install/setup.bash
ros2 run rqt_image_view rqt_image_view /frugalnav/down_cam/annotated
```

That window shows the downward camera image with **detected markers outlined** and a HUD
(`markers=…  blur=…  feats=…` and the live `FIX (x, y) m  err=…`). It's the drone
literally seeing and locking onto the tags.

- Want the raw image with nothing drawn? Use `/frugalnav/down_cam/image_raw` instead.
- If the rqt window won't paint in WSLg, click its taskbar icon (see Troubleshooting).

---

## 4. REAL demo (camera perception — compact arena)

The drone flies **autonomously on real vision**: a real downward camera → real ArUco
detection → the blind navigator (which *estimates* the wind, never sees its true value).

**Terminal 1 — launch, HELD, ready for mission control** (`start_paused:=true` waits for you
to press PLAY, so you can place the drone and set weather first):
```bash
source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.sh
source /mnt/c/Users/parth/Downloads/drone/ros2_ws/install/setup.bash
ros2 launch frugalnav_ros real_demo.launch.py map:=real start_paused:=true gui:=true
```
- Drop `start_paused:=true` if you just want it to fly itself immediately.
- Drop `gui:=true` (or set `gui:=false`) to skip the heavy Gazebo 3D window and watch in RViz only.

**Terminal 2 — MISSION CONTROL (play / pause / reset / place the drone / weather):**
```bash
source /opt/ros/humble/setup.bash
source /mnt/c/Users/parth/Downloads/drone/ros2_ws/install/setup.bash
ros2 run frugalnav_ros frugalnav_mission_control.py
```

**Terminal 3 — the drone feed (see Section 3):**
```bash
source /opt/ros/humble/setup.bash
source /mnt/c/Users/parth/Downloads/drone/ros2_ws/install/setup.bash
ros2 run rqt_image_view rqt_image_view /frugalnav/down_cam/annotated
```

One-click alternative: double-click `Downloads\drone\run_real_demo.bat`.

### Mission-control keys (Terminal 2 — real demos only)

| Key | Action |
|-----|--------|
| **SPACE** | PLAY / PAUSE (toggle) — pause truly freezes the drone, wind included |
| **R** | RESET — teleport the drone back to its start and hold |
| **I / K** | move the start point **north / south** (teleport, while held) |
| **J / L** | move the start point **west / east** |
| **1 / 2** | AUTO (fly to target) / MANUAL (fly it yourself) |
| **W A S D** | fly (MANUAL mode); **X** = stop |
| **] / [** | wind stronger / weaker |
| **T** | rain / gust on-off |
| **G** | weather master on-off |
| **Q** | quit |

> Typical flow: launch with `start_paused:=true` → in mission control, nudge the drone with
> **I/J/K/L** to your chosen start, dial wind with **] [** and **T/G**, then press **SPACE**
> to fly. **R** any time to rewind to that start. (The old `frugalnav_teleop.py` still works
> too, but mission control is the one built for the real demo.)

---

## 5. REAL DENSE map (camera perception — big cluttered arena)

Same real pipeline as Section 4, but a **44 m corridor packed with 14 tall obstacles and
20 unique ArUco tiles**. The drone threads the whole field on vision alone, staying frugal
(only a handful of corrections despite many markers in view).

**Terminal 1 — launch (held for mission control):**
```bash
source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.sh
source /mnt/c/Users/parth/Downloads/drone/ros2_ws/install/setup.bash
ros2 launch frugalnav_ros real_demo.launch.py map:=real_dense start_paused:=true gui:=true
```
(drop `start_paused:=true` to fly immediately; drop `gui:=true` to watch in RViz only.)

**Terminal 2 — mission control (same keys as Section 4):**
```bash
source /opt/ros/humble/setup.bash
source /mnt/c/Users/parth/Downloads/drone/ros2_ws/install/setup.bash
ros2 run frugalnav_ros frugalnav_mission_control.py
```

**Terminal 3 — drone feed:**
```bash
source /opt/ros/humble/setup.bash
source /mnt/c/Users/parth/Downloads/drone/ros2_ws/install/setup.bash
ros2 run rqt_image_view rqt_image_view /frugalnav/down_cam/annotated
```

### What you should see in the real demos

- **RViz**: green true path, cyan estimate, yellow fix corrections, cyan wind arrow, and a
  HUD reading `REAL VISION NAV  mode=…  U=…  fixes=…  est-err=…  wind_est=(…, …)`.
- **Camera window**: markers outlined as the drone passes over them, `FIX … err=…` ticking.
- **Terminal 1 log**: `FIX from N marker(s) … ERROR=… m` and `pos=(…) fixes=… wind_est=(…)`.

---

## 6. DEMO 3 — full realism ("knows nothing")

Same maps (`real`, `real_dense`) but the navigator is handed **nothing**: real optical-flow
VIO for motion, real ArUco fixes for position, a **real 360° laser** for obstacles (no map),
wind inferred. Adds a **forward camera** you can watch.

**Terminal 1 — launch (held for mission control):**
```bash
source /opt/ros/humble/setup.bash
source /usr/share/gazebo/setup.sh
source /mnt/c/Users/parth/Downloads/drone/ros2_ws/install/setup.bash
ros2 launch frugalnav_ros real3_demo.launch.py map:=real start_paused:=true gui:=true
```
(use `map:=real_dense` for the cluttered arena; drop `start_paused:=true` to fly immediately.)

**Terminal 2 — mission control (same keys as Section 4):**
```bash
source /opt/ros/humble/setup.bash
source /mnt/c/Users/parth/Downloads/drone/ros2_ws/install/setup.bash
ros2 run frugalnav_ros frugalnav_mission_control.py
```

**Terminal 3 — downward feed (AVL + VIO):**
```bash
source /opt/ros/humble/setup.bash
source /mnt/c/Users/parth/Downloads/drone/ros2_ws/install/setup.bash
ros2 run rqt_image_view rqt_image_view /frugalnav/down_cam/annotated
```

**Terminal 4 — FRONT camera (obstacle view):**
```bash
source /opt/ros/humble/setup.bash
source /mnt/c/Users/parth/Downloads/drone/ros2_ws/install/setup.bash
ros2 run rqt_image_view rqt_image_view /frugalnav/front_cam/annotated
```

> Why a laser (not camera vision) for obstacles: reading distances is a few comparisons —
> RISC-V cheap. Camera-based obstacle detection is the CPU-heavy thing we avoid. The front
> camera is just for you to watch; it's not in the navigator's loop.

A full illustrated write-up is in **`docs/FrugalNav_Guide.pdf`**.

## Troubleshooting

**A window shows only as a taskbar icon (WSLg quirk).**
1. Click the taskbar icon — often it just needs a click to paint.
2. Still hidden? It may be off-screen: hover the taskbar icon → right-click the thumbnail →
   **Move** → press an arrow key, move the mouse, and the window snaps to the cursor.
3. Nuclear option — in **PowerShell**: `wsl --shutdown`, wait ~8 s, relaunch. For the real
   demos you don't need the Gazebo window at all; RViz + the camera feed tell the whole story.

**RViz opens but the arena/obstacles don't draw ("map did not load").** Fixed — the scene is
now published latched (transient-local). If you ever see it again, make sure you rebuilt:
```bash
source /opt/ros/humble/setup.bash
cd /mnt/c/Users/parth/Downloads/drone/ros2_ws && colcon build --symlink-install --packages-select frugalnav_ros
```

**The drone jerks / fights itself.** Two launches are running. Ctrl+C both, run the
cleanup line from Section 0, and start exactly one.

**Regenerate the real maps** (after editing the generators):
```bash
py "C:\Users\parth\Downloads\drone\ros2_ws\src\frugalnav_ros\tools\gen_aruco_media.py"
py "C:\Users\parth\Downloads\drone\ros2_ws\src\frugalnav_ros\tools\gen_maps.py" real
```
then rebuild (see above). Passing `real` rebuilds only `real` + `real_dense`; the sandbox
`demo`/`canopy` worlds are left untouched.
