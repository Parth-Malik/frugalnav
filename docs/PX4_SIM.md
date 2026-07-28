# FrugalNav on PX4 SITL — the complete-simulation / deployment path

This is the honest account of running FrugalNav against a **real PX4 autopilot** in
software-in-the-loop (SITL). It is the same navigator that flies Demo 3 in Gazebo — the
only thing that changes is the *platform adapter* underneath it, which is the whole point
of the seam.

## What actually runs

```
PX4 SITL (flight stack + Gazebo vehicle + EKF2)
        │  uXRCE-DDS  (/fmu/… topics)
   Micro-XRCE-DDS Agent  ⇄  ROS 2
        │
   frugalnav_px4_bridge   /fmu/out/vehicle_local_position → /frugalnav/truth + /frugalnav/vio
        │
   frugalnav_dwa_node (platform:=px4)
        │  Px4Platform: streams OffboardControlMode + TrajectorySetpoint,
        │  commands OFFBOARD + ARM, sends scheduler-gated fixes as vehicle_visual_odometry
        ▼
   PX4 flies the offboard velocity setpoints → the DWA navigator flies the vehicle
```

The DWA navigator, the controller, the fusion and **the uncertainty scheduler are byte-for-byte
the same** as the Gazebo demo. `platform:=px4` swaps `SimPlatform` (Gazebo planar_move +
teleport) for `Px4Platform` (PX4 offboard). That is the deployability claim made concrete.

## One-command bringup

```bash
bash ros2_ws/src/frugalnav_ros/tools/px4_sitl_bringup.sh
```

First run clones and builds **PX4-Autopilot v1.14** and the **Micro-XRCE-DDS Agent** under
`~/frugalnav_px4` (heavy: 15–30 min, needs network + ~3 GB). Re-runs skip straight to
launching. `HEADLESS=1 bash …` runs PX4 without the Gazebo GUI.

It then starts, in order: the XRCE agent (`udp4 -p 8888`), PX4 SITL (`gz_x500` quad), and
the FrugalNav `px4_offboard.launch.py`. You should see the drone **arm, take off to 5 m,
and fly to the target** — the navigator commanding it the whole way.

### Doing it by hand (three terminals)

```bash
# 1 — agent
MicroXRCEAgent udp4 -p 8888
# 2 — PX4 SITL
cd ~/frugalnav_px4/PX4-Autopilot && make px4_sitl gz_x500
# 3 — FrugalNav (needs px4_msgs built in the workspace, see below)
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
ros2 launch frugalnav_ros px4_offboard.launch.py
```

## The one build dependency: `px4_msgs`

The PX4 adapter and bridge import `px4_msgs`. Build it once into the workspace (the release
that matches PX4 v1.14):

```bash
cd ros2_ws/src && git clone --depth 1 -b release/1.14 https://github.com/PX4/px4_msgs.git
cd .. && source /opt/ros/humble/setup.bash && colcon build --packages-select px4_msgs frugalnav_ros
```

`frugalnav_ros` builds and runs **without** `px4_msgs` — the import is lazy, inside
`Px4Platform`, so the Gazebo demos are unaffected. It is only needed for `platform:=px4`.

## Frames (the easy thing to get wrong)

PX4 is **NED** (x=north, y=east, z=down); our world is **ENU-style** (x=east, y=north, up).
The adapter and bridge convert consistently: world `(x,y)` ⇄ NED `(north=y, east=x)`, and
altitude `z` ⇄ `down=-z`. A world **+x** command drives the vehicle **east** in both sims.

## What is genuinely done vs. what remains

**Done — the control + deployment path:**
- `Px4Platform` is a complete offboard client: setpoint stream, **OFFBOARD engage + ARM**
  startup, altitude gap-closing, and a `send_vision()` that publishes each scheduler-gated
  fix as `VehicleOdometry` on `/fmu/in/vehicle_visual_odometry` (external vision into EKF2).
- The bridge + launch let the **unmodified** DWA navigator fly the PX4 vehicle to a target.
- One-command bringup that installs and wires the whole stack.

**Remaining — the perception layer in the PX4 world:**
The Gazebo Demo-3 perception (downward camera → ArUco, the 360° laser) lives in *our*
Gazebo Classic world and drone model. PX4 SITL spawns *its* `x500` model in *its* `gz`
world, which has no ArUco tiles and no laser. So against bare SITL the navigator flies on
PX4's EKF2 state (accurate in SITL) rather than our vision — it proves the control and
offboard path, not the vision pipeline. To make the vision loop run in PX4 SITL you would:
1. add a downward camera + a 2-D lidar to the PX4 airframe's `gz` model,
2. drop the ArUco tile models (and a marker→world map) into the PX4 `gz` world,
3. run `frugalnav_perception.py` and `frugalnav_vio.py` against those camera/scan topics,
   and feed the fixes through `send_vision()` (already wired).
That is a world-and-model porting job, not a navigator change — the seam holds.

**Honest status:** simulation-validated control/offboard integration with a real PX4 flight
stack; the vision-in-SITL step above is scoped but not yet built.
