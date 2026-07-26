# Running FrugalNav on a real drone

This explains, honestly, what it takes to fly this on real hardware — what already
works unchanged, the one part you swap, and what still needs doing. It does **not**
claim the code is flight-tested. It is simulation-validated and structured so that
porting is a driver-and-tuning job rather than a rewrite.

## The idea that makes it portable

Every navigation node consumes **standard ROS 2 sensor messages**, not anything
Gazebo-specific. On a real drone you launch real sensor drivers that publish the same
message types on the same topics, and the nodes do not change:

| Topic | Type | In simulation | On a real drone |
|-------|------|---------------|-----------------|
| `/frugalnav/down_cam/image_raw` (+ `camera_info`) | `sensor_msgs/Image`, `CameraInfo` | Gazebo camera | a real downward camera driver (`usb_cam`, `gscam`, …) |
| `/frugalnav/imu` | `sensor_msgs/Imu` | Gazebo IMU | the flight controller's IMU, or a standalone IMU |
| `/frugalnav/height` | `sensor_msgs/Range` | Gazebo ray | a downward ToF / lidar-lite (VL53L1X, TFmini, …) |
| `/frugalnav/scan` | `sensor_msgs/LaserScan` | Gazebo ray | a 2-D lidar (RPLIDAR A1, …) |

Because the perception, VIO and navigator nodes read only these, **they run on real
hardware unmodified.** The camera pipeline already handles lens distortion (the
coefficients flow from `camera_info` into `solvePnP`), the VIO de-rotates the flow with
the gyro, and the altitude scale comes from the rangefinder.

## What already works unchanged

- **`frugalnav_perception.py`** — real ArUco detection + `solvePnP`. Point it at a real
  camera and calibrate that camera (below). Nothing else changes.
- **`frugalnav_vio.py`** — real optical-flow odometry. Uses the gyro to de-rotate the
  flow and the rangefinder for metric scale. Both are real sensor topics.
- **`frugalnav_real3_node.py`** — scheduler + fusion + laser avoidance + wind estimate.
  Reads `/frugalnav/vio`, `/frugalnav/fix`, `/frugalnav/cues`, `/frugalnav/scan`,
  `/frugalnav/imu`. All hardware-portable.
- **`core/`** — no ROS, no OS, no allocation in the hot loop; the C++ port in `cpp/` is
  the same logic for an embedded target.

## The one part you swap: the motion output (already abstracted)

This is the only genuinely sim-specific piece, and it now lives behind a **platform
adapter** so nothing in the navigator changes. The nav (`frugalnav_real3_node.py`) only
ever calls three methods on `self.platform`:

```
set_velocity(vx, vy)   world-frame horizontal velocity setpoint  [m/s]
set_altitude(z)        desired altitude setpoint                  [m]
go_to(x, y, z)         reposition / reset to a pose
```

`scripts/frugalnav_platform.py` implements two adapters:

- **`SimPlatform`** (default, `platform:=sim`) — velocity to `/frugalnav/nav_cmd`
  (planar_move), altitude/pose by teleport. This is the simulation.
- **`Px4Platform`** (`platform:=px4`) — streams the standard PX4 offboard messages
  (`OffboardControlMode` heartbeat + `TrajectorySetpoint`) over uXRCE-DDS, with `go_to`
  flying to a pose instead of teleporting. Needs `px4_msgs` and a running PX4 (≥ v1.14).

To fly it for real you launch with `platform:=px4`, run PX4 + the DDS agent, and handle
arming / the switch into OFFBOARD mode (a `VehicleCommand` startup routine). The
navigation code, the scheduler, the perception and the VIO are all identical between the
two — that is the whole point of the seam.

### A subtlety worth knowing

PX4 has its own state estimator (EKF2). The cleanest integration is **not** to run your
own fusion in parallel but to feed the ArUco fixes in as external vision:

- publish each fix to `/fmu/in/vehicle_visual_odometry`,
- and let the scheduler decide **when** to publish one.

That is arguably a stronger framing of the whole project: *the scheduler gates external
vision updates into a production flight controller's estimator, cutting update rate at
equal accuracy.* It is more work, but it is the industrially-correct architecture.

## What still needs doing before it flies

Be honest about these — none are hidden:

1. **Camera calibration.** Run `camera_calibration` (ROS) with a checkerboard once to get
   the real intrinsics + distortion. This replaces the simulator-fit mounting
   calibration; the code already consumes `camera_info`, so it is a one-time bench step.
2. **Flight dynamics + tuning.** PX4 handles attitude and thrust, but the wind
   feed-forward gain, the scheduler threshold `tau`, `q_per_metre`, and the laser
   repulsion gains were tuned in sim and will need re-tuning on the airframe.
3. **Marker survey.** Every ArUco tile's world position must be measured (tape measure or
   total station) and written into the scene/marker map. Fixes are only as good as this.
4. **Reactive-avoidance caveat.** The laser avoider is a potential field with no global
   plan — it can be trapped by a concave dead-end. For real obstacle fields, add a
   planner (e.g. a local costmap) in front of it.
5. **Flight testing.** Simulation cannot substitute for it. Start with the drone
   tethered / at low altitude over a marker field before any autonomous run.

## What is genuinely done

- The algorithm (scheduler + fusion + control) and its portable C++ core.
- Real perception, real optical-flow VIO with gyro de-rotation and altimeter scale, real
  laser obstacle sensing — all on standard sensor topics.
- A multi-seed, EuRoC-backed evaluation of the frugality claim.
- Everything except the platform adapter and the hardware bring-up above.
