#!/usr/bin/env python3
"""
Platform adapters -- the seam between the (platform-independent) navigator and the thing
that actually moves the vehicle.

The navigator only ever calls three methods:
    set_velocity(vx, vy)   world-frame horizontal velocity setpoint  [m/s]
    set_altitude(z)        desired altitude setpoint                  [m]
    go_to(x, y, z)         reposition / reset to a pose

    * SimPlatform  drives the Gazebo model (planar_move for velocity, set_entity_state
                   for altitude/teleport) -- identical to the old inline behaviour.
    * Px4Platform  drives a real vehicle over PX4 offboard (uXRCE-DDS). It publishes the
                   standard PX4 setpoints; on hardware go_to() flies to a pose instead of
                   teleporting. Selected with launch arg platform:=px4; needs px4_msgs.

Swapping the two changes nothing in the navigator, which is the point: the sim is one
deployment target and a real PX4 drone is another.
"""
import numpy as np


def make(node, name='sim'):
    return Px4Platform(node) if name == 'px4' else SimPlatform(node)


# A fourth method, send_vision(xy, yaw, cov), lets the navigator hand a scheduler-gated
# landmark fix to the platform. On sim it is a no-op (the sim already knows truth); on PX4
# it publishes the fix as external vision into EKF2 -- the "scheduler gates external vision
# into a production estimator" architecture. Every adapter therefore defines all four.


class SimPlatform:
    """Gazebo target: velocity -> /frugalnav/nav_cmd (planar_move), altitude/pose -> teleport."""

    def __init__(self, node):
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from gazebo_msgs.srv import SetEntityState
        self.node = node
        self._Twist = Twist
        self._Req = SetEntityState.Request
        self._cur = None
        self.cmd_pub = node.create_publisher(Twist, '/frugalnav/nav_cmd', 10)
        self.tele = node.create_client(SetEntityState, '/gazebo/set_entity_state')
        node.create_subscription(Odometry, '/frugalnav/truth', self._on_truth, 10)

    def _on_truth(self, m):
        self._cur = (m.pose.pose.position.x, m.pose.pose.position.y)

    def set_velocity(self, vx, vy):
        t = self._Twist(); t.linear.x = float(vx); t.linear.y = float(vy)
        self.cmd_pub.publish(t)

    def set_altitude(self, z):
        # planar_move can't drive z, so the sim applies altitude by moving the model.
        if self._cur is not None and self.tele.service_is_ready():
            self._teleport(self._cur[0], self._cur[1], z, block=False)

    def go_to(self, x, y, z):
        self._teleport(x, y, z, block=True)

    def send_vision(self, xy, yaw=0.0, cov=0.25):
        pass                    # the sim already owns truth; nothing to feed

    def _teleport(self, x, y, z, block):
        if block and not self.tele.wait_for_service(timeout_sec=0.3):
            self.node.get_logger().warn('set_entity_state not ready; teleport skipped'); return
        req = self._Req()
        req.state.name = 'frugalnav_drone'
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = float(z)
        req.state.pose.orientation.w = 1.0
        req.state.reference_frame = 'world'
        self.tele.call_async(req)


class Px4Platform:
    """Real/SITL PX4 offboard target. Needs px4_msgs and a PX4 autopilot on uXRCE-DDS.

    A complete offboard bringup, not just a setpoint stream:
      1. stream OffboardControlMode (>2 Hz, mandatory) + a TrajectorySetpoint,
      2. once a second of setpoints has been streamed, command OFFBOARD mode and ARM
         (PX4 rejects OFFBOARD unless setpoints are already flowing),
      3. keep streaming; set_velocity/set_altitude/go_to update the live setpoint.
    Frames: PX4 is NED. World-ENU altitude z -> setpoint z = -z; ENU (vx,vy) map to NED
    (vy, vx) [east,north] so a world +x command drives PX4 north-east consistently with
    the sim. send_vision() feeds a scheduler-gated fix to EKF2 as external vision.
    """

    def __init__(self, node):
        from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint, VehicleCommand,
                                  VehicleLocalPosition, VehicleStatus, VehicleOdometry)
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.node = node
        self._Off, self._Sp, self._Cmd, self._Odo = (OffboardControlMode, TrajectorySetpoint,
                                                      VehicleCommand, VehicleOdometry)
        self.vx = self.vy = 0.0
        self.target_z = -5.0            # NED down-negative: 5 m altitude
        self.cur_z = -5.0
        self.goal = None               # (n, e, d) position setpoint for go_to
        self.ticks = 0; self.engaged = False; self.nav_state = None
        self.off_pub = node.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.sp_pub = node.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.cmd_pub = node.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)
        self.vis_pub = node.create_publisher(VehicleOdometry, '/fmu/in/vehicle_visual_odometry', qos)
        node.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self._on_pos, qos)
        node.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self._on_status, qos)
        node.create_timer(0.05, self._stream)   # 20 Hz offboard stream
        node.get_logger().info('Px4Platform: streaming offboard setpoints; will arm + engage OFFBOARD')

    def _on_pos(self, m): self.cur_z = m.z
    def _on_status(self, m): self.nav_state = m.nav_state
    def _stamp(self): return int(self.node.get_clock().now().nanoseconds / 1000)

    # -- navigator-facing interface (world ENU) ----------------------------
    def set_velocity(self, vx, vy):
        self.vx, self.vy, self.goal = float(vx), float(vy), None

    def set_altitude(self, z):
        self.target_z = -float(z)

    def go_to(self, x, y, z):
        self.goal = (float(y), float(x), -float(z))   # ENU (x,y,z) -> NED (north=y, east=x, down=-z)

    def send_vision(self, xy, yaw=0.0, cov=0.25):
        """Publish a scheduler-gated landmark fix as external vision for EKF2 (NED)."""
        o = self._Odo(); o.timestamp = o.timestamp_sample = self._stamp()
        o.pose_frame = self._Odo.POSE_FRAME_NED
        o.position = [float(xy[1]), float(xy[0]), self.cur_z]      # ENU->NED north,east,down
        o.q = [float('nan')] * 4
        o.position_variance = [float(cov)] * 3
        o.velocity_variance = [float('nan')] * 3
        self.vis_pub.publish(o)

    # -- offboard bringup + setpoint stream --------------------------------
    def _cmd(self, command, p1=0.0, p2=0.0):
        c = self._Cmd(); c.timestamp = self._stamp()
        c.command = command; c.param1 = float(p1); c.param2 = float(p2)
        c.target_system = 1; c.target_component = 1
        c.source_system = 1; c.source_component = 1; c.from_external = True
        self.cmd_pub.publish(c)

    def _stream(self):
        self.ticks += 1
        off = self._Off(); off.position = self.goal is not None; off.velocity = self.goal is None
        off.timestamp = self._stamp(); self.off_pub.publish(off)
        sp = self._Sp(); sp.timestamp = off.timestamp
        nan = float('nan')
        if self.goal is not None:
            sp.position = [self.goal[0], self.goal[1], self.goal[2]]
            sp.velocity = [nan, nan, nan]
        else:
            vz = float(np.clip(0.6 * (self.target_z - self.cur_z), -1.0, 1.0))   # close altitude gap
            sp.position = [nan, nan, nan]
            sp.velocity = [self.vy, self.vx, vz]        # NED: north=vy(ENU), east=vx(ENU)
        self.sp_pub.publish(sp)
        # after ~1 s of streamed setpoints, engage OFFBOARD (custom main mode 6) and arm
        if not self.engaged and self.ticks == 20:
            self._cmd(self._Cmd.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self._cmd(self._Cmd.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            self.engaged = True
            self.node.get_logger().info('Px4Platform: sent OFFBOARD + ARM')
