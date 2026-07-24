#!/usr/bin/env python3
"""
Demo 2 navigator. Flies on camera-derived signals:

    /frugalnav/fix    absolute position from real ArUco detection
    /frugalnav/cues   [blur, features, n_markers] measured from the image
    /frugalnav/truth  corrupted into a drifting VIO velocity (see step())

Wind is not given: the drone compares the velocity it commanded to the velocity it
achieved and feeds the difference forward. Corrections are frugal -- even with a
marker in view, a fix is only spent when the scheduler's U says so.

Demo 3 (frugalnav_real3_node.py) replaces the simulated VIO and the obstacle map
with a real optical-flow front-end and a laser.
"""
import os
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped
from std_msgs.msg import Float32MultiArray, String
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from gazebo_msgs.srv import SetEntityState                     # teleport the drone (reset / reposition)

sys.path.insert(0, '/mnt/c/Users/parth/Downloads/drone')      # the FrugalNav Python core
from core.uncertainty_scheduler import UncertaintyScheduler, SchedulerConfig
from core.state_fusion import StateFusion
from core.controller import TargetCentricController, ControllerConfig
from core.obstacle_avoidance import ObstacleAvoidance, AvoidanceConfig, ttc_from_range
from core.types import LandmarkFix


class RealNav(Node):
    def __init__(self):
        super().__init__('frugalnav_real_node')
        scene = self.declare_parameter('scene_file', '').value
        self.B, self.start, self.obst = np.array([0., 0.]), np.array([0., 0.]), []
        self.load_scene(scene)

        self.ctrl = TargetCentricController(self.B, ControllerConfig(kp=0.6, v_max=2.2, arrive_tol=1.5))
        # q grows the fused covariance realistically fast, so sigma_pos reaches the
        # floor and the scheduler re-corrects along the route (else it drifts blind).
        self.fusion = StateFusion(init_xy=tuple(self.start)); self.fusion.q_per_metre = 0.09
        self.sched = UncertaintyScheduler(SchedulerConfig(tau=0.45, sigma_pos_floor=0.7))
        self.avoid = ObstacleAvoidance(AvoidanceConfig(ttc_trigger=3.0, ttc_release=4.5, gain=5.0))

        # VIO sim (a slow velocity bias makes the dead-reckon drift -> needs fixes)
        self.rng = np.random.default_rng(0)
        self.vbias = np.zeros(2)
        self.true = None; self.prev_true = None
        self.fix = None; self.fix_t = -1.0; self.fix_used_t = -1.0
        self.blur = 300.0; self.feats = 60.0; self.prev_feats = 60.0
        self.wind_est = np.zeros(2); self.v_cmd = np.zeros(2)
        self.teleop_vel = np.zeros(2)                # manual WASD velocity (world XY)
        self.mode = 'auto'
        self.paused = bool(self.declare_parameter('start_paused', False).value)
        self.home = self.start.copy()                # where RESET returns / repositioning base
        self.spawn_z = 5.0
        self.true_path, self.est_path, self.corr = [], [], []
        self.fixes = 0; self.arrived = False
        # service to teleport the Gazebo drone (real RESET + reposition-before-start)
        self.tele_cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')

        self.cmd_pub = self.create_publisher(Twist, '/frugalnav/nav_cmd', 10)
        self.viz_pub = self.create_publisher(MarkerArray, '/frugalnav/viz', 10)
        # LATCHED (transient-local): the scene is static and published once, so a
        # late-joining RViz still receives it; also matches RViz's transient-local
        # request (a volatile publisher here -> "incompatible QoS" -> map never draws).
        latched = rclpy.qos.QoSProfile(
            depth=1, durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.scene_pub = self.create_publisher(MarkerArray, '/frugalnav/scene', latched)
        self.create_subscription(Odometry, '/frugalnav/truth', self.on_truth, 20)
        self.create_subscription(PointStamped, '/frugalnav/fix', self.on_fix, 10)
        self.create_subscription(Float32MultiArray, '/frugalnav/cues', self.on_cues, 10)
        self.create_subscription(String, '/frugalnav/ctrl', self.on_ctrl, 10)
        self.create_subscription(Twist, '/frugalnav/teleop', self.on_teleop, 10)
        self.publish_scene()
        self.timer = self.create_timer(0.05, self.step)     # 20 Hz
        self.get_logger().info(f'blind nav up: target {self.B}, {len(self.obst)} obstacles')

    def load_scene(self, p):
        if not (p and os.path.exists(p)):
            return
        for line in open(p):
            t = line.split()
            if not t:
                continue
            if t[0] == 'target': self.B = np.array([float(t[1]), float(t[2])])
            elif t[0] == 'start': self.start = np.array([float(t[1]), float(t[2])])
            elif t[0] == 'pillar': self.obst.append((float(t[1]), float(t[2]), float(t[3])))

    def on_truth(self, m): self.true = np.array([m.pose.pose.position.x, m.pose.pose.position.y])
    def on_fix(self, m):
        self.fix = np.array([m.point.x, m.point.y]); self.fix_t = self.now()
    def on_cues(self, m):
        if len(m.data) >= 2: self.blur, self.feats = float(m.data[0]), float(m.data[1])
    def on_ctrl(self, m):
        c = m.data
        if c in ('auto', 'manual'): self.mode = c; self.arrived = False
        elif c == 'pause': self.paused = True
        elif c in ('play', 'resume'): self.paused = False
        elif c == 'reset': self.reset_home()
        elif c == 'move_north': self.reposition([0.0, 1.0])
        elif c == 'move_south': self.reposition([0.0, -1.0])
        elif c == 'move_east': self.reposition([1.0, 0.0])
        elif c == 'move_west': self.reposition([-1.0, 0.0])
    def on_teleop(self, m): self.teleop_vel = np.array([m.linear.x, m.linear.y])
    def now(self): return self.get_clock().now().nanoseconds * 1e-9

    def teleport(self, xy):
        # move the real Gazebo drone to xy (keeps its flight altitude)
        if not self.tele_cli.wait_for_service(timeout_sec=0.3):
            self.get_logger().warn('set_entity_state not ready; teleport skipped'); return
        req = SetEntityState.Request()
        req.state.name = 'frugalnav_drone'
        req.state.pose.position.x = float(xy[0])
        req.state.pose.position.y = float(xy[1])
        req.state.pose.position.z = float(self.spawn_z)
        req.state.pose.orientation.w = 1.0
        req.state.reference_frame = 'world'
        self.tele_cli.call_async(req)

    def reset_home(self):
        # RESET: teleport the drone home, resync the estimate, clear trails, and HOLD
        self.teleport(self.home)
        self.fusion.state.xy = self.home.copy(); self.prev_true = None
        self.true_path.clear(); self.est_path.clear(); self.corr.clear()
        self.fixes = 0; self.arrived = False; self.paused = True
        self.get_logger().info(f'RESET -> home {self.home}, holding')

    def reposition(self, delta):
        # move the start/home position (and the drone) before launching the mission
        self.home = self.home + np.array(delta, float)
        self.teleport(self.home)
        self.fusion.state.xy = self.home.copy(); self.prev_true = None
        self.get_logger().info(f'home -> {self.home}')

    def step(self):
        if self.true is None or self.paused:
            self.pub_cmd(0, 0); return
        if self.prev_true is None:
            self.prev_true = self.true.copy(); return
        dt = 0.05

        # --- simulated VIO: measured relative motion of the drone (drifts) ---
        actual_vel = (self.true - self.prev_true) / dt
        self.vbias += self.rng.normal(0, 0.002, 2)
        vio_vel = actual_vel + self.vbias + self.rng.normal(0, 0.02, 2)

        # --- WIND ESTIMATE: what we commanded vs what we actually did ---
        # actual_vel = nav_cmd + wind, so (actual_vel - nav_cmd) = wind. Low-pass it.
        self.wind_est = 0.92 * self.wind_est + 0.08 * (actual_vel - self.v_cmd)

        # --- dead-reckon the fused estimate on VIO motion ---
        self.fusion.predict(vio_vel * dt)
        est = self.fusion.state.xy

        # --- scheduler on MEASURED cues ---
        floss = max(0.0, self.prev_feats - self.feats); self.prev_feats = self.feats
        cues = dict(sigma_pos=self.fusion.sigma_pos(), sigma_head=0.01,
                    blur=self.blur, feature_loss=floss, imu_bias=0.02,
                    active_features=self.feats)
        U, trig, reason, _ = self.sched.compute(cues)

        # --- FRUGAL correction: only when U fires AND a fresh camera fix exists ---
        fix_fresh = (self.fix is not None and self.now() - self.fix_t < 0.5
                     and self.fix_t > self.fix_used_t)
        if trig and fix_fresh:
            self.fusion.update(LandmarkFix(xy=self.fix, yaw=0.0,
                               covariance=np.eye(2) * 0.25, marker_id=0), gain=None)
            self.sched.reset_after_fix(); self.vbias *= 0.2
            self.fix_used_t = self.fix_t; self.fixes += 1; self.corr.append(self.fusion.state.xy.copy())
        est = self.fusion.state.xy

        # --- control (target-centric + obstacle avoidance), then WIND FEEDFORWARD ---
        if self.mode == 'manual':
            # fly by WASD; keep obstacle repulsion so manual flight can't crash a pillar
            v_ctrl = self.add_repulsion(est, self.teleop_vel.copy())
        elif self.mode == 'auto' and self.ctrl.arrived(est):
            self.arrived = True; v_ctrl = np.zeros(2)
        else:
            seek = self.B - est; n = np.linalg.norm(seek) or 1.0; seek_hat = seek / n
            ttc, bearing, near = self.nearest_obstacle(seek_hat)
            evade = self.avoid.update(seek_hat, ttc, bearing)
            v_ctrl = self.apf(est, seek_hat) + evade
            s = np.linalg.norm(v_ctrl)
            if s > self.ctrl.cfg.v_max * 1.8:
                v_ctrl = v_ctrl / s * self.ctrl.cfg.v_max * 1.8
        self.v_cmd = v_ctrl - self.wind_est                 # feedforward cancels wind
        self.pub_cmd(self.v_cmd[0], self.v_cmd[1])

        self.true_path.append(self.true.copy()); self.est_path.append(est.copy())
        self.last = dict(U=U, est=est)
        self.publish_viz(U)
        self._logc = getattr(self, '_logc', 0) + 1
        if self._logc % 40 == 0:
            terr = float(np.linalg.norm(est - self.true))
            self.get_logger().info(
                f'pos=({self.true[0]:.1f},{self.true[1]:.1f}) U={U:.2f} fixes={self.fixes} '
                f'est_err={terr:.2f}m wind_est=({self.wind_est[0]:.2f},{self.wind_est[1]:.2f})')
        self.prev_true = self.true.copy()

    def nearest_obstacle(self, seek_hat):
        best, ttc, bearing = 1e9, np.inf, 0.0
        for (ox, oy, r) in self.obst:
            to = np.array([ox, oy]) - self.true; d = np.linalg.norm(to); ds = d - r
            if ds > 6 or d < 1e-6: continue
            toh = to / d; along = float(seek_hat @ toh)
            if along < -0.2 or ds > best: continue
            best = ds; closing = max(along, 0.35) * self.ctrl.cfg.v_max
            ttc = ttc_from_range(ds, closing)
            bearing = float(np.arctan2(seek_hat[0]*toh[1]-seek_hat[1]*toh[0], along))
        return ttc, bearing, best < 1e9

    def apf(self, pos, seek_hat):
        return self.add_repulsion(pos, seek_hat * self.ctrl.cfg.v_max)

    def add_repulsion(self, pos, v):
        # push the velocity away from every nearby obstacle (potential field). Shared
        # by AUTO seeking and MANUAL flying so neither can drive into a pillar.
        for (ox, oy, r) in self.obst:
            to = pos - np.array([ox, oy]); d = np.linalg.norm(to); ds = d - r
            if ds >= 3.6: continue
            away = to / max(d, 1e-6)
            v = v + away * min(6.0, 4.5 * (1.0 / max(ds, 0.2) - 1.0 / (r + 3.6)))
            into = -float(v @ away)
            if into > 0: v = v + away * into
        return v

    def pub_cmd(self, x, y):
        t = Twist(); t.linear.x = float(x); t.linear.y = float(y); self.cmd_pub.publish(t)

    def mk(self, i, typ, s, r, g, b, a):
        m = Marker(); m.header.frame_id = 'world'; m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'nav'; m.id = i; m.type = typ; m.action = 0
        m.scale.x = m.scale.y = m.scale.z = s
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a; m.pose.orientation.w = 1.0
        return m

    def line(self, i, pts, r, g, b):
        m = self.mk(i, Marker.LINE_STRIP, 0.15, r, g, b, 0.95); m.scale.x = 0.15
        for p in pts:
            from geometry_msgs.msg import Point
            q = Point(); q.x, q.y, q.z = float(p[0]), float(p[1]), 0.1; m.points.append(q)
        return m

    def publish_scene(self):
        a = MarkerArray()
        for j, (ox, oy, r) in enumerate(self.obst):
            c = self.mk(30 + j, Marker.CYLINDER, 1.0, 0.4, 0.4, 0.45, 0.8)
            c.scale.x = c.scale.y = 2 * r; c.scale.z = 12.0
            c.pose.position.x, c.pose.position.y, c.pose.position.z = ox, oy, 6.0; a.markers.append(c)
        tgt = self.mk(2, Marker.CYLINDER, 1.2, 0.98, 0.75, 0.14, 0.95); tgt.scale.z = 3.0
        tgt.pose.position.x, tgt.pose.position.y, tgt.pose.position.z = float(self.B[0]), float(self.B[1]), 1.5
        a.markers.append(tgt); self.scene_pub.publish(a)

    def publish_viz(self, U):
        from geometry_msgs.msg import Point
        a = MarkerArray()
        a.markers.append(self.line(200, self.true_path, 0.2, 0.83, 0.44))
        a.markers.append(self.line(201, self.est_path, 0.22, 0.74, 0.97))
        fx = self.mk(203, Marker.SPHERE_LIST, 0.9, 0.99, 0.85, 0.14, 1.0)
        for c in self.corr:
            q = Point(); q.x, q.y, q.z = float(c[0]), float(c[1]), 0.2; fx.points.append(q)
        a.markers.append(fx)
        # estimated wind arrow (cyan) at the drone
        wm = self.mk(206, Marker.ARROW, 0.3, 0.4, 0.9, 1.0, 0.9); wm.scale.x = 0.3; wm.scale.y = 0.6
        p0 = Point(); p0.x, p0.y, p0.z = float(self.true[0]), float(self.true[1]), 3.0
        p1 = Point(); p1.x = float(self.true[0] + self.wind_est[0]*2); p1.y = float(self.true[1] + self.wind_est[1]*2); p1.z = 3.0
        wm.points = [p0, p1]; a.markers.append(wm)
        tx = self.mk(204, Marker.TEXT_VIEW_FACING, 2.0, 0.92, 0.95, 0.98, 1.0)
        tx.pose.position.x, tx.pose.position.y, tx.pose.position.z = float(self.start[0]), float(self.start[1]), 6.0
        terr = np.linalg.norm(self.last['est'] - self.true) if self.true is not None else 0.0
        tx.text = (f"REAL VISION NAV  mode={self.mode}\nU={U:.2f} fixes={self.fixes} "
                   f"est-err={terr:.2f}m\nwind_est=({self.wind_est[0]:.2f},{self.wind_est[1]:.2f}) m/s")
        a.markers.append(tx); self.viz_pub.publish(a)


def main():
    rclpy.init(); rclpy.spin(RealNav()); rclpy.shutdown()


if __name__ == '__main__':
    main()
