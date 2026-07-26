#!/usr/bin/env python3
"""
Demo 3 navigator. Flies on measured signals only:

    /frugalnav/vio    velocity from optical flow
    /frugalnav/fix    absolute ArUco fixes, spent frugally by the scheduler
    /frugalnav/cues   blur / feature count, feeds the scheduler
    /frugalnav/scan   laser ranges for obstacles (there is no obstacle map)

Wind is inferred from (measured velocity - commanded velocity). Outputs
/frugalnav/nav_cmd. Ground truth is only used to draw the error and to teleport on
reset, never for a control decision.
"""
import os
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped
from std_msgs.msg import Float32MultiArray, String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Imu
from visualization_msgs.msg import Marker, MarkerArray
from gazebo_msgs.srv import SetEntityState

def _repo_root():
    """Locate the repo so `core/` is importable, from either a symlink or copy install."""
    env = os.environ.get('FRUGALNAV_ROOT')
    if env and os.path.isdir(os.path.join(env, 'core')):
        return env
    for start in (os.path.realpath(__file__), os.path.abspath(__file__)):
        d = os.path.dirname(start)
        for _ in range(8):
            if os.path.isdir(os.path.join(d, 'core')):
                return d
            d = os.path.dirname(d)
    raise RuntimeError('cannot find the FrugalNav core; set FRUGALNAV_ROOT to the repo root')


sys.path.insert(0, _repo_root())
from core.uncertainty_scheduler import UncertaintyScheduler, SchedulerConfig
from core.state_fusion import StateFusion
from core.controller import TargetCentricController, ControllerConfig
from core.types import LandmarkFix

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))   # sibling module
import frugalnav_platform


class Real3Nav(Node):
    def __init__(self):
        super().__init__('frugalnav_real3_node')
        scene = self.declare_parameter('scene_file', '').value
        self.B, self.start = np.array([0., 0.]), np.array([0., 0.])
        # obstacle map: RViz REFERENCE ONLY, never used to navigate
        self.ref_obst = []            # (x, y, r)            round pillars
        self.ref_boxes = []           # (x, y, sx, sy, h)    city buildings
        self.load_scene(scene)

        self.ctrl = TargetCentricController(self.B, ControllerConfig(kp=0.6, v_max=2.0, arrive_tol=1.5))
        # q grows the covariance faster (real optical-flow VIO drifts faster than 0.09/m
        # under load) and tau triggers sooner, so the FIRST fix lands before the estimate
        # has drifted metres -- otherwise the drone navigates on a stale position into a
        # block. This only tunes the live demo; the frugality evaluation uses the core
        # scheduler's own defaults, so the headline result is unchanged.
        self.fusion = StateFusion(init_xy=tuple(self.start)); self.fusion.q_per_metre = 0.14
        self.sched = UncertaintyScheduler(SchedulerConfig(tau=0.28, sigma_pos_floor=0.6))

        self.true = None                         # truth: error/teleport reference ONLY
        self.vio_vel = np.zeros(2)               # measured velocity (optical-flow VIO)
        self.vio_t = -1.0                        # last VIO stamp (VIO only publishes once calibrated)
        self.fix = None; self.fix_t = -1.0; self.fix_used_t = -1.0
        self.blur = 300.0; self.feats = 60.0; self.prev_feats = 60.0
        self.scan = None
        self.gyro_mag = 0.0                       # smoothed |angular velocity| from the IMU
        self.wind_est = np.zeros(2); self.v_cmd = np.zeros(2)
        self.teleop_vel = np.zeros(2)
        self.mode = 'auto'
        self.paused = bool(self.declare_parameter('start_paused', False).value)
        self.home = self.start.copy(); self.spawn_z = 5.0
        # altitude: cruise high, drop under a haze bank when the image degrades
        self.cruise_z, self.low_z = 5.0, 2.6
        self.alt = 5.0
        self.poor_vis = 0
        self.true_path, self.est_path, self.corr = [], [], []
        self.fixes = 0; self.arrived = False; self.nearest = np.inf
        self.arrival_confirmed = False

        # platform adapter: 'sim' drives Gazebo (default), 'px4' drives a real drone.
        # The navigation code below is identical for both.
        self.platform = frugalnav_platform.make(self, self.declare_parameter('platform', 'sim').value)
        self.viz_pub = self.create_publisher(MarkerArray, '/frugalnav/viz', 10)
        latched = rclpy.qos.QoSProfile(
            depth=1, durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.scene_pub = self.create_publisher(MarkerArray, '/frugalnav/scene', latched)
        self.create_subscription(Odometry, '/frugalnav/truth', self.on_truth, 20)
        self.create_subscription(Twist, '/frugalnav/vio', self.on_vio, 20)
        self.create_subscription(PointStamped, '/frugalnav/fix', self.on_fix, 10)
        self.create_subscription(Float32MultiArray, '/frugalnav/cues', self.on_cues, 10)
        self.create_subscription(LaserScan, '/frugalnav/scan', self.on_scan, 10)
        self.create_subscription(Imu, '/frugalnav/imu', self.on_imu, 20)
        self.create_subscription(String, '/frugalnav/ctrl', self.on_ctrl, 10)
        self.create_subscription(Twist, '/frugalnav/teleop', self.on_teleop, 10)
        self.publish_scene()
        self.timer = self.create_timer(0.05, self.step)      # 20 Hz
        self.get_logger().info(f'demo-3 blind nav up: target {self.B} (VIO + laser, no map)')

    def load_scene(self, p):
        if not (p and os.path.exists(p)):
            return
        for line in open(p):
            t = line.split()
            if not t:
                continue
            if t[0] == 'target': self.B = np.array([float(t[1]), float(t[2])])
            elif t[0] == 'start': self.start = np.array([float(t[1]), float(t[2])])
            elif t[0] == 'pillar': self.ref_obst.append((float(t[1]), float(t[2]), float(t[3])))
            elif t[0] == 'building':
                self.ref_boxes.append(tuple(float(v) for v in t[1:6]))

    def on_truth(self, m): self.true = np.array([m.pose.pose.position.x, m.pose.pose.position.y])
    def on_vio(self, m):
        # clamp (a real UAV can't exceed a few m/s -> reject flow glitches) + light smoothing
        v = np.clip(np.array([m.linear.x, m.linear.y]), -4.0, 4.0)
        self.vio_vel = 0.5 * self.vio_vel + 0.5 * v; self.vio_t = self.now()
    def on_fix(self, m): self.fix = np.array([m.point.x, m.point.y]); self.fix_t = self.now()
    def on_cues(self, m):
        if len(m.data) >= 2: self.blur, self.feats = float(m.data[0]), float(m.data[1])
    def on_scan(self, m): self.scan = m
    def on_imu(self, m):
        w = m.angular_velocity
        self.gyro_mag = 0.8 * self.gyro_mag + 0.2 * float(np.hypot(w.x, np.hypot(w.y, w.z)))
    def on_teleop(self, m): self.teleop_vel = np.array([m.linear.x, m.linear.y])
    def on_ctrl(self, m):
        c = m.data
        if c in ('auto', 'manual'):
            self.mode = c; self.arrived = False; self.arrival_confirmed = False
        elif c == 'pause': self.paused = True
        elif c in ('play', 'resume'): self.paused = False
        elif c == 'reset': self.reset_home()
        elif c == 'move_north': self.reposition([0.0, 1.0])
        elif c == 'move_south': self.reposition([0.0, -1.0])
        elif c == 'move_east': self.reposition([1.0, 0.0])
        elif c == 'move_west': self.reposition([-1.0, 0.0])
    def now(self): return self.get_clock().now().nanoseconds * 1e-9

    # ---- reset / reposition -> the platform (sim teleports, PX4 flies there) ----
    def teleport(self, xy):
        self.platform.go_to(float(xy[0]), float(xy[1]), self.spawn_z)

    def reset_home(self):
        self.teleport(self.home)
        self.fusion.state.xy = self.home.copy()
        self.true_path.clear(); self.est_path.clear(); self.corr.clear()
        self.fixes = 0; self.arrived = False; self.paused = True
        self.arrival_confirmed = False
        self.get_logger().info(f'RESET -> home {self.home}, holding')

    def reposition(self, delta):
        self.home = self.home + np.array(delta, float)
        self.teleport(self.home)
        self.fusion.state.xy = self.home.copy()
        self.get_logger().info(f'home -> {self.home}')

    def scan_avoid(self, base):
        # Potential field from the laser. Bin the scan into sectors and use only the
        # nearest return per sector -- one push per obstacle rather than one per ray,
        # otherwise a wide pillar out-pushes the seek velocity and the drone stalls.
        v = base.copy()
        self.nearest = np.inf
        if self.scan is None:
            return v
        r = np.asarray(self.scan.ranges, float)
        nR = len(r)
        if nR == 0:
            return v
        amin = self.scan.angle_min; ainc = self.scan.angle_increment
        # react within 6 m: wider berth avoids getting boxed into a local minimum between
        # two pillars (a tighter radius traps the drone -- verified), and keeps a big margin.
        rmax, nsec = 6.0, 24
        seclen = max(1, nR // nsec)
        vmax = self.ctrl.cfg.v_max
        nearest_away = None
        for s in range(0, nR, seclen):
            seg = r[s:s + seclen]
            m = np.isfinite(seg) & (seg > 0.05) & (seg < rmax)
            if not m.any():
                continue
            j = int(np.argmin(np.where(m, seg, np.inf)))
            r_i = float(seg[j]); a_i = amin + (s + j) * ainc          # yaw=0 -> world bearing
            away = -np.array([np.cos(a_i), np.sin(a_i)])
            if r_i < self.nearest:
                self.nearest = r_i; nearest_away = away
            # Capped well below the seek speed: in clutter several sectors contribute at
            # once, and if their sum can outrun the seek the drone simply stops dead.
            v = v + away * min(2.6, 3.0 * (1.0 / max(r_i - 0.5, 0.2) - 1.0 / rmax))
            # Cancel the component heading into an obstacle so we slide around it -- but
            # only when genuinely close. Applying it out at the full reaction radius vetoes
            # approach toward anything in front, which in clutter means never moving.
            if r_i < 3.0:
                into = -float(v @ away)
                if into > 0:
                    v = v + away * into
        # Tangential term: circle the nearest obstacle toward the goal side. Without it
        # seek and repulsion can balance exactly and the drone sits still.
        if nearest_away is not None and self.nearest < rmax:
            bhat = base / (np.linalg.norm(base) + 1e-9)
            tang = np.array([-nearest_away[1], nearest_away[0]])
            if tang @ bhat < 0:
                tang = -tang
            v = v + tang * min(2.0, 2.2 * (1.0 - self.nearest / rmax))
        n2 = np.linalg.norm(v)
        if n2 > vmax * 1.8:
            v = v / n2 * vmax * 1.8
        return v

    def step(self):
        if self.true is None or self.paused:
            self.pub_cmd(0, 0)
            if self.true is not None:
                self.last = dict(U=0.0, est=self.fusion.state.xy)
                self.publish_viz(0.0)
            return
        dt = 0.05

        # VIO stays silent until it has calibrated. Until then, dead-reckon on the
        # commanded motion and skip the wind feed-forward -- feeding an unscaled
        # velocity into the wind loop makes it diverge.
        vio_ok = (self.now() - self.vio_t) < 0.5
        if vio_ok:
            self.wind_est = 0.92 * self.wind_est + 0.08 * (self.vio_vel - self.v_cmd)
            # tighter clamp: real wind here is ~0.8 m/s, so bound the estimate near that.
            # A 3 m/s clamp let a diverged estimate shove the drone through obstacle
            # avoidance and pin it against a block.
            self.wind_est = np.clip(self.wind_est, -1.5, 1.5)
            motion = self.vio_vel
        else:
            self.wind_est = np.zeros(2)
            motion = self.v_cmd
        self.fusion.predict(motion * dt)
        est = self.fusion.state.xy

        # --- scheduler on measured image cues ---
        floss = max(0.0, self.prev_feats - self.feats); self.prev_feats = self.feats
        # inertial cue from the real gyro: ~0.02 at rest, rising as the drone rotates
        # (rotation degrades the flow, so the scheduler should trust it less)
        imu_bias = min(0.2, 0.02 + 0.4 * self.gyro_mag)
        cues = dict(sigma_pos=self.fusion.sigma_pos(), sigma_head=0.01,
                    blur=self.blur, feature_loss=floss, imu_bias=imu_bias, active_features=self.feats)
        U, trig, reason, _ = self.sched.compute(cues)

        # --- FRUGAL correction: only when U fires AND a fresh camera fix exists ---
        fix_fresh = (self.fix is not None and self.now() - self.fix_t < 0.5
                     and self.fix_t > self.fix_used_t)
        # Once we believe we have arrived, spend one fix to confirm it. Without this the
        # drone stops, uncertainty stops growing (it scales with distance travelled), no
        # fix ever fires again, and whatever drift it had becomes the permanent miss.
        confirming = (self.mode == 'auto' and not self.arrival_confirmed
                      and self.ctrl.arrived(est))
        if (trig or confirming) and fix_fresh:
            self.fusion.update(LandmarkFix(xy=self.fix, yaw=0.0,
                               covariance=np.eye(2) * 0.25, marker_id=0), gain=None)
            self.sched.reset_after_fix()
            self.fix_used_t = self.fix_t; self.fixes += 1; self.corr.append(self.fusion.state.xy.copy())
            if confirming and self.ctrl.arrived(self.fusion.state.xy):
                self.arrival_confirmed = True    # still arrived after the fix -> we're there
        est = self.fusion.state.xy

        # --- control: seek target with laser avoidance, then WIND FEEDFORWARD ---
        if self.mode == 'manual':
            v_ctrl = self.scan_avoid(self.teleop_vel.copy())
        elif self.mode == 'auto' and self.ctrl.arrived(est):
            self.arrived = True; v_ctrl = np.zeros(2); self.nearest = np.inf
        else:
            seek = self.B - est; n = np.linalg.norm(seek) or 1.0
            v_ctrl = self.scan_avoid(seek / n * self.ctrl.cfg.v_max)
        # Apply only 0.6 of the wind estimate. With full feedforward the wind loop is a
        # pure integrator (the v_cmd term cancels the 0.92 leak) and runs to the clamp;
        # at 0.6 the effective factor drops below 1, so it's a stable leaky estimator that
        # still cancels most of the wind but can't run away and jam the drone on a block.
        self.v_cmd = v_ctrl - 0.6 * self.wind_est
        self.pub_cmd(self.v_cmd[0], self.v_cmd[1])

        self.hold_altitude()
        self.true_path.append(self.true.copy()); self.est_path.append(est.copy())
        self.last = dict(U=U, est=est)
        self.publish_viz(U)
        self._logc = getattr(self, '_logc', 0) + 1
        if self._logc % 40 == 0:
            terr = float(np.linalg.norm(est - self.true))
            nz = self.nearest if np.isfinite(self.nearest) else -1.0
            self.get_logger().info(
                f'pos=({self.true[0]:.1f},{self.true[1]:.1f}) U={U:.2f} fixes={self.fixes} '
                f'est_err={terr:.2f}m wind_est=({self.wind_est[0]:.2f},{self.wind_est[1]:.2f}) '
                f'nearest_obst={nz:.1f}m alt={self.alt:.1f}m feats={self.feats:.0f}')

    def pub_cmd(self, x, y):
        self.platform.set_velocity(x, y)

    def hold_altitude(self):
        # Decide the desired altitude from measured image CONTRAST (Laplacian variance):
        # under a haze bank the ground washes out and contrast collapses, so drop below it,
        # then climb back once it clears. Contrast (not feature count) because feature count
        # also falls with altitude, so the drone would read its own descent as fog. The
        # platform then realises the altitude (sim moves the model; PX4 commands it).
        if self.blur < 120:
            self.poor_vis = min(self.poor_vis + 1, 40)
        elif self.blur > 250:
            self.poor_vis = max(self.poor_vis - 1, 0)
        want = self.low_z if self.poor_vis > 8 else self.cruise_z
        if abs(want - self.alt) < 0.05:
            return                                # at target altitude: don't disturb motion
        self.alt += max(-0.10, min(0.10, want - self.alt))    # ease, don't snap
        self.platform.set_altitude(self.alt)

    # ---- visualization ----
    def mk(self, i, typ, s, r, g, b, a):
        m = Marker(); m.header.frame_id = 'world'; m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'nav'; m.id = i; m.type = typ; m.action = 0
        m.scale.x = m.scale.y = m.scale.z = s
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a; m.pose.orientation.w = 1.0
        return m

    def line(self, i, pts, r, g, b):
        from geometry_msgs.msg import Point
        m = self.mk(i, Marker.LINE_STRIP, 0.15, r, g, b, 0.95); m.scale.x = 0.15
        for p in pts:
            q = Point(); q.x, q.y, q.z = float(p[0]), float(p[1]), 0.1; m.points.append(q)
        return m

    def publish_scene(self):
        a = MarkerArray()
        # faint REFERENCE obstacles so you can see where they are. The navigator never
        # reads these -- it avoids purely from the live laser (the red hit dots).
        # Boxes win when a map provides them, so a city is not drawn as circles.
        for j, (ox, oy, sx, sy, h) in enumerate(self.ref_boxes):
            b = self.mk(60 + j, Marker.CUBE, 1.0, 0.5, 0.53, 0.58, 0.30)
            b.scale.x, b.scale.y, b.scale.z = sx, sy, h
            b.pose.position.x, b.pose.position.y, b.pose.position.z = ox, oy, h / 2
            a.markers.append(b)
        for j, (ox, oy, r) in enumerate(self.ref_obst):
            c = self.mk(300 + j, Marker.CYLINDER, 1.0, 0.5, 0.53, 0.58, 0.30)
            c.scale.x = c.scale.y = 2 * r; c.scale.z = 12.0
            c.pose.position.x, c.pose.position.y, c.pose.position.z = ox, oy, 6.0
            a.markers.append(c)
        tgt = self.mk(2, Marker.CYLINDER, 1.2, 0.98, 0.75, 0.14, 0.95); tgt.scale.z = 3.0
        tgt.pose.position.x, tgt.pose.position.y, tgt.pose.position.z = float(self.B[0]), float(self.B[1]), 1.5
        a.markers.append(tgt); self.scene_pub.publish(a)

    def publish_viz(self, U):
        from geometry_msgs.msg import Point
        a = MarkerArray()
        # the drone itself (blue cube) so it's easy to find in RViz
        dm = self.mk(210, Marker.CUBE, 0.9, 0.22, 0.74, 0.97, 1.0)
        dm.pose.position.x, dm.pose.position.y, dm.pose.position.z = float(self.true[0]), float(self.true[1]), 0.6
        a.markers.append(dm)
        a.markers.append(self.line(200, self.true_path, 0.2, 0.83, 0.44))
        a.markers.append(self.line(201, self.est_path, 0.22, 0.74, 0.97))
        fx = self.mk(203, Marker.SPHERE_LIST, 0.9, 0.99, 0.85, 0.14, 1.0)
        for c in self.corr:
            q = Point(); q.x, q.y, q.z = float(c[0]), float(c[1]), 0.2; fx.points.append(q)
        a.markers.append(fx)
        # live laser hits (what the drone actually detects as obstacles)
        hits = self.mk(205, Marker.SPHERE_LIST, 0.5, 0.98, 0.35, 0.25, 0.9)
        if self.scan is not None and self.true is not None:
            r = np.asarray(self.scan.ranges, float)
            m = np.isfinite(r) & (r > 0.05) & (r < 6.0)
            if m.any():
                idx = np.where(m)[0]
                ang = self.scan.angle_min + idx * self.scan.angle_increment
                for a_i, r_i in zip(ang, r[m]):
                    q = Point(); q.x = float(self.true[0] + r_i * np.cos(a_i))
                    q.y = float(self.true[1] + r_i * np.sin(a_i)); q.z = 2.0; hits.points.append(q)
        a.markers.append(hits)
        wm = self.mk(206, Marker.ARROW, 0.3, 0.4, 0.9, 1.0, 0.9); wm.scale.x = 0.3; wm.scale.y = 0.6
        p0 = Point(); p0.x, p0.y, p0.z = float(self.true[0]), float(self.true[1]), 3.0
        p1 = Point(); p1.x = float(self.true[0] + self.wind_est[0] * 2)
        p1.y = float(self.true[1] + self.wind_est[1] * 2); p1.z = 3.0
        wm.points = [p0, p1]; a.markers.append(wm)
        tx = self.mk(204, Marker.TEXT_VIEW_FACING, 2.0, 0.92, 0.95, 0.98, 1.0)
        tx.pose.position.x, tx.pose.position.y, tx.pose.position.z = float(self.start[0]), float(self.start[1]), 6.0
        terr = np.linalg.norm(self.last['est'] - self.true) if self.true is not None else 0.0
        nz = self.nearest if np.isfinite(self.nearest) else -1.0
        tx.text = (f"DEMO 3  REAL VIO + LASER  mode={self.mode}\n"
                   f"U={U:.2f} fixes={self.fixes} est-err={terr:.2f}m\n"
                   f"wind_est=({self.wind_est[0]:.2f},{self.wind_est[1]:.2f})  nearest={nz:.1f}m")
        a.markers.append(tx); self.viz_pub.publish(a)


def main():
    rclpy.init(); rclpy.spin(Real3Nav()); rclpy.shutdown()


if __name__ == '__main__':
    main()
