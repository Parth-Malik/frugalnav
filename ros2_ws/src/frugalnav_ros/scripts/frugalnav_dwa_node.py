#!/usr/bin/env python3
"""
Demo 3 STRONG navigator (frugalnav_dwa_node).

Same measured-only inputs as the baseline (`frugalnav_real3_node.py`), but guidance,
avoidance and the estimator are rebuilt on established methods instead of hand-tuned
reactions:

  * Local planner -- Dynamic Window Approach (Fox, Burgard & Thrun, 1997), the
    sampling planner behind ROS Nav2's DWB. Each tick it samples world-frame velocity
    candidates, rolls each out over a short horizon, and scores them by goal progress +
    obstacle clearance + speed + smoothness, then commands the best. Scoring whole
    short trajectories (not an instantaneous potential gradient) is what escapes the
    local minima that trapped the baseline between clustered pillars.

  * Estimator -- the project's tight StateFusion (2x2 Kalman filter) driven with a
    CONSISTENT process model: covariance grows with distance AND time and inflates when
    the VIO is stale or the image is low-texture, so sigma_pos tracks the true drift.
    Honest uncertainty is what lets the scheduler fire a fix BEFORE the estimate
    silently walks off -- the real-map failure was an over-confident filter (U stayed
    low, no fix fired, the drone drifted while believing it was fine).

  * Disturbance rejection -- a velocity-tracking PI with anti-windup closes the loop on
    (desired - measured) velocity to cancel a steady wind. This replaces the open-loop
    feed-forward that algebraically became a pure integrator and flung the drone.

  * Scheduler -- UNCHANGED. Rohan's UncertaintyScheduler still decides WHEN to spend a
    landmark fix; this navigator only makes the signals it reads honest.

Output goes through the platform adapter (sim | px4), so the same node flies Gazebo and
a real PX4 vehicle.
"""
import os
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped, Point
from std_msgs.msg import Float32MultiArray, String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Imu
from visualization_msgs.msg import Marker, MarkerArray


def _repo_root():
    """Find the repo so `core/` imports, from a symlink or copy install."""
    env = os.environ.get('FRUGALNAV_ROOT')
    if env and os.path.isdir(os.path.join(env, 'core')):
        return env
    for start in (os.path.realpath(__file__), os.path.abspath(__file__)):
        d = os.path.dirname(start)
        for _ in range(8):
            if os.path.isdir(os.path.join(d, 'core')):
                return d
            d = os.path.dirname(d)
    raise RuntimeError('cannot find the FrugalNav core; set FRUGALNAV_ROOT')


sys.path.insert(0, _repo_root())
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))   # for frugalnav_platform
import frugalnav_platform
from core.uncertainty_scheduler import UncertaintyScheduler, SchedulerConfig
from core.state_fusion import StateFusion, FusionConfig
from core.controller import TargetCentricController, ControllerConfig
from core.types import LandmarkFix


class DwaNav(Node):
    def __init__(self):
        super().__init__('frugalnav_dwa_node')
        scene = self.declare_parameter('scene_file', '').value
        self.B, self.start = np.array([0., 0.]), np.array([0., 0.])
        self.ref_obst, self.ref_boxes = [], []          # RViz reference only
        self.load_scene(scene)

        # --- estimator: consistent process noise so sigma_pos tracks true drift ---
        self.ctrl = TargetCentricController(self.B, ControllerConfig(v_max=2.0, arrive_tol=1.5))
        self.fusion = StateFusion(init_xy=tuple(self.start), cfg=FusionConfig(q_per_metre=0.05))
        self.q_time = 0.02        # variance/s added every tick (grows P even when still)
        self.q_stale = 0.15       # extra variance/s while the VIO is stale (blind coasting)
        self.q_lowtex = 0.05      # extra variance/s when the image is low-texture
        # scheduler unchanged in structure; thresholds a touch tighter for a drift-prone VIO
        self.sched = UncertaintyScheduler(SchedulerConfig(tau=0.35, sigma_pos_floor=1.0,
                                                          feature_floor=25))

        # --- DWA parameters ---
        self.dwa = dict(n_head=24, speeds=(0.5, 1.0, 1.5, 2.0), horizon=1.6, steps=8,
                        r_safe=1.4, clear_cap=6.0,
                        w_goal=0.60, w_clear=0.28, w_speed=0.07, w_smooth=0.05)

        self.true = None
        self.vio_vel = np.zeros(2); self.vio_t = -1.0
        self.fix = None; self.fix_t = -1.0; self.fix_used_t = -1.0
        self.blur = 300.0; self.feats = 60.0; self.prev_feats = 60.0
        self.scan = None; self.nearest = np.inf
        self.gyro_mag = 0.0
        self.wind_i = np.zeros(2); self.v_cmd = np.zeros(2); self.v_prev = np.zeros(2)
        self.Ki = 0.8; self.wind_max = 1.8      # velocity-PI integral gain + anti-windup clamp
        self.teleop_vel = np.zeros(2)
        self.mode = 'auto'
        self.paused = bool(self.declare_parameter('start_paused', False).value)
        self.home = self.start.copy(); self.spawn_z = 5.0
        self.cruise_z, self.low_z, self.alt = 5.0, 2.6, 5.0
        self.poor_vis = 0
        self.true_path, self.est_path, self.corr = [], [], []
        self.fixes = 0; self.arrived = False; self.arrival_confirmed = False
        self.last = dict(U=0.0, est=self.start.copy())
        self._last_t = None

        # platform adapter: 'sim' -> Gazebo, 'px4' -> real PX4 offboard. Nav is identical.
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
        self.timer = self.create_timer(0.05, self.step)     # 20 Hz
        self.get_logger().info(f'DWA nav up: target {self.B} (DWA + consistent EKF, no map)')

    # ---------------------------------------------------------------- scene / io
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
            elif t[0] == 'building': self.ref_boxes.append(tuple(float(v) for v in t[1:6]))

    def on_truth(self, m): self.true = np.array([m.pose.pose.position.x, m.pose.pose.position.y])
    def on_vio(self, m):
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
            self.mode = c; self.arrived = False; self.arrival_confirmed = False; self.wind_i = np.zeros(2)
        elif c == 'pause': self.paused = True
        elif c in ('play', 'resume'): self.paused = False
        elif c == 'reset': self.reset_home()
        elif c == 'move_north': self.reposition([0.0, 1.0])
        elif c == 'move_south': self.reposition([0.0, -1.0])
        elif c == 'move_east': self.reposition([1.0, 0.0])
        elif c == 'move_west': self.reposition([-1.0, 0.0])
    def now(self): return self.get_clock().now().nanoseconds * 1e-9

    def reset_home(self):
        self.platform.go_to(float(self.home[0]), float(self.home[1]), self.spawn_z)
        self.fusion.state.xy = self.home.copy(); self.fusion.state.covariance = np.eye(2) * 0.01
        self.wind_i = np.zeros(2); self.arrived = False; self.arrival_confirmed = False
        self.true_path, self.est_path, self.corr = [], [], []

    def reposition(self, d):
        self.home = self.home + np.array(d, float)
        self.platform.go_to(float(self.home[0]), float(self.home[1]), self.spawn_z)
        self.fusion.state.xy = self.home.copy()

    # ---------------------------------------------------------------- planner
    def _obst_points(self, p0):
        """World-frame points of the nearest laser return per sector (yaw=0)."""
        if self.scan is None:
            self.nearest = np.inf; return None
        r = np.asarray(self.scan.ranges, float); n = len(r)
        if n == 0:
            self.nearest = np.inf; return None
        amin, ainc = self.scan.angle_min, self.scan.angle_increment
        rmax, nsec = self.dwa['clear_cap'], 24
        seclen = max(1, n // nsec)
        pts = []; self.nearest = np.inf
        for s in range(0, n, seclen):
            seg = r[s:s + seclen]
            m = np.isfinite(seg) & (seg > 0.05) & (seg < rmax)
            if not m.any():
                continue
            j = int(np.argmin(np.where(m, seg, np.inf)))
            ri = float(seg[j]); ai = amin + (s + j) * ainc
            if ri < self.nearest:
                self.nearest = ri
            pts.append(p0 + ri * np.array([np.cos(ai), np.sin(ai)]))
        return np.array(pts) if pts else None

    def _dwa(self, p0, goal, obst):
        """Dynamic Window Approach over world-frame velocities. Returns the best v_des."""
        cfg = self.dwa
        dirs = np.linspace(0.0, 2 * np.pi, cfg['n_head'], endpoint=False)
        cand = [np.zeros(2)]
        for s in cfg['speeds']:
            for a in dirs:
                cand.append(s * np.array([np.cos(a), np.sin(a)]))
        V = np.array(cand)                                   # (N,2)
        T, K = cfg['horizon'], cfg['steps']
        ts = np.linspace(T / K, T, K)                        # (K,)
        roll = p0[None, None, :] + V[:, None, :] * ts[None, :, None]   # (N,K,2)

        d0 = float(np.linalg.norm(p0 - goal))
        dend = np.linalg.norm(roll[:, -1, :] - goal, axis=1)
        progress = d0 - dend                                 # closer endpoint = better
        if obst is not None and len(obst):
            diff = roll[:, :, None, :] - obst[None, None, :, :]        # (N,K,M,2)
            dmin = np.sqrt((diff ** 2).sum(-1)).min(axis=(1, 2))      # (N,)
        else:
            dmin = np.full(len(V), cfg['clear_cap'])
        feasible = dmin > cfg['r_safe']
        speed = np.linalg.norm(V, axis=1)
        smooth = -np.linalg.norm(V - self.v_prev, axis=1)

        def norm(x):
            lo, hi = float(np.min(x)), float(np.max(x))
            return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

        score = (cfg['w_goal'] * norm(progress) + cfg['w_clear'] * norm(np.clip(dmin, 0, cfg['clear_cap']))
                 + cfg['w_speed'] * norm(speed) + cfg['w_smooth'] * norm(smooth))
        if not feasible.any():
            return V[int(np.argmax(dmin))]        # boxed in -> take the safest (max clearance)
        score[~feasible] = -1e9
        return V[int(np.argmax(score))]

    def _manual_safe(self, v):
        """Let teleop through but veto the component driving into a close obstacle."""
        obst = self._obst_points(self.fusion.state.xy)
        if obst is None or self.nearest > 2.5 or np.linalg.norm(v) < 1e-6:
            return v
        d = self.fusion.state.xy
        near = obst[np.argmin(np.linalg.norm(obst - d, axis=1))]
        away = (d - near); away /= (np.linalg.norm(away) + 1e-9)
        into = -float(v @ away)
        return v + away * into if into > 0 else v

    # ---------------------------------------------------------------- main loop
    def step(self):
        if self.true is None:
            return
        now = self.now()
        dt = 0.05 if self._last_t is None else float(np.clip(now - self._last_t, 0.005, 0.2))
        self._last_t = now
        if self.paused:
            self.publish_viz(self.last['U']); return

        # --- predict with a consistent (distance + time + quality) process model ---
        vio_ok = (now - self.vio_t) < 0.5
        motion = (self.vio_vel if vio_ok else self.v_cmd) * dt
        extra = self.q_time * dt
        if not vio_ok:
            extra += self.q_stale * dt
        if self.feats < 30:
            extra += self.q_lowtex * dt
        self.fusion.predict(motion, extra_var=extra, t=now)
        est = self.fusion.state.xy

        # --- scheduler: fire a fix on honest uncertainty (structure unchanged) ---
        floss = max(0.0, self.prev_feats - self.feats) / dt; self.prev_feats = self.feats
        cues = dict(sigma_pos=self.fusion.sigma_pos(), sigma_head=0.0, feature_loss=floss,
                    blur=self.blur, imu_bias=0.0, active_features=self.feats)
        U, trig, reason, _ = self.sched.compute(cues)
        fix_fresh = (self.fix is not None and now - self.fix_t < 0.5 and self.fix_t > self.fix_used_t)
        confirming = (self.mode == 'auto' and not self.arrival_confirmed and self.ctrl.arrived(est))
        if (trig or confirming) and fix_fresh:
            self.fusion.update(LandmarkFix(xy=self.fix, yaw=0.0, covariance=np.eye(2) * 0.25,
                                           marker_id=0), gain=None)     # optimal blend, stays consistent
            self.platform.send_vision(self.fix, cov=0.25)              # on px4 -> EKF2 external vision
            self.sched.reset_after_fix()
            self.fix_used_t = self.fix_t; self.fixes += 1
            self.corr.append(self.fusion.state.xy.copy())
            if confirming and self.ctrl.arrived(self.fusion.state.xy):
                self.arrival_confirmed = True
        est = self.fusion.state.xy

        # --- guidance: DWA (auto) or safety-filtered teleop (manual) ---
        if self.mode == 'manual':
            v_des = self._manual_safe(self.teleop_vel.copy())
        elif self.mode == 'auto' and self.ctrl.arrived(est):
            self.arrived = True; v_des = np.zeros(2)
        else:
            v_des = self._dwa(est, self.B, self._obst_points(est))
        self.v_prev = v_des

        # --- velocity-tracking PI: cancel a steady wind without open-loop wind-up ---
        if self.mode == 'auto' and vio_ok and np.linalg.norm(v_des) > 1e-6:
            e = v_des - self.vio_vel
            self.wind_i = np.clip(self.wind_i + self.Ki * e * dt, -self.wind_max, self.wind_max)
        else:
            self.wind_i *= 0.98
        self.v_cmd = v_des + self.wind_i
        s = np.linalg.norm(self.v_cmd)
        if s > self.ctrl.cfg.v_max * 1.5:
            self.v_cmd = self.v_cmd / s * self.ctrl.cfg.v_max * 1.5
        self.platform.set_velocity(float(self.v_cmd[0]), float(self.v_cmd[1]))

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
                f'est_err={terr:.2f}m wind_est=({self.wind_i[0]:.2f},{self.wind_i[1]:.2f}) '
                f'nearest_obst={nz:.1f}m alt={self.alt:.1f}m feats={self.feats:.0f}')

    def hold_altitude(self):
        # Contrast (Laplacian variance), not feature count: features also fall with
        # altitude, so counting them makes the drone read its own descent as fog.
        if self.blur < 120:
            self.poor_vis = min(self.poor_vis + 1, 40)
        elif self.blur > 250:
            self.poor_vis = max(self.poor_vis - 1, 0)
        want = self.low_z if self.poor_vis > 8 else self.cruise_z
        if abs(want - self.alt) < 0.05:
            return
        self.alt += max(-0.10, min(0.10, want - self.alt))
        self.platform.set_altitude(self.alt)

    # ---------------------------------------------------------------- viz
    def mk(self, i, typ, s, r, g, b, a):
        m = Marker(); m.header.frame_id = 'world'; m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'nav'; m.id = i; m.type = typ; m.action = 0
        m.scale.x = m.scale.y = m.scale.z = s
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a; m.pose.orientation.w = 1.0
        return m

    def line(self, i, pts, r, g, b):
        m = self.mk(i, Marker.LINE_STRIP, 0.15, r, g, b, 0.95); m.scale.x = 0.15
        for p in pts:
            q = Point(); q.x, q.y, q.z = float(p[0]), float(p[1]), 0.1; m.points.append(q)
        return m

    def publish_scene(self):
        a = MarkerArray()
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
        a = MarkerArray()
        dm = self.mk(210, Marker.CUBE, 0.9, 0.22, 0.74, 0.97, 1.0)
        dm.pose.position.x, dm.pose.position.y, dm.pose.position.z = float(self.true[0]), float(self.true[1]), 0.6
        a.markers.append(dm)
        a.markers.append(self.line(200, self.true_path, 0.2, 0.83, 0.44))
        a.markers.append(self.line(201, self.est_path, 0.22, 0.74, 0.97))
        fx = self.mk(203, Marker.SPHERE_LIST, 0.9, 0.99, 0.85, 0.14, 1.0)
        for c in self.corr:
            q = Point(); q.x, q.y, q.z = float(c[0]), float(c[1]), 0.2; fx.points.append(q)
        a.markers.append(fx)
        hits = self.mk(205, Marker.SPHERE_LIST, 0.5, 0.98, 0.35, 0.25, 0.9)
        if self.scan is not None:
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
        p1 = Point(); p1.x = float(self.true[0] + self.wind_i[0] * 2)
        p1.y = float(self.true[1] + self.wind_i[1] * 2); p1.z = 3.0
        wm.points = [p0, p1]; a.markers.append(wm)
        tx = self.mk(204, Marker.TEXT_VIEW_FACING, 2.0, 0.92, 0.95, 0.98, 1.0)
        tx.pose.position.x, tx.pose.position.y, tx.pose.position.z = float(self.start[0]), float(self.start[1]), 6.0
        terr = float(np.linalg.norm(self.last['est'] - self.true))
        nz = self.nearest if np.isfinite(self.nearest) else -1.0
        tx.text = (f"DEMO 3  DWA NAV  mode={self.mode}\n"
                   f"U={U:.2f} fixes={self.fixes} est-err={terr:.2f}m\n"
                   f"wind_est=({self.wind_i[0]:.2f},{self.wind_i[1]:.2f})  nearest={nz:.1f}m")
        a.markers.append(tx); self.viz_pub.publish(a)


def main():
    rclpy.init(); rclpy.spin(DwaNav()); rclpy.shutdown()


if __name__ == '__main__':
    main()
