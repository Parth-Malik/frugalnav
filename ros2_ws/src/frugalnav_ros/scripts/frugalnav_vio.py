#!/usr/bin/env python3
"""
Visual-inertial odometry from the downward camera.

Tracks sparse features with Lucas-Kanade optical flow, DE-ROTATES the flow using the
gyro (a real drone tilts to move, which swings the camera -- without this the flow
would read that tilt as translation), and scales it to a metric velocity with the
altimeter (v = flow * altitude / fx). Publishes /frugalnav/vio (Twist, world m/s).

Inputs are all standard sensor messages, so on real hardware you point it at the real
camera / IMU / rangefinder topics unchanged:
    /frugalnav/down_cam/image_raw  + camera_info   (sensor_msgs/Image, CameraInfo)
    /frugalnav/imu                                 (sensor_msgs/Imu)
    /frugalnav/height                              (sensor_msgs/Range)

Integrating this velocity drifts, which is the point: the ArUco fixes bound the drift.
Cheap on purpose -- sparse LK on ~90 features, no dense flow or stereo.
"""
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, Imu, Range
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge


class VIO(Node):
    def __init__(self):
        super().__init__('frugalnav_vio')
        self.bridge = CvBridge()
        self.K = None
        self.prev = None
        self.prev_t = None
        self.prev_pts = None
        self.alt = 5.0
        self.A = None
        self.calib = []
        self.skip_next = False
        self.prev_truth = None
        self.prev_truth_t = None
        self.truth_vel = None
        self.omega = np.zeros(3)          # body angular velocity, from the gyro
        self.have_height = False
        # camera-optical <- body rotation for the downward camera (pose pitch +90 deg)
        self.R_CB = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
        self.pub = self.create_publisher(Twist, '/frugalnav/vio', 10)
        self.create_subscription(CameraInfo, '/frugalnav/down_cam/camera_info', self.on_info, 5)
        self.create_subscription(Image, '/frugalnav/down_cam/image_raw', self.on_img, 5)
        self.create_subscription(Imu, '/frugalnav/imu', self.on_imu, 20)
        self.create_subscription(Range, '/frugalnav/height', self.on_height, 10)
        self.create_subscription(Odometry, '/frugalnav/truth', self.on_truth, 10)
        self.get_logger().info('VIO up: visual-inertial odometry on the downward camera')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_info(self, m):
        if self.K is None:
            self.K = np.array(m.k, float).reshape(3, 3)

    def on_imu(self, m):
        w = m.angular_velocity
        self.omega = np.array([w.x, w.y, w.z])

    def on_height(self, m):
        # real altimeter -> the metric scale for the flow
        if 0.1 < m.range < 40.0:
            self.alt = max(0.3, float(m.range)); self.have_height = True

    def on_truth(self, m):
        # Runtime altitude now comes from /frugalnav/height. Truth is used only for the
        # one-time flow calibration (and as an altitude fallback if no altimeter is up),
        # never for the runtime velocity.
        p = np.array([m.pose.pose.position.x, m.pose.pose.position.y])
        if not self.have_height:
            self.alt = max(0.5, float(m.pose.pose.position.z))
        t = self.now()
        if self.prev_truth is not None:
            step = np.linalg.norm(p - self.prev_truth)
            if step > 1.0:                      # teleport (reset/reposition) -> drop the flow frame
                self.skip_next = True
            dt = t - self.prev_truth_t
            if dt > 1e-3:
                self.truth_vel = (p - self.prev_truth) / dt
        self.prev_truth = p
        self.prev_truth_t = t

    def on_img(self, msg):
        if self.K is None:
            return
        gray = cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg, 'bgr8'), cv2.COLOR_BGR2GRAY)
        t = self.now()
        vel_cam = np.zeros(2)
        ok = False
        if self.skip_next:
            self.skip_next = False
        elif self.prev is not None and self.prev_pts is not None and len(self.prev_pts) > 0:
            dt = max(1e-3, t - self.prev_t)
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.prev_pts, None)
            if nxt is not None and st is not None:
                good_new = nxt[st.flatten() == 1]
                good_old = self.prev_pts[st.flatten() == 1]
                if len(good_new) >= 6:
                    # LK returns (N,1,2); flatten so the median is a 2-vector
                    flow = np.median((good_new - good_old).reshape(-1, 2), axis=0)  # px/frame
                    # DE-ROTATE: remove the flow the camera's own rotation produced. At the
                    # image centre the rotational flow is f*dt*[-wc_y, wc_x], where wc is the
                    # angular velocity in the camera frame. Zero when the drone isn't turning
                    # (as in this planar sim); essential once it tilts to move.
                    wc = self.R_CB @ self.omega
                    flow = flow - self.K[0, 0] * dt * np.array([-wc[1], wc[0]])
                    if np.linalg.norm(flow) < 120:                  # reject tracking glitches
                        disp = flow * (self.alt / self.K[0, 0])     # metric ground displacement
                        vel_cam = disp / dt
                        ok = True
        # (re)seed features every frame -- cheap and keeps the tracker fresh
        self.prev_pts = cv2.goodFeaturesToTrack(gray, 90, 0.01, 8)
        self.prev = gray
        self.prev_t = t

        if not ok:
            return
        if self.A is not None:
            # calibrated: publish the real measured world velocity
            v = self.A @ vel_cam
            out = Twist(); out.linear.x = float(v[0]); out.linear.y = float(v[1])
            self.pub.publish(out)
        else:
            # collect calibration only while genuinely moving, then fit cam->world.
            # Do NOT publish until calibrated -- the nav bootstraps on commanded motion,
            # so it never feeds an un-scaled velocity into the wind loop (that diverges).
            if self.truth_vel is not None and 0.25 < np.linalg.norm(self.truth_vel) < 4.0:
                self.calib.append((vel_cam[0], vel_cam[1],
                                   self.truth_vel[0], self.truth_vel[1]))
            if len(self.calib) >= 40:
                M = np.array([[c[0], c[1]] for c in self.calib])
                b = np.array([[c[2], c[3]] for c in self.calib])
                # ORTHOGONAL Procrustes: the camera-image -> world-axes map is a fixed
                # rotation/reflection (yaw is 0, and we already scaled by altitude/fx).
                # Constraining A to be orthogonal makes it generalize to ANY motion
                # direction even though calibration motion is mostly one-directional --
                # a plain least-squares 2x2 overfits that one direction and diverges.
                Uo, _, Vto = np.linalg.svd(M.T @ b)
                self.A = (Uo @ Vto).T
                self.get_logger().info(f'VIO calibrated (orthogonal) from {len(self.calib)} samples')


def main():
    rclpy.init(); rclpy.spin(VIO()); rclpy.shutdown()


if __name__ == '__main__':
    main()
