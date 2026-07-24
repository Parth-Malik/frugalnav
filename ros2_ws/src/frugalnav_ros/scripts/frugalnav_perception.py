#!/usr/bin/env python3
"""
Turns the downward camera image into the signals the navigator needs.

    /frugalnav/fix                 absolute world position, from ArUco detection +
                                   solvePnP against the known marker map
    /frugalnav/cues                [blur, feature count, n_markers] for the scheduler
    /frugalnav/down_cam/annotated  same image with detections drawn, for viewing

Fix error is logged against /frugalnav/truth when available, as a sanity check.
"""
import os
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge


def rot_from_rvec(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3))
    return R


def make_T(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.asarray(t, float).reshape(3); return T


def inv_T(T):
    R = T[:3, :3]; t = T[:3, 3]; Ti = np.eye(4)
    Ti[:3, :3] = R.T; Ti[:3, 3] = -R.T @ t; return Ti


class Perception(Node):
    def __init__(self):
        super().__init__('frugalnav_perception')
        scene = self.declare_parameter('scene_file', '').value
        # detectable black pattern is smaller than the 2.6 m tile: the texture is a
        # 600 px marker inside an 80 px white quiet-zone border -> 600/760 * 2.6 = 2.05 m.
        self.msize = float(self.declare_parameter('marker_size', 2.05).value)
        self.markers = self.load_map(scene)       # id -> (x, y)
        self.get_logger().info(f'loaded {len(self.markers)} mapped ArUco markers')

        # camera->body extrinsic for the downward camera (optical axes in body frame).
        # Gazebo cam pose (0 0 -0.12 0 1.5708 0): optical +Z = body down, +X = body -Y,
        # +Y = body -X. Overridable if the mounting changes.
        R_BC = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
        self.T_BC = make_T(R_BC, [0.0, 0.0, -0.12])

        self.K = None; self.dist = None
        self.bridge = CvBridge()
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        if hasattr(cv2.aruco, 'DetectorParameters_create'):
            self._p = cv2.aruco.DetectorParameters_create()
            self._p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # sub-pixel corners
            self._detect = lambda g: cv2.aruco.detectMarkers(g, d, parameters=self._p)
        else:
            pr = cv2.aruco.DetectorParameters()
            pr.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            det = cv2.aruco.ArucoDetector(d, pr)
            self._detect = lambda g: det.detectMarkers(g)

        s = self.msize / 2.0
        self.obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)

        self.truth = None
        self.fix_pub = self.create_publisher(PointStamped, '/frugalnav/fix', 10)
        self.cue_pub = self.create_publisher(Float32MultiArray, '/frugalnav/cues', 10)
        self.ann_pub = self.create_publisher(Image, '/frugalnav/down_cam/annotated', 5)
        self.create_subscription(CameraInfo, '/frugalnav/down_cam/camera_info', self.on_info, 5)
        self.create_subscription(Image, '/frugalnav/down_cam/image_raw', self.on_img, 5)
        self.create_subscription(Odometry, '/frugalnav/truth', self.on_truth, 5)
        self._n = 0

    def load_map(self, path):
        m = {}
        if path and os.path.exists(path):
            for line in open(path):
                t = line.split()
                if t and t[0] == 'amarker':
                    m[int(t[1])] = (float(t[2]), float(t[3]))
        return m

    def on_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, float).reshape(3, 3)
            self.dist = np.array(msg.d, float) if len(msg.d) else np.zeros(5)
            self.get_logger().info('camera intrinsics received')

    def on_truth(self, msg):
        self.truth = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def on_img(self, msg):
        if self.K is None:
            return
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- MEASURED cues ---
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        feats = cv2.goodFeaturesToTrack(gray, 300, 0.01, 7)
        nfeat = 0 if feats is None else len(feats)

        # --- real ArUco detection + absolute fix ---
        if not hasattr(self, 'A'):
            self.A = None; self.calib = []      # one-time camera-mounting calibration
        corners, ids, _ = self._detect(gray)
        fixes = []
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                if int(i) not in self.markers:
                    continue
                ok, rvec, tvec = cv2.solvePnP(self.obj, c.reshape(-1, 2).astype(np.float32),
                                              self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not ok:
                    continue
                tvx, tvy = float(tvec[0]), float(tvec[1])
                mx, my = self.markers[int(i)]
                if self.A is not None:
                    # drone = marker - (calibrated image->world offset)
                    off = self.A @ np.array([tvx, tvy, 1.0])
                    fixes.append(np.array([mx - off[0], my - off[1]]))
                else:
                    # pre-calibration: analytic extrinsic + collect a truth sample
                    T_WB = (make_T(np.eye(3), [mx, my, 0.0]) @ inv_T(make_T(rot_from_rvec(rvec), tvec))
                            @ inv_T(self.T_BC))
                    fixes.append(T_WB[:2, 3])
                    if self.truth is not None:
                        self.calib.append((tvx, tvy, mx - self.truth[0], my - self.truth[1]))
            if self.A is None and len(self.calib) >= 40:
                M = np.array([[s[0], s[1], 1.0] for s in self.calib])
                b = np.array([[s[2], s[3]] for s in self.calib])
                self.A = np.linalg.lstsq(M, b, rcond=None)[0].T      # 2x3 image->world offset
                res = float(np.sqrt(np.mean(np.sum((M @ self.A.T - b) ** 2, axis=1))))
                self.get_logger().info(
                    f'CALIBRATED camera mounting from {len(self.calib)} samples, residual={res:.2f} m')

        cue = Float32MultiArray()
        cue.data = [blur, float(nfeat), float(len(fixes))]
        self.cue_pub.publish(cue)

        # annotated feed for the operator view
        ann = img.copy()
        if ids is not None and len(ids):
            cv2.aruco.drawDetectedMarkers(ann, corners, ids)
        cv2.putText(ann, f'markers={len(fixes)}  blur={blur:.0f}  feats={nfeat}',
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 240, 60), 2)
        if fixes:
            fxy = np.mean(fixes, axis=0)
            hud = f'FIX ({fxy[0]:.1f}, {fxy[1]:.1f}) m'
            if self.truth is not None:
                hud += f'  err={np.hypot(fxy[0]-self.truth[0], fxy[1]-self.truth[1]):.2f} m'
            cv2.putText(ann, hud, (10, img.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 210, 255), 2)
        self.ann_pub.publish(self.bridge.cv2_to_imgmsg(ann, 'bgr8'))

        self._n += 1
        if fixes:
            xy = np.mean(fixes, axis=0)
            ps = PointStamped(); ps.header.stamp = self.get_clock().now().to_msg()
            ps.header.frame_id = 'world'; ps.point.x = float(xy[0]); ps.point.y = float(xy[1])
            self.fix_pub.publish(ps)
            if self._n % 10 == 0:
                err = ''
                if self.truth is not None:
                    e = np.hypot(xy[0] - self.truth[0], xy[1] - self.truth[1])
                    err = f' | truth=({self.truth[0]:.2f},{self.truth[1]:.2f}) ERROR={e:.2f} m'
                self.get_logger().info(
                    f'FIX from {len(fixes)} marker(s): ({xy[0]:.2f},{xy[1]:.2f}){err} '
                    f'| blur={blur:.0f} feats={nfeat}')
        elif self._n % 15 == 0:
            self.get_logger().info(f'no mapped markers in view | blur={blur:.0f} feats={nfeat}')


def main():
    rclpy.init()
    rclpy.spin(Perception())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
