#!/usr/bin/env python3
"""
Forward camera view with the laser proximity drawn on top, so you can watch the drone
react to obstacles ahead. Publishes /frugalnav/front_cam/annotated:

    ros2 run rqt_image_view rqt_image_view /frugalnav/front_cam/annotated

This is display only -- the overlay comes from a few laser ranges and the navigator
never reads this image.
"""
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge


class FrontView(Node):
    def __init__(self):
        super().__init__('frugalnav_front_view')
        self.bridge = CvBridge()
        self.ahead = None
        self.pub = self.create_publisher(Image, '/frugalnav/front_cam/annotated', 5)
        self.create_subscription(LaserScan, '/frugalnav/scan', self.on_scan, 10)
        self.create_subscription(Image, '/frugalnav/front_cam/image_raw', self.on_img, 5)

    def on_scan(self, m):
        r = np.asarray(m.ranges, float)
        good = np.isfinite(r) & (r > 0.05) & (r < m.range_max)
        idx = np.where(good)[0]
        if len(idx):
            ang = m.angle_min + idx * m.angle_increment
            rel = np.arctan2(np.sin(ang - np.pi), np.cos(ang - np.pi))   # camera faces -X
            fwd = np.abs(rel) < 0.7
            self.ahead = float(r[idx][fwd].min()) if fwd.any() else None
        else:
            self.ahead = None

    def on_img(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w = img.shape[:2]
        cv2.putText(img, 'FRONT / APPROACH CAM', (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
        a = self.ahead
        if a is not None and a < 6.0:
            close = a < 3.0
            col = (40, 40, 255) if close else (40, 200, 255)
            cv2.putText(img, f'OBSTACLE AHEAD: {a:.1f} m', (10, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            if close:
                cv2.rectangle(img, (6, 6), (w - 6, h - 6), (40, 40, 255), 6)
                cv2.putText(img, 'AVOIDING', (w // 2 - 80, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (40, 40, 255), 3)
        else:
            cv2.putText(img, 'PATH CLEAR', (10, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 220, 60), 2)
        self.pub.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))


def main():
    rclpy.init(); rclpy.spin(FrontView()); rclpy.shutdown()


if __name__ == '__main__':
    main()
