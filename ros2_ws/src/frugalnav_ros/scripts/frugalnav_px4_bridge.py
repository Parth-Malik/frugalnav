#!/usr/bin/env python3
"""
PX4 SITL bridge -- lets the SAME navigator fly a PX4 vehicle.

The navigator consumes generic topics (/frugalnav/truth for the reference pose and
/frugalnav/vio for measured velocity). In the Gazebo demo those come from our world and
optical-flow node. Against PX4 SITL they come from the autopilot's own estimator (EKF2):
this node republishes /fmu/out/vehicle_local_position into exactly those topics, so the
DWA navigator runs unchanged with platform:=px4.

Frames: PX4 is NED (x=north, y=east, z=down); our world is ENU-style (x=east, y=north).
So world.x = ned.y (east) and world.y = ned.x (north), and likewise for velocity.
"""
import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleLocalPosition


class Px4Bridge(Node):
    def __init__(self):
        super().__init__('frugalnav_px4_bridge')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.truth = self.create_publisher(Odometry, '/frugalnav/truth', 10)
        self.vio = self.create_publisher(Twist, '/frugalnav/vio', 10)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position',
                                 self.on_pos, qos)
        self.get_logger().info('PX4 bridge up: /fmu vehicle_local_position -> /frugalnav/truth,/vio')

    def on_pos(self, m):
        od = Odometry(); od.header.frame_id = 'world'
        od.pose.pose.position.x = float(m.y)     # east  -> world x
        od.pose.pose.position.y = float(m.x)     # north -> world y
        od.pose.pose.position.z = float(-m.z)    # down  -> up
        od.pose.pose.orientation.w = 1.0
        self.truth.publish(od)
        v = Twist(); v.linear.x = float(m.vy); v.linear.y = float(m.vx)
        self.vio.publish(v)


def main():
    rclpy.init(); rclpy.spin(Px4Bridge()); rclpy.shutdown()


if __name__ == '__main__':
    main()
