#!/usr/bin/env python3
"""
FrugalNav external WIND / environment node -- the disturbance the navigator cannot
see. It sits between the nav and the drone:

    /frugalnav/nav_cmd  --(+ wind)-->  /frugalnav/cmd_vel  --> the drone

The nav publishes what it WANTS; this node adds a real gusting wind and forwards the
sum to the drone. The nav never subscribes to the wind here -- it can only infer it
by comparing the velocity it commanded to the velocity it actually achieved. Wind is
user-adjustable from the teleop (] / [ stronger/weaker, T gust on/off, G on/off);
the true wind is published on /frugalnav/wind_true for the RViz comparison only.
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String


class Wind(Node):
    def __init__(self):
        super().__init__('frugalnav_wind')
        self.speed = float(self.declare_parameter('wind', 0.8).value)
        self.direction = 2.4
        self.gust = True
        self.on = True
        self.t = 0.0
        self.last_nav = np.zeros(2)
        self.wind = np.zeros(2)
        self.cmd_pub = self.create_publisher(Twist, '/frugalnav/cmd_vel', 10)
        self.wind_pub = self.create_publisher(Vector3, '/frugalnav/wind_true', 10)
        self.create_subscription(Twist, '/frugalnav/nav_cmd', self.on_nav, 20)
        self.create_subscription(String, '/frugalnav/ctrl', self.on_ctrl, 10)
        self.create_timer(0.05, self.tick)
        self.get_logger().info(f'external wind node up: base {self.speed:.1f} m/s (nav cannot see it)')

    def on_nav(self, m):
        self.last_nav = np.array([m.linear.x, m.linear.y])

    def on_ctrl(self, c):
        d = c.data
        if d == 'wind_up': self.speed = min(6.0, self.speed + 0.4)
        elif d == 'wind_down': self.speed = max(0.0, self.speed - 0.4)
        elif d == 'rain' or d == 'gust': self.gust = not self.gust
        elif d == 'weather': self.on = not self.on

    def tick(self):
        self.t += 0.05
        if self.on:
            g = (0.7 + 0.3 * math.sin(self.t * 0.7)) if self.gust else 1.0
            sp = self.speed * g + (0.15 * math.sin(self.t * 2.3) if self.gust else 0.0)
            dirn = self.direction + 0.3 * math.sin(self.t * 0.25)
            self.wind = np.array([sp * math.cos(dirn), sp * math.sin(dirn)])
        else:
            self.wind = np.zeros(2)
        out = Twist()
        out.linear.x = float(self.last_nav[0] + self.wind[0])   # nav command + wind -> drone
        out.linear.y = float(self.last_nav[1] + self.wind[1])
        self.cmd_pub.publish(out)
        v = Vector3(); v.x, v.y = float(self.wind[0]), float(self.wind[1]); self.wind_pub.publish(v)


def main():
    rclpy.init(); rclpy.spin(Wind()); rclpy.shutdown()


if __name__ == '__main__':
    main()
