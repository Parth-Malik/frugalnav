#!/usr/bin/env python3
"""
FrugalNav keyboard teleop / mission control. Run in its own terminal; keypresses
fly the drone and switch modes on the interactive node.

    ros2 run frugalnav_ros frugalnav_teleop.py

Publishes:
    /frugalnav/teleop  (geometry_msgs/Twist)  -- manual velocity (world XY)
    /frugalnav/ctrl    (std_msgs/String)      -- auto|manual|euroc|reset|pause|resume
"""
import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

SPEED = 2.0

BANNER = """
============================================================
  FrugalNav  --  keyboard mission control
============================================================
  FLY (MANUAL mode):   W = north(+Y)   S = south(-Y)
                       A = west(-X)    D = east(+X)
                       SPACE / K = stop

  MODES:   1 = AUTO    (scheduler flies to the target)
           2 = MANUAL  (you fly with WASD)
           3 = EUROC   (replay the real EuRoC MH_01 flight)

  R = RESET  (teleport drone back to start = "rewind")
  P = PAUSE / RESUME       G = weather ON/OFF (wind + fog)
  Q or Ctrl-C = quit
------------------------------------------------------------
  Tip: press 2 first to take manual control, then fly with WASD.
============================================================
"""


class Teleop(Node):
    def __init__(self):
        super().__init__('frugalnav_teleop')
        self.cmd = self.create_publisher(Twist, '/frugalnav/teleop', 10)
        self.ctrl = self.create_publisher(String, '/frugalnav/ctrl', 10)
        self.paused = False

    def vel(self, x, y):
        t = Twist(); t.linear.x = float(x); t.linear.y = float(y); self.cmd.publish(t)

    def say(self, s):
        m = String(); m.data = s; self.ctrl.publish(m)


def get_key(settings, timeout=0.1):
    tty.setraw(sys.stdin.fileno())
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    key = sys.stdin.read(1) if r else ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def main():
    rclpy.init()
    node = Teleop()
    settings = termios.tcgetattr(sys.stdin)
    sys.stdout.write(BANNER)
    sys.stdout.flush()
    try:
        while rclpy.ok():
            k = get_key(settings)
            if k:
                k = k.lower()
                if k in ('\x03', 'q'):
                    break
                elif k == 'w': node.vel(0, SPEED)
                elif k == 's': node.vel(0, -SPEED)
                elif k == 'a': node.vel(-SPEED, 0)
                elif k == 'd': node.vel(SPEED, 0)
                elif k in (' ', 'k'): node.vel(0, 0)
                elif k == '1': node.say('auto'); print(' -> AUTO\r')
                elif k == '2': node.say('manual'); print(' -> MANUAL (fly with WASD)\r')
                elif k == '3': node.say('euroc'); print(' -> EUROC replay\r')
                elif k == 'r': node.say('reset'); print(' -> RESET (drone back to start)\r')
                elif k == 'p':
                    node.paused = not node.paused
                    node.say('pause' if node.paused else 'resume')
                    print(' -> PAUSED\r' if node.paused else ' -> RESUMED\r')
                elif k == 'g': node.say('weather'); print(' -> toggled WEATHER (wind + fog)\r')
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.vel(0, 0)
        rclpy.shutdown()


if __name__ == '__main__':
    main()
