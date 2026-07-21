#!/usr/bin/env python3
"""
FrugalNav keyboard mission control + weather simulator. Run in its own terminal.

    ros2 run frugalnav_ros frugalnav_teleop.py

Publishes:
    /frugalnav/teleop  (geometry_msgs/Twist)  -- manual velocity (world XY)
    /frugalnav/ctrl    (std_msgs/String)      -- modes, reset/pause, altitude, weather
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
================================================================
  FrugalNav  --  mission control + weather simulator
================================================================
  FLY (manual):   W north   S south   A west   D east   SPACE stop
  ALTITUDE:       U up       N down     M = auto (from visibility)
  MODES:          1 AUTO     2 MANUAL   3 EUROC replay
  WEATHER:        ] wind+    [ wind-      (gusting disturbance)
                  - fog+     = clearer    (visibility)
                  T rain on/off           G weather master on/off
  SIM:            R RESET (rewind)   P pause/resume   Q quit
----------------------------------------------------------------
  Press 2 to take manual control, then fly with WASD.
  Watch the HUD (top of RViz) for mode / wind / vis / rain / alt.
================================================================
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


KEYMAP = {
    '1': ('auto', 'AUTO'), '2': ('manual', 'MANUAL (fly with WASD)'), '3': ('euroc', 'EUROC replay'),
    'r': ('reset', 'RESET (rewind to start)'),
    'u': ('alt_up', 'altitude UP'), 'n': ('alt_down', 'altitude DOWN'), 'm': ('alt_auto', 'altitude AUTO'),
    ']': ('wind_up', 'wind +'), '[': ('wind_down', 'wind -'),
    '-': ('fog_up', 'more fog'), '=': ('fog_down', 'clearer'),
    't': ('rain', 'toggle rain'), 'g': ('weather', 'toggle weather master'),
}


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
    sys.stdout.write(BANNER); sys.stdout.flush()
    try:
        while rclpy.ok():
            k = get_key(settings)
            if k:
                if k in ('\x03', 'q', 'Q'):
                    break
                kl = k.lower()
                if kl == 'w': node.vel(0, SPEED)
                elif kl == 's': node.vel(0, -SPEED)
                elif kl == 'a': node.vel(-SPEED, 0)
                elif kl == 'd': node.vel(SPEED, 0)
                elif k in (' ', 'k'): node.vel(0, 0)
                elif kl == 'p':
                    node.paused = not node.paused
                    node.say('pause' if node.paused else 'resume')
                    print(' -> PAUSED\r' if node.paused else ' -> RESUMED\r')
                elif k in KEYMAP or kl in KEYMAP:
                    cmd, label = KEYMAP.get(k, KEYMAP.get(kl))
                    node.say(cmd); print(f' -> {label}\r')
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.vel(0, 0)
        rclpy.shutdown()


if __name__ == '__main__':
    main()
