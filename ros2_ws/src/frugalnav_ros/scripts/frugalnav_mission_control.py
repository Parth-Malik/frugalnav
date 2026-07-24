#!/usr/bin/env python3
"""
Control panel for the camera demos: play/pause, reset, place the drone before
starting, manual flight, wind and weather. Run in its own terminal:

    ros2 run frugalnav_ros frugalnav_mission_control.py

Pair with a demo launched held, so it waits for PLAY:

    ros2 launch frugalnav_ros real_demo.launch.py map:=real gui:=true start_paused:=true

Publishes /frugalnav/ctrl (String commands) and /frugalnav/teleop (manual velocity).
The weather keys change the actual wind node, not the navigator's knowledge of it.
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
  FrugalNav  --  REAL demo mission control
================================================================
  RUN:        SPACE  play / pause (toggle)     R  reset (home + hold)
  PLACE:      I north   K south   J west   L east   (move start, while held)
  MODE:       1 AUTO (fly to target)    2 MANUAL (fly it yourself)
  FLY:        W A S D  move    X  stop        (MANUAL mode)
  WEATHER:    ] wind+    [ wind-     T rain/gust on-off    G weather on-off
  QUIT:       Q
----------------------------------------------------------------
  Starts HELD. Place the drone with I/J/K/L, set weather, then SPACE to fly.
  Watch RViz (path + wind arrow) and the camera feed window.
================================================================
"""


class Mission(Node):
    def __init__(self):
        super().__init__('frugalnav_mission_control')
        self.ctrl = self.create_publisher(String, '/frugalnav/ctrl', 10)
        self.fly = self.create_publisher(Twist, '/frugalnav/teleop', 10)
        self.playing = False

    def say(self, s):
        m = String(); m.data = s; self.ctrl.publish(m)

    def vel(self, x, y):
        t = Twist(); t.linear.x = float(x); t.linear.y = float(y); self.fly.publish(t)


def get_key(settings, timeout=0.1):
    tty.setraw(sys.stdin.fileno())
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    key = sys.stdin.read(1) if r else ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


# single-key -> (ctrl string, on-screen label); movement/fly handled separately
SAY = {
    'r': ('reset', 'RESET (home + hold)'),
    'i': ('move_north', 'place NORTH'), 'k': ('move_south', 'place SOUTH'),
    'j': ('move_west', 'place WEST'), 'l': ('move_east', 'place EAST'),
    '1': ('auto', 'AUTO'), '2': ('manual', 'MANUAL (fly with WASD)'),
    ']': ('wind_up', 'wind +'), '[': ('wind_down', 'wind -'),
    't': ('rain', 'rain / gust toggle'), 'g': ('weather', 'weather on/off'),
}


def main():
    rclpy.init()
    node = Mission()
    settings = termios.tcgetattr(sys.stdin)
    sys.stdout.write(BANNER); sys.stdout.flush()
    node.say('pause')                       # make sure the drone is held on startup
    try:
        while rclpy.ok():
            k = get_key(settings)
            if k:
                if k in ('\x03', 'q', 'Q'):
                    break
                kl = k.lower()
                if k == ' ' or kl == 'p':
                    node.playing = not node.playing
                    node.say('play' if node.playing else 'pause')
                    print(f" -> {'PLAY' if node.playing else 'PAUSE'}\r")
                elif kl == 'w': node.vel(0, SPEED)
                elif kl == 's': node.vel(0, -SPEED)
                elif kl == 'a': node.vel(-SPEED, 0)
                elif kl == 'd': node.vel(SPEED, 0)
                elif kl == 'x': node.vel(0, 0)
                elif k in SAY or kl in SAY:
                    cmd, label = SAY.get(k, SAY.get(kl))
                    if cmd == 'reset':
                        node.playing = False
                    node.say(cmd); print(f' -> {label}\r')
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.vel(0, 0); node.say('pause')
        rclpy.shutdown()


if __name__ == '__main__':
    main()
