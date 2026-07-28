#!/bin/bash
# ---------------------------------------------------------------------------
# FrugalNav on PX4 SITL -- one-command bringup of the complete simulation.
#
# Stands up, in order:
#   1. PX4-Autopilot SITL (the flight stack + a Gazebo vehicle)   -> port 8888 uXRCE
#   2. the Micro-XRCE-DDS Agent (bridges PX4 <-> ROS 2 /fmu topics)
#   3. the FrugalNav px4 launch (bridge + DWA navigator, platform:=px4)
#
# First run clones + builds PX4 and the agent (heavy, 15-30 min, needs network).
# Re-runs skip straight to launching. Everything is installed under ~/frugalnav_px4.
#
#   bash tools/px4_sitl_bringup.sh            # gz (modern gazebo) x500 quad
#   HEADLESS=1 bash tools/px4_sitl_bringup.sh # no gazebo GUI
# ---------------------------------------------------------------------------
set -e
ROOT="${PX4_ROOT:-$HOME/frugalnav_px4}"
WS="${FRUGALNAV:-/mnt/c/Users/parth/Downloads/drone}/ros2_ws"
mkdir -p "$ROOT"

echo "== [1/4] PX4-Autopilot =="
if [ ! -d "$ROOT/PX4-Autopilot" ]; then
  git clone --recursive -b v1.14.0 https://github.com/PX4/PX4-Autopilot.git "$ROOT/PX4-Autopilot"
  bash "$ROOT/PX4-Autopilot/Tools/setup/ubuntu.sh" --no-nuttx    # host SITL deps only
fi

echo "== [2/4] Micro-XRCE-DDS Agent =="
if ! command -v MicroXRCEAgent >/dev/null 2>&1; then
  if [ ! -d "$ROOT/Micro-XRCE-DDS-Agent" ]; then
    git clone -b v2.4.2 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git "$ROOT/Micro-XRCE-DDS-Agent"
  fi
  ( cd "$ROOT/Micro-XRCE-DDS-Agent" && mkdir -p build && cd build \
      && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j"$(nproc)" && sudo make install && sudo ldconfig )
fi

echo "== [3/4] start SITL + agent =="
MicroXRCEAgent udp4 -p 8888 >/tmp/xrce_agent.log 2>&1 &
echo "  agent pid $! (log /tmp/xrce_agent.log)"
export PX4_GZ_STANDALONE=0
( cd "$ROOT/PX4-Autopilot" && \
  if [ "${HEADLESS:-0}" = "1" ]; then HEADLESS=1 make px4_sitl gz_x500 >/tmp/px4_sitl.log 2>&1 & \
  else make px4_sitl gz_x500 >/tmp/px4_sitl.log 2>&1 & fi )
echo "  PX4 SITL starting (log /tmp/px4_sitl.log); waiting 25 s for it to come up..."
sleep 25

echo "== [4/4] FrugalNav px4 stack =="
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
echo "  launching bridge + DWA navigator (platform:=px4). Ctrl-C to stop everything."
ros2 launch frugalnav_ros px4_offboard.launch.py

# on exit, tear down the background sims
trap 'pkill -f MicroXRCEAgent; pkill -f px4; pkill -f "gz sim"' EXIT
