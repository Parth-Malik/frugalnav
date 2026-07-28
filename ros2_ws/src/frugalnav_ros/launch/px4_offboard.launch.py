"""FrugalNav on PX4 SITL (complete simulation of the deployment path).

PX4 SITL provides the vehicle + EKF2 state; the uXRCE-DDS agent bridges the /fmu topics.
This launch adds the FrugalNav side:

    px4 bridge  : /fmu vehicle_local_position -> /frugalnav/truth + /frugalnav/vio
    DWA nav     : the SAME stronger navigator, platform:=px4 -> streams offboard setpoints,
                  arms + engages OFFBOARD, flies to the scene target.

Run PX4 SITL and the agent first (see tools/px4_sitl_bringup.sh), then:
    ros2 launch frugalnav_ros px4_offboard.launch.py
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg = get_package_share_directory('frugalnav_ros')
    scene = os.path.join(pkg, 'config', 'px4_scene.txt')
    return LaunchDescription([
        DeclareLaunchArgument('scene_file', default_value=scene),
        Node(package='frugalnav_ros', executable='frugalnav_px4_bridge.py',
             name='frugalnav_px4_bridge', output='screen'),
        Node(package='frugalnav_ros', executable='frugalnav_dwa_node.py',
             name='frugalnav_dwa_node', output='screen',
             parameters=[{'scene_file': LaunchConfiguration('scene_file'), 'platform': 'px4'}]),
    ])
