"""FrugalNav on the EuRoC MH_01 dataset, in RViz.

Replays the real ground-truth trajectory and runs the uncertainty scheduler over
it, showing truth (green) vs pure-VIO drift (red) vs uncertainty-aware (cyan).

    ros2 launch frugalnav_ros euroc_demo.launch.py
    ros2 launch frugalnav_ros euroc_demo.launch.py gt_csv:=/path/to/data.csv rate_hz:=90.0
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_EUROC_REL = 'datasets/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv'


def _default_csv():
    """EuRoC ground truth: $FRUGALNAV_EUROC, else <repo>/datasets/... if we can find it."""
    env = os.environ.get('FRUGALNAV_EUROC')
    if env:
        return env
    root = os.environ.get('FRUGALNAV_ROOT')
    if not root:
        d = os.path.dirname(os.path.realpath(__file__))
        for _ in range(8):
            if os.path.isdir(os.path.join(d, 'core')):
                root = d
                break
            d = os.path.dirname(d)
    return os.path.join(root, _EUROC_REL) if root else _EUROC_REL


DEFAULT_CSV = _default_csv()


def generate_launch_description():
    pkg = get_package_share_directory('frugalnav_ros')
    rviz_cfg = os.path.join(pkg, 'rviz', 'frugalnav.rviz')

    node = Node(package='frugalnav_ros', executable='frugalnav_euroc_node',
                name='frugalnav_euroc_node', output='screen',
                parameters=[{'gt_csv': LaunchConfiguration('gt_csv'),
                             'rate_hz': LaunchConfiguration('rate_hz'),
                             'stride': 10, 'fix_every_m': 1.5}])

    rviz_node = Node(package='rviz2', executable='rviz2', name='rviz2',
                     arguments=['-d', rviz_cfg], output='screen',
                     condition=IfCondition(LaunchConfiguration('rviz')))

    return LaunchDescription([
        DeclareLaunchArgument('gt_csv', default_value=DEFAULT_CSV),
        DeclareLaunchArgument('rate_hz', default_value='60.0'),
        DeclareLaunchArgument('rviz', default_value='true'),
        node, rviz_node,
    ])
