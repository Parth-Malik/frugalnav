"""FrugalNav Gazebo + RViz demo.

Starts Gazebo with the FrugalNav world, spawns the holonomic drone, launches the
C++ FrugalNav node (which flies the drone by driving /frugalnav/cmd_vel from the
real uncertainty scheduler), and opens RViz.

    ros2 launch frugalnav_ros gazebo_demo.launch.py            # full GUI
    ros2 launch frugalnav_ros gazebo_demo.launch.py gui:=false rviz:=false   # headless
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            TimerAction, GroupAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('frugalnav_ros')
    gazebo_ros = get_package_share_directory('gazebo_ros')
    world = os.path.join(pkg, 'worlds', 'frugalnav.world')
    model = os.path.join(pkg, 'models', 'frugalnav_drone', 'model.sdf')
    rviz_cfg = os.path.join(pkg, 'rviz', 'frugalnav.rviz')

    gui = LaunchConfiguration('gui')
    rviz = LaunchConfiguration('rviz')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros, 'launch', 'gazebo.launch.py')),
        launch_arguments={'world': world, 'gui': gui, 'verbose': 'true'}.items())

    spawn = Node(package='gazebo_ros', executable='spawn_entity.py',
                 arguments=['-entity', 'frugalnav_drone', '-file', model,
                            '-x', '58', '-y', '24', '-z', '0.5'],
                 output='screen')

    brain = Node(package='frugalnav_ros', executable='frugalnav_gazebo_node',
                 name='frugalnav_gazebo_node', output='screen')

    rviz_node = Node(package='rviz2', executable='rviz2', name='rviz2',
                     arguments=['-d', rviz_cfg], output='screen',
                     condition=IfCondition(rviz))

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true', description='Gazebo GUI'),
        DeclareLaunchArgument('rviz', default_value='true', description='launch RViz'),
        gazebo,
        TimerAction(period=4.0, actions=[spawn]),        # after Gazebo is up
        TimerAction(period=7.0, actions=[brain]),        # after the drone exists
        GroupAction([rviz_node]),
    ])
