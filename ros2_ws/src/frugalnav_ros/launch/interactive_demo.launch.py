"""FrugalNav interactive Gazebo + RViz demo (arena world).

Starts the arena, spawns the drone, launches the interactive multi-mode node, and
opens RViz. Fly / switch modes / reset from a SECOND terminal:

    ros2 launch frugalnav_ros interactive_demo.launch.py     # terminal 1 (this)
    ros2 run   frugalnav_ros frugalnav_teleop.py             # terminal 2 (keyboard)

Headless (no GUI):  ... interactive_demo.launch.py gui:=false rviz:=false
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('frugalnav_ros')
    gazebo_ros = get_package_share_directory('gazebo_ros')
    world = os.path.join(pkg, 'worlds', 'frugalnav_arena.world')
    model = os.path.join(pkg, 'models', 'frugalnav_drone', 'model.sdf')
    rviz_cfg = os.path.join(pkg, 'rviz', 'frugalnav.rviz')

    gui = LaunchConfiguration('gui')
    rviz = LaunchConfiguration('rviz')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros, 'launch', 'gazebo.launch.py')),
        launch_arguments={'world': world, 'gui': gui, 'verbose': 'true'}.items())

    spawn = Node(package='gazebo_ros', executable='spawn_entity.py',
                 arguments=['-entity', 'frugalnav_drone', '-file', model,
                            '-x', '58', '-y', '24', '-z', '0.5'], output='screen')

    brain = Node(package='frugalnav_ros', executable='frugalnav_interactive_node',
                 name='frugalnav_interactive_node', output='screen')

    rviz_node = Node(package='rviz2', executable='rviz2', name='rviz2',
                     arguments=['-d', rviz_cfg], output='screen',
                     condition=IfCondition(rviz))

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
        gazebo,
        TimerAction(period=4.0, actions=[spawn]),
        TimerAction(period=7.0, actions=[brain]),
        rviz_node,
    ])
