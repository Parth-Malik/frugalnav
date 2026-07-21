"""FrugalNav interactive Gazebo + RViz demo. Pick a map with map:=demo|canopy.

    ros2 launch frugalnav_ros interactive_demo.launch.py               # demo map
    ros2 launch frugalnav_ros interactive_demo.launch.py map:=canopy   # dense forest

Fly / switch modes / set weather from a SECOND terminal:
    ros2 run frugalnav_ros frugalnav_teleop.py
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, TimerAction,
                            OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context, *args, **kwargs):
    pkg = get_package_share_directory('frugalnav_ros')
    gazebo_ros = get_package_share_directory('gazebo_ros')
    mp = LaunchConfiguration('map').perform(context)
    world = os.path.join(pkg, 'worlds', f'{mp}.world')
    scene = os.path.join(pkg, 'config', f'{mp}_scene.txt')
    model = os.path.join(pkg, 'models', 'frugalnav_drone', 'model.sdf')
    rviz_cfg = os.path.join(pkg, 'rviz', 'frugalnav.rviz')

    # start position per map (drone spawns on the start pad)
    sx, sy = ('58', '24') if mp == 'demo' else ('56', '15')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros, 'launch', 'gazebo.launch.py')),
        launch_arguments={'world': world, 'gui': LaunchConfiguration('gui'),
                          'verbose': 'true'}.items())
    spawn = Node(package='gazebo_ros', executable='spawn_entity.py',
                 arguments=['-entity', 'frugalnav_drone', '-file', model,
                            '-x', sx, '-y', sy, '-z', '5.0'], output='screen')
    brain = Node(package='frugalnav_ros', executable='frugalnav_interactive_node',
                 name='frugalnav_interactive_node', output='screen',
                 parameters=[{'scene_file': scene}])
    rviz = Node(package='rviz2', executable='rviz2', name='rviz2',
                arguments=['-d', rviz_cfg], output='screen',
                condition=IfCondition(LaunchConfiguration('rviz')))
    return [gazebo, TimerAction(period=4.0, actions=[spawn]),
            TimerAction(period=7.0, actions=[brain]), rviz]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map', default_value='demo', description='demo | canopy'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
        OpaqueFunction(function=_setup),
    ])
