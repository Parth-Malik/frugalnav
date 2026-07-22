"""FrugalNav REAL perception demo -- the drone flies on real vision.

Wires the whole real loop, all separate from the sandbox demos:
  Gazebo camera drone over an ArUco-textured world
    -> perception (real cv2.aruco detect + solvePnP + blur/feature cues)
    -> blind navigator (scheduler + fusion + control; wind ESTIMATED, not known)
    -> external wind node (adds unknown wind) -> the drone
  visualized in RViz.

    ros2 launch frugalnav_ros real_demo.launch.py            # headless-ok
    ros2 launch frugalnav_ros real_demo.launch.py gui:=true  # watch Gazebo
Fly / switch modes / change wind from a second terminal:
    ros2 run frugalnav_ros frugalnav_teleop.py
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            TimerAction, SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('frugalnav_ros')
    gazebo_ros = get_package_share_directory('gazebo_ros')
    world = os.path.join(pkg, 'worlds', 'real.world')
    model = os.path.join(pkg, 'models', 'frugalnav_drone_cam', 'model.sdf')
    scene = os.path.join(pkg, 'config', 'real_scene.txt')
    media = os.path.join(pkg, 'media')
    rviz_cfg = os.path.join(pkg, 'rviz', 'frugalnav.rviz')
    res = os.environ.get('GAZEBO_RESOURCE_PATH', '')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros, 'launch', 'gazebo.launch.py')),
        launch_arguments={'world': world, 'gui': LaunchConfiguration('gui'),
                          'verbose': 'true'}.items())
    spawn = Node(package='gazebo_ros', executable='spawn_entity.py',
                 arguments=['-entity', 'frugalnav_drone', '-file', model,
                            '-x', '28', '-y', '10', '-z', '5'], output='screen')
    perception = Node(package='frugalnav_ros', executable='frugalnav_perception.py',
                      name='frugalnav_perception', output='screen',
                      parameters=[{'scene_file': scene}])
    nav = Node(package='frugalnav_ros', executable='frugalnav_real_node.py',
               name='frugalnav_real_node', output='screen',
               parameters=[{'scene_file': scene}])
    wind = Node(package='frugalnav_ros', executable='frugalnav_wind.py',
                name='frugalnav_wind', output='screen')
    rviz = Node(package='rviz2', executable='rviz2', name='rviz2',
                arguments=['-d', rviz_cfg], output='screen',
                condition=IfCondition(LaunchConfiguration('rviz')))

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='true'),
        SetEnvironmentVariable('GAZEBO_RESOURCE_PATH',
                               media + ':/usr/share/gazebo-11:' + res),
        gazebo,
        TimerAction(period=4.0, actions=[spawn]),
        # give Gazebo + the camera a moment before the perception/nav/wind nodes
        TimerAction(period=8.0, actions=[perception, nav, wind]),
        rviz,
    ])
