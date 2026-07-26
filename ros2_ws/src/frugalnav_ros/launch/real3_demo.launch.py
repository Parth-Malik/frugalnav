"""Demo 3: drone with a downward camera, a forward camera and a 360 deg laser.

    gazebo -> perception (ArUco fixes + image cues)
           -> vio        (velocity from optical flow)
           -> real3 nav  (obstacles from the laser, wind inferred)
           -> wind       (adds the disturbance) -> drone

    ros2 launch frugalnav_ros real3_demo.launch.py
    ros2 launch frugalnav_ros real3_demo.launch.py map:=real_dense gui:=true
    ros2 launch frugalnav_ros real3_demo.launch.py start_paused:=true
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            TimerAction, SetEnvironmentVariable, OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

START = {'real': ('28', '10', '5'), 'real_dense': ('46', '15', '5'),
         'city': ('64', '22', '5')}


def _setup(context, *args, **kwargs):
    pkg = get_package_share_directory('frugalnav_ros')
    gazebo_ros = get_package_share_directory('gazebo_ros')
    mp = LaunchConfiguration('map').perform(context)
    if mp not in START:
        mp = 'real'
    world = os.path.join(pkg, 'worlds', f'{mp}.world')
    scene = os.path.join(pkg, 'config', f'{mp}_scene.txt')
    model = os.path.join(pkg, 'models', 'frugalnav_drone_real3', 'model.sdf')
    rviz_cfg = os.path.join(pkg, 'rviz', 'frugalnav.rviz')
    sx, sy, sz = START[mp]
    sp = ParameterValue(LaunchConfiguration('start_paused'), value_type=bool)

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros, 'launch', 'gazebo.launch.py')),
        launch_arguments={'world': world, 'gui': LaunchConfiguration('gui'),
                          'verbose': 'true'}.items())
    spawn = Node(package='gazebo_ros', executable='spawn_entity.py',
                 arguments=['-entity', 'frugalnav_drone', '-file', model,
                            '-x', sx, '-y', sy, '-z', sz], output='screen')
    # Run on Gazebo's /clock so the VIO's frame-to-frame dt is consistent even when a
    # loaded WSL session slows the sim (wall-clock dt spikes are what corrupt the flow
    # velocity and diverge the estimate). This is launch config only -- no node code changes.
    SIM = {'use_sim_time': True}
    perception = Node(package='frugalnav_ros', executable='frugalnav_perception.py',
                      name='frugalnav_perception', output='screen',
                      parameters=[{'scene_file': scene}, SIM])
    vio = Node(package='frugalnav_ros', executable='frugalnav_vio.py',
               name='frugalnav_vio', output='screen', parameters=[SIM])
    nav = Node(package='frugalnav_ros', executable='frugalnav_real3_node.py',
               name='frugalnav_real3_node', output='screen',
               parameters=[{'scene_file': scene, 'start_paused': sp}, SIM])
    front = Node(package='frugalnav_ros', executable='frugalnav_front_view.py',
                 name='frugalnav_front_view', output='screen', parameters=[SIM])
    wind = Node(package='frugalnav_ros', executable='frugalnav_wind.py',
                name='frugalnav_wind', output='screen', parameters=[{'start_paused': sp}, SIM])
    rviz = Node(package='rviz2', executable='rviz2', name='rviz2',
                arguments=['-d', rviz_cfg], output='screen',
                condition=IfCondition(LaunchConfiguration('rviz')))

    return [gazebo,
            TimerAction(period=4.0, actions=[spawn]),
            TimerAction(period=8.0, actions=[perception, vio, nav, front, wind]),
            rviz]


def generate_launch_description():
    pkg = get_package_share_directory('frugalnav_ros')
    media = os.path.join(pkg, 'media')
    res = os.environ.get('GAZEBO_RESOURCE_PATH', '')
    return LaunchDescription([
        DeclareLaunchArgument('map', default_value='real',
                              description='real | real_dense | city'),
        DeclareLaunchArgument('gui', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('start_paused', default_value='false',
                              description='hold the drone until mission control presses PLAY'),
        SetEnvironmentVariable('GAZEBO_RESOURCE_PATH',
                               media + ':/usr/share/gazebo-11:' + res),
        OpaqueFunction(function=_setup),
    ])
