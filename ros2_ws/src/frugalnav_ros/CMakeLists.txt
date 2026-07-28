cmake_minimum_required(VERSION 3.8)
project(frugalnav_ros)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)   # ROS package (the cpp/ RISC-V port stays C++14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

set(DEPS rclcpp std_msgs geometry_msgs nav_msgs visualization_msgs
         tf2 tf2_ros tf2_geometry_msgs)

# The FrugalNav portable core (header-only, vendored from cpp/frugalnav/).
include_directories(include)

add_executable(frugalnav_gazebo_node src/frugalnav_gazebo_node.cpp)
ament_target_dependencies(frugalnav_gazebo_node ${DEPS})

add_executable(frugalnav_euroc_node src/frugalnav_euroc_node.cpp)
ament_target_dependencies(frugalnav_euroc_node ${DEPS})

find_package(gazebo_msgs REQUIRED)
add_executable(frugalnav_interactive_node src/frugalnav_interactive_node.cpp)
ament_target_dependencies(frugalnav_interactive_node ${DEPS} gazebo_msgs)

install(TARGETS frugalnav_gazebo_node frugalnav_euroc_node frugalnav_interactive_node
        DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS scripts/frugalnav_teleop.py scripts/frugalnav_perception.py
                scripts/frugalnav_real_node.py scripts/frugalnav_wind.py
                scripts/frugalnav_mission_control.py scripts/frugalnav_vio.py
                scripts/frugalnav_real3_node.py scripts/frugalnav_front_view.py
                scripts/frugalnav_platform.py scripts/frugalnav_dwa_node.py
                scripts/frugalnav_px4_bridge.py
        DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch worlds models rviz config media
        DESTINATION share/${PROJECT_NAME})

ament_package()
