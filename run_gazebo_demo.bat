@echo off
REM ============================================================
REM  FrugalNav - Gazebo + RViz demo (double-click to run)
REM  Opens Gazebo and RViz on your Windows desktop via WSLg.
REM  The drone homes to the target using the uncertainty scheduler.
REM  Close the Gazebo/RViz windows, or press Ctrl+C here, to stop.
REM ============================================================
echo.
echo  Launching FrugalNav Gazebo + RViz demo in WSL...
echo  (First launch can take 20-40s while Gazebo + RViz start. Be patient.)
echo.
wsl -d Ubuntu-22.04 -e bash -c "source /opt/ros/humble/setup.bash && source /mnt/c/Users/parth/Downloads/drone/ros2_ws/install/setup.bash && ros2 launch frugalnav_ros gazebo_demo.launch.py"
echo.
echo  Demo stopped.
pause
