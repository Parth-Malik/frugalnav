@echo off
REM ============================================================
REM  FrugalNav - INTERACTIVE Gazebo + RViz demo (double-click)
REM  Opens the sim in its own window, and keyboard control HERE.
REM
REM  Press 2 to take manual control, then fly with W A S D.
REM  1=auto  2=manual  3=euroc  R=reset(rewind)  P=pause  Q=quit
REM ============================================================
set ROSENV=source /opt/ros/humble/setup.bash ^&^& source /mnt/c/Users/parth/Downloads/drone/ros2_ws/install/setup.bash
echo.
echo  Starting FrugalNav interactive sim (Gazebo + RViz) in a new window...
start "FrugalNav Sim" wsl -d Ubuntu-22.04 -e bash -c "%ROSENV% && ros2 launch frugalnav_ros interactive_demo.launch.py"
echo  Waiting ~18s for Gazebo + RViz to come up...
timeout /t 18 /nobreak >nul
echo.
echo  === KEYBOARD CONTROL (this window) ===
echo  Press 2 to take manual control, then fly with W A S D.
echo.
wsl -d Ubuntu-22.04 -e bash -c "%ROSENV% && ros2 run frugalnav_ros frugalnav_teleop.py"
echo.
echo  Control stopped. Close the sim window to fully exit.
pause
