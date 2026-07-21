@echo off
REM ============================================================
REM  FrugalNav - CANOPY MAP (dense forest, GPS-denied scenario)
REM  Sim in a new window; keyboard + weather control HERE.
REM  1 AUTO  2 MANUAL(WASD)  3 EUROC   R rewind  P pause
REM  U/N altitude  M auto-alt   ] [ wind   - = fog   T rain   G weather
REM ============================================================
set ROSENV=source /opt/ros/humble/setup.bash ^&^& source /mnt/c/Users/parth/Downloads/drone/ros2_ws/install/setup.bash
echo.
echo  Starting FrugalNav CANOPY map (Gazebo + RViz) in a new window...
start "FrugalNav Canopy" wsl -d Ubuntu-22.04 -e bash -c "%ROSENV% && ros2 launch frugalnav_ros interactive_demo.launch.py map:=canopy"
echo  Waiting ~18s for Gazebo + RViz...
timeout /t 18 /nobreak >nul
echo.
echo  === KEYBOARD + WEATHER CONTROL (this window) ===
wsl -d Ubuntu-22.04 -e bash -c "%ROSENV% && ros2 run frugalnav_ros frugalnav_teleop.py"
echo.
echo  Control stopped. Close the sim window to fully exit.
pause
