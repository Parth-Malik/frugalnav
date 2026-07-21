@echo off
REM ============================================================
REM  FrugalNav - INTERACTIVE demo map (Gazebo + RViz) - double-click
REM  Sim in its own window; keyboard + weather control HERE.
REM  (For the dense forest, run run_canopy_map.bat instead.)
REM
REM  1 AUTO  2 MANUAL(WASD)  3 EUROC   R rewind  P pause  Q quit
REM  U/N altitude  M auto-alt   ] [ wind   - = fog   T rain   G weather
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
