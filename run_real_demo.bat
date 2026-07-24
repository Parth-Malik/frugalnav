@echo off
REM ============================================================
REM  FrugalNav - REAL VISION demo (double-click)
REM  The drone flies on a REAL downward camera: real ArUco detection,
REM  real blur measurement, wind it can only ESTIMATE (never knows).
REM  Sim + RViz in a new window; keyboard control HERE.
REM
REM  1 AUTO  2 MANUAL(WASD)   R reset   P pause
REM  ] [ wind stronger/weaker   T gust   G weather on/off
REM ============================================================
REM  Resolve this repo's WSL path from the location of this .bat (no hardcoded paths)
set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"
for /f "usebackq delims=" %%p in (`wsl -d Ubuntu-22.04 wslpath -a "%HERE%"`) do set "REPO=%%p"
set ROSENV=source /opt/ros/humble/setup.bash ^&^& source /usr/share/gazebo/setup.sh ^&^& source %REPO%/ros2_ws/install/setup.bash
echo.
echo  Starting FrugalNav REAL vision demo (Gazebo + RViz) in a new window...
start "FrugalNav Real" wsl -d Ubuntu-22.04 -e bash -c "%ROSENV% && ros2 launch frugalnav_ros real_demo.launch.py gui:=true"
echo  Waiting ~22s for Gazebo + camera + perception + nav to come up...
timeout /t 22 /nobreak >nul
echo.
echo  === KEYBOARD CONTROL (this window) ===
echo  1=auto 2=manual(WASD)  R=reset P=pause  ] [ wind  T gust  G weather
echo.
wsl -d Ubuntu-22.04 -e bash -c "%ROSENV% && ros2 run frugalnav_ros frugalnav_teleop.py"
echo.
echo  Control stopped. Close the sim window to fully exit.
pause
