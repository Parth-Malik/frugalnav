@echo off
REM ============================================================
REM  FrugalNav - EuRoC MH_01 demo in RViz (double-click to run)
REM  Runs the real uncertainty scheduler over the real EuRoC
REM  trajectory: green=truth, red=pure VIO drift, amber=fixed,
REM  cyan=uncertainty-aware. RViz opens on your Windows desktop.
REM  Press Ctrl+C here (or close RViz) to stop.
REM ============================================================
echo.
echo  Launching FrugalNav EuRoC RViz demo in WSL...
echo.
REM  Resolve this repo's WSL path from the location of this .bat (no hardcoded paths)
set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"
for /f "usebackq delims=" %%p in (`wsl -d Ubuntu-22.04 wslpath -a "%HERE%"`) do set "REPO=%%p"
wsl -d Ubuntu-22.04 -e bash -c "source /opt/ros/humble/setup.bash && source %REPO%/ros2_ws/install/setup.bash && ros2 launch frugalnav_ros euroc_demo.launch.py"
echo.
echo  Demo stopped.
pause
