# FrugalNav Setup Guide

This project relies on a lean Python stack. There is no ROS or Gazebo required on the critical path.

## 1. Prerequisites (Windows Users)
Ensure WSL (Windows Subsystem for Linux) is installed running Ubuntu 22.04:
`wsl --install -d Ubuntu-22.04`

## 2. System Packages
Update your package manager and install the Python virtual environment tools:
sudo apt update && sudo apt upgrade -y
sudo apt install python3-venv python3-pip -y

## 3. Environment & Dependencies
Create the environment and install the core stack (NumPy, OpenCV, Matplotlib, Evo):
python3 -m venv env
source env/bin/activate
pip install numpy opencv-python matplotlib evo

## 4. Repository Structure
Your working directory should look like this:
frugalnav/
├── core/       # Portable navigation logic (No ROS/OS deps)
├── harness/    # Throwaway dataset readers and plotting
├── tests/      # Unit tests
└── config/     # Landmark world-frame maps
