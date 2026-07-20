"""
Portable navigation core (Week 3 slice).

Everything in this package is intended to be dependency-light and to port to
C++/Eigen and ultimately an ultra-low-power RISC-V SoC (see the project plan,
section 6). Only `aruco_detector` touches OpenCV, because detection is
inherently an image operation; every module downstream of a MarkerSighting is
pure NumPy so the hot loop stays allocation-free and portable.
"""
