cmake_minimum_required(VERSION 3.10)
project(frugalnav_core CXX)

# The portable core is header-only, scalar, and dependency-free (no Eigen needed
# for the scheduler / controller / fusion / avoidance -- only the geometry modules
# would use Eigen, and those are not on the RISC-V hot path). C++14 so it builds on
# older cross-toolchains (e.g. the GAP9 / PULP GCC).
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

add_executable(frugalnav_demo main.cpp)
target_include_directories(frugalnav_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Cross-compiling for the PULP/GAP9 RISC-V target would instead point CMAKE at the
# riscv32 toolchain, e.g.:
#   cmake -DCMAKE_TOOLCHAIN_FILE=riscv.cmake -DCMAKE_CXX_COMPILER=riscv32-unknown-elf-g++
# The headers themselves need no change: no heap, no exceptions, no STL containers.
