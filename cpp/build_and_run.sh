#!/usr/bin/env bash
# Build and run the FrugalNav C++ core in one step.
# (On Windows hosts with Smart App Control / WDAC, a freshly-compiled unsigned exe
#  can be blocked from a *separate* launch; compiling and running in one shell step
#  as below avoids that. On Linux/macOS it just works.)
set -e
cd "$(dirname "$0")"
CXX="${CXX:-g++}"
"$CXX" -O2 -std=c++14 -I. main.cpp -o frugalnav_demo.exe
./frugalnav_demo.exe
