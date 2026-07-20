# FrugalNav — C++ core port (Week 7)

A header-only C++14 port of the **portable decision hot loop** — the part of the
system that runs every frame and is targeted at the ultra-low-power RISC-V SoC
(GAP9). It is deliberately dependency-free: **no heap allocation, no exceptions,
no STL containers in the hot path** — just fixed-size structs and scalar/2×2 math.

| Header | Ports (Python source) | What it is |
|---|---|---|
| `frugalnav/uncertainty_scheduler.hpp` | `core/uncertainty_scheduler.py` | **the contribution** — 5-cue U, two-tier trigger, hysteresis |
| `frugalnav/nav_core.hpp` | `core/controller.py`, `state_fusion.py`, `obstacle_avoidance.py` | controller, 2×2 EKF fusion, TTC avoidance |
| `main.cpp` | the `__main__` self-tests | cross-check + one hot-loop step + benchmark |

## Build & run

```bash
# one step (recommended on Windows; avoids Smart App Control blocking a re-launch)
bash build_and_run.sh

# or directly
g++ -O2 -std=c++14 -I. main.cpp -o frugalnav_demo && ./frugalnav_demo

# or via CMake
cmake -S . -B build && cmake --build build && ./build/frugalnav_demo
```

Verified with **g++ 6.3.0 (MinGW)** on Windows. No third-party libraries.

## What `main.cpp` proves

1. **Bit-faithful port.** The scheduler self-test reproduces the five cases from
   `core/uncertainty_scheduler.py` and the computed `U` matches the Python
   reference to 4 decimals (e.g. drifting state `U = 0.666`). The C++ cross-check
   is what caught a stale, too-sparse case in the Python self-test (now fixed).
2. **Full hot loop runs.** One integrated step (predict → schedule → fuse →
   avoid → command) executes with fixed state and no allocation.
3. **Measured latency** (this x86 host, `-O2`): scheduler ≈ **77 ns/step**, full
   decision hot loop ≈ **93 ns/step**. These are *host* nanoseconds — the GAP9
   projection (cycle model + clock/IPC derating) is in
   [`../profiling/riscv_feasibility.md`](../profiling/riscv_feasibility.md).

## Cross-compiling for GAP9 / PULP

The headers need **no change**. Point a RISC-V toolchain at them:

```bash
riscv32-unknown-elf-g++ -O2 -std=c++14 -I. main.cpp -o frugalnav_core.elf
```

Everything downstream of a `MarkerSighting` / `Cues` struct is scalar and
allocation-free, which is exactly the property that makes the memory budget
(≈20 KB landmark map + small VIO state, well inside GAP9's ~1.6 MB L2) hold.
