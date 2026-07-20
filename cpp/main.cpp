// cpp/main.cpp
// -----------------------------------------------------------------------------
// Cross-checks the C++ port against the Python reference (same 5 scheduler cases
// as core/uncertainty_scheduler.py's self-test), exercises the full decision hot
// loop once, and BENCHMARKS the hot loop so the RISC-V feasibility study has real
// measured per-call latency rather than only a model.
//
// Build (any of):
//   g++ -O2 -std=c++14 -I. main.cpp -o frugalnav_demo         # direct
//   cmake -S . -B build && cmake --build build                # via CMake
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>

#include "frugalnav/uncertainty_scheduler.hpp"
#include "frugalnav/nav_core.hpp"

using namespace frugalnav;

static const char* reason_str(Reason r) {
    switch (r) {
        case Reason::HardFloorSigma:     return "hard_floor_sigma";
        case Reason::ObservabilityFloor: return "observability_floor";
        case Reason::SoftU:              return "soft_U";
        default:                         return "none";
    }
}

static void scheduler_selftest() {
    std::printf("== scheduler self-test (must match uncertainty_scheduler.py) ==\n");
    UncertaintyScheduler sch;

    // 1) calm -> low U, no trigger
    Cues c1; c1.sigma_pos = 0.1f; c1.feature_loss = 2.0f; c1.active_features = 120;
    auto r1 = sch.compute(c1);
    std::printf("  calm:       U=%.3f trigger=%d (%s)\n", r1.U, r1.trigger, reason_str(r1.reason));
    assert(r1.U < 0.2f && !r1.trigger);

    // 2) drifting below the hard floor -> soft trigger. A realistic drifting state
    //    raises several leading cues together (heading cov, blur, gyro bias), which
    //    is what carries U over tau before sigma_pos reaches the hard floor.
    Cues c2; c2.sigma_pos = 1.2f; c2.sigma_head = 0.20f; c2.feature_loss = 25.0f;
    c2.blur = 180.0f; c2.imu_bias = 0.08f; c2.active_features = 80;
    auto r2 = sch.compute(c2);
    std::printf("  drifting:   U=%.3f trigger=%d (%s)\n", r2.U, r2.trigger, reason_str(r2.reason));
    assert(r2.trigger && r2.reason == Reason::SoftU);
    sch.reset_after_fix();

    // 3) right after a fix -> refractory suppresses the soft trigger
    auto r3 = sch.compute(c2);
    std::printf("  refractory: U=%.3f trigger=%d (%s)  <- suppressed\n",
                r3.U, r3.trigger, reason_str(r3.reason));
    assert(!r3.trigger);

    // 4) hard floor overrides refractory
    Cues c4; c4.sigma_pos = 2.5f; c4.feature_loss = 25.0f; c4.active_features = 80;
    auto r4 = sch.compute(c4);
    std::printf("  hardfloor:  U=%.3f trigger=%d (%s)\n", r4.U, r4.trigger, reason_str(r4.reason));
    assert(r4.trigger && r4.reason == Reason::HardFloorSigma);

    // 5) observability floor: too few features -> force a fix
    Cues c5; c5.sigma_pos = 0.3f; c5.feature_loss = 5.0f; c5.active_features = 12;
    auto r5 = sch.compute(c5);
    std::printf("  few-feat:   U=%.3f trigger=%d (%s)\n", r5.U, r5.trigger, reason_str(r5.reason));
    assert(r5.trigger && r5.reason == Reason::ObservabilityFloor);

    std::printf("  all scheduler self-tests passed.\n\n");
}

static void hot_loop_once() {
    std::printf("== one full decision hot-loop step ==\n");
    Controller ctrl; ctrl.B = {0, 0};
    StateFusion fus; fus.xy = {58, 24};
    ObstacleAvoidance av;
    UncertaintyScheduler sch;

    fus.predict({-0.9f, -0.4f}, 0.05f);            // VIO relative motion
    Cues c; c.sigma_pos = fus.sigma_pos(); c.feature_loss = 8; c.blur = 210;
    c.active_features = 90;
    auto sr = sch.compute(c);
    if (sr.trigger) { fus.update({57.1f, 23.6f}, 0.02f * 0.02f); sch.reset_after_fix(); }
    Vec2 seek = ctrl.B - fus.xy;
    Vec2 evade = av.update(seek, ttc_from_range(3.0f, 2.0f), 0.3f);
    VelocityCmd cmd = ctrl.command(fus.xy, evade);
    std::printf("  U=%.3f trigger=%d  sigma=%.3f  evading=%d  cmd=(%.3f, %.3f)\n\n",
                sr.U, sr.trigger, fus.sigma_pos(), av.evading, cmd.vx, cmd.vy);
}

static void benchmark() {
    std::printf("== hot-loop benchmark (measured on this host) ==\n");
    UncertaintyScheduler sch;
    Controller ctrl; StateFusion fus; ObstacleAvoidance av;
    Cues c; c.sigma_pos = 0.8f; c.feature_loss = 10; c.blur = 220; c.active_features = 90;

    const long N = 5000000;
    volatile float sink = 0.0f;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (long i = 0; i < N; ++i) {
        c.sigma_pos = 0.5f + 0.5f * std::sin(i * 1e-4f);      // vary the input
        auto r = sch.compute(c);
        fus.predict({0.06f, 0.0f}, 0.0f);
        Vec2 ev = av.update({-1.0f, 0.0f}, r.trigger ? 1.0f : 5.0f, 0.2f);
        VelocityCmd cmd = ctrl.command(fus.xy, ev);
        sink += r.U + cmd.vx + fus.a;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    std::printf("  full hot loop: %.1f ns/step over %ld steps  (sink=%.1f)\n",
                ns / N, N, (float)sink);

    // scheduler-only
    auto t2 = std::chrono::high_resolution_clock::now();
    for (long i = 0; i < N; ++i) {
        c.sigma_pos = 0.5f + 0.5f * std::sin(i * 1e-4f);
        auto r = sch.compute(c);
        sink += r.U;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double ns2 = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    std::printf("  scheduler only: %.1f ns/step (measured on this x86 host)\n", ns2 / N);
    std::printf("  NOTE: these are HOST nanoseconds. The GAP9 projection (cycle-count\n"
                "        model + clock/IPC derating) is in profiling/riscv_feasibility.md;\n"
                "        do not read host ns as target cycles.\n\n");
}

int main() {
    scheduler_selftest();
    hot_loop_once();
    benchmark();
    std::printf("FrugalNav C++ core OK.\n");
    return 0;
}
