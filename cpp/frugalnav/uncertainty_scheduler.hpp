// frugalnav/uncertainty_scheduler.hpp
// -----------------------------------------------------------------------------
// C++ port of the headline contribution: the Uncertainty-Aware Landmark Scheduler
// (see ../../core/uncertainty_scheduler.py). This is the module the plan targets
// for the ultra-low-power RISC-V SoC (GAP9): scalar math only, header-only, NO
// dynamic allocation, NO exceptions, NO STL containers in the hot path -- just a
// handful of floats and a fixed-size std::array of weights. It compiles to a few
// hundred bytes of code and runs in well under a microsecond per call.
//
// The logic is line-for-line faithful to the Python reference so the two can be
// cross-checked bit-for-bit on the same inputs (see main.cpp).
//
// Build: header-only, C++14. No third-party dependencies (no Eigen needed -- the
// scheduler is pure scalar arithmetic; Eigen is only for the geometry modules).
#ifndef FRUGALNAV_UNCERTAINTY_SCHEDULER_HPP
#define FRUGALNAV_UNCERTAINTY_SCHEDULER_HPP

#include <array>
#include <cstdint>

namespace frugalnav {

// Map a raw cue into [0,1] using calibrated [lo,hi] bounds. invert=true for cues
// where a HIGHER raw number means LESS uncertainty (blur / image sharpness).
inline float normalize(float value, float lo, float hi, bool invert = false) {
    if (hi == lo) return 0.0f;
    float x = (value - lo) / (hi - lo);
    if (x < 0.0f) x = 0.0f;
    if (x > 1.0f) x = 1.0f;
    return invert ? (1.0f - x) : x;
}

// The glass-box signals the scheduler consumes each tick (fixed POD struct --
// "sensor struct in", plan section 6). active_features defaults to "plenty".
struct Cues {
    float sigma_pos      = 0.0f;   // metres of position std-dev (from state fusion)
    float sigma_head     = 0.0f;   // radians of heading std-dev
    float feature_loss   = 0.0f;   // features lost per second (leading indicator)
    float blur           = 300.0f; // raw Laplacian variance (INVERTED inside)
    float imu_bias       = 0.0f;   // rad/s gyro-bias magnitude
    float active_features = 9999.0f;
};

// Calibrated [lo,hi] range for each cue (tuned on EuRoC MH_01, matching the .py).
struct CueBounds {
    float sigma_pos_lo   = 0.05f,  sigma_pos_hi   = 2.0f;
    float sigma_head_lo  = 0.01f,  sigma_head_hi  = 0.30f;
    float feature_loss_lo = 0.0f,  feature_loss_hi = 30.0f;
    float blur_lo        = 150.0f, blur_hi        = 300.0f;   // inverted
    float imu_bias_lo    = 0.0f,   imu_bias_hi    = 0.15f;
};

// Five weights (sum = 1) + thresholds. Matches SchedulerConfig in the .py.
struct SchedulerConfig {
    // order: sigma_pos, sigma_head, blur, feature_loss, imu_bias
    std::array<float, 5> weights = {{0.45f, 0.20f, 0.20f, 0.10f, 0.05f}};
    float tau              = 0.45f;   // soft threshold on U in [0,1]
    float sigma_pos_floor  = 1.5f;    // hard floor: force a fix above this (m)
    int   feature_floor    = 20;      // observability floor: force a fix below this
    int   refractory_ticks = 15;      // suppress re-trigger for N ticks after a fix
    CueBounds bounds;
};

enum class Reason : std::uint8_t { None, HardFloorSigma, ObservabilityFloor, SoftU };

// Command/telemetry struct out.
struct SchedulerResult {
    float U = 0.0f;
    bool  trigger = false;
    Reason reason = Reason::None;
    std::array<float, 5> components = {{0, 0, 0, 0, 0}};  // normalized cues, same order
};

class UncertaintyScheduler {
public:
    explicit UncertaintyScheduler(const SchedulerConfig& cfg = SchedulerConfig())
        : cfg_(cfg), refractory_(0) {}

    // Call right after a correction is applied.
    void reset_after_fix() { refractory_ = cfg_.refractory_ticks; }

    SchedulerResult compute(const Cues& c) {
        const CueBounds& b = cfg_.bounds;
        SchedulerResult r;
        r.components[0] = normalize(c.sigma_pos,    b.sigma_pos_lo,    b.sigma_pos_hi);
        r.components[1] = normalize(c.sigma_head,   b.sigma_head_lo,   b.sigma_head_hi);
        r.components[2] = normalize(c.blur,         b.blur_lo,         b.blur_hi, true);
        r.components[3] = normalize(c.feature_loss, b.feature_loss_lo, b.feature_loss_hi);
        r.components[4] = normalize(c.imu_bias,     b.imu_bias_lo,     b.imu_bias_hi);

        float U = 0.0f;
        for (int i = 0; i < 5; ++i) U += cfg_.weights[i] * r.components[i];
        r.U = U;

        if (refractory_ > 0) --refractory_;

        // two-tier trigger + hysteresis (identical to the Python reference)
        if (c.sigma_pos > cfg_.sigma_pos_floor) {
            r.trigger = true; r.reason = Reason::HardFloorSigma;
        } else if (c.active_features < (float)cfg_.feature_floor) {
            r.trigger = true; r.reason = Reason::ObservabilityFloor;
        } else if (refractory_ == 0 && U > cfg_.tau) {
            r.trigger = true; r.reason = Reason::SoftU;
        }
        return r;
    }

private:
    SchedulerConfig cfg_;
    int refractory_;
};

}  // namespace frugalnav
#endif  // FRUGALNAV_UNCERTAINTY_SCHEDULER_HPP
