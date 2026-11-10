// cpp/v2_check.cpp -- checks the C++ FrugalNav v2 logic against the Python numbers
// (tests/test_frugalnav_v2.py). Build: g++ -std=c++14 -O2 -I. v2_check.cpp -o v2_check
#include <cstdio>
#include "frugalnav/uncertainty_scheduler.hpp"
#include "frugalnav/nav_core.hpp"
using namespace frugalnav;
int main() {
    UncertaintyScheduler s(SchedulerConfig::v2(1.0f));
    s.reset_after_fix();
    Cues c; c.sigma_pos = 1.2f; c.active_features = 80;
    bool a = s.compute(c).trigger;          // expect false (cooldown)
    c.sigma_pos = 2.5f;
    bool b = s.compute(c).trigger;          // expect true (emergency)
    StateFusion f; f.a = f.c = 0.01f; f.use_v2();
    for (int i = 0; i < 50; ++i) f.predict({0.2f, 0.0f});
    float expect = 0.01f + 0.02f * 10.0f + 8.0e-4f * 100.0f;   // same as the Python test
    std::printf("cooldown_blocks=%d emergency=%d var=%.5f expect=%.5f\n", !a, b, f.a, expect);
    return (!a && b && std::fabs(f.a - expect) < 1e-4f) ? 0 : 1;
}
