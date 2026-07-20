// frugalnav/nav_core.hpp
// -----------------------------------------------------------------------------
// The rest of the portable decision hot loop in C++ (see ../../core/controller.py,
// state_fusion.py, obstacle_avoidance.py). Fixed 2-vector / 2x2 state, closed-form
// 2x2 inverse, no heap, no STL containers. This is what runs every frame on the
// RISC-V core alongside the scheduler; the pixel front-ends (ArUco, optical flow)
// live off-core and only hand these functions small scalars/vectors.
#ifndef FRUGALNAV_NAV_CORE_HPP
#define FRUGALNAV_NAV_CORE_HPP

#include <cmath>
#include <limits>

namespace frugalnav {

struct Vec2 { float x = 0.0f, y = 0.0f; };
inline Vec2 operator+(Vec2 a, Vec2 b) { return {a.x + b.x, a.y + b.y}; }
inline Vec2 operator-(Vec2 a, Vec2 b) { return {a.x - b.x, a.y - b.y}; }
inline Vec2 operator*(float s, Vec2 a) { return {s * a.x, s * a.y}; }
inline float norm(Vec2 a) { return std::sqrt(a.x * a.x + a.y * a.y); }

// ---------------------------- controller ------------------------------------
struct VelocityCmd { float vx = 0.0f, vy = 0.0f; };

struct Controller {
    Vec2  B = {0.0f, 0.0f};   // target, world-frame constant
    float kp = 0.6f;
    float v_max = 2.0f;
    float arrive_tol = 1.0f;

    VelocityCmd command(Vec2 est, Vec2 evasion = {0, 0}) const {
        Vec2 o = est - B;                    // target-centric offset
        Vec2 v = (-kp) * o + evasion;
        float s = norm(v);
        if (s > v_max) v = (v_max / s) * v;  // speed clamp
        return {v.x, v.y};
    }
    bool arrived(Vec2 est) const { return norm(est - B) < arrive_tol; }
};

// --------------------------- state fusion (2x2) -----------------------------
// Symmetric 2x2 covariance held as (a=Pxx, b=Pxy, c=Pyy).
struct StateFusion {
    Vec2  xy = {0, 0};
    float a = 0.01f, b = 0.0f, c = 0.01f;   // covariance
    float q_per_metre = 0.02f;
    float p_floor = 1e-4f;

    void predict(Vec2 d, float extra_var = 0.0f) {
        xy = xy + d;
        float q = q_per_metre * norm(d) + extra_var;
        a += q; c += q;                      // isotropic process growth
    }

    // Kalman update with an isotropic measurement covariance R = r*I.
    void update(Vec2 z, float r) {
        // K = P (P+R)^-1 ; S = P + R (symmetric 2x2), inverse in closed form.
        float sa = a + r, sb = b, sc = c + r;
        float det = sa * sc - sb * sb;
        if (det < 1e-12f) det = 1e-12f;
        float ia =  sc / det, ib = -sb / det, ic =  sa / det;   // S^-1
        // K = P * S^-1
        float k11 = a * ia + b * ib, k12 = a * ib + b * ic;
        float k21 = b * ia + c * ib, k22 = b * ib + c * ic;
        Vec2 innov = z - xy;
        xy.x += k11 * innov.x + k12 * innov.y;
        xy.y += k21 * innov.x + k22 * innov.y;
        // P = (I - K) P
        float na = (1 - k11) * a - k12 * b;
        float nb = (1 - k11) * b - k12 * c;
        float nc = -k21 * b + (1 - k22) * c;
        a = na + p_floor; b = 0.5f * (nb + (-k21 * a + (1 - k22) * b)); c = nc + p_floor;
    }

    float sigma_pos() const {                // sqrt of larger eigenvalue of P
        float tr = a + c, d = a - c;
        float lam = 0.5f * (tr + std::sqrt(d * d + 4 * b * b));
        return std::sqrt(lam > 0 ? lam : 0.0f);
    }
};

// ------------------------- obstacle avoidance -------------------------------
inline float ttc_from_range(float distance, float closing_speed) {
    if (closing_speed <= 1e-6f) return std::numeric_limits<float>::infinity();
    return (distance > 0 ? distance : 0.0f) / closing_speed;
}
inline float ttc_from_looming(float s_prev, float s_now, float dt) {
    if (dt <= 0 || s_prev <= 0 || s_now <= s_prev)
        return std::numeric_limits<float>::infinity();
    return s_now / ((s_now - s_prev) / dt);
}

struct ObstacleAvoidance {
    float ttc_trigger = 2.0f, ttc_release = 3.0f, ttc_min = 0.4f, gain = 2.0f;
    float default_side = 1.0f;
    bool  evading = false;

    Vec2 update(Vec2 seek_dir, float ttc, float bearing) {
        if (evading) { if (ttc > ttc_release) evading = false; }
        else         { if (ttc < ttc_trigger) evading = true; }
        if (!evading) return {0, 0};

        float n = norm(seek_dir);
        Vec2 s = (n > 1e-9f) ? (1.0f / n) * seek_dir : Vec2{1, 0};
        Vec2 left = {-s.y, s.x};
        Vec2 perp;
        if (bearing > 1e-3f)       perp = {-left.x, -left.y};   // obstacle left -> go right
        else if (bearing < -1e-3f) perp = left;                // obstacle right -> go left
        else                       perp = {-default_side * left.x, -default_side * left.y};

        float urg = (ttc_trigger - ttc) / (ttc_trigger - ttc_min > 1e-6f
                                           ? ttc_trigger - ttc_min : 1e-6f);
        if (urg < 0) urg = 0; if (urg > 1) urg = 1;
        return (gain * urg) * perp;
    }
};

}  // namespace frugalnav
#endif  // FRUGALNAV_NAV_CORE_HPP
