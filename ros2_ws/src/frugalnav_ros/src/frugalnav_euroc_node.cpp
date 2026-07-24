// frugalnav_euroc_node.cpp
// -----------------------------------------------------------------------------
// FrugalNav on REAL data: replays the EuRoC MAV MH_01 ground-truth trajectory and
// runs the real uncertainty scheduler over it, visualised in RViz. THREE estimators
// share one drift realisation so both the accuracy and the frugality show directly:
//
//   GREEN  = ground truth (EuRoC)
//   RED    = pure VIO, never corrected            -> drifts away
//   AMBER  = fixed-period, correct at EVERY marker -> accurate but spends every fix
//   CYAN   = uncertainty-aware                     -> matches amber with far fewer fixes
//
// The scheduler / state-fusion are the unmodified portable core (cpp/frugalnav/);
// only the ArUco detection is stubbed as a perfect absolute fix.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"

#include "frugalnav/uncertainty_scheduler.hpp"
#include "frugalnav/nav_core.hpp"

using namespace std::chrono_literals;
using frugalnav::Vec2;
static double d2(Vec2 a, Vec2 b) { return std::hypot(a.x - b.x, a.y - b.y); }

class FrugalNavEurocNode : public rclcpp::Node {
public:
  FrugalNavEurocNode() : Node("frugalnav_euroc_node"), rng_(7) {
    // EuRoC ground truth: $FRUGALNAV_EUROC, else $FRUGALNAV_ROOT/datasets/..., else relative
    const char* euroc = std::getenv("FRUGALNAV_EUROC");
    const char* root = std::getenv("FRUGALNAV_ROOT");
    const std::string rel = "datasets/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv";
    std::string def = euroc ? std::string(euroc)
                            : (root ? std::string(root) + "/" + rel : rel);
    csv_ = declare_parameter<std::string>("gt_csv", def);
    stride_ = declare_parameter<int>("stride", 10);            // 200 Hz -> 20 Hz
    fix_every_m_ = declare_parameter<double>("fix_every_m", 1.5);
    rate_hz_ = declare_parameter<double>("rate_hz", 60.0);     // playback speed

    if (!load_csv()) {
      RCLCPP_FATAL(get_logger(), "Could not read GT csv: %s", csv_.c_str());
      throw std::runtime_error("euroc csv not found");
    }
    Vec2 o = gt_[0];
    for (auto &p : gt_) { p.x -= o.x; p.y -= o.y; }            // start near origin
    place_markers();

    for (auto f : {&none_, &fixed_, &unc_}) { f->xy = gt_[0]; f->q_per_metre = 0.035f; }
    // Sigma-driven selective scheduler: fire when the fused position std has grown
    // past a small floor (~0.45 m) -> adaptive, spends fixes only as drift demands,
    // so it skips markers while the estimate is still confident.
    frugalnav::SchedulerConfig scfg;
    scfg.tau = 0.7f; scfg.sigma_pos_floor = 0.45f; scfg.refractory_ticks = 10;
    sched_ = std::make_unique<frugalnav::UncertaintyScheduler>(scfg);

    scene_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "/frugalnav/scene", rclcpp::QoS(1).transient_local());
    viz_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/frugalnav/viz", 10);
    u_pub_ = create_publisher<std_msgs::msg::Float32>("/frugalnav/U", 10);
    tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    publish_scene();
    timer_ = create_wall_timer(
        std::chrono::milliseconds(std::max(1, int(1000.0 / rate_hz_))), [this] { step(); });
    RCLCPP_INFO(get_logger(), "EuRoC replay: %zu poses, %zu virtual markers.",
                gt_.size(), markers_.size());
  }

private:
  bool load_csv() {
    std::ifstream f(csv_);
    if (!f) return false;
    std::string line; long i = 0;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;
      if (i++ % stride_ != 0) continue;
      std::stringstream ss(line); std::string c; std::vector<double> v;
      while (std::getline(ss, c, ',') && v.size() < 3) v.push_back(std::atof(c.c_str()));
      if (v.size() >= 3) gt_.push_back({float(v[1]), float(v[2])});     // p_x, p_y
    }
    return gt_.size() > 10;
  }
  void place_markers() {
    double acc = 1e9; Vec2 prev = gt_[0];
    for (auto &p : gt_) { acc += d2(p, prev); prev = p;
      if (acc >= fix_every_m_) { markers_.push_back(p); acc = 0; } }
  }
  // Distance-scaled VIO drift. The SAME noise realisation is applied to all three
  // estimators, but each keeps an INDEPENDENT heading accumulator (only the ones
  // that correct reset it) -- so 'none' really does drift while the corrected ones
  // stay locked, which is the whole comparison.
  struct Noise { double dyaw, nx, ny, sd; };
  Noise draw(Vec2 delta) {
    double sd = std::hypot(delta.x, delta.y);
    std::normal_distribution<double> n(0.0, 1.0);
    return {n(rng_) * yaw_rw_ * std::sqrt(std::max(sd, 1e-6)),
            n(rng_) * add_ * sd, n(rng_) * add_ * sd, sd};
  }
  Vec2 apply(Vec2 delta, double &yaw, const Noise &z) {
    yaw += yaw_bias_ * z.sd + z.dyaw;
    double c = std::cos(yaw), s = std::sin(yaw);
    return {float(c * delta.x - s * delta.y + z.nx),
            float(s * delta.x + c * delta.y + z.ny)};
  }
  // Markers are REUSABLE (a surveyed tile can be re-detected on a revisit), so one
  // is almost always available along the path -- that keeps the covariance bounded
  // and lets the scheduler genuinely CHOOSE to skip markers when it is confident.
  bool marker_in_view() {
    Vec2 p = gt_[std::min(k_, gt_.size() - 1)];
    for (auto &m : markers_) if (d2(p, m) <= 0.8) return true;
    return false;
  }
  void snap(frugalnav::StateFusion &f, Vec2 t) {
    std::normal_distribution<double> fn(0.0, 0.02);
    f.update({t.x + float(fn(rng_)), t.y + float(fn(rng_))}, 0.02f * 0.02f);
  }

  void step() {
    if (k_ >= gt_.size()) { finish(); return; }
    Vec2 t = gt_[k_];
    if (k_ == 0) { prev_ = t; ++k_; return; }
    Vec2 td{t.x - prev_.x, t.y - prev_.y};
    double sd = std::hypot(td.x, td.y);
    double d = std::min(1.0, sd * 20.0 / 2.5);                  // motion-blur proxy (20 Hz data, 2.5 m/s)

    Noise z = draw(td);                                         // one noise draw...
    none_.predict(apply(td, yaw_none_, z), 0.0f);               // ...independent yaw each
    fixed_.predict(apply(td, yaw_fixed_, z), 0.0f);
    unc_.predict(apply(td, yaw_unc_, z), float(0.02 * d));

    bool marker = marker_in_view();
    // fixed-period baseline: correct on every marker pass (cooldown = one fix per pass)
    if (marker && k_ - last_fix_k_ >= 14) {
      snap(fixed_, t); yaw_fixed_ *= 0.15; last_fix_k_ = k_; ++fixed_count_;
    }

    // uncertainty-aware: correct only when the scheduler fires AND a marker is in view
    frugalnav::Cues cues;
    cues.sigma_pos = unc_.sigma_pos();
    cues.blur = float(std::max(5.0, 300.0 * (1.0 - 0.25 * d)));  // mild leading cues
    cues.feature_loss = float(3.0 * d);
    cues.imu_bias = float(0.02 + 0.02 * d);
    cues.sigma_head = 0.01f; cues.active_features = 150.0f;
    auto r = sched_->compute(cues);
    if (r.trigger && marker && k_ - last_unc_k_ >= 6) {
      snap(unc_, t); sched_->reset_after_fix(); yaw_unc_ *= 0.15;
      last_unc_k_ = k_; corr_.push_back(unc_.xy); ++fixes_;
    }

    true_path_.push_back(t);
    none_path_.push_back(none_.xy); fixed_path_.push_back(fixed_.xy); unc_path_.push_back(unc_.xy);
    last_U_ = r.U; drone_ = t;
    publish_viz(); broadcast_tf();
    prev_ = t; ++k_;
  }

  void finish() {
    if (done_) return; done_ = true;
    auto peak = [&](const std::vector<Vec2>&e){ double m=0; for(size_t i=0;i<e.size();++i) m=std::max(m,d2(e[i],true_path_[i])); return m; };
    RCLCPP_INFO(get_logger(),
      "EuRoC MH_01 done | none: final %.2f peak %.2f (0 fixes) | "
      "fixed-period: final %.2f peak %.2f (%zu fixes) | "
      "uncertainty: final %.2f peak %.2f (%d fixes)  -> %.0f%% fewer fixes at similar accuracy",
      d2(none_path_.back(), true_path_.back()), peak(none_path_),
      d2(fixed_path_.back(), true_path_.back()), peak(fixed_path_), (size_t)fixed_count_,
      d2(unc_path_.back(), true_path_.back()), peak(unc_path_), fixes_,
      100.0 * (1.0 - double(fixes_) / std::max<size_t>(1, (size_t)fixed_count_)));
  }

  visualization_msgs::msg::Marker base(int id, int type, double s, float r, float g, float b, float a) {
    visualization_msgs::msg::Marker m; m.header.frame_id = "world"; m.header.stamp = now();
    m.ns = "euroc"; m.id = id; m.type = type; m.action = 0;
    m.scale.x = s; m.scale.y = s; m.scale.z = s;
    m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = a; m.pose.orientation.w = 1; return m;
  }
  visualization_msgs::msg::Marker line(int id, const std::vector<Vec2>&pts, float r, float g, float b) {
    auto m = base(id, visualization_msgs::msg::Marker::LINE_STRIP, 0.03, r, g, b, 0.95);
    for (auto&p:pts){geometry_msgs::msg::Point q;q.x=p.x;q.y=p.y;q.z=0;m.points.push_back(q);} return m;
  }
  void publish_scene() {
    visualization_msgs::msg::MarkerArray a;
    auto mk = base(1, visualization_msgs::msg::Marker::SPHERE_LIST, 0.12, 0.9, 0.9, 0.95, 1.0);
    for (auto&p:markers_){geometry_msgs::msg::Point q;q.x=p.x;q.y=p.y;q.z=0;mk.points.push_back(q);}
    a.markers.push_back(mk); scene_pub_->publish(a);
  }
  void publish_viz() {
    visualization_msgs::msg::MarkerArray a;
    a.markers.push_back(line(10, true_path_, 0.20, 0.83, 0.44));   // green truth
    a.markers.push_back(line(11, none_path_, 0.98, 0.30, 0.32));   // red no-correction
    a.markers.push_back(line(16, fixed_path_, 0.96, 0.65, 0.14));  // amber fixed-period
    a.markers.push_back(line(12, unc_path_, 0.22, 0.74, 0.97));    // cyan uncertainty
    auto dr = base(13, visualization_msgs::msg::Marker::SPHERE, 0.18, 0.22, 0.74, 0.97, 1.0);
    dr.pose.position.x = drone_.x; dr.pose.position.y = drone_.y; a.markers.push_back(dr);
    auto fx = base(14, visualization_msgs::msg::Marker::SPHERE_LIST, 0.16, 0.99, 0.85, 0.14, 1.0);
    for (auto&c:corr_){geometry_msgs::msg::Point q;q.x=c.x;q.y=c.y;q.z=0;fx.points.push_back(q);}
    a.markers.push_back(fx);
    auto tx = base(15, visualization_msgs::msg::Marker::TEXT_VIEW_FACING, 0.35, 0.92, 0.95, 0.98, 1.0);
    tx.pose.position.x = gt_[0].x; tx.pose.position.y = gt_[0].y; tx.pose.position.z = 1.2;
    char b[160]; std::snprintf(b, sizeof(b), "EuRoC MH_01  U=%.2f  uncertainty fixes=%d / fixed=%zu",
                               last_U_, fixes_, (size_t)fixed_count_);
    tx.text = b; a.markers.push_back(tx);
    viz_pub_->publish(a);
    std_msgs::msg::Float32 u; u.data = last_U_; u_pub_->publish(u);
  }
  void broadcast_tf() {
    geometry_msgs::msg::TransformStamped t; t.header.stamp = now();
    t.header.frame_id = "world"; t.child_frame_id = "euroc_drone";
    t.transform.translation.x = drone_.x; t.transform.translation.y = drone_.y;
    t.transform.rotation.w = 1.0; tf_->sendTransform(t);
  }

  std::string csv_; int stride_ = 10; double fix_every_m_ = 1.5, rate_hz_ = 60.0;
  std::vector<Vec2> gt_, markers_, true_path_, none_path_, fixed_path_, unc_path_, corr_;
  size_t k_ = 0, last_fix_k_ = 0, last_unc_k_ = 0; Vec2 prev_{}, drone_{};
  int fixes_ = 0, fixed_count_ = 0; float last_U_ = 0; bool done_ = false;
  double yaw_none_ = 0, yaw_fixed_ = 0, yaw_unc_ = 0;
  double yaw_bias_ = 0.0035, yaw_rw_ = 0.015, add_ = 0.010;
  frugalnav::StateFusion none_, fixed_, unc_;
  std::unique_ptr<frugalnav::UncertaintyScheduler> sched_;
  std::mt19937 rng_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr scene_pub_, viz_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr u_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try { rclcpp::spin(std::make_shared<FrugalNavEurocNode>()); }
  catch (const std::exception &e) { RCLCPP_ERROR(rclcpp::get_logger("euroc"), "%s", e.what()); }
  rclcpp::shutdown();
  return 0;
}
