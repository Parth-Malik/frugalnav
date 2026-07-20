// frugalnav_gazebo_node.cpp
// -----------------------------------------------------------------------------
// The FrugalNav brain, closing the loop in a live Gazebo simulation.
//
//   Gazebo (planar_move + p3d)  ── /frugalnav/truth ─►  THIS NODE  ── /frugalnav/cmd_vel ─►  Gazebo
//                                                          │
//                                        the REAL C++ core (cpp/frugalnav/*.hpp):
//                                        StateFusion.predict → UncertaintyScheduler.compute
//                                        → (if fire & marker in view) StateFusion.update
//                                        → ObstacleAvoidance → TargetCentricController
//                                                          │
//                                            RViz  ◄── markers / paths / U / corrections / TF
//
// Gazebo supplies ground-truth motion (physics); this node treats it as the VIO
// input, injects realistic drift, and lets the scheduler decide WHEN to snap the
// estimate back with an absolute landmark fix. The scheduler/fusion/controller/
// avoidance are the unmodified portable core; only the ArUco *detection* is stubbed
// as a perfect absolute fix (that geometry is validated in the Python corrector).
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "std_msgs/msg/float32.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"

#include "frugalnav/uncertainty_scheduler.hpp"
#include "frugalnav/nav_core.hpp"

using namespace std::chrono_literals;
using frugalnav::Vec2;

static double dist2(Vec2 a, Vec2 b) { return std::hypot(a.x - b.x, a.y - b.y); }

class FrugalNavGazeboNode : public rclcpp::Node {
public:
  FrugalNavGazeboNode() : Node("frugalnav_gazebo_node"), rng_(1) {
    // --- world (matches harness/integrated_sim.py) ---
    B_ = {0.0f, 0.0f};
    start_ = {58.0f, 24.0f};
    const double fr[] = {0.14, 0.28, 0.42, 0.60, 0.75, 0.88};
    for (double f : fr) markers_.push_back(lerp(start_, B_, f));
    obstacle_ = pt_lateral(0.52, 1.5);   obstacle_r_ = 2.3f;
    hard_c_   = lerp(start_, B_, 0.34);   hard_r_ = 9.0f;

    // --- core modules (the contribution) ---
    controller_.B = B_; controller_.kp = 0.6f; controller_.v_max = 2.0f;
    controller_.arrive_tol = 1.2f;
    fusion_.xy = start_; fusion_.a = fusion_.c = 0.05f * 0.05f;
    fusion_.q_per_metre = 0.03f;
    avoider_.ttc_trigger = 3.6f; avoider_.ttc_release = 5.0f;
    avoider_.ttc_min = 0.5f; avoider_.gain = 5.0f;
    frugalnav::SchedulerConfig scfg; scfg.tau = 0.45f;
    scheduler_ = std::make_unique<frugalnav::UncertaintyScheduler>(scfg);

    // --- ROS I/O ---
    cmd_pub_  = create_publisher<geometry_msgs::msg::Twist>("/frugalnav/cmd_vel", 10);
    u_pub_    = create_publisher<std_msgs::msg::Float32>("/frugalnav/U", 10);
    scene_pub_= create_publisher<visualization_msgs::msg::MarkerArray>(
        "/frugalnav/scene", rclcpp::QoS(1).transient_local());   // latched
    viz_pub_  = create_publisher<visualization_msgs::msg::MarkerArray>("/frugalnav/viz", 10);
    tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    truth_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "/frugalnav/truth", 20,
        [this](nav_msgs::msg::Odometry::SharedPtr m) {
          true_.x = m->pose.pose.position.x;
          true_.y = m->pose.pose.position.y;
          have_truth_ = true;
        });

    publish_scene();
    timer_ = create_wall_timer(33ms, [this] { step(); });
    RCLCPP_INFO(get_logger(),
      "FrugalNav Gazebo node up. Homing (%.0f,%.0f)->(0,0) through a hard patch and an obstacle.",
      start_.x, start_.y);
  }

private:
  // ---- geometry helpers ----
  static Vec2 lerp(Vec2 a, Vec2 b, double f) {
    return {float(a.x + (b.x - a.x) * f), float(a.y + (b.y - a.y) * f)};
  }
  Vec2 pt_lateral(double f, double lat) {
    Vec2 seg{B_.x - start_.x, B_.y - start_.y};
    double n = std::hypot(seg.x, seg.y);
    Vec2 axis{float(seg.x / n), float(seg.y / n)};
    Vec2 left{-axis.y, axis.x};
    Vec2 p = lerp(start_, B_, f);
    return {float(p.x + lat * left.x), float(p.y + lat * left.y)};
  }
  double difficulty(Vec2 p) {
    double d = dist2(p, hard_c_);
    return d >= hard_r_ ? 0.0 : hard_peak_ * (1.0 - d / hard_r_);
  }
  // drift model (matches harness/detour_sim.py DriftModel.corrupt)
  Vec2 drift_corrupt(Vec2 delta, double rate_mult) {
    std::normal_distribution<double> n01(0.0, 1.0);
    yaw_err_   += (yaw_bias_ + n01(rng_) * yaw_rw_) * rate_mult;
    log_scale_ += n01(rng_) * scale_rw_ * rate_mult;
    double s = std::exp(log_scale_), c = std::cos(yaw_err_), sn = std::sin(yaw_err_);
    Vec2 out{float(s * (c * delta.x - sn * delta.y) + n01(rng_) * add_noise_ * rate_mult),
             float(s * (sn * delta.x + c * delta.y) + n01(rng_) * add_noise_ * rate_mult)};
    return out;
  }

  void step() {
    if (!have_truth_ || arrived_) { publish_cmd(0, 0); return; }
    if (!inited_) { prev_true_ = true_; inited_ = true; return; }

    double d = difficulty(true_);

    // obstacle sensed? (within range AND still ahead) -> also a maneuver drift bump
    Vec2 est = fusion_.xy;
    Vec2 seek{B_.x - est.x, B_.y - est.y};
    double seek_n = std::max(1e-9, (double)std::hypot(seek.x, seek.y));
    Vec2 seek_hat{float(seek.x / seek_n), float(seek.y / seek_n)};
    float ttc = std::numeric_limits<float>::infinity(); float bearing = 0.0f;
    bool maneuvering = obstacle_cue(seek_hat, ttc, bearing);

    // 1) PREDICT: dead-reckon the fused state on drift-corrupted true motion
    double man = maneuvering ? 3.5 : 1.0;
    Vec2 td{true_.x - prev_true_.x, true_.y - prev_true_.y};
    Vec2 vio_delta = drift_corrupt(td, (1.0 + 8.0 * d) * man);
    fusion_.predict(vio_delta, float(0.16 * d + (maneuvering ? 0.10 : 0.0)));

    // 2) CUES + SCHEDULE (the contribution)
    frugalnav::Cues cues;
    cues.sigma_pos = fusion_.sigma_pos();
    cues.feature_loss = float(std::max(0.0, prev_feat_ - (150.0 * (1.0 - 0.85 * d))) / 0.033);
    prev_feat_ = 150.0 * (1.0 - 0.85 * d);
    cues.blur = float(std::max(5.0, 300.0 * (1.0 - 0.85 * d)));
    cues.imu_bias = float(0.02 + 0.05 * d);
    cues.sigma_head = 0.01f;
    cues.active_features = float(prev_feat_);
    auto r = scheduler_->compute(cues);

    // 3) CORRECT: only if fired AND a mapped marker is in view
    bool corrected = false;
    if (r.trigger) {
      int mi = marker_in_view();
      if (mi >= 0) {
        std::normal_distribution<double> fn(0.0, 0.03);
        fusion_.update({float(true_.x + fn(rng_)), float(true_.y + fn(rng_))}, 0.03f * 0.03f);
        scheduler_->reset_after_fix();
        yaw_err_ *= 0.1; log_scale_ *= 0.1;          // marker re-anchors heading
        used_.insert(mi); corrections_.push_back(fusion_.xy); corrected = true;
      }
    }
    est = fusion_.xy;

    // 4) AVOID + 5) COMMAND (both real core)
    Vec2 evade = avoider_.update({B_.x - est.x, B_.y - est.y}, ttc, bearing);
    auto cmd = controller_.command(est, evade);
    publish_cmd(cmd.vx, cmd.vy);

    // telemetry
    est_path_.push_back(est);
    true_path_.push_back(true_);
    last_U_ = r.U; last_trigger_ = r.trigger; last_evading_ = avoider_.evading;
    publish_viz();
    broadcast_tf(est);

    if (controller_.arrived(est)) {
      arrived_ = true; publish_cmd(0, 0);
      double miss = dist2(true_, B_);
      RCLCPP_INFO(get_logger(),
        "ARRIVED. true miss=%.2f m | fixes=%zu | est-error=%.2f m", miss,
        corrections_.size(), dist2(true_, est));
    }
    prev_true_ = true_;
  }

  bool obstacle_cue(Vec2 seek_hat, float &ttc, float &bearing) {
    Vec2 to{obstacle_.x - float(true_.x), obstacle_.y - float(true_.y)};
    double dc = std::hypot(to.x, to.y), dsurf = dc - obstacle_r_;
    if (dsurf > obstacle_sense_) return false;
    Vec2 toh{float(to.x / std::max(1e-9, dc)), float(to.y / std::max(1e-9, dc))};
    double along = seek_hat.x * toh.x + seek_hat.y * toh.y;
    if (along < -0.2) return false;
    double closing = std::max(along, 0.35) * controller_.v_max;
    ttc = frugalnav::ttc_from_range(float(dsurf), float(closing));
    double cross = seek_hat.x * toh.y - seek_hat.y * toh.x;
    bearing = float(std::atan2(cross, std::max(-1.0, std::min(1.0, along))));
    return true;
  }
  int marker_in_view() {
    for (size_t i = 0; i < markers_.size(); ++i)
      if (!used_.count(int(i)) && dist2(true_, markers_[i]) <= sensing_r_) return int(i);
    return -1;
  }

  // ---- ROS publishing ----
  void publish_cmd(double vx, double vy) {
    geometry_msgs::msg::Twist t; t.linear.x = vx; t.linear.y = vy; cmd_pub_->publish(t);
  }
  visualization_msgs::msg::Marker base(int id, int type, double sx, double sy, double sz,
                                       float r, float g, float b, float a) {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = "world"; m.header.stamp = now();
    m.ns = "frugalnav"; m.id = id; m.type = type; m.action = 0;
    m.scale.x = sx; m.scale.y = sy; m.scale.z = sz;
    m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = a;
    m.pose.orientation.w = 1.0; return m;
  }
  void publish_scene() {
    visualization_msgs::msg::MarkerArray arr;
    auto hard = base(100, visualization_msgs::msg::Marker::CYLINDER,
                     2 * hard_r_, 2 * hard_r_, 0.05, 0.96, 0.62, 0.04, 0.18);
    hard.pose.position.x = hard_c_.x; hard.pose.position.y = hard_c_.y; arr.markers.push_back(hard);
    auto obs = base(101, visualization_msgs::msg::Marker::CYLINDER,
                    2 * obstacle_r_, 2 * obstacle_r_, 2.0, 0.33, 0.33, 0.36, 0.9);
    obs.pose.position.x = obstacle_.x; obs.pose.position.y = obstacle_.y; obs.pose.position.z = 1.0;
    arr.markers.push_back(obs);
    for (size_t i = 0; i < markers_.size(); ++i) {
      auto mk = base(110 + int(i), visualization_msgs::msg::Marker::CUBE, 0.8, 0.8, 0.1,
                     0.9, 0.9, 0.95, 1.0);
      mk.pose.position.x = markers_[i].x; mk.pose.position.y = markers_[i].y; mk.pose.position.z = 0.05;
      arr.markers.push_back(mk);
    }
    auto tgt = base(102, visualization_msgs::msg::Marker::CYLINDER, 1.2, 1.2, 3.0,
                    0.98, 0.75, 0.14, 0.95);
    tgt.pose.position.x = B_.x; tgt.pose.position.y = B_.y; tgt.pose.position.z = 1.5;
    arr.markers.push_back(tgt);
    scene_pub_->publish(arr);
  }
  visualization_msgs::msg::Marker line(int id, const std::vector<Vec2> &pts,
                                       float r, float g, float b) {
    auto m = base(id, visualization_msgs::msg::Marker::LINE_STRIP, 0.18, 0, 0, r, g, b, 0.95);
    for (auto &p : pts) { geometry_msgs::msg::Point q; q.x = p.x; q.y = p.y; q.z = 0.1; m.points.push_back(q); }
    return m;
  }
  void publish_viz() {
    visualization_msgs::msg::MarkerArray arr;
    arr.markers.push_back(line(200, true_path_, 0.20, 0.83, 0.44));   // green truth
    arr.markers.push_back(line(201, est_path_, 0.22, 0.74, 0.97));    // cyan estimate
    auto drone = base(202, visualization_msgs::msg::Marker::SPHERE, 1.2, 1.2, 1.2,
                      0.22, 0.74, 0.97, 1.0);
    drone.pose.position.x = true_.x; drone.pose.position.y = true_.y; drone.pose.position.z = 0.6;
    arr.markers.push_back(drone);
    auto fixes = base(203, visualization_msgs::msg::Marker::SPHERE_LIST, 0.9, 0.9, 0.9,
                      0.99, 0.85, 0.14, 1.0);
    for (auto &c : corrections_) { geometry_msgs::msg::Point q; q.x = c.x; q.y = c.y; q.z = 0.2; fixes.points.push_back(q); }
    arr.markers.push_back(fixes);
    auto txt = base(204, visualization_msgs::msg::Marker::TEXT_VIEW_FACING, 1, 1, 2.2,
                    0.92, 0.95, 0.98, 1.0);
    txt.pose.position.x = start_.x - 4; txt.pose.position.y = start_.y + 3; txt.pose.position.z = 3;
    char buf[160];
    std::snprintf(buf, sizeof(buf), "uncertainty-aware\nU=%.2f  fixes=%zu%s",
                  last_U_, corrections_.size(), last_evading_ ? "\nDETOUR" : "");
    txt.text = buf; arr.markers.push_back(txt);
    viz_pub_->publish(arr);
    std_msgs::msg::Float32 u; u.data = last_U_; u_pub_->publish(u);
  }
  void broadcast_tf(Vec2 est) {
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = now(); t.header.frame_id = "world"; t.child_frame_id = "base_link";
    t.transform.translation.x = true_.x; t.transform.translation.y = true_.y; t.transform.rotation.w = 1.0;
    tf_->sendTransform(t);
    geometry_msgs::msg::TransformStamped e = t; e.child_frame_id = "frugalnav_estimate";
    e.transform.translation.x = est.x; e.transform.translation.y = est.y; tf_->sendTransform(e);
  }

  // ---- state ----
  Vec2 B_, start_, obstacle_, hard_c_, prev_true_{}, true_{};
  float obstacle_r_ = 2.3f, hard_r_ = 9.0f, sensing_r_ = 4.0f, obstacle_sense_ = 6.0f;
  double hard_peak_ = 0.9, prev_feat_ = 150.0;
  std::vector<Vec2> markers_, est_path_, true_path_, corrections_;
  std::set<int> used_;
  // drift params (match integrated_sim)
  double yaw_err_ = 0, log_scale_ = 0;
  double yaw_bias_ = 0.00003, yaw_rw_ = 0.00020, scale_rw_ = 0.00010, add_noise_ = 0.0018;
  bool have_truth_ = false, inited_ = false, arrived_ = false, last_trigger_ = false, last_evading_ = false;
  float last_U_ = 0.0f;
  std::mt19937 rng_;
  frugalnav::Controller controller_; frugalnav::StateFusion fusion_;
  frugalnav::ObstacleAvoidance avoider_;
  std::unique_ptr<frugalnav::UncertaintyScheduler> scheduler_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr u_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr scene_pub_, viz_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr truth_sub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FrugalNavGazeboNode>());
  rclcpp::shutdown();
  return 0;
}
