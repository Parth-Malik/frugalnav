// frugalnav_interactive_node.cpp
// -----------------------------------------------------------------------------
// Interactive FrugalNav brain for the Gazebo arena. Three modes, live-switchable
// from the keyboard teleop:
//
//   AUTO    the uncertainty scheduler flies the drone to the target, weaving the
//           pillar slalom (obstacle avoidance) and correcting drift at markers.
//   MANUAL  YOU fly with WASD; the estimator still runs, so you watch VIO drift
//           accumulate as you fly and see where/when the scheduler would correct.
//   EUROC   the drone flies the real EuRoC MH_01 trajectory (as moving waypoints)
//           through the arena, with the scheduler running on it.
//
// Controls come in on /frugalnav/ctrl (std_msgs/String): auto|manual|euroc|reset|
// pause|resume. RESET teleports the drone back to start (via /gazebo/set_entity_state)
// and clears the estimator -- the practical "rewind". The scheduler / fusion /
// controller / avoidance are the unmodified C++ core (cpp/frugalnav/).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/string.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "gazebo_msgs/srv/set_entity_state.hpp"

#include "frugalnav/uncertainty_scheduler.hpp"
#include "frugalnav/nav_core.hpp"
#include "frugalnav/scene_arena.hpp"

using namespace std::chrono_literals;
using frugalnav::Vec2;
namespace scene = frugalnav::scene;
static double dd(Vec2 a, Vec2 b) { return std::hypot(a.x - b.x, a.y - b.y); }

enum class Mode { AUTO, MANUAL, EUROC };

class FrugalNavInteractive : public rclcpp::Node {
public:
  FrugalNavInteractive() : Node("frugalnav_interactive_node"), rng_(1) {
    euroc_csv_ = declare_parameter<std::string>("gt_csv",
        "/mnt/c/Users/parth/Downloads/drone/datasets/MH_01_easy/"
        "mav0/state_groundtruth_estimate0/data.csv");

    B_ = {scene::TARGET_X, scene::TARGET_Y};
    start_ = {scene::START_X, scene::START_Y};
    hard_c_ = {scene::HARD_X, scene::HARD_Y};
    for (auto &m : scene::MARKERS) markers_.push_back({m[0], m[1]});
    load_euroc();

    controller_.B = B_; controller_.kp = 0.6f; controller_.v_max = 2.2f;
    controller_.arrive_tol = 1.4f;
    avoider_.ttc_trigger = 3.4f; avoider_.ttc_release = 4.8f;
    avoider_.ttc_min = 0.5f; avoider_.gain = 5.5f;
    reset_estimator();

    cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("/frugalnav/cmd_vel", 10);
    u_pub_ = create_publisher<std_msgs::msg::Float32>("/frugalnav/U", 10);
    scene_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "/frugalnav/scene", rclcpp::QoS(1).transient_local());
    viz_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/frugalnav/viz", 10);
    tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    set_state_ = create_client<gazebo_msgs::srv::SetEntityState>("/gazebo/set_entity_state");

    truth_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "/frugalnav/truth", 20, [this](nav_msgs::msg::Odometry::SharedPtr m) {
          true_.x = m->pose.pose.position.x; true_.y = m->pose.pose.position.y;
          have_truth_ = true; });
    teleop_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        "/frugalnav/teleop", 20, [this](geometry_msgs::msg::Twist::SharedPtr m) {
          teleop_ = {float(m->linear.x), float(m->linear.y)}; teleop_age_ = 0; });
    ctrl_sub_ = create_subscription<std_msgs::msg::String>(
        "/frugalnav/ctrl", 20, [this](std_msgs::msg::String::SharedPtr m) { on_ctrl(m->data); });

    publish_scene();
    timer_ = create_wall_timer(33ms, [this] { step(); });
    RCLCPP_INFO(get_logger(), "Interactive node up. Mode=AUTO. Use the teleop window to fly / switch modes / reset.");
  }

private:
  // ---------- setup ----------
  void load_euroc() {
    std::ifstream f(euroc_csv_); if (!f) return;
    std::string line; long i = 0; std::vector<Vec2> raw;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;
      if (i++ % 10 != 0) continue;
      std::stringstream ss(line); std::string c; std::vector<double> v;
      while (std::getline(ss, c, ',') && v.size() < 3) v.push_back(std::atof(c.c_str()));
      if (v.size() >= 3) raw.push_back({float(v[1]), float(v[2])});
    }
    if (raw.size() < 10) return;
    Vec2 o = raw[0]; float sc = 2.2f;                 // scale up, place near start
    for (auto &p : raw) euroc_.push_back({start_.x + (p.x - o.x) * sc,
                                          start_.y + (p.y - o.y) * sc});
    double acc = 1e9; Vec2 prev = euroc_[0];
    for (auto &p : euroc_) { acc += dd(p, prev); prev = p;
      if (acc >= 3.0) { euroc_markers_.push_back(p); acc = 0; } }
  }
  void reset_estimator() {
    fusion_ = frugalnav::StateFusion(); fusion_.xy = start_; fusion_.q_per_metre = 0.03f;
    frugalnav::SchedulerConfig s; s.tau = 0.45f; s.sigma_pos_floor = 0.9f;
    scheduler_ = std::make_unique<frugalnav::UncertaintyScheduler>(s);
    yaw_ = 0; log_scale_ = 0; prev_true_ = start_; inited_ = false; arrived_ = false;
    est_path_.clear(); true_path_.clear(); corr_.clear(); fixes_ = 0;
    used_.assign(std::max(markers_.size(), euroc_markers_.size()), 0);
    euroc_wp_ = 0;
  }
  void teleport_to_start() {
    if (!set_state_->service_is_ready()) {
      RCLCPP_WARN(get_logger(), "set_entity_state not ready; estimator reset only.");
      return;
    }
    auto req = std::make_shared<gazebo_msgs::srv::SetEntityState::Request>();
    req->state.name = "frugalnav_drone"; req->state.reference_frame = "world";
    req->state.pose.position.x = start_.x; req->state.pose.position.y = start_.y;
    req->state.pose.position.z = 0.5; req->state.pose.orientation.w = 1.0;
    set_state_->async_send_request(req);
  }

  void on_ctrl(const std::string &c) {
    if (c == "auto") { mode_ = Mode::AUTO; arrived_ = false; }
    else if (c == "manual") mode_ = Mode::MANUAL;
    else if (c == "euroc") { mode_ = Mode::EUROC; reset_estimator(); teleport_to_start(); }
    else if (c == "reset") { reset_estimator(); teleport_to_start(); }
    else if (c == "pause") paused_ = true;
    else if (c == "resume") paused_ = false;
    RCLCPP_INFO(get_logger(), "ctrl=%s -> mode=%s paused=%d", c.c_str(), mode_str(), paused_);
  }
  const char *mode_str() { return mode_ == Mode::AUTO ? "AUTO" :
                                  mode_ == Mode::MANUAL ? "MANUAL" : "EUROC"; }

  // ---------- scene helpers ----------
  double difficulty(Vec2 p) {
    double d = dd(p, hard_c_);
    return d >= scene::HARD_R ? 0.0 : 0.9 * (1.0 - d / scene::HARD_R);
  }
  Vec2 drift_corrupt(Vec2 delta, double rate) {
    std::normal_distribution<double> n(0, 1);
    yaw_ += (yaw_bias_ + n(rng_) * yaw_rw_) * rate;
    log_scale_ += n(rng_) * scale_rw_ * rate;
    double s = std::exp(log_scale_), c = std::cos(yaw_), sn = std::sin(yaw_);
    return {float(s * (c * delta.x - sn * delta.y) + n(rng_) * add_ * rate),
            float(s * (sn * delta.x + c * delta.y) + n(rng_) * add_ * rate)};
  }
  const std::vector<Vec2> &active_markers() { return mode_ == Mode::EUROC ? euroc_markers_ : markers_; }
  int marker_in_view() {
    auto &mk = active_markers();
    for (size_t i = 0; i < mk.size(); ++i)
      if (!used_[i] && dd(true_, mk[i]) <= 4.0) return int(i);
    return -1;
  }
  // nearest threatening pillar -> (ttc, bearing); false if clear
  bool obstacle_cue(Vec2 seek_hat, float &ttc, float &bearing) {
    double best = 1e9; bool found = false;
    for (auto &p : scene::PILLARS) {
      Vec2 to{p.x - float(true_.x), p.y - float(true_.y)};
      double dc = std::hypot(to.x, to.y), ds = dc - p.r;
      if (ds > 6.0) continue;
      Vec2 toh{float(to.x / std::max(1e-9, dc)), float(to.y / std::max(1e-9, dc))};
      double along = seek_hat.x * toh.x + seek_hat.y * toh.y;
      if (along < -0.2) continue;
      if (ds < best) {
        best = ds; found = true;
        double closing = std::max(along, 0.35) * controller_.v_max;
        ttc = frugalnav::ttc_from_range(float(ds), float(closing));
        double cross = seek_hat.x * toh.y - seek_hat.y * toh.x;
        bearing = float(std::atan2(cross, std::max(-1.0, std::min(1.0, along))));
      }
    }
    return found;
  }

  // ---------- the loop ----------
  void step() {
    if (!have_truth_) return;
    if (paused_) { publish_cmd(0, 0); publish_viz(); return; }
    if (!inited_) { prev_true_ = true_; inited_ = true; }

    double d = difficulty(true_);
    Vec2 est = fusion_.xy;

    // target for this mode
    Vec2 tgt = B_;
    if (mode_ == Mode::EUROC && !euroc_.empty()) {
      while (euroc_wp_ + 1 < euroc_.size() && dd(true_, euroc_[euroc_wp_]) < 2.0) ++euroc_wp_;
      tgt = euroc_[std::min(euroc_wp_, euroc_.size() - 1)];
    }
    Vec2 seek{tgt.x - float((mode_ == Mode::EUROC ? true_.x : est.x)),
              tgt.y - float((mode_ == Mode::EUROC ? true_.y : est.y))};
    double sn = std::max(1e-9, (double)std::hypot(seek.x, seek.y));
    Vec2 seek_hat{float(seek.x / sn), float(seek.y / sn)};

    float ttc = std::numeric_limits<float>::infinity(); float bearing = 0;
    bool maneuvering = (mode_ != Mode::MANUAL) && obstacle_cue(seek_hat, ttc, bearing);

    // 1) PREDICT the estimator on drift-corrupted true motion (all flying modes)
    double man = maneuvering ? 3.0 : 1.0;
    Vec2 td{true_.x - prev_true_.x, true_.y - prev_true_.y};
    fusion_.predict(drift_corrupt(td, (1.0 + 8.0 * d) * man),
                    float(0.16 * d + (maneuvering ? 0.10 : 0.0)));

    // 2) SCHEDULE
    frugalnav::Cues cues;
    cues.sigma_pos = fusion_.sigma_pos();
    cues.blur = float(std::max(5.0, 300.0 * (1.0 - 0.85 * d)));
    cues.feature_loss = float(15.0 * d);
    cues.imu_bias = float(0.02 + 0.05 * d);
    cues.active_features = float(150.0 * (1.0 - 0.85 * d)); cues.sigma_head = 0.01f;
    auto r = scheduler_->compute(cues);

    // 3) CORRECT
    if (r.trigger) {
      int mi = marker_in_view();
      if (mi >= 0) {
        std::normal_distribution<double> fn(0, 0.03);
        fusion_.update({float(true_.x + fn(rng_)), float(true_.y + fn(rng_))}, 0.03f * 0.03f);
        scheduler_->reset_after_fix(); yaw_ *= 0.1; log_scale_ *= 0.1;
        used_[mi] = 1; corr_.push_back(fusion_.xy); ++fixes_;
      }
    }
    est = fusion_.xy;

    // 4/5) COMMAND per mode
    Vec2 cmd{0, 0};
    if (mode_ == Mode::MANUAL) {
      teleop_age_ += 1; if (teleop_age_ > 12) teleop_ = {0, 0};   // decay if no keypress
      cmd = teleop_;
    } else {
      Vec2 evade = avoider_.update(seek_hat, ttc, bearing);
      auto vc = controller_.command((mode_ == Mode::EUROC ? true_ : est), evade);
      // in EUROC we steer by truth toward the moving waypoint (faithful replay)
      if (mode_ == Mode::EUROC) {
        Vec2 v{controller_.v_max * seek_hat.x + evade.x, controller_.v_max * seek_hat.y + evade.y};
        float s = std::hypot(v.x, v.y); if (s > controller_.v_max) v = {v.x/s*controller_.v_max, v.y/s*controller_.v_max};
        cmd = v;
      } else cmd = {vc.vx, vc.vy};
      if (mode_ == Mode::AUTO && controller_.arrived(est)) { arrived_ = true; cmd = {0, 0}; }
      if (mode_ == Mode::EUROC && euroc_wp_ + 1 >= euroc_.size()) cmd = {0, 0};
    }
    publish_cmd(cmd.x, cmd.y);

    est_path_.push_back(est); true_path_.push_back(true_);
    last_U_ = r.U; last_evading_ = avoider_.evading;
    publish_viz(); broadcast_tf(est);
    prev_true_ = true_;
  }

  // ---------- publishing ----------
  void publish_cmd(double x, double y) {
    geometry_msgs::msg::Twist t; t.linear.x = x; t.linear.y = y; cmd_pub_->publish(t);
  }
  visualization_msgs::msg::Marker mk(int id, int type, double sx, double sy, double sz,
                                     float r, float g, float b, float a) {
    visualization_msgs::msg::Marker m; m.header.frame_id = "world"; m.header.stamp = now();
    m.ns = "fn"; m.id = id; m.type = type; m.action = 0;
    m.scale.x = sx; m.scale.y = sy; m.scale.z = sz;
    m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = a; m.pose.orientation.w = 1; return m;
  }
  void publish_scene() {
    visualization_msgs::msg::MarkerArray a;
    auto hard = mk(1, visualization_msgs::msg::Marker::CYLINDER, 2*scene::HARD_R, 2*scene::HARD_R, 0.05,
                   0.96, 0.62, 0.04, 0.16);
    hard.pose.position.x = hard_c_.x; hard.pose.position.y = hard_c_.y; a.markers.push_back(hard);
    int id = 10;
    for (auto &p : scene::PILLARS) {
      auto c = mk(id++, visualization_msgs::msg::Marker::CYLINDER, 2*p.r, 2*p.r, 2.5, 0.4, 0.4, 0.45, 0.85);
      c.pose.position.x = p.x; c.pose.position.y = p.y; c.pose.position.z = 1.25; a.markers.push_back(c);
    }
    for (auto &m : markers_) {
      auto c = mk(id++, visualization_msgs::msg::Marker::CUBE, 0.9, 0.9, 0.1, 0.9, 0.9, 0.95, 1);
      c.pose.position.x = m.x; c.pose.position.y = m.y; c.pose.position.z = 0.06; a.markers.push_back(c);
    }
    auto tgt = mk(2, visualization_msgs::msg::Marker::CYLINDER, 1.2, 1.2, 3, 0.98, 0.75, 0.14, 0.95);
    tgt.pose.position.x = B_.x; tgt.pose.position.y = B_.y; tgt.pose.position.z = 1.5; a.markers.push_back(tgt);
    scene_pub_->publish(a);
  }
  visualization_msgs::msg::Marker line(int id, const std::vector<Vec2>&pts, float r, float g, float b) {
    auto m = mk(id, visualization_msgs::msg::Marker::LINE_STRIP, 0.18, 0, 0, r, g, b, 0.95);
    for (auto&p:pts){geometry_msgs::msg::Point q;q.x=p.x;q.y=p.y;q.z=0.12;m.points.push_back(q);} return m;
  }
  void publish_viz() {
    visualization_msgs::msg::MarkerArray a;
    a.markers.push_back(line(200, true_path_, 0.20, 0.83, 0.44));
    a.markers.push_back(line(201, est_path_, 0.22, 0.74, 0.97));
    if (mode_ == Mode::EUROC) {                          // show the EuRoC route
      auto rt = line(205, euroc_, 0.55, 0.55, 0.62); rt.color.a = 0.35; a.markers.push_back(rt);
    }
    auto dr = mk(202, visualization_msgs::msg::Marker::SPHERE, 1.3, 1.3, 1.3, 0.22, 0.74, 0.97, 1);
    dr.pose.position.x = true_.x; dr.pose.position.y = true_.y; dr.pose.position.z = 0.6;
    if (last_evading_) { dr.color.r = 0.98; dr.color.g = 0.45; dr.color.b = 0.45; }
    a.markers.push_back(dr);
    auto fx = mk(203, visualization_msgs::msg::Marker::SPHERE_LIST, 0.9, 0.9, 0.9, 0.99, 0.85, 0.14, 1);
    for (auto&c:corr_){geometry_msgs::msg::Point q;q.x=c.x;q.y=c.y;q.z=0.2;fx.points.push_back(q);}
    a.markers.push_back(fx);
    auto tx = mk(204, visualization_msgs::msg::Marker::TEXT_VIEW_FACING, 1, 1, 2.4, 0.92, 0.95, 0.98, 1);
    tx.pose.position.x = start_.x; tx.pose.position.y = start_.y + 4; tx.pose.position.z = 3.5;
    char b[160]; std::snprintf(b, sizeof(b), "MODE: %s%s\nU=%.2f  fixes=%d",
                               mode_str(), paused_ ? " (PAUSED)" : "", last_U_, fixes_);
    tx.text = b; a.markers.push_back(tx);
    viz_pub_->publish(a);
    std_msgs::msg::Float32 u; u.data = last_U_; u_pub_->publish(u);
  }
  void broadcast_tf(Vec2 est) {
    geometry_msgs::msg::TransformStamped t; t.header.stamp = now();
    t.header.frame_id = "world"; t.child_frame_id = "base_link";
    t.transform.translation.x = true_.x; t.transform.translation.y = true_.y; t.transform.rotation.w = 1;
    tf_->sendTransform(t);
    auto e = t; e.child_frame_id = "frugalnav_estimate";
    e.transform.translation.x = est.x; e.transform.translation.y = est.y; tf_->sendTransform(e);
  }

  // ---------- state ----------
  std::string euroc_csv_;
  Mode mode_ = Mode::AUTO; bool paused_ = false;
  Vec2 B_, start_, hard_c_, prev_true_{}, true_{}, teleop_{0, 0};
  int teleop_age_ = 999;
  std::vector<Vec2> markers_, euroc_, euroc_markers_, est_path_, true_path_, corr_;
  std::vector<int> used_; size_t euroc_wp_ = 0;
  double yaw_ = 0, log_scale_ = 0;
  double yaw_bias_ = 0.00004, yaw_rw_ = 0.00022, scale_rw_ = 0.00012, add_ = 0.0022;
  bool have_truth_ = false, inited_ = false, arrived_ = false, last_evading_ = false;
  float last_U_ = 0; int fixes_ = 0;
  std::mt19937 rng_;
  frugalnav::Controller controller_; frugalnav::StateFusion fusion_;
  frugalnav::ObstacleAvoidance avoider_;
  std::unique_ptr<frugalnav::UncertaintyScheduler> scheduler_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr u_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr scene_pub_, viz_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr truth_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr teleop_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr ctrl_sub_;
  rclcpp::Client<gazebo_msgs::srv::SetEntityState>::SharedPtr set_state_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FrugalNavInteractive>());
  rclcpp::shutdown();
  return 0;
}
