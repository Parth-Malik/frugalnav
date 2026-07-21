// frugalnav_interactive_node.cpp
// -----------------------------------------------------------------------------
// Interactive FrugalNav brain. Loads its scene (obstacles/markers/target) from a
// file param, so one binary flies either map (demo or dense-canopy). Full manual
// weather + altitude control, a stern potential-field avoider for the tall (14 m)
// obstacles, and the three flight modes. The scheduler / fusion / controller are
// the unmodified C++ core (cpp/frugalnav/).
//
//   MODES:  AUTO (scheduler flies + weaves obstacles) | MANUAL (WASD) | EUROC replay
//   WEATHER (manual): wind up/down, fog up/down (visibility), rain toggle, master.
//   ALTITUDE: manual up/down, or auto (computed from visibility: clear=high, fog=low).
//   RESET = teleport to start (rewind); PAUSE.
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

using namespace std::chrono_literals;
using frugalnav::Vec2;
static double dd(Vec2 a, Vec2 b) { return std::hypot(a.x - b.x, a.y - b.y); }
struct Obst { float x, y, r; };
enum class Mode { AUTO, MANUAL, EUROC };

class FrugalNavInteractive : public rclcpp::Node {
public:
  FrugalNavInteractive() : Node("frugalnav_interactive_node"), rng_(1) {
    euroc_csv_ = declare_parameter<std::string>("gt_csv",
        "/mnt/c/Users/parth/Downloads/drone/datasets/MH_01_easy/"
        "mav0/state_groundtruth_estimate0/data.csv");
    std::string scene = declare_parameter<std::string>("scene_file", "");
    if (!scene.empty()) load_scene(scene);
    if (pillars_.empty() && markers_.empty())
      RCLCPP_WARN(get_logger(), "no scene loaded (scene_file='%s')", scene.c_str());
    load_euroc();

    controller_.B = B_; controller_.kp = 0.6f; controller_.v_max = 2.2f;
    controller_.arrive_tol = 1.5f;
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
          true_z_ = m->pose.pose.position.z; have_truth_ = true; });
    teleop_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        "/frugalnav/teleop", 20, [this](geometry_msgs::msg::Twist::SharedPtr m) {
          teleop_ = {float(m->linear.x), float(m->linear.y)}; teleop_age_ = 0; });
    ctrl_sub_ = create_subscription<std_msgs::msg::String>(
        "/frugalnav/ctrl", 20, [this](std_msgs::msg::String::SharedPtr m) { on_ctrl(m->data); });

    publish_scene();
    timer_ = create_wall_timer(33ms, [this] { step(); });
    RCLCPP_INFO(get_logger(), "Interactive node up: %zu obstacles, %zu markers. Fly from the teleop window.",
                pillars_.size(), markers_.size());
  }

private:
  // ---------- scene / setup ----------
  void load_scene(const std::string &path) {
    std::ifstream f(path);
    if (!f) { RCLCPP_ERROR(get_logger(), "cannot open scene_file %s", path.c_str()); return; }
    std::string line, tok;
    while (std::getline(f, line)) {
      std::stringstream ss(line); ss >> tok;
      if (tok == "target") { float x, y; ss >> x >> y; B_ = {x, y}; }
      else if (tok == "start") { float x, y; ss >> x >> y; start_ = {x, y}; }
      else if (tok == "hard") { ss >> hard_x_ >> hard_y_ >> hard_r_; }
      else if (tok == "pillar") { float x, y, r; ss >> x >> y >> r; pillars_.push_back({x, y, r}); }
      else if (tok == "marker") { float x, y; ss >> x >> y; markers_.push_back({x, y}); }
    }
  }
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
    Vec2 o = raw[0]; float sc = 2.2f;
    for (auto &p : raw) euroc_.push_back({start_.x + (p.x - o.x) * sc, start_.y + (p.y - o.y) * sc});
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
    used_.assign(std::max(markers_.size(), euroc_markers_.size()) + 1, 0); euroc_wp_ = 0;
  }
  void teleport(Vec2 xy, double z) {
    if (!set_state_->service_is_ready()) return;
    auto req = std::make_shared<gazebo_msgs::srv::SetEntityState::Request>();
    req->state.name = "frugalnav_drone"; req->state.reference_frame = "world";
    req->state.pose.position.x = xy.x; req->state.pose.position.y = xy.y;
    req->state.pose.position.z = z; req->state.pose.orientation.w = 1.0;
    req->state.twist.linear.x = vel_.x; req->state.twist.linear.y = vel_.y;
    set_state_->async_send_request(req);
  }
  void on_ctrl(const std::string &c) {
    if (c == "auto") { mode_ = Mode::AUTO; arrived_ = false; }
    else if (c == "manual") mode_ = Mode::MANUAL;
    else if (c == "euroc") { mode_ = Mode::EUROC; reset_estimator(); teleport(start_, 5.0); }
    else if (c == "reset") { reset_estimator(); teleport(start_, alt()); }
    else if (c == "pause") paused_ = true;
    else if (c == "resume") paused_ = false;
    else if (c == "weather") weather_on_ = !weather_on_;
    else if (c == "wind_up") wind_speed_ = std::min(6.0, wind_speed_ + 0.4);
    else if (c == "wind_down") wind_speed_ = std::max(0.0, wind_speed_ - 0.4);
    else if (c == "fog_up") visibility_ = std::max(0.1, visibility_ - 0.12);   // more fog
    else if (c == "fog_down") visibility_ = std::min(1.0, visibility_ + 0.12); // clearer
    else if (c == "rain") rain_ = !rain_;
    else if (c == "alt_up") { alt_manual_ = true; alt_cmd_ = std::min(12.0, alt_cmd_ + 1.0); }
    else if (c == "alt_down") { alt_manual_ = true; alt_cmd_ = std::max(1.0, alt_cmd_ - 1.0); }
    else if (c == "alt_auto") alt_manual_ = false;
    RCLCPP_INFO(get_logger(), "ctrl=%s | mode=%s wx=%.1f vis=%.2f rain=%d altM=%d",
                c.c_str(), mode_str(), wind_speed_, visibility_, rain_, alt_manual_);
  }
  const char *mode_str() { return mode_ == Mode::AUTO ? "AUTO" : mode_ == Mode::MANUAL ? "MANUAL" : "EUROC"; }

  // ---------- environment ----------
  double eff_vis() { if (!weather_on_) return 1.0; return std::clamp(visibility_ * (rain_ ? 0.6 : 1.0), 0.1, 1.0); }
  double alt() { return alt_manual_ ? alt_cmd_ : (2.0 + 6.0 * eff_vis()); }
  void update_wind() {
    gust_t_ += 0.033;
    if (!weather_on_) { wind_ = {0, 0}; return; }
    double sp = wind_speed_ * (0.7 + 0.3 * std::sin(gust_t_ * 0.7)) + 0.15 * std::sin(gust_t_ * 2.3);
    double dir = wind_dir_ + 0.3 * std::sin(gust_t_ * 0.25);
    wind_ = {float(sp * std::cos(dir)), float(sp * std::sin(dir))};
  }
  void apply_altitude() {
    if (++alt_tick_ < 10) return; alt_tick_ = 0;
    double a = alt();
    if (std::fabs(true_z_ - a) > 0.3) teleport(true_, a);
  }
  double marker_range() { double v = eff_vis(); return (2.6 + 3.4 * v) + (8.0 - alt()) * 0.22; }
  double difficulty(Vec2 p) {
    double d = std::hypot(p.x - hard_x_, p.y - hard_y_);
    return d >= hard_r_ ? 0.0 : 0.9 * (1.0 - d / hard_r_);
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
    auto &mk = active_markers(); double R = marker_range();
    for (size_t i = 0; i < mk.size(); ++i)
      if (!used_[i] && dd(true_, mk[i]) <= R) return int(i);
    return -1;
  }
  // STERN potential field over all obstacles (tall -> real hazards). Larger obstacles
  // repel from farther; the seek component heading INTO an obstacle is cancelled.
  Vec2 auto_command(Vec2 seek_hat) {
    Vec2 v{seek_hat.x * controller_.v_max, seek_hat.y * controller_.v_max};
    bool near = false;
    const double PAD = 3.6, K = 4.5, MAXR = 6.0;
    for (auto &p : pillars_) {
      double tx = true_.x - p.x, ty = true_.y - p.y;
      double dist = std::hypot(tx, ty), ds = dist - p.r, INFL = p.r + PAD;
      if (ds >= PAD) continue;
      near = true;
      double ax = tx / std::max(1e-6, dist), ay = ty / std::max(1e-6, dist);
      double mag = std::min(MAXR, K * (1.0 / std::max(ds, 0.2) - 1.0 / INFL));
      v.x += ax * mag; v.y += ay * mag;
      double into = -(v.x * ax + v.y * ay);
      if (into > 0) { v.x += ax * into; v.y += ay * into; }   // fully cancel into-obstacle motion
    }
    last_evading_ = near;
    double s = std::hypot(v.x, v.y), cap = controller_.v_max * 2.0;
    if (s > cap) { v.x = v.x / s * cap; v.y = v.y / s * cap; }
    return v;
  }

  // ---------- loop ----------
  void step() {
    if (!have_truth_) return;
    update_wind();
    if (paused_) { publish_cmd(0, 0); publish_viz(); return; }
    if (!inited_) { prev_true_ = true_; inited_ = true; }
    vel_ = {float((true_.x - prev_true_.x) / 0.033), float((true_.y - prev_true_.y) / 0.033)};
    apply_altitude();

    double d = difficulty(true_);
    double v = eff_vis();
    double vd = std::min(1.0, d + (1.0 - v) * 0.8 + (rain_ ? 0.15 : 0.0));

    Vec2 est = fusion_.xy, tgt = B_;
    if (mode_ == Mode::EUROC && !euroc_.empty()) {
      while (euroc_wp_ + 1 < euroc_.size() && dd(true_, euroc_[euroc_wp_]) < 2.0) ++euroc_wp_;
      tgt = euroc_[std::min(euroc_wp_, euroc_.size() - 1)];
    }
    Vec2 base{tgt.x - float((mode_ == Mode::EUROC ? true_.x : est.x)),
              tgt.y - float((mode_ == Mode::EUROC ? true_.y : est.y))};
    double sn = std::max(1e-9, (double)std::hypot(base.x, base.y));
    Vec2 seek_hat{float(base.x / sn), float(base.y / sn)};

    fusion_.predict(drift_corrupt({true_.x - prev_true_.x, true_.y - prev_true_.y}, 1.0 + 8.0 * vd),
                    float(0.16 * vd));

    frugalnav::Cues cues;
    cues.sigma_pos = fusion_.sigma_pos();
    cues.blur = float(std::max(5.0, 300.0 * (1.0 - 0.85 * d) * (0.35 + 0.65 * v)));
    cues.feature_loss = float(15.0 * d + (1.0 - v) * 12.0 + (rain_ ? 6.0 : 0.0));
    cues.imu_bias = float(0.02 + 0.05 * d);
    cues.active_features = float(150.0 * (1.0 - 0.85 * vd)); cues.sigma_head = 0.01f;
    auto r = scheduler_->compute(cues);

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

    Vec2 cmd{0, 0};
    if (mode_ == Mode::MANUAL) {
      teleop_age_ += 1; if (teleop_age_ > 12) teleop_ = {0, 0};
      cmd = teleop_; last_evading_ = false;
    } else if (mode_ == Mode::EUROC) {
      cmd = {float(controller_.v_max * seek_hat.x), float(controller_.v_max * seek_hat.y)};
      last_evading_ = false;
      if (euroc_wp_ + 1 >= euroc_.size()) cmd = {0, 0};
    } else {
      if (controller_.arrived(est)) { arrived_ = true; cmd = {0, 0}; last_evading_ = false; }
      else cmd = auto_command(seek_hat);
    }
    cmd.x += wind_.x; cmd.y += wind_.y;
    publish_cmd(cmd.x, cmd.y);

    est_path_.push_back(est); true_path_.push_back(true_);
    last_U_ = r.U; publish_viz(); broadcast_tf(est);
    prev_true_ = true_;
  }

  // ---------- publishing ----------
  void publish_cmd(double x, double y) {
    geometry_msgs::msg::Twist t; t.linear.x = x; t.linear.y = y; cmd_pub_->publish(t);
  }
  visualization_msgs::msg::Marker mkr(int id, int type, double sx, double sy, double sz,
                                      float r, float g, float b, float a) {
    visualization_msgs::msg::Marker m; m.header.frame_id = "world"; m.header.stamp = now();
    m.ns = "fn"; m.id = id; m.type = type; m.action = 0;
    m.scale.x = sx; m.scale.y = sy; m.scale.z = sz;
    m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = a; m.pose.orientation.w = 1; return m;
  }
  void publish_scene() {
    visualization_msgs::msg::MarkerArray a; int id = 20;
    auto hard = mkr(1, visualization_msgs::msg::Marker::CYLINDER, 2*hard_r_, 2*hard_r_, 0.05, 0.96, 0.62, 0.04, 0.16);
    hard.pose.position.x = hard_x_; hard.pose.position.y = hard_y_; a.markers.push_back(hard);
    for (auto &p : pillars_) {
      auto c = mkr(id++, visualization_msgs::msg::Marker::CYLINDER, 2*p.r, 2*p.r, 12.0, 0.4, 0.4, 0.45, 0.75);
      c.pose.position.x = p.x; c.pose.position.y = p.y; c.pose.position.z = 6.0; a.markers.push_back(c);
    }
    for (auto &m : markers_) {
      auto c = mkr(id++, visualization_msgs::msg::Marker::CUBE, 0.9, 0.9, 0.1, 0.9, 0.9, 0.95, 1);
      c.pose.position.x = m.x; c.pose.position.y = m.y; c.pose.position.z = 0.06; a.markers.push_back(c);
    }
    auto tgt = mkr(2, visualization_msgs::msg::Marker::CYLINDER, 1.2, 1.2, 3, 0.98, 0.75, 0.14, 0.95);
    tgt.pose.position.x = B_.x; tgt.pose.position.y = B_.y; tgt.pose.position.z = 1.5; a.markers.push_back(tgt);
    scene_pub_->publish(a);
  }
  visualization_msgs::msg::Marker line(int id, const std::vector<Vec2>&pts, float r, float g, float b) {
    auto m = mkr(id, visualization_msgs::msg::Marker::LINE_STRIP, 0.18, 0, 0, r, g, b, 0.95);
    for (auto&p:pts){geometry_msgs::msg::Point q;q.x=p.x;q.y=p.y;q.z=0.12;m.points.push_back(q);} return m;
  }
  void publish_viz() {
    visualization_msgs::msg::MarkerArray a;
    a.markers.push_back(line(200, true_path_, 0.20, 0.83, 0.44));
    a.markers.push_back(line(201, est_path_, 0.22, 0.74, 0.97));
    if (mode_ == Mode::EUROC) { auto rt = line(205, euroc_, 0.55, 0.55, 0.62); rt.color.a = 0.35; a.markers.push_back(rt); }
    double A = std::max(0.5, alt());
    auto dr = mkr(202, visualization_msgs::msg::Marker::SPHERE, 1.3, 1.3, 1.3, 0.22, 0.74, 0.97, 1);
    dr.pose.position.x = true_.x; dr.pose.position.y = true_.y; dr.pose.position.z = A;
    if (last_evading_) { dr.color.r = 0.98; dr.color.g = 0.45; dr.color.b = 0.45; }
    a.markers.push_back(dr);
    auto fx = mkr(203, visualization_msgs::msg::Marker::SPHERE_LIST, 0.9, 0.9, 0.9, 0.99, 0.85, 0.14, 1);
    for (auto&c:corr_){geometry_msgs::msg::Point q;q.x=c.x;q.y=c.y;q.z=0.2;fx.points.push_back(q);}
    a.markers.push_back(fx);
    if (weather_on_ && (std::hypot(wind_.x, wind_.y) > 0.05)) {
      auto w = mkr(206, visualization_msgs::msg::Marker::ARROW, 0.3, 0.6, 0.6, 0.6, 0.8, 1.0, 0.9);
      geometry_msgs::msg::Point p0, p1; p0.x = true_.x; p0.y = true_.y; p0.z = A + 2;
      p1.x = true_.x + wind_.x * 1.5; p1.y = true_.y + wind_.y * 1.5; p1.z = A + 2;
      w.points.push_back(p0); w.points.push_back(p1); a.markers.push_back(w);
    }
    auto tx = mkr(204, visualization_msgs::msg::Marker::TEXT_VIEW_FACING, 1, 1, 2.0, 0.92, 0.95, 0.98, 1);
    tx.pose.position.x = start_.x; tx.pose.position.y = start_.y; tx.pose.position.z = A + 5;
    char b[240]; std::snprintf(b, sizeof(b),
        "MODE %s%s  fixes=%d U=%.2f\nweather %s  wind=%.1f  vis=%.0f%%  rain=%s\nalt=%.1fm (%s)",
        mode_str(), paused_ ? " PAUSED" : "", fixes_, last_U_,
        weather_on_ ? "ON" : "off", std::hypot(wind_.x, wind_.y), eff_vis() * 100.0,
        rain_ ? "ON" : "off", alt(), alt_manual_ ? "manual" : "auto");
    tx.text = b; a.markers.push_back(tx);
    viz_pub_->publish(a);
    std_msgs::msg::Float32 u; u.data = last_U_; u_pub_->publish(u);
  }
  void broadcast_tf(Vec2 est) {
    geometry_msgs::msg::TransformStamped t; t.header.stamp = now();
    t.header.frame_id = "world"; t.child_frame_id = "base_link";
    t.transform.translation.x = true_.x; t.transform.translation.y = true_.y;
    t.transform.translation.z = std::max(0.5, alt()); t.transform.rotation.w = 1;
    tf_->sendTransform(t);
    auto e = t; e.child_frame_id = "frugalnav_estimate";
    e.transform.translation.x = est.x; e.transform.translation.y = est.y; tf_->sendTransform(e);
  }

  // ---------- state ----------
  std::string euroc_csv_;
  Mode mode_ = Mode::AUTO; bool paused_ = false;
  // scene
  Vec2 B_{0, 0}, start_{0, 0}; double hard_x_ = 0, hard_y_ = 0, hard_r_ = 1;
  std::vector<Obst> pillars_; std::vector<Vec2> markers_, euroc_, euroc_markers_;
  // weather (manual) + altitude
  bool weather_on_ = true, rain_ = false, alt_manual_ = false;
  double wind_speed_ = 0.4, wind_dir_ = 2.4, visibility_ = 0.85, alt_cmd_ = 6.0, gust_t_ = 0;
  Vec2 wind_{0, 0};
  // runtime
  Vec2 prev_true_{}, true_{}, teleop_{0, 0}, vel_{0, 0};
  int teleop_age_ = 999, alt_tick_ = 0; float true_z_ = 0.5f;
  std::vector<Vec2> est_path_, true_path_, corr_; std::vector<int> used_; size_t euroc_wp_ = 0;
  double yaw_ = 0, log_scale_ = 0;
  double yaw_bias_ = 0.00004, yaw_rw_ = 0.00022, scale_rw_ = 0.00012, add_ = 0.0022;
  bool have_truth_ = false, inited_ = false, arrived_ = false, last_evading_ = false;
  float last_U_ = 0; int fixes_ = 0;
  std::mt19937 rng_;
  frugalnav::Controller controller_; frugalnav::StateFusion fusion_;
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
