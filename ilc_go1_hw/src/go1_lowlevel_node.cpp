// go1_lowlevel_node: the Go1's low-level interface in ROS 1 (on the robot's own computer), over unitree_legged_sdk 3.8.
//
// The same topics and safety as the ROS 2 go1_bridge.py (ilc_quad), so the jump controllers see the robot exactly as they
// saw the simulated one, plus the state-estimation inputs the car's hardware layer has (ilc_f1tenth_hw):
//
//   UDP in       LowState (every cycle, rate_hz)
//   subscribes   joint_cmd      std_msgs/Float64MultiArray, [q, dq, kp, kd, tau] x 12, field-major, canonical order
//   publishes    joint_states   sensor_msgs/JointState, canonical order and names; effort = tauEst
//                foot_contacts  std_msgs/Float64MultiArray, raw foot forces FL FR RL RR
//                imu            sensor_msgs/Imu (frame imu_link = the trunk origin): orientation in ROS order (x, y, z, w)
//                               -- the SDK's quaternion order is DETECTED against its own rpy (see go1_core.h) unless
//                               quat_order is set; gyro rad/s; accelerometer m/s^2 raw (gravity included)
//                leg_odom       geometry_msgs/TwistWithCovarianceStamped (frame trunk): the trunk velocity from the joint
//                               angles and rates of the stance feet and the gyro, v = -(J dq + w x p_foot); published
//                               while >= min_contacts feet are down; covariance grows with the stance feet's disagreement
//                zupt           geometry_msgs/TwistWithCovarianceStamped: zero velocity while all feet are down and joints
//                               and trunk are still (the car's ZUPT)
//                contact_state  std_msgs/UInt8MultiArray, the hysteresis contact flags FL FR RL RR
//   service      clear_fault    std_srvs/Trigger: release a latched joint-limit fault
//   UDP out      LowCmd, one per cycle
//
// Safety (independent of the controller, as go1_bridge.py): watchdog (no joint_cmd for cmd_timeout -> damping), joint
// limits (a joint > limit_margin past its MJCF range latches damping until /clear_fault), clipping (q to the range, kp/kd
// to [0, kp_max/kd_max], tau to the torque limits), the SDK's PowerProtect at power_level. Until the first LowState the
// SDK's idle command goes out (passive motors).
#include <ros/ros.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/UInt8MultiArray.h>
#include <std_srvs/Trigger.h>

#include <atomic>
#include <mutex>
#include <string>

#include "ilc_go1_hw/go1_core.h"
#include "unitree_legged_sdk/unitree_legged_sdk.h"

namespace sdk = UNITREE_LEGGED_SDK;
using namespace go1hw;

class Go1LowLevel {
 public:
  explicit Go1LowLevel(ros::NodeHandle& nh, ros::NodeHandle& pnh)
      : udp_(sdk::LOWLEVEL, pnh.param("local_port", 8090), pnh.param<std::string>("robot_ip", "192.168.123.10").c_str(),
             pnh.param("robot_port", 8007)),
        safe_(sdk::LeggedType::Go1) {
    rate_hz_ = pnh.param("rate_hz", 500.0);
    power_level_ = pnh.param("power_level", 10);
    cmd_timeout_ = pnh.param("cmd_timeout", 0.05);
    damp_kd_ = pnh.param("damp_kd", 2.0);
    limit_margin_ = pnh.param("limit_margin", 0.20);     // rear calves pass full extension at takeoff (see go1_bridge.py)
    kp_max_ = pnh.param("kp_max", 100.0);
    kd_max_ = pnh.param("kd_max", 10.0);
    std::string qo = pnh.param<std::string>("quat_order", "auto");   // auto | xyzw | wxyz
    quat_order_ = qo == "xyzw" ? +1 : qo == "wxyz" ? -1 : 0;
    contacts_ = ContactDetector(pnh.param("contact_on", 60.0), pnh.param("contact_off", 40.0));
    zupt_ = ZeroVelocityDetector(pnh.param("zupt_max_dq", 0.05), pnh.param("zupt_max_gyro", 0.05),
                                 pnh.param("zupt_hold_cycles", 50));
    min_contacts_ = pnh.param("min_contacts", 2);
    odom_sigma_ = pnh.param("leg_odom_sigma", 0.05);       // m/s, one stance foot
    odom_every_ = std::max(1, pnh.param("leg_odom_decimation", 2));
    imu_frame_ = pnh.param<std::string>("imu_frame", "imu_link");
    trunk_frame_ = pnh.param<std::string>("trunk_frame", "trunk");
    gyro_var_ = pnh.param("gyro_var", 1e-4);
    acc_var_ = pnh.param("acc_var", 4e-2);
    orient_var_ = pnh.param("orientation_var", 1e-2);      // the IMU attitude drifts under acceleration (SDK note)
    if (power_level_ < 1 || power_level_ > 10) throw std::runtime_error("power_level must be 1..10");

    udp_.InitCmdData(cmd_);
    udp_.InitCmdData(idle_);
    cmd_.levelFlag = idle_.levelFlag = sdk::LOWLEVEL;

    sub_cmd_ = nh.subscribe("joint_cmd", 1, &Go1LowLevel::onJointCmd, this, ros::TransportHints().tcpNoDelay());
    pub_js_ = nh.advertise<sensor_msgs::JointState>("joint_states", 1);
    pub_ff_ = nh.advertise<std_msgs::Float64MultiArray>("foot_contacts", 1);
    pub_imu_ = nh.advertise<sensor_msgs::Imu>("imu", 1);
    pub_odom_ = nh.advertise<geometry_msgs::TwistWithCovarianceStamped>("leg_odom", 1);
    pub_zupt_ = nh.advertise<geometry_msgs::TwistWithCovarianceStamped>("zupt", 1);
    pub_contact_ = nh.advertise<std_msgs::UInt8MultiArray>("contact_state", 1);
    srv_clear_ = nh.advertiseService("clear_fault", &Go1LowLevel::onClearFault, this);
    ROS_INFO("go1 low-level up: UDP %s at %.0f Hz, power level %d/10, quaternion order %s; damping until commands",
             pnh.param<std::string>("robot_ip", "192.168.123.10").c_str(), rate_hz_, power_level_, qo.c_str());
  }

  void spin() {
    ros::Rate rate(rate_hz_);
    while (ros::ok()) {
      ros::spinOnce();
      cycle();
      rate.sleep();
    }
  }

 private:
  void onJointCmd(const std_msgs::Float64MultiArray::ConstPtr& msg) {
    if (msg->data.size() != size_t(JOINT_CMD_LEN)) {
      ROS_WARN_ONCE("ignoring joint_cmd of length %zu (expected %d)", msg->data.size(), JOINT_CMD_LEN);
      return;
    }
    std::lock_guard<std::mutex> lk(mtx_);
    for (int i = 0; i < JOINT_CMD_LEN; ++i) jcmd_[i] = msg->data[i];
    have_cmd_ = true;
    cmd_time_ = ros::WallTime::now();
  }

  bool onClearFault(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& res) {
    res.success = !fault_.empty();
    res.message = fault_.empty() ? "no fault" : "cleared: " + fault_;
    fault_.clear();
    return true;
  }

  void cycle() {
    udp_.Recv();
    udp_.GetRecv(state_);
    std::array<double, NU> q{}, dq{}, tau{};
    bool any = false;
    for (int i = 0; i < NU; ++i) {
      const auto& m = state_.motorState[CANONICAL_TO_MOTOR[i]];
      q[i] = m.q;
      dq[i] = m.dq;
      tau[i] = m.tauEst;
      any = any || q[i] != 0.0;
    }
    if (!any) {                               // nothing from the robot yet: passive motors
      udp_.SetSend(idle_);
      udp_.Send();
      return;
    }
    ros::Time now = ros::Time::now();
    ++tick_;
    checkLimits(q);

    // joints + foot forces
    sensor_msgs::JointState js;
    js.header.stamp = now;
    js.name.assign(joint_names().begin(), joint_names().end());
    js.position.assign(q.begin(), q.end());
    js.velocity.assign(dq.begin(), dq.end());
    js.effort.assign(tau.begin(), tau.end());
    pub_js_.publish(js);
    std::array<double, NLEG> force{};
    std_msgs::Float64MultiArray ff;
    for (int l = 0; l < NLEG; ++l) {
      force[l] = state_.footForce[FOOT_FROM_UNITREE[l]];
      ff.data.push_back(force[l]);
    }
    pub_ff_.publish(ff);
    auto contact = contacts_.update(force);
    std_msgs::UInt8MultiArray cs;
    for (bool c : contact) cs.data.push_back(c ? 1 : 0);
    pub_contact_.publish(cs);

    // IMU (once the quaternion order is known)
    std::array<float, 4> qraw;
    std::array<float, 3> rpy;
    for (int k = 0; k < 4; ++k) qraw[k] = state_.imu.quaternion[k];
    for (int k = 0; k < 3; ++k) rpy[k] = state_.imu.rpy[k];
    if (quat_order_ == 0) {
      quat_order_ = detector_.add(qraw, rpy);
      if (quat_order_ != 0)
        ROS_WARN("IMU quaternion order detected: %s (mean rpy error xyzw %.3f rad, wxyz %.3f rad over %d samples); "
                 "pin it with quat_order:=%s", quat_order_ > 0 ? "xyzw" : "wxyz", detector_.mean_err_xyzw(),
                 detector_.mean_err_wxyz(), detector_.samples(), quat_order_ > 0 ? "xyzw" : "wxyz");
    }
    Vec3 gyro = {state_.imu.gyroscope[0], state_.imu.gyroscope[1], state_.imu.gyroscope[2]};
    if (quat_order_ != 0) {
      Quat qq = quat_from(qraw, quat_order_ > 0);
      sensor_msgs::Imu imu;
      imu.header.stamp = now;
      imu.header.frame_id = imu_frame_;
      imu.orientation.x = qq.x;
      imu.orientation.y = qq.y;
      imu.orientation.z = qq.z;
      imu.orientation.w = qq.w;
      imu.angular_velocity.x = gyro[0];
      imu.angular_velocity.y = gyro[1];
      imu.angular_velocity.z = gyro[2];
      imu.linear_acceleration.x = state_.imu.accelerometer[0];
      imu.linear_acceleration.y = state_.imu.accelerometer[1];
      imu.linear_acceleration.z = state_.imu.accelerometer[2];
      for (int k = 0; k < 3; ++k) {
        imu.orientation_covariance[4 * k] = orient_var_;
        imu.angular_velocity_covariance[4 * k] = gyro_var_;
        imu.linear_acceleration_covariance[4 * k] = acc_var_;
      }
      pub_imu_.publish(imu);
    }

    // leg odometry from the joint angles / rates of the stance feet
    if (tick_ % odom_every_ == 0) {
      LegOdom lo = leg_odometry(q, dq, gyro, contact);
      if (lo.n_contact >= min_contacts_) {
        geometry_msgs::TwistWithCovarianceStamped tw;
        tw.header.stamp = now;
        tw.header.frame_id = trunk_frame_;
        tw.twist.twist.linear.x = lo.v[0];
        tw.twist.twist.linear.y = lo.v[1];
        tw.twist.twist.linear.z = lo.v[2];
        double var = odom_sigma_ * odom_sigma_ / lo.n_contact + lo.spread * lo.spread;   // feet that slip disagree
        for (int k = 0; k < 3; ++k) tw.twist.covariance[7 * k] = var;
        for (int k = 3; k < 6; ++k) tw.twist.covariance[7 * k] = 1e6;                   // angular: not measured here
        pub_odom_.publish(tw);
      }
      if (zupt_.update(dq, gyro, contact)) {
        geometry_msgs::TwistWithCovarianceStamped z;
        z.header.stamp = now;
        z.header.frame_id = trunk_frame_;
        for (int k = 0; k < 6; ++k) z.twist.covariance[7 * k] = 1e-4;
        pub_zupt_.publish(z);
      }
    }

    fillCommand();
    safe_.PowerProtect(cmd_, state_, power_level_);
    udp_.SetSend(cmd_);
    udp_.Send();
  }

  void checkLimits(const std::array<double, NU>& q) {
    int bad = joint_past_limit(q, limit_margin_);
    if (bad >= 0 && fault_.empty()) {
      fault_ = joint_names()[bad] + "=" + std::to_string(q[bad]) + " past its limit";
      ROS_ERROR("damping latched: %s; /clear_fault to release", fault_.c_str());
    }
  }

  void fillCommand() {
    std::array<double, JOINT_CMD_LEN> c;
    bool fresh;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      c = jcmd_;
      fresh = have_cmd_ && (ros::WallTime::now() - cmd_time_).toSec() < cmd_timeout_;
    }
    bool damp = !fault_.empty() || !fresh;
    if (damp != was_damping_) {
      ROS_INFO("%s", damp ? "damping" : "following joint_cmd");
      was_damping_ = damp;
    }
    for (int i = 0; i < NU; ++i) {
      MotorTarget t = damp ? damping(damp_kd_)
                           : sanitize(i, c[i], c[NU + i], c[2 * NU + i], c[3 * NU + i], c[4 * NU + i], kp_max_, kd_max_);
      auto& m = cmd_.motorCmd[CANONICAL_TO_MOTOR[i]];
      m.mode = 0x0A;                          // closed-loop servo
      m.q = t.q;
      m.dq = t.dq;
      m.Kp = t.kp;
      m.Kd = t.kd;
      m.tau = t.tau;
    }
  }

  sdk::UDP udp_;
  sdk::Safety safe_;
  sdk::LowCmd cmd_{}, idle_{};
  sdk::LowState state_{};
  double rate_hz_, cmd_timeout_, damp_kd_, limit_margin_, kp_max_, kd_max_, odom_sigma_;
  double gyro_var_, acc_var_, orient_var_;
  int power_level_, quat_order_, min_contacts_, odom_every_;
  long tick_ = 0;
  std::string imu_frame_, trunk_frame_, fault_;
  QuatOrderDetector detector_;
  ContactDetector contacts_;
  ZeroVelocityDetector zupt_;
  std::mutex mtx_;
  std::array<double, JOINT_CMD_LEN> jcmd_{};
  bool have_cmd_ = false, was_damping_ = true;
  ros::WallTime cmd_time_;
  ros::Subscriber sub_cmd_;
  ros::Publisher pub_js_, pub_ff_, pub_imu_, pub_odom_, pub_zupt_, pub_contact_;
  ros::ServiceServer srv_clear_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "go1_lowlevel");
  ros::NodeHandle nh, pnh("~");
  try {
    Go1LowLevel node(nh, pnh);
    node.spin();
  } catch (const std::exception& e) {
    ROS_FATAL("%s", e.what());
    return 1;
  }
  return 0;
}
