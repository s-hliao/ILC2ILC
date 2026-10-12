// go1_core.h: the Go1 hardware layer's robot-independent core -- no ROS, no SDK, so it is unit-tested off the robot
// (test/test_go1_core.cpp). Everything here mirrors the simulation the controllers were trained and tested in:
//   * joint order / names / limits: ilc_quad.sim_quad_model (Menagerie unitree_go1/go1.xml) and ilc_quad.go2_lowcmd
//     (CANONICAL_TO_MOTOR, FOOT_FROM_UNITREE, the "no target" sentinels), as the ROS 2 go1_bridge.py uses them;
//   * leg kinematics: the MJCF's bodies (hip (+-0.1881, +-0.04675, 0), thigh offset +-0.08 in y, thigh and calf 0.213,
//     hip about x, thigh / calf about y; foot = the calf's foot sphere centre, radius 0.023);
//   * joint_cmd: [q, dq, kp, kd, tau] x 12, field-major, canonical order (sim_quad_model.JOINT_CMD_FIELDS).
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <string>

namespace go1hw {

constexpr int NU = 12;                    // joints
constexpr int NLEG = 4;                   // canonical legs FL, FR, RL, RR
constexpr int JOINT_CMD_LEN = 5 * NU;
constexpr float POS_STOP_F = 2.146e9f;    // Unitree's "no position target"
constexpr float VEL_STOP_F = 16000.0f;    // and "no velocity target"

// canonical joint i lives in Unitree motor slot CANONICAL_TO_MOTOR[i] (slots FR 0-2, FL 3-5, RR 6-8, RL 9-11)
constexpr std::array<int, NU> CANONICAL_TO_MOTOR = {3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8};
// canonical leg i (FL FR RL RR) reads Unitree footForce[FOOT_FROM_UNITREE[i]] (Unitree legs FR FL RR RL)
constexpr std::array<int, NLEG> FOOT_FROM_UNITREE = {1, 0, 3, 2};
inline const std::array<std::string, NU>& joint_names() {
  static const std::array<std::string, NU> n = {
      "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
      "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"};
  return n;
}
// MJCF joint ranges and actuator torque limits, canonical order
constexpr std::array<double, 3> RANGE_LO = {-0.863, -0.686, -2.818};
constexpr std::array<double, 3> RANGE_HI = {0.863, 4.501, -0.888};
constexpr std::array<double, 3> TAU_LIMIT = {23.7, 23.7, 35.55};

// ---------------------------------------------------------------------------------------------------------------
// quaternions. The SDK's header says imu.quaternion is (w, x, y, z); a recorded Go1 standing at yaw -0.72 read
// (-0.010, -0.008, -0.352, 0.936), i.e. (x, y, z, w). The order is therefore DETECTED against imu.rpy (QuatOrderDetector).
struct Quat { double w, x, y, z; };

inline Quat quat_from(const std::array<float, 4>& q, bool xyzw) {
  return xyzw ? Quat{q[3], q[0], q[1], q[2]} : Quat{q[0], q[1], q[2], q[3]};
}

inline std::array<double, 3> quat_to_rpy(const Quat& q) {     // ZYX (yaw, pitch, roll), as Unitree's rpy
  double sp = 2.0 * (q.w * q.y - q.z * q.x);
  sp = sp > 1.0 ? 1.0 : (sp < -1.0 ? -1.0 : sp);
  return {std::atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y)), std::asin(sp),
          std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))};
}

inline double wrap(double a) { return std::atan2(std::sin(a), std::cos(a)); }

inline double rpy_error(const Quat& q, const std::array<float, 3>& rpy) {
  auto r = quat_to_rpy(q);
  double e = 0.0;
  for (int i = 0; i < 3; ++i) e += std::fabs(wrap(r[i] - rpy[i]));
  return e;
}

// Accumulates samples until one order explains imu.rpy clearly better than the other.
class QuatOrderDetector {
 public:
  explicit QuatOrderDetector(int min_samples = 50, double margin = 0.2) : min_(min_samples), margin_(margin) {}
  // returns +1 once decided xyzw, -1 once decided wxyz, 0 while undecided
  int add(const std::array<float, 4>& q, const std::array<float, 3>& rpy) {
    double n = 0.0;
    for (float v : q) n += double(v) * v;
    if (std::fabs(n - 1.0) > 0.05) return decided_;          // not a unit quaternion yet (no data)
    e_xyzw_ += rpy_error(quat_from(q, true), rpy);
    e_wxyz_ += rpy_error(quat_from(q, false), rpy);
    ++n_;
    if (!decided_ && n_ >= min_) {
      double a = e_xyzw_ / n_, b = e_wxyz_ / n_;
      if (a + margin_ < b) decided_ = +1;
      else if (b + margin_ < a) decided_ = -1;
    }
    return decided_;
  }
  double mean_err_xyzw() const { return n_ ? e_xyzw_ / n_ : NAN; }
  double mean_err_wxyz() const { return n_ ? e_wxyz_ / n_ : NAN; }
  int samples() const { return n_; }

 private:
  int min_;
  double margin_;
  double e_xyzw_ = 0.0, e_wxyz_ = 0.0;
  int n_ = 0, decided_ = 0;
};

// ---------------------------------------------------------------------------------------------------------------
// leg kinematics in the trunk frame (the MJCF's trunk body; the IMU site is its origin)
using Vec3 = std::array<double, 3>;
constexpr double HIP_X = 0.1881, HIP_Y = 0.04675, THIGH_Y = 0.08, L_THIGH = 0.213, L_CALF = 0.213, FOOT_R = 0.023;
constexpr std::array<double, NLEG> SX = {+1, +1, -1, -1};      // FL FR RL RR
constexpr std::array<double, NLEG> SY = {+1, -1, +1, -1};

// foot sphere centre of canonical leg `leg` for its joints (hip, thigh, calf)
inline Vec3 foot_position(int leg, double q0, double q1, double q2) {
  // thigh frame: Ry(q1); the knee at (0, 0, -L_THIGH), the foot at Ry(q2) (0, 0, -L_CALF) from it
  double s1 = std::sin(q1), c1 = std::cos(q1), s12 = std::sin(q1 + q2), c12 = std::cos(q1 + q2);
  // Ry(a) (0, 0, -l) = (-l sin a, 0, -l cos a)
  double px = -L_THIGH * s1 - L_CALF * s12;
  double pz = -L_THIGH * c1 - L_CALF * c12;
  double py = SY[leg] * THIGH_Y;
  // hip: Rx(q0)
  double s0 = std::sin(q0), c0 = std::cos(q0);
  return {SX[leg] * HIP_X + px, SY[leg] * HIP_Y + c0 * py - s0 * pz, s0 * py + c0 * pz};
}

// d foot / d (q0, q1, q2): columns
inline std::array<Vec3, 3> foot_jacobian(int leg, double q0, double q1, double q2) {
  double s1 = std::sin(q1), c1 = std::cos(q1), s12 = std::sin(q1 + q2), c12 = std::cos(q1 + q2);
  double s0 = std::sin(q0), c0 = std::cos(q0);
  double py = SY[leg] * THIGH_Y;
  double pz = -L_THIGH * c1 - L_CALF * c12;
  // d/dq0 of (Rx(q0) (px, py, pz)) = (0, -s0 py - c0 pz, c0 py - s0 pz)
  Vec3 j0 = {0.0, -s0 * py - c0 * pz, c0 * py - s0 * pz};
  // d/dq1: px' = -Lt c1 - Lc c12, pz' = Lt s1 + Lc s12; d/dq2: px' = -Lc c12, pz' = Lc s12
  double dpx1 = -L_THIGH * c1 - L_CALF * c12, dpz1 = L_THIGH * s1 + L_CALF * s12;
  double dpx2 = -L_CALF * c12, dpz2 = L_CALF * s12;
  Vec3 j1 = {dpx1, -s0 * dpz1, c0 * dpz1};
  Vec3 j2 = {dpx2, -s0 * dpz2, c0 * dpz2};
  return {j0, j1, j2};
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

// ---------------------------------------------------------------------------------------------------------------
// leg odometry: a foot in contact is at rest in the world, so the trunk's velocity in its own frame is
//   v = -(J(q) dq + w x p_foot)     (w: the gyro, body frame)
// averaged over the stance feet. Joint arrays are canonical.
struct LegOdom {
  Vec3 v{0, 0, 0};
  int n_contact = 0;
  double spread = 0.0;                    // max deviation of a stance foot's estimate from the mean (m/s)
};

inline LegOdom leg_odometry(const std::array<double, NU>& q, const std::array<double, NU>& dq, const Vec3& gyro,
                            const std::array<bool, NLEG>& contact) {
  LegOdom out;
  std::array<Vec3, NLEG> est{};
  for (int l = 0; l < NLEG; ++l) {
    if (!contact[l]) continue;
    const double *ql = &q[3 * l], *dl = &dq[3 * l];
    Vec3 p = foot_position(l, ql[0], ql[1], ql[2]);
    auto J = foot_jacobian(l, ql[0], ql[1], ql[2]);
    Vec3 w_x_p = cross(gyro, p);
    for (int k = 0; k < 3; ++k)
      est[l][k] = -(J[0][k] * dl[0] + J[1][k] * dl[1] + J[2][k] * dl[2] + w_x_p[k]);
    for (int k = 0; k < 3; ++k) out.v[k] += est[l][k];
    ++out.n_contact;
  }
  if (out.n_contact == 0) return out;
  for (int k = 0; k < 3; ++k) out.v[k] /= out.n_contact;
  for (int l = 0; l < NLEG; ++l) {
    if (!contact[l]) continue;
    double d = 0.0;
    for (int k = 0; k < 3; ++k) d += (est[l][k] - out.v[k]) * (est[l][k] - out.v[k]);
    out.spread = std::fmax(out.spread, std::sqrt(d));
  }
  return out;
}

// foot contact from the raw force sensors, with hysteresis (on above `on`, off below `off`)
class ContactDetector {
 public:
  ContactDetector(double on = 60.0, double off = 40.0) : on_(on), off_(off) {}
  std::array<bool, NLEG> update(const std::array<double, NLEG>& force) {
    for (int l = 0; l < NLEG; ++l) {
      if (state_[l] && force[l] < off_) state_[l] = false;
      else if (!state_[l] && force[l] > on_) state_[l] = true;
    }
    return state_;
  }

 private:
  double on_, off_;
  std::array<bool, NLEG> state_{{false, false, false, false}};
};

// zero-velocity: every foot down, joints and trunk still for `hold` consecutive cycles
class ZeroVelocityDetector {
 public:
  ZeroVelocityDetector(double max_dq = 0.05, double max_gyro = 0.05, int hold = 50)
      : max_dq_(max_dq), max_gyro_(max_gyro), hold_(hold) {}
  bool update(const std::array<double, NU>& dq, const Vec3& gyro, const std::array<bool, NLEG>& contact) {
    bool still = contact[0] && contact[1] && contact[2] && contact[3];
    for (double v : dq) still = still && std::fabs(v) < max_dq_;
    for (double w : gyro) still = still && std::fabs(w) < max_gyro_;
    count_ = still ? count_ + 1 : 0;
    return count_ >= hold_;
  }

 private:
  double max_dq_, max_gyro_;
  int hold_, count_ = 0;
};

// ---------------------------------------------------------------------------------------------------------------
// the command the motors get for one canonical joint (the ROS 2 go1_bridge's _fill_command, verbatim in effect)
struct MotorTarget { float q, dq, kp, kd, tau; };

inline MotorTarget sanitize(int joint, double q, double dq, double kp, double kd, double tau, double kp_max,
                            double kd_max) {
  int j = joint % 3;
  q = std::fmin(std::fmax(q, RANGE_LO[j]), RANGE_HI[j]);
  kp = std::fmin(std::fmax(kp, 0.0), kp_max);
  kd = std::fmin(std::fmax(kd, 0.0), kd_max);
  tau = std::fmin(std::fmax(tau, -TAU_LIMIT[j]), TAU_LIMIT[j]);
  // a joint with kp = 0 takes no position target, likewise dq when kd = 0 (no action on a stale target)
  return {kp > 0 ? float(q) : POS_STOP_F, kd > 0 ? float(dq) : VEL_STOP_F, float(kp), float(kd), float(tau)};
}

inline MotorTarget damping(double damp_kd) { return {POS_STOP_F, 0.0f, 0.0f, float(damp_kd), 0.0f}; }

// index of the first joint more than `margin` past its MJCF range, or -1
inline int joint_past_limit(const std::array<double, NU>& q, double margin) {
  for (int i = 0; i < NU; ++i) {
    int j = i % 3;
    if (q[i] < RANGE_LO[j] - margin || q[i] > RANGE_HI[j] + margin) return i;
  }
  return -1;
}

}  // namespace go1hw
