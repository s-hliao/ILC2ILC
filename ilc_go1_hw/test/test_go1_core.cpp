// Off-robot tests of go1_core.h (no ROS, no SDK):  g++ -std=c++14 -O2 -I../include test_go1_core.cpp -o t && ./t
#include <cstdio>
#include <cstdlib>
#include <random>

#include "fk_reference.h"
#include "ilc_go1_hw/go1_core.h"

using namespace go1hw;
static int fails = 0;
#define CHECK(cond, ...)                       \
  do {                                         \
    if (!(cond)) {                             \
      ++fails;                                 \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
      std::printf(__VA_ARGS__);                \
      std::printf("\n");                       \
    }                                          \
  } while (0)

int main() {
  // 1. forward kinematics == MuJoCo's foot sites (Menagerie go1.xml), 20 random configurations x 4 legs
  double worst = 0.0;
  for (int n = 0; n < FK_N; ++n)
    for (int l = 0; l < NLEG; ++l) {
      Vec3 p = foot_position(l, FK_Q[n][3 * l], FK_Q[n][3 * l + 1], FK_Q[n][3 * l + 2]);
      for (int k = 0; k < 3; ++k) worst = std::fmax(worst, std::fabs(p[k] - FK_FEET[n][l][k]));
    }
  CHECK(worst < 1e-6, "FK vs MuJoCo: worst %.3g m", worst);
  std::printf("FK vs MuJoCo: worst error %.2e m over %d configurations\n", worst, FK_N);

  // 2. analytic Jacobian == central finite differences
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  double jw = 0.0;
  for (int t = 0; t < 200; ++t) {
    int l = t % 4;
    double q[3] = {0.8 * U(rng), 1.2 + 1.5 * U(rng), -1.8 + 0.8 * U(rng)};
    auto J = foot_jacobian(l, q[0], q[1], q[2]);
    for (int j = 0; j < 3; ++j) {
      double qp[3] = {q[0], q[1], q[2]}, qm[3] = {q[0], q[1], q[2]};
      qp[j] += 1e-6;
      qm[j] -= 1e-6;
      Vec3 a = foot_position(l, qp[0], qp[1], qp[2]), b = foot_position(l, qm[0], qm[1], qm[2]);
      for (int k = 0; k < 3; ++k) jw = std::fmax(jw, std::fabs((a[k] - b[k]) / 2e-6 - J[j][k]));
    }
  }
  CHECK(jw < 1e-7, "Jacobian vs finite differences: worst %.3g", jw);
  std::printf("Jacobian vs finite differences: worst %.2e\n", jw);

  // 3. leg odometry: a trunk moving at v with rate w over feet fixed in the world -> the joint rates that keep the feet
  //    still; leg_odometry must return v
  {
    std::array<double, NU> q{}, dq{};
    for (int l = 0; l < NLEG; ++l) { q[3 * l] = 0.1 * (l - 1.5); q[3 * l + 1] = 0.9; q[3 * l + 2] = -1.7; }
    Vec3 v = {0.4, -0.1, 0.25}, w = {0.3, -0.5, 0.2};
    for (int l = 0; l < NLEG; ++l) {
      // the foot's velocity in the body frame must be -(v + w x p): solve J dq = -(v + w x p)
      Vec3 p = foot_position(l, q[3 * l], q[3 * l + 1], q[3 * l + 2]);
      Vec3 wxp = cross(w, p);
      Vec3 rhs = {-(v[0] + wxp[0]), -(v[1] + wxp[1]), -(v[2] + wxp[2])};
      auto J = foot_jacobian(l, q[3 * l], q[3 * l + 1], q[3 * l + 2]);
      // 3x3 solve (Cramer); columns J[0..2]
      auto det = [](const Vec3& a, const Vec3& b, const Vec3& c) {
        return a[0] * (b[1] * c[2] - b[2] * c[1]) - b[0] * (a[1] * c[2] - a[2] * c[1]) + c[0] * (a[1] * b[2] - a[2] * b[1]);
      };
      double D = det(J[0], J[1], J[2]);
      dq[3 * l] = det(rhs, J[1], J[2]) / D;
      dq[3 * l + 1] = det(J[0], rhs, J[2]) / D;
      dq[3 * l + 2] = det(J[0], J[1], rhs) / D;
    }
    LegOdom o = leg_odometry(q, dq, w, {{true, true, true, true}});
    double e = std::fabs(o.v[0] - v[0]) + std::fabs(o.v[1] - v[1]) + std::fabs(o.v[2] - v[2]);
    CHECK(e < 1e-9 && o.n_contact == 4 && o.spread < 1e-9, "leg odometry: |v - v_true| = %.3g, spread %.3g", e, o.spread);
    LegOdom o2 = leg_odometry(q, dq, w, {{true, false, false, true}});
    CHECK(o2.n_contact == 2 && std::fabs(o2.v[0] - v[0]) < 1e-9, "leg odometry, two feet");
    std::printf("leg odometry: recovered v = (%.3f, %.3f, %.3f), true (0.400, -0.100, 0.250)\n", o.v[0], o.v[1], o.v[2]);
  }

  // 4. quaternion order: the recorded Go1 standing still (SDK header claims wxyz; the data is xyzw)
  {
    QuatOrderDetector det(50);
    std::array<float, 4> q = {-0.010f, -0.008f, -0.352f, 0.936f};
    std::array<float, 3> rpy = {-0.012f, -0.022f, -0.720f};
    int d = 0;
    for (int i = 0; i < 60; ++i) d = det.add(q, rpy);
    CHECK(d == +1, "recorded Go1 quaternion: expected xyzw, got %d (err xyzw %.3f, wxyz %.3f)", d, det.mean_err_xyzw(),
          det.mean_err_wxyz());
    std::printf("quaternion order on the recorded Go1 data: %s (mean rpy error xyzw %.3f rad, wxyz %.3f rad)\n",
                d == 1 ? "xyzw" : d == -1 ? "wxyz" : "undecided", det.mean_err_xyzw(), det.mean_err_wxyz());
    // a true wxyz stream must be detected as wxyz
    QuatOrderDetector det2(50);
    std::array<float, 4> qw = {0.936f, -0.010f, -0.008f, -0.352f};
    for (int i = 0; i < 60; ++i) d = det2.add(qw, rpy);
    CHECK(d == -1, "wxyz stream: expected wxyz, got %d", d);
    // no data (all zeros) never decides
    QuatOrderDetector det3(5);
    for (int i = 0; i < 20; ++i) d = det3.add({0, 0, 0, 0}, {0, 0, 0});
    CHECK(d == 0, "zero quaternion must not decide");
  }

  // 5. commands: clipping, stop sentinels, limits
  {
    MotorTarget m = sanitize(2, -3.5, 1.0, 150.0, 20.0, 99.0, 100.0, 10.0);
    CHECK(m.q == float(RANGE_LO[2]) && m.kp == 100.0f && m.kd == 10.0f && m.tau == float(TAU_LIMIT[2]), "clip");
    MotorTarget z = sanitize(0, 0.3, 0.5, 0.0, 0.0, 1.0, 100.0, 10.0);
    CHECK(z.q == POS_STOP_F && z.dq == VEL_STOP_F && z.tau == 1.0f, "kp = kd = 0 -> stop sentinels");
    std::array<double, NU> q{};
    for (int l = 0; l < NLEG; ++l) { q[3 * l + 1] = 0.8; q[3 * l + 2] = -1.6; }
    CHECK(joint_past_limit(q, 0.2) == -1, "standing pose within limits");
    q[11] = -0.888 + 0.25;                   // RR calf over-extended past -0.888 by 0.25 > 0.20
    CHECK(joint_past_limit(q, 0.2) == 11, "calf past limit");
    std::array<double, JOINT_CMD_LEN> cmd{};
    cmd[11] = -1.8;                           // valid calf target, back into range
    cmd[2 * NU + 11] = 20.0;
    CHECK(command_recovers_limits(q, cmd, 0.2), "inward position command permits recovery");
    cmd[11] = -0.5;                           // raw outward target is clipped to RANGE_HI, still inward from current q
    CHECK(command_recovers_limits(q, cmd, 0.2), "clipped target permits recovery");
    cmd[2 * NU + 11] = 0.0;
    cmd[3 * NU + 11] = 2.0;
    CHECK(command_recovers_limits(q, cmd, 0.2), "pure damping over an out-of-range joint is accepted");
    cmd[4 * NU + 11] = 1.0;
    CHECK(!command_recovers_limits(q, cmd, 0.2), "feedforward torque without a position servo is not recovery");
    cmd[4 * NU + 11] = 0.0;
    cmd[NU + 11] = 1.0;
    CHECK(!command_recovers_limits(q, cmd, 0.2), "damping toward a nonzero velocity is not recovery");
    cmd[NU + 11] = 0.0;
    cmd[2 * NU + 11] = 20.0;
    cmd[11] = -0.5;
    q[11] = -0.888 - 0.5;                     // inside the range: an outward target is not checked here
    CHECK(!command_recovers_limits(q, cmd, 0.2), "nothing out of range -> not a recovery");
    q[11] = -1.6;
    q[3] = 1.13244;                           // measured folded Go1 FR hip
    cmd = {};
    cmd[3] = q[3];                            // sanitize() clips this target to RANGE_HI
    cmd[2 * NU + 3] = 1.0;
    CHECK(command_recovers_limits(q, cmd, 0.2), "folded Go1 hip can move back into range");
  }

  // 6. contact hysteresis, zero-velocity
  {
    ContactDetector c(60, 40);
    auto s = c.update({70, 50, 30, 61});
    CHECK(s[0] && !s[1] && !s[2] && s[3], "contact on");
    s = c.update({45, 50, 30, 39});
    CHECK(s[0] && !s[1] && !s[2] && !s[3], "contact hysteresis");
    ZeroVelocityDetector z(0.05, 0.05, 3);
    std::array<double, NU> dq{};
    bool r = false;
    for (int i = 0; i < 3; ++i) r = z.update(dq, {0.01, 0, 0}, {{true, true, true, true}});
    CHECK(r, "zupt after hold");
    CHECK(!z.update(dq, {0.2, 0, 0}, {{true, true, true, true}}), "zupt drops on rotation");
  }

  std::printf(fails ? "%d FAILED\n" : "all tests passed\n", fails);
  return fails ? 1 : 0;
}
