#!/usr/bin/env python3
"""
net_runner: flies an ILC2Real / DistILC car network on the real F1TENTH (ROS 2), the counterpart of LLA-MPC's onboard
controller (LLA-MPC-online llampc/scripts/llampc_node.py: odometry from the EKF, a fixed-rate control timer, an
AckermannDriveStamped command) for the networks' current-controlled drive.

Every control period (40 Hz, DT = 0.025 s as the networks are trained) it builds the measured planar state
    [X, Y, psi, vx, vy, r]  /odometry/filtered (the EKF of ilc_f1tenth_hw, body frame cg; psi unwrapped)
    omega_w                 /sensors/core state.speed (ERPM) / pole_pairs -> motor rad/s / gear (11.82)
    I                       /sensors/core state.avg_iq (the measured motor current, A)
    delta                   the last steering command
and runs it through ilc2real/policy_runtime.Controller -- the controller step of real_car.run, checked identical on
replayed simulated trials (test_policy_runtime.py): plan projection, observation features, the network, the slow-start
launch controller. The command goes out on /mpc/drive (drive_relay.py forwards it to /drive with its watchdog) as
drive.jerk = 2.0 (current mode, the patched vesc_ackermann), drive.acceleration = motor current A, drive.steering_angle.

SAFETY: a trial stops -- duty-cycle 0 (brake) for brake_s, then silence (the relay keeps braking) -- when
    |e_y| > safety_ey (0.5 m: the room's envelope)      'wall'
    |e_psi| > 1.2 rad                                    'spin'
    vx < 0.3 m/s after the launch                        'stall'
    the odometry or the VESC is older than stale_s       'stale'
    /net_runner/stop (std_srvs/Trigger), or Ctrl-C      'stopped'
and ends normally after the requested laps.

TRIALS come from hw_stage.py --backend manual (or by hand) as requests in session_dir:
    req_<k>.json  {"plan": <bank plan name>, "policy": <npz path>, "laps": 2.05, "launch": {"v_hand": 1.0, "current": 8.0}}
The runner announces each pending request; place the car at the start and press Enter in this terminal (or call
/net_runner/start). The trial is written as res_<k>.npz (real_car.run's dict: x, obs, mu, a, err, idx, applied, laps,
failed, crash, ...) and res_<k>.json (summary). Plans come from the same bank as the simulations (F1T_PLANS, env_tight.sh).

    source <ilc2real>/env_tight.sh && ros2 run ilc_f1tenth_hw net_runner.py --ros-args -p session_dir:=<dir>
"""
import glob
import json
import math
import os
import sys
import threading
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_srvs.srv import Trigger

ILC2REAL = os.environ.get('ILC2REAL_DIR') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                                                          'ilc_f1tenth', 'ilc2real')
sys.path.insert(0, ILC2REAL)
import bank                                      # noqa: E402
import car_model as cm                           # noqa: E402
import policy_runtime as prt                     # noqa: E402

try:
    from vesc_msgs.msg import VescStateStamped   # the car's VESC driver (f1tenth_system/vesc)
except ImportError:                              # a desk test without the VESC driver
    VescStateStamped = None


class NetRunner(Node):
    def __init__(self):
        super().__init__('net_runner')
        p = self.declare_parameter
        p('session_dir', os.path.expanduser('~/car_session'))
        p('rate_hz', 1.0 / prt.DT)
        p('safety_ey', 0.5)
        p('odom_topic', '/odometry/filtered')
        p('vesc_topic', '/sensors/core')
        p('drive_topic', '/mpc/drive')
        p('pole_pairs', 2.0)
        p('gear_ratio', 11.82)
        p('current_field', 'avg_iq')
        p('stale_s', 0.1)
        p('brake_s', 1.0)
        p('stdin_start', True)
        p('require_vesc', True)
        g = lambda n: self.get_parameter(n).value
        self.sdir = g('session_dir')
        os.makedirs(self.sdir, exist_ok=True)
        self.safety_ey, self.stale_s, self.brake_s = float(g('safety_ey')), float(g('stale_s')), float(g('brake_s'))
        self.pole_pairs, self.gear, self.cur_field = float(g('pole_pairs')), float(g('gear_ratio')), g('current_field')
        self.require_vesc = bool(g('require_vesc')) and VescStateStamped is not None
        self.s_max = float(cm.NOMINAL['s_max'])
        self.i_lo, self.i_hi = float(cm.A_OFF[1] - cm.A_SCALE[1]), float(cm.A_OFF[1] + cm.A_SCALE[1])

        self.odom, self.odom_t, self.psi = None, 0.0, None
        self.vesc, self.vesc_t = (0.0, 0.0), 0.0
        self.delta_cmd = 0.0
        self.lock = threading.Lock()
        self.pending = None                      # (k, request) announced, waiting for start
        self.trial = None                        # the trial in flight: dict(k, req, ctl, laps_m)
        self.brake_until = 0.0
        self.stop_reason = ''

        self.create_subscription(Odometry, g('odom_topic'), self.on_odom, 10)
        if VescStateStamped is not None:
            self.create_subscription(VescStateStamped, g('vesc_topic'), self.on_vesc, 10)
        elif bool(g('require_vesc')):
            self.get_logger().warn('vesc_msgs not importable: omega_w / I read as 0 (desk test only)')
        self.pub = self.create_publisher(AckermannDriveStamped, g('drive_topic'), 10)
        self.create_service(Trigger, '~/start', self.on_start)
        self.create_service(Trigger, '~/stop', self.on_stop)
        self.create_timer(1.0 / float(g('rate_hz')), self.control)
        self.create_timer(0.5, self.poll_requests)
        if bool(g('stdin_start')):
            threading.Thread(target=self.stdin_loop, daemon=True).start()
        self.get_logger().info(f"net_runner: session {self.sdir}, {g('rate_hz'):.0f} Hz, safety envelope "
                               f"{self.safety_ey:.2f} m, plans {bank.PLAN_DIR}")

    # ---- inputs ---------------------------------------------------------------------------------------------------
    def on_odom(self, msg):
        q = msg.pose.pose.orientation
        psi = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        if self.psi is not None:                 # unwrap against the last heading
            psi = self.psi + math.atan2(math.sin(psi - self.psi), math.cos(psi - self.psi))
        self.psi = psi
        t = msg.twist.twist
        self.odom = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, psi, t.linear.x, t.linear.y, t.angular.z])
        self.odom_t = time.monotonic()

    def on_vesc(self, msg):
        motor_rad_s = msg.state.speed / self.pole_pairs * 2.0 * math.pi / 60.0
        self.vesc = (motor_rad_s / self.gear, float(getattr(msg.state, self.cur_field)))
        self.vesc_t = time.monotonic()

    def xmeas(self):
        return np.r_[self.odom, self.vesc[0], self.vesc[1], self.delta_cmd]

    # ---- trials ---------------------------------------------------------------------------------------------------
    def poll_requests(self):
        with self.lock:
            if self.trial is not None or self.pending is not None:
                return
            for f in sorted(glob.glob(os.path.join(self.sdir, 'req_*.json'))):
                k = os.path.basename(f)[4:-5]
                if os.path.exists(os.path.join(self.sdir, f'res_{k}.json')):
                    continue
                try:
                    req = json.load(open(f))
                except (OSError, ValueError):
                    return                       # being written
                self.pending = (k, req)
                self.get_logger().info(f"trial {k}: plan {req['plan']}, {req.get('laps', 2.05)} laps, policy "
                                       f"{os.path.basename(req['policy'])} -- place the car at the start, then Enter "
                                       f"(or /net_runner/start)")
                return

    def start_pending(self):
        with self.lock:
            if self.pending is None or self.trial is not None:
                return False, 'nothing pending' if self.pending is None else 'a trial is flying'
            if self.odom is None or time.monotonic() - self.odom_t > self.stale_s:
                return False, 'no fresh odometry'
            if self.require_vesc and time.monotonic() - self.vesc_t > self.stale_s:
                return False, 'no fresh VESC state'
            k, req = self.pending
            plan = bank.load_plan(req['plan'])
            plan['name'] = req['plan']
            ctl = prt.Controller(plan, prt.load_policy(req['policy']), launch=req.get('launch', dict(v_hand=1.0, current=8.0)),
                                 x0=self.odom)
            self.trial = dict(k=k, req=req, ctl=ctl, laps_m=float(req.get('laps', 2.05)) * ctl.pl.L, t0=time.monotonic())
            self.pending, self.stop_reason, self.delta_cmd = None, '', 0.0
            self.get_logger().info(f'trial {k} started')
            return True, f'trial {k} started'

    def finish(self, crash):
        tr = self.trial
        rec = tr['ctl'].record(failed=bool(crash), crash=crash, safety_ey=self.safety_ey)
        rec['seconds'] = time.monotonic() - tr['t0']
        k = tr['k']
        np.savez(os.path.join(self.sdir, f'res_{k}.npz'), **{a: np.asarray(v) for a, v in rec.items() if v is not None})
        summ = dict(plan=tr['req']['plan'], laps=float(rec['laps']), failed=rec['failed'], crash=crash, steps=rec['steps'],
                    peak_ey=float(rec['peak_ey']), handover=int(rec['handover']), seconds=rec['seconds'])
        json.dump(summ, open(os.path.join(self.sdir, f'res_{k}.json'), 'w'), indent=1)
        self.get_logger().info(f"trial {k} {'STOPPED: ' + crash if crash else 'done'}: {summ['laps']:.2f} laps, "
                               f"peak |e_y| {summ['peak_ey']:.3f} m")
        self.trial = None
        self.brake_until = time.monotonic() + self.brake_s

    # ---- the control period ---------------------------------------------------------------------------------------
    def control(self):
        now = time.monotonic()
        with self.lock:
            tr = self.trial
            if tr is None:
                if now < self.brake_until:
                    self.send_brake()
                return
            stale = now - self.odom_t > self.stale_s or (self.require_vesc and now - self.vesc_t > self.stale_s)
            if self.stop_reason or stale:
                self.finish(self.stop_reason or 'stale')
                self.send_brake()
                return
            x = self.xmeas()
            out = tr['ctl'].step(x)
            crash = tr['ctl'].check_stop(out['err'], x, self.safety_ey)
            if crash:
                self.finish(crash)
                self.send_brake()
                return
            if out['progress'] >= tr['laps_m']:
                self.finish('')
                self.send_brake()
                return
            steer = float(np.clip(out['cmd'][0], -self.s_max, self.s_max))
            current = float(np.clip(out['cmd'][1], self.i_lo, self.i_hi))
            self.delta_cmd = steer
            self.send(steer, current)

    def send(self, steer, current):
        m = AckermannDriveStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'base_link'
        m.drive.steering_angle = steer
        m.drive.speed = 0.0
        m.drive.acceleration = current           # motor current (A) in current mode
        m.drive.jerk = 2.0                       # the patched vesc_ackermann: 2.0 = current mode
        self.pub.publish(m)

    def send_brake(self):
        m = AckermannDriveStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.drive.steering_angle = 0.0
        m.drive.speed = 0.0
        m.drive.acceleration = 0.0               # duty cycle 0: the VESC brakes
        m.drive.jerk = 1.0                       # 1.0 = duty-cycle mode
        self.pub.publish(m)

    # ---- operator -------------------------------------------------------------------------------------------------
    def on_start(self, req, res):
        res.success, res.message = self.start_pending()
        return res

    def on_stop(self, req, res):
        with self.lock:
            flying = self.trial is not None
            self.stop_reason = 'stopped' if flying else ''
        res.success, res.message = flying, 'stopping' if flying else 'no trial flying'
        return res

    def stdin_loop(self):
        for _ in sys.stdin:
            ok, msg = self.start_pending()
            self.get_logger().info(msg)


def main():
    rclpy.init()
    node = NetRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        with node.lock:
            if node.trial is not None:
                node.finish('stopped')
        for _ in range(int(node.brake_s / 0.025)):
            node.send_brake()
            time.sleep(0.025)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
