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
    |e_y| > safety_ey (0.5 m, 2026-10-31 user: tightened back from 1.0)      'wall'
    |e_psi| > 1.2 rad                                    'spin'
    vx < 0.3 m/s after the launch                        'stall'   (only with -p stall_stop:=true since 2026-11-08)
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
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy

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
    # (2026-11-05) the announced trial, mirrored to <session>/.pending.json so car_pos.sh checks the car against the trial the
    # runner will actually fly (a re-run is announced ahead of the queue). Every assignment of self.pending writes it.
    @property
    def pending(self):
        return getattr(self, '_pending', None)

    @pending.setter
    def pending(self, v):
        self._pending = v
        f = os.path.join(self.sdir, '.pending.json')
        try:
            if v is None:
                if os.path.exists(f):
                    os.remove(f)
            else:
                k, req = v
                json.dump(dict(k=k, plan=req.get('plan'), tag=req.get('tag'), rerun_of=req.get('rerun_of')), open(f + '.tmp', 'w'))
                os.replace(f + '.tmp', f)
        except OSError:
            pass

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
        # (2026-11-04, user: "check the car's starting position is reasonable for the relevant track, and reset the steering
        # to straight before starting each trial") the start check against the request's plan, at its start (index 0, where
        # the simulated trials start: car_session_sim / real_car.run s0 = 0) -- it only refuses a start, never changes one
        p('stall_stop', False)                   # (2026-11-08) True: stop on 'stall' (vx < 0.3 after the launch); off: mocap
                                                 # glitches made the EKF speed jump and ended healthy runs as 'stall'
        p('start_max_dist', 0.25)                # m: car position to the plan's start point
        p('start_max_ey', 0.10)                  # m: lateral offset from the plan at its start
        p('start_max_dpsi_deg', 20.0)            # deg: heading vs the plan's start heading
        p('start_max_speed', 0.10)               # m/s: at rest
        p('straight_s', 0.5)                     # s: steering held straight (brake) after Enter, before the trial
        p('announce_straight_s', 1.0)            # s: steering straight (brake) when a request is announced
        # (2026-11-04 21:40, user) the track as laid out in the room: (dx, dy) m in the EKF frame added to every plan's path
        # points (xy). Exact: the observation, the plan states z and the scoring are all path-relative (replayed recorded
        # trials shifted 0.5 m: commands equal to 4e-14). car_runner.sh passes ~/car_runs/track_offset.txt.
        p('track_offset', [0.0, 0.0])
        # (user) mocap-loss stop: the on-car EKF keeps publishing predicted odometry when mocap stops (WiFi drop 20:45), which
        # the odometry age check never sees. No start without fresh mocap; a trial ends 'mocap lost' after mocap_lost_s.
        # (2026-11-15) the mocap frames optitrack_node ACCEPTED (/optitrack/pose_ok: flips, upside-down solves and position
        # jumps dropped), not the raw stream: a stream of rejected frames is a mocap loss, not fresh mocap.
        p('mocap_topic', '/optitrack/pose_ok')
        p('mocap_lost_s', 0.5)
        # (2026-11-05, user: "launch with a pure-pursuit or better algorithm instead of the initial 8 A flat") the launch on the
        # car: the fixed launch current (8 A) left the car creeping at 0.1 m/s for 3-25 s from rest (sensorless start).
        # launch_mode 'speed': pure pursuit steering on the trial's path + the VESC's closed-loop speed mode, ramped at
        # launch_accel to launch_speed_gain x v_hand; the hand-over to the network is unchanged (Controller: measured speed
        # >= v_hand). 'current': the old launch (the Controller's own command).
        p('launch_mode', 'speed')
        p('launch_accel', 1.0)                   # m/s^2
        p('launch_speed_gain', 1.2)              # x v_hand: the speed target, so the measured speed crosses v_hand
        p('launch_lookahead', 0.5)               # m: pure pursuit
        g = lambda n: self.get_parameter(n).value
        self.sdir = g('session_dir')
        os.makedirs(self.sdir, exist_ok=True)
        self.safety_ey, self.stale_s, self.brake_s = float(g('safety_ey')), float(g('stale_s')), float(g('brake_s'))
        self.pole_pairs, self.gear, self.cur_field = float(g('pole_pairs')), float(g('gear_ratio')), g('current_field')
        self.require_vesc = bool(g('require_vesc')) and VescStateStamped is not None
        self.stall_stop = bool(g('stall_stop'))
        self.start_lim = (float(g('start_max_dist')), float(g('start_max_ey')), math.radians(float(g('start_max_dpsi_deg'))),
                          float(g('start_max_speed')))
        self.straight_s, self.announce_straight_s = float(g('straight_s')), float(g('announce_straight_s'))
        self.prestart = None                     # dict(k, req, plan, until): Enter pressed, steering held straight
        self.track_offset = np.asarray(g('track_offset'), float).reshape(2)
        self.mocap_lost_s, self.mocap_t = float(g('mocap_lost_s')), 0.0
        self.mocap_topic = g('mocap_topic')
        self.launch_mode = g('launch_mode')
        self.launch_accel, self.launch_gain, self.launch_ld = float(g('launch_accel')), float(g('launch_speed_gain')), float(g('launch_lookahead'))
        self.wheelbase = float(cm.NOMINAL['lf'] + cm.NOMINAL['lr'])
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
        self.create_subscription(PoseStamped, g('mocap_topic'), self.on_mocap, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
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
                               f"{self.safety_ey:.2f} m, plans {bank.PLAN_DIR}, track offset ({self.track_offset[0]:+.2f}, "
                               f"{self.track_offset[1]:+.2f}) m, mocap-loss stop {self.mocap_lost_s:.1f} s on {g('mocap_topic')}, "
                               f"launch {self.launch_mode}")

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

    def on_mocap(self, msg):
        self.mocap_t = time.monotonic()

    def on_vesc(self, msg):
        motor_rad_s = msg.state.speed / self.pole_pairs * 2.0 * math.pi / 60.0
        self.vesc = (motor_rad_s / self.gear, float(getattr(msg.state, self.cur_field)))
        self.vesc_t = time.monotonic()

    def xmeas(self):
        return np.r_[self.odom, self.vesc[0], self.vesc[1], self.delta_cmd]

    # ---- trials ---------------------------------------------------------------------------------------------------
    def poll_requests(self):
        with self.lock:
            if self.trial is not None or self.pending is not None or self.prestart is not None:
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
                self.brake_until = max(self.brake_until, time.monotonic() + self.announce_straight_s)   # wheels straight
                # (2026-11-07, user: "print out the semantics of the run it's currently on, i need to name the mocap")
                name = req.get('tag') or self.derived_name(k, req)
                take = os.path.basename(os.path.normpath(self.sdir)) + f"_{k}" + (f"_{name}" if name else '')
                self.get_logger().info(f"trial {k}: {name or '(no tag)'}{'' if req.get('tag') else ' (untagged request: named here)'} | plan {req['plan']}, {req.get('laps', 2.05)} laps, "
                                       f"policy {os.path.basename(req['policy'])}" + (f" | re-run of {req['rerun_of']}" if req.get('rerun_of') else ''))
                self.get_logger().info(f">>> MOCAP TAKE: {take}  -- place the car at the start, then Enter (or /net_runner/start)")
                return

    def net_of(self, req):
        """'rma' / 'ppodr' from a request's policy file (RMA: the actor saved with mode 'rma'), '' if unreadable."""
        try:
            d = np.load(req['policy'], allow_pickle=False)
            return 'rma' if ('mode' in d.files and str(d['mode']) == 'rma') else 'ppodr'
        except Exception:
            return ''

    def derived_name(self, k, req):
        """(2026-11-07) a name for an untagged request (hw_stage stage trials queued without one): <net>_<tire>_stage_t<n> (as hw_stage.py tags them), n = 1 +
        the untagged requests of the same network before it in this session."""
        net = self.net_of(req)
        if not net:
            return ''
        n = 1
        for f in sorted(glob.glob(os.path.join(self.sdir, 'req_*.json'))):
            k2 = os.path.basename(f)[4:-5]
            if k2 >= k:
                break
            try:
                q = json.load(open(f))
            except (OSError, ValueError):
                continue
            if not q.get('tag') and self.net_of(q) == net:
                n += 1
        tire = next((t for t in ('plastic', 'rubber') if t in os.path.basename(os.path.normpath(self.sdir))), '')
        return '_'.join(x for x in (net, tire, 'stage') if x) + f'_t{n}'

    def start_pending(self):
        with self.lock:
            if self.prestart is not None:
                return False, 'a trial is starting'
            if self.pending is None or self.trial is not None:
                return False, 'nothing pending' if self.pending is None else 'a trial is flying'
            ok, msg = self.inputs_ok()
            if not ok:
                return False, msg
            k, req = self.pending
            plan = bank.load_plan(req['plan'])
            plan['name'] = req['plan']
            plan['xy'] = np.asarray(plan['xy']) + self.track_offset          # the track as laid out in the room
            ok, msg = self.start_check(plan)
            if not ok:
                return False, f'trial {k} NOT started: {msg} -- re-place the car, then Enter again'
            self.prestart = dict(k=k, req=req, plan=plan, until=time.monotonic() + self.straight_s)
            self.pending = None
            return True, f'trial {k}: start position ok ({msg}); steering straight {self.straight_s:.1f} s, then the trial'

    def inputs_ok(self):
        if self.odom is None or time.monotonic() - self.odom_t > self.stale_s:
            return False, 'no fresh odometry'
        if self.require_vesc and time.monotonic() - self.vesc_t > self.stale_s:
            return False, 'no fresh VESC state'
        if time.monotonic() - self.mocap_t > self.mocap_lost_s:
            return False, f'no fresh accepted mocap ({self.mocap_topic}): Motive tracking the car? WiFi up? optitrack_node dropping frames?'
        return True, ''

    def start_check(self, plan):
        """The car against the plan's start (index 0): distance, lateral offset, heading, speed."""
        pl = prt.sn.PlanNP(plan)
        x = self.odom
        s_off, ey, _ = pl.path_coords(0, x)
        dpsi = math.atan2(math.sin(x[2] - pl.theta[0]), math.cos(x[2] - pl.theta[0]))
        dist, v = float(np.hypot(*(x[:2] - pl.xy[0]))), float(np.hypot(x[3], x[4]))
        dmax, eymax, psimax, vmax = self.start_lim
        msg = (f'{dist:.2f} m from the start (along {s_off:+.2f}, lateral {ey:+.2f}), heading {math.degrees(dpsi):+.0f} deg, '
               f'speed {v:.2f} m/s')
        bad = [w for w, b in (('too far from the start', dist > dmax), (f'lateral beyond {eymax:.2f} m', abs(ey) > eymax),
                              (f'heading beyond {math.degrees(psimax):.0f} deg', abs(dpsi) > psimax),
                              ('not at rest', v > vmax)) if b]
        start = f'start ({pl.xy[0][0]:+.2f}, {pl.xy[0][1]:+.2f}) heading {math.degrees(pl.theta[0]):+.0f} deg'
        return (not bad), (msg + ('; ' + ', '.join(bad) + f'; plan {start}' if bad else ''))

    def begin_trial(self):
        """After the straight-steering hold: re-check, then the trial (called from control, the lock held)."""
        ps, self.prestart = self.prestart, None
        ok, msg = self.inputs_ok()
        if ok:
            ok, msg = self.start_check(ps['plan'])
        if not ok:
            self.pending = (ps['k'], ps['req'])
            self.get_logger().warn(f"trial {ps['k']} NOT started: {msg} -- re-place the car, then Enter again")
            return
        k, req, plan = ps['k'], ps['req'], ps['plan']
        # (2026-11-05) start on the plan's START LINE (index 0, where the start check put the car; the simulated trials start
        # at s0 = 0), not at the nearest plan point: at the figure-eight's crossing the nearest point was the OTHER branch
        # (trial 1105b/00003: e_psi +67.6 deg from step 0 -> 'wall' 0.3 s after the hand-over)
        ctl = prt.Controller(plan, prt.load_policy(req['policy']), launch=req.get('launch', dict(v_hand=1.0, current=8.0)),
                             idx0=0)
        self.trial = dict(k=k, req=req, ctl=ctl, laps_m=float(req.get('laps', 2.05)) * ctl.pl.L, t0=time.monotonic(), launch_sent=[])
        self.stop_reason, self.delta_cmd = '', 0.0
        self.get_logger().info(f'trial {k} started')

    def finish(self, crash):
        tr = self.trial
        rec = tr['ctl'].record(failed=bool(crash), crash=crash, safety_ey=self.safety_ey)
        rec['seconds'] = time.monotonic() - tr['t0']
        rec['launch_mode'] = self.launch_mode
        if tr.get('launch_sent'):
            rec['launch_sent'] = np.asarray(tr['launch_sent'])      # [t, steering, speed command] actually sent in the launch
        k = tr['k']
        np.savez(os.path.join(self.sdir, f'res_{k}.npz'), **{a: np.asarray(v) for a, v in rec.items() if v is not None})
        summ = dict(plan=tr['req']['plan'], laps=float(rec['laps']), failed=rec['failed'], crash=crash, steps=rec['steps'],
                    peak_ey=float(rec['peak_ey']), handover=int(rec['handover']), seconds=rec['seconds'],
                    track_offset=[float(v) for v in self.track_offset], launch_mode=self.launch_mode)
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
                if self.prestart is not None:        # wheels straight, at rest, then the trial
                    self.send_brake()
                    if now >= self.prestart['until']:
                        self.begin_trial()
                elif now < self.brake_until:
                    self.send_brake()
                return
            stale = now - self.odom_t > self.stale_s or (self.require_vesc and now - self.vesc_t > self.stale_s)
            if not (self.stop_reason or stale) and now - self.mocap_t > self.mocap_lost_s:
                self.stop_reason = 'mocap lost'
            if self.stop_reason or stale:
                self.finish(self.stop_reason or 'stale')
                self.send_brake()
                return
            x = self.xmeas()
            out = tr['ctl'].step(x)
            crash = tr['ctl'].check_stop(out['err'], x, self.safety_ey)
            if crash == 'stall' and not self.stall_stop:
                crash = ''
            if crash:
                self.finish(crash)
                self.send_brake()
                return
            if out['progress'] >= tr['laps_m']:
                self.finish('')
                self.send_brake()
                return
            if self.launch_mode == 'speed' and not out['launched']:
                steer, v_cmd = self.launch_cmd(tr, x, out['idx'], now)
                self.delta_cmd = steer
                self.send_speed(steer, v_cmd)
                tr['launch_sent'].append([now - tr['t0'], steer, v_cmd])
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

    def launch_cmd(self, tr, x, idx, now):
        """The launch (launch_mode 'speed'): pure pursuit on the trial's path, VESC speed mode ramped from rest."""
        pl = tr['ctl'].pl
        g = pl.xy[(idx + int(round(self.launch_ld / pl.ds))) % pl.M] - x[:2]
        gx, gy = math.cos(x[2]) * g[0] + math.sin(x[2]) * g[1], -math.sin(x[2]) * g[0] + math.cos(x[2]) * g[1]
        steer = float(np.clip(math.atan2(2 * self.wheelbase * gy, gx * gx + gy * gy), -self.s_max, self.s_max))
        v_target = self.launch_gain * float(tr['ctl'].launch['v_hand'])
        v_cmd = min(v_target, 0.3 + self.launch_accel * (now - tr['t0']))
        return steer, v_cmd

    def send_speed(self, steer, speed):
        m = AckermannDriveStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'base_link'
        m.drive.steering_angle = steer
        m.drive.speed = float(speed)             # jerk 0 + speed: the VESC's closed-loop speed mode (ackermann_to_vesc)
        m.drive.acceleration = 0.0
        m.drive.jerk = 0.0
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
            if self.prestart is not None:            # stop during the straight hold: back to pending
                self.pending, self.prestart = (self.prestart['k'], self.prestart['req']), None
        res.success, res.message = flying, 'stopping' if flying else 'no trial flying'
        return res

    def stdin_loop(self):
        # Enter: start the pending trial. r + Enter: re-fly the last finished trial (tag <tag>_rerun, not scored as it).
        # R + Enter: re-fly it IN PLACE of the last one (its request is tagged VOID_..., the re-run keeps the tag). (2026-11-05)
        for line in sys.stdin:
            c = line.strip()
            if c in ('r', 'R'):
                ok, msg = self.rerun_last(replace=(c == 'R'))
            else:
                ok, msg = self.start_pending()
            self.get_logger().info(msg)

    def rerun_last(self, replace):
        """Queue the last finished trial again, as a new request that becomes the pending one (ahead of the queue)."""
        with self.lock:
            if self.trial is not None or self.prestart is not None:
                return False, 'a trial is flying: re-run after it ends'
            done = sorted(f for f in glob.glob(os.path.join(self.sdir, 'res_*.json')))
            if not done:
                return False, 'no finished trial to re-run'
            k_old = os.path.basename(done[-1])[4:-5]
            f_old = os.path.join(self.sdir, f'req_{k_old}.json')
            try:
                q = json.load(open(f_old))
            except (OSError, ValueError) as e:
                return False, f'cannot read req_{k_old}.json: {e}'
            tag = q.get('tag', '')
            if tag.startswith('VOID_') or tag.startswith('CANCELLED_'):          # (11-05: R pressed twice re-ran a void trial)
                return False, f'trial {k_old} is {tag.split("_")[0]} (already re-run or cancelled): nothing re-queued'
            new = dict(q, rerun_of=k_old, tag=tag if replace else tag + '_rerun')
            k_new = f"{len(glob.glob(os.path.join(self.sdir, 'req_*.json'))):05d}"   # the hw_eval_requests numbering
            if os.path.exists(os.path.join(self.sdir, f'req_{k_new}.json')):
                return False, f'req_{k_new}.json exists: not re-running (numbering clash)'
            tmp = os.path.join(self.sdir, f'.req_{k_new}.json')
            json.dump(new, open(tmp, 'w'), indent=1)
            os.replace(tmp, os.path.join(self.sdir, f'req_{k_new}.json'))
            if replace:
                q['tag'], q['void'] = 'VOID_' + tag, f're-flown as {k_new} (R in net_runner)'
                json.dump(q, open(f_old + '.tmp', 'w'), indent=1)
                os.replace(f_old + '.tmp', f_old)
            self.pending = (k_new, new)
            self.brake_until = max(self.brake_until, time.monotonic() + self.announce_straight_s)
            return True, (f"trial {k_new}: RE-RUN of {k_old} ({q.get('plan', new['plan'])}, {new.get('laps', 2.05)} laps, tag "
                          f"{new['tag']}{'; ' + k_old + ' now VOID' if replace else ''}) -- place the car at the start, then Enter")


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
        node.pending = None                      # no stale .pending.json for car_pos.sh after the runner stops
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
