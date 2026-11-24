#!/usr/bin/env python3
"""(2026-11-24) AUTONOMOUS mode: after 's', key 'a' walks --auto-walks goals on its own: each next goal is drawn from where the robot
now stands (a point in the room --auto-min..--auto-max m away, any direction, at least --goal-margin m inside --bounds; seeded),
the network walks there and stands, --auto-pause s later the next walk starts. 'd' (or Ctrl-C) damps and stops the sequence;
any abort stops it. Logs: <dir>/auto/walk_<j>.npz (meta: goal_world).
(2026-11-24) GOAL-CONDITIONED walker (walker_core_goal: the network walks to the goal and turns; goals at 8 bearings around
the robot, incl. behind, --dist m from where it stands; place the robot near the middle of the area). Copied from walker_node.py.
The Go1 goal walker on the real robot (2026-11-21): ROS 2 node + operator prompt, on the workstation next to go1_bridge_imu.py.

    source /opt/ros/humble/setup.bash; source ~/ros2_ws/install/setup.bash
    python3 walker_node.py stage RUN_DIR            # follow stage_goal_hw.py's requests (RUN_DIR/it<k>/REQUEST.json, policy.npz)
    python3 walker_node.py eval OUT_DIR --policy P.npz --walks 6 --speed 1.0   # zero-shot / final evaluation walks
    common: --pose-topic /dl_go1/pose --mocap-offset 0.010 0 -0.0025 --mocap-body-yaw-deg 89.07 (the jump calibration, quad_env_1102.sh)

What it does (hw/walker_core.py has the walker itself; tested against the MJX walker: hw/test_walker_core.py, hw/sim_closed_loop.py):
  STAND UP   's': motor targets ramp from the measured joints to the home pose over --standup-s, PD kp 35 / kd 0.5 (the training's)
  WALK       'w': the next owed walk. The walk frame = the start pose, turned so the robot faces the goal: the goal is ALWAYS
             --dist (2.5) m straight ahead of where the robot stands. 50 Hz: mocap pose (MocapProcessor: calibration + filtered
             velocity), IMU gyro, encoders -> observation -> network -> motor targets home + 0.5 a; the standing controller (a = 0)
             takes over 0.5 s after the reference reaches the goal; logged 0.3 s more; then it stands.
  SAFETY     abort -> damping (kp 0, kd --damp-kd) on: tilt > 45 deg, trunk 12 cm below its start height, mocap older than 0.1 s,
             leaving --bounds; plus go1_bridge's own (joint limits latch damping, no command for 50 ms -> damping).
             'd' at the prompt (or Ctrl-C) damps at once.
  LOGS       one npz per walk (walker_core.WalkLog: the simulated rollout's fields + raw signals), written atomically to
             <job>/walks/walk_<j>.npz; 'v' voids the last walk (renamed VOID_..., re-walked next).
Keys: s stand up | w walk | d damp | l lie down (ramp to a crouch, then damp) | v void last walk | i info | q quit (damps).
"""
import argparse
import glob
import json
import math
import os
import sys
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import walker_core_m1121 as C                                                  # noqa: E402
import walker_core_goal as CG                                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=('stage', 'eval'))
ap.add_argument('dir', help='stage: the run folder stage_goal_hw.py mirrors here; eval: the output folder')
ap.add_argument('--policy', default='', help='eval: the network (npz from export_policy.py / stage_goal_hw.py)')
ap.add_argument('--walks', type=int, default=8, help='eval: walks (bearings 0, 45, -45, 90, -90, 135, -135, 180 deg, repeated)')
ap.add_argument('--speed', type=float, default=1.0)
ap.add_argument('--dist', type=float, default=1.5, help='m (the tracked area is 4.3 m in x: 1.5 fits all 8 bearings from its middle)')
ap.add_argument('--goal-margin', type=float, default=0.4, help='m: refuse a walk whose goal is closer than this to --bounds')
ap.add_argument('--auto-walks', type=int, default=8, help="'a': walks in one autonomous sequence")
ap.add_argument('--auto-min', type=float, default=1.0)
ap.add_argument('--auto-max', type=float, default=3.0)
ap.add_argument('--auto-pause', type=float, default=1.5, help='s standing between autonomous walks')
ap.add_argument('--auto-seed', type=int, default=0)
ap.add_argument('--pose-topic', default='/dl_go1/pose')
ap.add_argument('--pose-latency', type=float, default=0.007, help='s, added to the pose age (logged; the 0.1 s stale check)')
ap.add_argument('--mocap-offset', type=float, nargs=3, default=[0.010, 0.0, -0.0025])
ap.add_argument('--mocap-body-yaw-deg', type=float, default=89.07)
ap.add_argument('--mocap-tau', type=float, default=0.02)
ap.add_argument('--mocap-spike-pct', type=float, default=15.0)
ap.add_argument('--bounds', type=float, nargs=4, default=[-2.39, 1.92, -2.56, 2.50], help='x_lo x_hi y_lo y_hi of the trunk (mocap world)')
ap.add_argument('--standup-s', type=float, default=2.5)
ap.add_argument('--damp-kd', type=float, default=2.0)
ap.add_argument('--max-tilt', type=float, default=45.0)
ap.add_argument('--max-drop', type=float, default=0.12)
ap.add_argument('--ilc-quad', default=os.path.expanduser('~/ros2_ws/src/ilc_quad'), help='for ilc_quad.mocap (the calibrated MocapProcessor)')
a = ap.parse_args()
sys.path.insert(0, a.ilc_quad)
from ilc_quad.mocap import MocapProcessor                               # noqa: E402

SENSOR_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                        durability=QoSDurabilityPolicy.VOLATILE)
CROUCH = np.array([0.1, 1.2, -2.6, -0.1, 1.2, -2.6, 0.1, 1.2, -2.6, -0.1, 1.2, -2.6])   # lie-down target (network order), then damping
CAN_NAMES = tuple(f'{l}_{j}_joint' for l in C.CAN_LEGS for j in ('hip', 'thigh', 'calf'))


# ---------------------------------------------------------------- jobs: what to walk next
BEARINGS = [0, 45, -45, 90, -90, 135, -135, 180]


def goals_eval(n, v):
    th = np.radians(np.array(BEARINGS * (n // len(BEARINGS) + 1))[:n])
    return [np.array([a.dist * math.cos(t), a.dist * math.sin(t)]) for t in th]


def next_job():
    """(policy path, goal, walk path, label) of the next owed walk, or None."""
    if a.mode == 'eval':
        wd = os.path.join(a.dir, 'walks'); os.makedirs(wd, exist_ok=True)
        for j, g in enumerate(goals_eval(a.walks, a.speed)):
            f = os.path.join(wd, f'walk_{j:02d}.npz')
            if not os.path.exists(f):
                return a.policy, g, f, f'eval walk {j + 1}/{a.walks}'
        return None
    reqs = sorted(glob.glob(os.path.join(a.dir, 'it*', 'REQUEST.json')), key=lambda f: int(f.split('/it')[-1].split('/')[0]))
    if not reqs:
        return None
    rq = reqs[-1]
    r = json.load(open(rq))
    it = os.path.dirname(rq)
    pol = os.path.join(it, r['policy'])
    for j, g in enumerate(r['goals']):
        f = os.path.join(it, r['walks'], f'walk_{j:02d}.npz')
        if not os.path.exists(f):
            return pol, np.asarray(g, float), f, f'stage iteration {r["iter"]} walk {j + 1}/{len(r["goals"])}'
    return None


# ---------------------------------------------------------------- the node
class WalkerNode(Node):
    def __init__(self):
        super().__init__('go1_walker')
        self.lock = threading.Lock()
        self.js = self.ff = self.imu = None
        self.t_js = self.t_imu = self.t_pose = -1.0
        self.pose = None                                                 # (origin, R, twist, w_mocap)
        self.mp = MocapProcessor(mocap_offset=a.mocap_offset, tau=a.mocap_tau, spike_pct=a.mocap_spike_pct,
                                 body_yaw=math.radians(a.mocap_body_yaw_deg))
        self.mode = 'passive'                                            # passive | standup | stand | walk | damp | liedown
        self.cmd = None                                                  # (q_target_pg, kp, kd)
        self.ramp = None
        self.ctl = None
        self.job = None
        self.last_saved = None
        self.status = ''
        self.create_subscription(JointState, 'joint_states', self.on_js, SENSOR_QOS)
        self.create_subscription(Float64MultiArray, 'foot_contacts', self.on_ff, SENSOR_QOS)
        self.create_subscription(Imu, 'imu', self.on_imu, SENSOR_QOS)
        self.create_subscription(PoseStamped, a.pose_topic, self.on_pose, SENSOR_QOS)
        self.pub = self.create_publisher(Float64MultiArray, 'joint_cmd', SENSOR_QOS)
        self.create_timer(C.DT, self.control)                            # 50 Hz policy
        self.create_timer(0.005, self.republish)                         # 200 Hz: the latest command (bridge watchdog 50 ms)

    @staticmethod
    def now():
        return time.monotonic()

    # -- sensors
    def on_js(self, m):
        if tuple(m.name) != CAN_NAMES:
            idx = [list(m.name).index(n) for n in CAN_NAMES]
        else:
            idx = list(range(12))
        q, dq, tau = (np.asarray(x, float)[idx] for x in (m.position, m.velocity, m.effort))
        with self.lock:
            self.js = (q[C.PG_FROM_CAN], dq[C.PG_FROM_CAN], tau[C.PG_FROM_CAN]); self.t_js = self.now()

    def on_ff(self, m):
        with self.lock:
            self.ff = np.asarray(m.data, float)[C.FOOT_PG_FROM_CAN]

    def on_imu(self, m):
        w = m.angular_velocity
        o = m.orientation
        with self.lock:
            self.imu = (np.array([w.x, w.y, w.z]), np.array([o.w, o.x, o.y, o.z])); self.t_imu = self.now()

    def on_pose(self, m):
        t = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
        if t <= 0:
            t = self.now()
        p = m.pose.position; o = m.pose.orientation
        _, origin, q_xyzw, twist = self.mp.update(t, [p.x, p.y, p.z], [o.x, o.y, o.z, o.w])
        R = C.quat_to_R(np.r_[q_xyzw[3], q_xyzw[:3]])
        with self.lock:
            self.pose = (origin, R, twist); self.t_pose = self.now()

    # -- commands
    def send(self, q_pg, kp, kd):
        q_can = np.asarray(q_pg)[C.CAN_FROM_PG]
        z = np.zeros(12)
        msg = Float64MultiArray()
        msg.data = list(np.r_[q_can, z, np.full(12, kp), np.full(12, kd), z].astype(float))
        self.pub.publish(msg)

    def republish(self):
        with self.lock:
            mode, cmd = self.mode, self.cmd
        if mode == 'passive' or cmd is None:
            return
        self.send(*cmd)

    def set_damp(self, why):
        with self.lock:
            self.mode, self.cmd = 'damp', (C.HOME, 0.0, a.damp_kd)
            self.status = f'DAMPING: {why}'
        print(f'\n!!! {self.status}', flush=True)

    def fresh(self):
        t = self.now()
        return dict(joints=t - self.t_js < 0.05, imu=t - self.t_imu < 0.05, mocap=t - self.t_pose < 0.05)

    # -- 50 Hz
    def control(self):
        with self.lock:
            mode, js, imu, pose, ff = self.mode, self.js, self.imu, self.pose, self.ff
            t_pose = self.t_pose
        t = self.now()
        if mode in ('standup', 'liedown'):
            r = self.ramp
            s = min(1.0, (t - r['t0']) / r['T'])
            q = (1 - s) * r['q0'] + s * r['q1']
            with self.lock:
                self.cmd = (q, C.KP, C.KD)
                if s >= 1.0:
                    if mode == 'standup':
                        self.mode = 'stand'
                    else:
                        self.mode, self.cmd = 'damp', (C.HOME, 0.0, a.damp_kd)
            return
        if mode != 'walk':
            return
        if js is None or imu is None or pose is None or pose[2] is None:
            self.set_damp('sensor missing during the walk'); self.finish('abort: sensor missing'); return
        q, dq, tau = js
        origin, R, twist = pose
        q_t, status = self.ctl.step(origin, R, twist[:3], imu[0], q, dq, tau, ff if ff is not None else np.zeros(4),
                                    t - t_pose + a.pose_latency, w_mocap=twist[3:6], imu_quat=imu[1], t_wall=t)
        if status.startswith('abort'):
            self.set_damp(status[7:]); self.finish(status); return
        with self.lock:
            self.cmd = (q_t, C.KP, C.KD)
        if status == 'done':
            self.finish('done')

    def finish(self, status):
        with self.lock:
            if self.mode == 'walk':
                self.mode = 'stand'; self.cmd = (C.HOME, C.KP, C.KD)
            js = self.js
        pol, g, path, label = self.job
        tmp = path[:-4] + '.part.npz'
        self.ctl.log.meta.update(status=status, label=label, robot='go1', tau_includes_kd=True, pose_topic=a.pose_topic,
                                 mocap_offset=a.mocap_offset, mocap_body_yaw_deg=a.mocap_body_yaw_deg, mocap_tau=a.mocap_tau,
                                 mocap_spike_pct=a.mocap_spike_pct, pose_latency=a.pose_latency)
        r = self.ctl.log.save(tmp, final_q=js[0] if js else None, final_tau=js[2] if js else None)
        os.replace(tmp, path)
        self.last_saved = path
        p = r['p'][:, :2]; d = np.linalg.norm(p - g[:2], axis=1); arr = np.where(d < CG.ARRIVE + 0.05)[0]
        u = g[:2] / np.linalg.norm(g[:2]); cross = np.abs(u[0] * p[:, 1] - u[1] * p[:, 0])
        k_end = arr[0] + 1 if len(arr) else len(p)
        dt = np.diff(r['t_wall']) if 't_wall' in r else np.array([0.02])
        print(f'\n>>> {label}: {status}; time to goal {arr[0] * C.DT if len(arr) and status == "done" else "-"} s, path '
              f'{100 * np.sqrt(np.mean(cross[:k_end] ** 2)):.1f} cm, final {100 * d[-1]:.1f} cm; policy period {1000 * dt.mean():.1f} '
              f'ms (max {1000 * dt.max():.1f}); saved {path}', flush=True)

    # -- operator actions (main thread)
    def standup(self, T=None, target=None, mode='standup'):
        with self.lock:
            js = self.js
        if js is None or not self.fresh()['joints']:
            print('!!! no joint_states (is go1_bridge_imu.py running?)'); return
        with self.lock:
            self.ramp = dict(t0=self.now(), T=T or a.standup_s, q0=js[0].copy(), q1=C.HOME if target is None else target)
            self.cmd = (js[0].copy(), C.KP, C.KD)
            self.mode = mode
        print(f'... {mode} over {T or a.standup_s:.1f} s')

    def auto_goal(self, j):
        """A world goal auto_min..auto_max m from the robot, any direction, inside the bounds by goal_margin -> (g in the robot's
        frame, gw world), or None."""
        with self.lock:
            p_w, R_w = self.pose[0], self.pose[1]
        yaw = C.yaw_of_R(R_w)
        rng = np.random.default_rng(1000 * a.auto_seed + j)
        lo_x, hi_x, lo_y, hi_y = a.bounds
        mg = a.goal_margin
        for _ in range(500):
            r, th = rng.uniform(a.auto_min, a.auto_max), rng.uniform(-math.pi, math.pi)
            gw = np.asarray(p_w)[:2] + r * np.array([math.cos(th), math.sin(th)])
            if lo_x + mg <= gw[0] <= hi_x - mg and lo_y + mg <= gw[1] <= hi_y - mg:
                return C.Rz(-yaw)[:2, :2] @ (gw - np.asarray(p_w)[:2]), gw
        return None

    def auto(self, n):
        wd = os.path.join(a.dir, 'auto'); os.makedirs(wd, exist_ok=True)
        self.auto_on = True
        done = 0
        print(f'>>> AUTONOMOUS: {n} walks ({a.auto_min}-{a.auto_max} m, any direction); d / Ctrl-C damps and stops', flush=True)
        while self.auto_on and done < n:
            time.sleep(a.auto_pause)
            if not self.auto_on or self.mode != 'stand':
                break
            j = len(glob.glob(os.path.join(wd, 'walk_*.npz')))
            gg = self.auto_goal(j)
            if gg is None:
                print('!!! no reachable goal inside the bounds from here: stopping'); break
            g, gw = gg
            self.walk(job=(a.policy, g, os.path.join(wd, f'walk_{j:02d}.npz'), f'auto walk {done + 1}/{n} (#{j})'))
            while self.mode == 'walk':
                time.sleep(0.05)
            done += 1
            if self.mode != 'stand':
                break
        self.auto_on = False
        print(f'\n>>> AUTONOMOUS sequence ended after {done} walk(s), mode {self.mode}', flush=True)

    def walk(self, job=None):
        if self.mode != 'stand':
            print(f'!!! stand up first (mode {self.mode})'); return
        f = self.fresh()
        if not all(f.values()):
            print(f'!!! stale sensors: {f}'); return
        job = job or next_job()
        if job is None:
            print('nothing owed (stage: waiting for the next REQUEST.json)'); return
        pol, g, path, label = job
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with self.lock:
            p_w, R_w = self.pose[0], self.pose[1]
        yaw = C.yaw_of_R(R_w)
        gw = np.asarray(p_w)[:2] + C.Rz(yaw)[:2, :2] @ np.asarray(g)[:2]           # the goal in the mocap world
        lo_x, hi_x, lo_y, hi_y = a.bounds
        mg = a.goal_margin
        if not (lo_x + mg <= gw[0] <= hi_x - mg and lo_y + mg <= gw[1] <= hi_y - mg):
            print(f'!!! the goal ({gw[0]:.2f}, {gw[1]:.2f}) is within {mg} m of the bounds {a.bounds}: move / turn the robot '
                  f'toward the middle (it is at ({p_w[0]:.2f}, {p_w[1]:.2f}), heading {math.degrees(yaw):.0f} deg)'); return
        policy = C.Policy(pol)
        ctl = CG.GoalWalkController(policy, g, max_tilt=a.max_tilt, max_drop=a.max_drop, bounds=a.bounds,
                                    meta=dict(label=label, policy=pol, started=time.time(), goal_world=gw.tolist()))
        with self.lock:
            self.ctl, self.job = ctl, job
            self.mode = 'walk'
        print(f'>>> {label}: goal {a.dist} m at bearing {math.degrees(math.atan2(g[1], g[0])):.0f} deg -> world ({gw[0]:.2f}, {gw[1]:.2f}); '
              f'at most {ctl.t_sw:.1f} s, stands {CG.POST_S} s after arriving; policy {pol}')


def main():
    rclpy.init()
    node = WalkerNode()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    th = threading.Thread(target=ex.spin, daemon=True)
    th.start()
    print(__doc__.split('\n')[0])
    print('keys: s stand up | a AUTONOMOUS walks | w one walk | d damp (stops auto) | l lie down | v void last walk | i info | q quit')
    node.auto_on = False
    try:
        while True:
            job = next_job()
            print(f'\n[{node.mode}] {node.status} next: {job[3] if job else "nothing owed"} > ', end='', flush=True)
            k = sys.stdin.readline().strip().lower()
            if k == 's':
                node.standup()
            elif k == 'w':
                node.walk()
                while node.mode == 'walk':
                    time.sleep(0.05)
            elif k == 'a':
                if a.mode != 'eval' or not a.policy:
                    print('!!! autonomous walks need eval mode and --policy')
                elif node.auto_on:
                    print('already running')
                else:
                    threading.Thread(target=node.auto, args=(a.auto_walks,), daemon=True).start()
            elif k == 'd':
                node.auto_on = False
                node.set_damp('operator')
            elif k == 'l':
                node.standup(T=2.0, target=CROUCH, mode='liedown')
            elif k == 'v':
                if node.last_saved and os.path.exists(node.last_saved):
                    d, b = os.path.split(node.last_saved)
                    os.replace(node.last_saved, os.path.join(d, f'VOID_{int(time.time())}_{b}'))
                    print(f'voided {node.last_saved}: it will be walked again'); node.last_saved = None
            elif k == 'i':
                with node.lock:
                    pose, imu = node.pose, node.imu
                print(f'fresh {node.fresh()}; mocap origin {None if pose is None else np.round(pose[0], 3)}, yaw '
                      f'{None if pose is None else round(math.degrees(C.yaw_of_R(pose[1])), 1)} deg, tilt '
                      f'{None if pose is None else round(C.tilt_deg(pose[1]), 1)} deg; gyro {None if imu is None else np.round(imu[0], 3)}')
            elif k == 'q':
                node.auto_on = False
                node.set_damp('quit'); time.sleep(0.5); break
    except KeyboardInterrupt:
        node.auto_on = False
        node.set_damp('Ctrl-C'); time.sleep(0.5)
    finally:
        ex.shutdown(); node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
