"""Closed-loop test of the hardware walker WITHOUT ROS (2026-11-21): CPU MuJoCo stands in for the Go1 and walker_core.WalkController
drives it from hardware-like signals, as walker_node.py does on the robot:
  mocap     the rigid body's raw pose (the jump calibration's offset and body yaw), POSE_HZ, latency, position / angle noise,
            through the workstation's MocapProcessor (ilc_quad/mocap.py, its filter settings = --tau / --spike-pct)
  IMU gyro  MuJoCo's gyro + noise;  joints  encoders;  tau  actuator force (tauEst)
The motor PD = the MJCF's position actuators (kp 35) + joint damping 0.5, 5 substeps of 4 ms per 20 ms policy step.
CPU MuJoCo is not MJX (solver / contact settings differ slightly): a sim-to-sim check on top.

    ~/venvs/pg0/bin/python hw/sim_closed_loop.py --npz hw/policies/nominal.npz --case nominal --walks 6 --save-dir /tmp/x
"""
import argparse, json, math, os, sys
import numpy as np
import mujoco
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, HERE)
import walker_core as C                                                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--npz', required=True)
ap.add_argument('--case', default='nominal', help='nominal | payload:m,dy,dz | foam:tc,width')
ap.add_argument('--walks', type=int, default=6)
ap.add_argument('--speed', type=float, default=1.0)
ap.add_argument('--dist', type=float, default=2.5)
ap.add_argument('--mocap', default=os.path.join(HERE, 'vendor'), help='directory holding the workstation ilc_quad/mocap.py copy (ws_mocap.py)')
ap.add_argument('--tau', type=float, default=0.02)
ap.add_argument('--spike-pct', type=float, default=0.0)
ap.add_argument('--true-vel', action='store_true', help='feed the true trunk velocity (no mocap estimator)')
ap.add_argument('--pose-hz', type=float, default=250.0)
ap.add_argument('--latency', type=float, default=0.007)
ap.add_argument('--pos-noise', type=float, default=0.0003)
ap.add_argument('--ang-noise-deg', type=float, default=0.1)
ap.add_argument('--gyro-noise', type=float, default=0.01)
ap.add_argument('--mocap-offset', type=float, nargs=3, default=[0.010, 0.0, -0.0025])
ap.add_argument('--body-yaw-deg', type=float, default=89.07)
ap.add_argument('--save-dir', default='')
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
sys.path.insert(0, a.mocap)
from ws_mocap import MocapProcessor                                     # noqa: E402

from mujoco_playground import registry                                   # noqa: E402
env = registry.load('Go1JoystickFlatTerrain', config=registry.get_default_config('Go1JoystickFlatTerrain'))
m = env.mj_model
assert abs(m.opt.timestep - 0.004) < 1e-9
SUB = int(round(C.DT / m.opt.timestep))
if a.case.startswith('payload:'):
    mp, dy, dz = [float(x) for x in a.case.split(':')[1].split(',')]
    M, c = m.body_mass[1], m.body_ipos[1].copy()
    r = c + np.array([0.0, dy, dz]); cn = (M * c + mp * r) / (M + mp); d_, dc = r - cn, cn - c
    dI = mp * np.array([d_[1]**2 + d_[2]**2, d_[0]**2 + d_[2]**2, d_[0]**2 + d_[1]**2]) + M * np.array([dc[1]**2 + dc[2]**2, dc[0]**2 + dc[2]**2, dc[0]**2 + dc[1]**2])
    m.body_mass[1] = M + mp; m.body_ipos[1] = cn; m.body_inertia[1] += dI
elif a.case.startswith('foam:'):
    tc, wd = [float(x) for x in a.case.split(':')[1].split(',')][:2]
    m.geom_solref[0] = [tc, 1.0]; m.geom_solimp[0] = [0.0, 0.95, wd, 0.5, 2.0]
d = mujoco.MjData(m)
pol = C.Policy(a.npz)
rng = np.random.default_rng(a.seed)
th_all = np.radians(np.random.default_rng(300).uniform(-15, 15, a.walks))
sens = lambda n: d.sensordata[m.sensor_adr[m.sensor(n).id]:m.sensor_adr[m.sensor(n).id] + 3]
off = np.asarray(a.mocap_offset); Cb = C.Rz(-math.radians(a.body_yaw_deg))      # trunk = body C  ->  body = trunk C^T


def to_ros(q):                                                           # w x y z -> x y z w
    return np.array([q[1], q[2], q[3], q[0]])


rows = []
os.makedirs(a.save_dir, exist_ok=True) if a.save_dir else None
for i, th in enumerate(th_all):
    g = np.array([a.dist * math.cos(th), a.dist * math.sin(th), a.speed])
    # start: somewhere in the room, any heading (the walk frame makes it irrelevant), home pose, at rest
    mujoco.mj_resetDataKeyframe(m, d, m.keyframe('home').id)
    yaw0 = rng.uniform(-math.pi, math.pi)
    d.qpos[0:2] = rng.uniform(-1, 1, 2); d.qpos[3:7] = [math.cos(yaw0 / 2), 0, 0, math.sin(yaw0 / 2)]
    d.ctrl[:] = C.HOME
    for _ in range(50):                                                  # settle on the feet (0.2 s), as the robot stands first
        mujoco.mj_step(m, d)
    proc = MocapProcessor(mocap_offset=off, tau=a.tau, spike_pct=a.spike_pct, body_yaw=math.radians(a.body_yaw_deg))
    queue, est, t_sim, t_last_pose, next_pose = [], None, 0.0, -1.0, -0.2   # mocap streams before the walk (the node waits for it)
    ctl = C.WalkController(pol, g, meta=dict(sim=vars(a)))
    status = 'walk'
    while status in ('walk', 'stand'):
        # mocap samples taken during the last control period, delivered after the latency
        while next_pose <= t_sim + 1e-9:
            Rt = C.quat_to_R(d.qpos[3:7])
            Rb = Rt @ Cb.T
            pb = d.qpos[0:3] + Rt @ off + rng.normal(0, a.pos_noise, 3)
            dR = C.Rz(0)  # small random rotation noise
            ax = rng.normal(0, math.radians(a.ang_noise_deg), 3); ang = np.linalg.norm(ax)
            if ang > 0:
                k_ = ax / ang; K = np.array([[0, -k_[2], k_[1]], [k_[2], 0, -k_[0]], [-k_[1], k_[0], 0]])
                dR = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
            queue.append((next_pose + a.latency, next_pose, pb, to_ros(C.R_to_quat(Rb @ dR))))
            next_pose += 1.0 / a.pose_hz
        while queue and queue[0][0] <= t_sim + 1e-9:
            _, t_cap, pb, qr = queue.pop(0)
            _, origin, q_xyzw, twist = proc.update(t_cap, pb, qr)
            est = (t_cap, origin, C.quat_to_R(np.r_[q_xyzw[3], q_xyzw[:3]]), twist)
            t_last_pose = t_sim
        t_cap, p_w, R_w, twist = est
        v_body = (C.quat_to_R(d.qpos[3:7]).T @ d.qvel[0:3]) if a.true_vel else (twist[:3] if twist is not None else np.zeros(3))
        gyro = sens('gyro') + rng.normal(0, a.gyro_noise, 3)
        q_t, status = ctl.step(p_w, R_w, v_body, gyro, d.qpos[7:].copy(), d.qvel[6:].copy(), d.actuator_force.copy(),
                               np.zeros(4), t_sim - t_last_pose + a.latency)
        if status.startswith('abort'):
            break
        d.ctrl[:] = q_t
        for _ in range(SUB):
            mujoco.mj_step(m, d)
        t_sim += C.DT
    r = ctl.log.arrays(final_q=d.qpos[7:].copy(), final_tau=d.actuator_force.copy())
    fell = status.startswith('abort')
    p = r['p'][:, :2]; dist = np.linalg.norm(p - g[:2], axis=1); arr = np.where(dist < 0.15)[0]
    u = g[:2] / np.linalg.norm(g[:2]); cross = np.abs(u[0] * p[:, 1] - u[1] * p[:, 0]); k_end = arr[0] + 1 if len(arr) else len(p)
    row = dict(walk=i, fell=fell, reason=status, t_goal=None if fell or not len(arr) else float(arr[0] * C.DT),
               path_cm=float(100 * np.sqrt(np.mean(cross[:k_end] ** 2))), final_cm=None if fell else float(100 * dist[-1]),
               v_est_err=float(np.sqrt(np.mean((r['v'][:, :3] - 0) ** 2))) if False else None)
    rows.append(row)
    if a.save_dir:
        ctl.log.save(os.path.join(a.save_dir, f'walk_{i:02d}.npz'), final_q=d.qpos[7:].copy(), final_tau=d.actuator_force.copy())
    print(json.dumps(row), flush=True)
tg = [r['t_goal'] for r in rows if r['t_goal'] is not None]
fin = [r['final_cm'] for r in rows if r['final_cm'] is not None]
print(json.dumps(dict(case=a.case, est='true' if a.true_vel else f'tau{a.tau}_spike{a.spike_pct}', walks=len(rows),
                      falls=sum(r['fell'] for r in rows), arrived=len(tg), t_goal=float(np.mean(tg)) if tg else None,
                      path_cm=float(np.mean([r['path_cm'] for r in rows])), final_cm=float(np.median(fin)) if fin else None)))
