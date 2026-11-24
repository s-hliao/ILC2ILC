"""A CPU-MuJoCo stand-in for the real Go1 (2026-11-21), driven exactly as walker_node.py drives the robot: walker_core.WalkController
on hardware-like signals (raw mocap rigid-body pose with offset / body yaw / latency / noise through the workstation MocapProcessor;
IMU gyro + noise; encoders; actuator force as tauEst WITHOUT the Kd dq term, since MuJoCo's damping is passive -> meta
tau_includes_kd False). Used by hw/sim_closed_loop.py and hw/stage_goal_hw.py --backend cpusim (the pipeline test)."""
import math, os, sys
import numpy as np
import mujoco
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import walker_core as C                                                   # noqa: E402


def to_ros(q):
    return np.array([q[1], q[2], q[3], q[0]])


class CpuRobot:
    def __init__(self, case='nominal', tau=0.02, spike_pct=15.0, true_vel=False, pose_hz=250.0, latency=0.007, pos_noise=0.0003,
                 ang_noise_deg=0.1, gyro_noise=0.01, mocap_offset=(0.010, 0.0, -0.0025), body_yaw_deg=89.07, seed=0,
                 mocap_dir=os.path.join(HERE, 'vendor')):
        sys.path.insert(0, mocap_dir)
        from ws_mocap import MocapProcessor
        from mujoco_playground import registry
        self.MP = MocapProcessor
        env = registry.load('Go1JoystickFlatTerrain', config=registry.get_default_config('Go1JoystickFlatTerrain'))
        m = self.m = env.mj_model
        assert abs(m.opt.timestep - 0.004) < 1e-9
        self.sub = int(round(C.DT / m.opt.timestep))
        if case.startswith('payload:'):
            mp, dy, dz = [float(x) for x in case.split(':')[1].split(',')]
            M, c = m.body_mass[1], m.body_ipos[1].copy()
            r = c + np.array([0.0, dy, dz]); cn = (M * c + mp * r) / (M + mp); d_, dc = r - cn, cn - c
            dI = mp * np.array([d_[1]**2 + d_[2]**2, d_[0]**2 + d_[2]**2, d_[0]**2 + d_[1]**2]) \
                + M * np.array([dc[1]**2 + dc[2]**2, dc[0]**2 + dc[2]**2, dc[0]**2 + dc[1]**2])
            m.body_mass[1] = M + mp; m.body_ipos[1] = cn; m.body_inertia[1] += dI
        elif case.startswith('foam:'):
            tc, wd = [float(x) for x in case.split(':')[1].split(',')][:2]
            m.geom_solref[0] = [tc, 1.0]; m.geom_solimp[0] = [0.0, 0.95, wd, 0.5, 2.0]
        elif case != 'nominal':
            raise ValueError(case)
        self.d = mujoco.MjData(m)
        self.o = dict(case=case, tau=tau, spike_pct=spike_pct, true_vel=true_vel, pose_hz=pose_hz, latency=latency, pos_noise=pos_noise,
                      ang_noise_deg=ang_noise_deg, gyro_noise=gyro_noise, mocap_offset=list(mocap_offset), body_yaw_deg=body_yaw_deg)
        self.rng = np.random.default_rng(seed)
        self.off = np.asarray(mocap_offset, float)
        self.Cb = C.Rz(-math.radians(body_yaw_deg))
        self.gyro_adr = m.sensor_adr[m.sensor('gyro').id]

    def _noisy_R(self):
        ax = self.rng.normal(0, math.radians(self.o['ang_noise_deg']), 3); ang = np.linalg.norm(ax)
        if ang == 0:
            return np.eye(3)
        k = ax / ang; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K

    def walk(self, policy, goal, **ctl_kw):
        """One walk from a standing start at a random spot / heading. Returns (controller, status, final_q, final_tau)."""
        m, d, o, rng = self.m, self.d, self.o, self.rng
        mujoco.mj_resetDataKeyframe(m, d, m.keyframe('home').id)
        yaw0 = rng.uniform(-math.pi, math.pi)
        d.qpos[0:2] = rng.uniform(-1, 1, 2); d.qpos[3:7] = [math.cos(yaw0 / 2), 0, 0, math.sin(yaw0 / 2)]
        d.ctrl[:] = C.HOME
        for _ in range(50):
            mujoco.mj_step(m, d)
        proc = self.MP(mocap_offset=self.off, tau=o['tau'], spike_pct=o['spike_pct'], body_yaw=math.radians(o['body_yaw_deg']))
        queue, est, t_sim, t_last, next_pose = [], None, 0.0, -1.0, -0.2
        ctl = C.WalkController(policy, goal, meta=dict(ctl_kw.pop('meta', {}), robot='cpusim', sim=o, tau_includes_kd=False), **ctl_kw)
        status = 'walk'
        while status in ('walk', 'stand'):
            while next_pose <= t_sim + 1e-9:
                Rt = C.quat_to_R(d.qpos[3:7])
                pb = d.qpos[0:3] + Rt @ self.off + rng.normal(0, o['pos_noise'], 3)
                queue.append((next_pose + o['latency'], next_pose, pb, to_ros(C.R_to_quat(Rt @ self.Cb.T @ self._noisy_R()))))
                next_pose += 1.0 / o['pose_hz']
            while queue and queue[0][0] <= t_sim + 1e-9:
                _, t_cap, pb, qr = queue.pop(0)
                _, origin, q_xyzw, twist = proc.update(t_cap, pb, qr)
                est = (origin, C.quat_to_R(np.r_[q_xyzw[3], q_xyzw[:3]]), twist)
                t_last = t_sim
            p_w, R_w, twist = est
            v_body = (C.quat_to_R(d.qpos[3:7]).T @ d.qvel[0:3]) if o['true_vel'] else (twist[:3] if twist is not None else np.zeros(3))
            gyro = d.sensordata[self.gyro_adr:self.gyro_adr + 3] + rng.normal(0, o['gyro_noise'], 3)
            q_t, status = ctl.step(p_w, R_w, v_body, gyro, d.qpos[7:].copy(), d.qvel[6:].copy(), d.actuator_force.copy(),
                                   np.zeros(4), t_sim - t_last + o['latency'])
            if status.startswith('abort'):
                break
            d.ctrl[:] = q_t
            for _ in range(self.sub):
                mujoco.mj_step(m, d)
            t_sim += C.DT
        return ctl, status, d.qpos[7:].copy(), d.actuator_force.copy()
