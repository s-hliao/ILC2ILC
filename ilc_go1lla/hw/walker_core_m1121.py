"""(FROZEN copy of walker_core.py as of 11-21, md5 6e9d0acf, imported by the 11-24 goal walker files so later edits to walker_core.py cannot change them.)
Go1 goal walker on hardware: everything the robot side needs, numpy only (no JAX / MuJoCo / ROS), 2026-11-21.

Mirrors ilc_go1lla/walkenv.py exactly (tests: hw/test_walker_core.py against the MJX walker):
  policy      Playground Go1 joystick PPO network (normalizer + MLP 512-256-128 swish, deterministic tanh(mean))
  command     the fixed goal-tracking law (trapezoidal straight-line reference, body-frame v = v_ref + K (p_ref - p), yaw to the line)
  obs (48)    [linvel at the IMU site (body), gyro, gravity in the body, q - home, dq, last action, command]  (Playground order)
  action      motor target q = home + 0.5 a, PD kp 35 / kd 0.5 on the motor drivers (the MJCF's position actuators + joint damping)
  hand-over   STAND_DELAY s after the reference reaches the goal the walker gives way to the standing controller (a = 0)
Orders: the network's joints are FR, FL, RR, RL (hip, thigh, calf); the Go1 bridge's ("canonical") are FL, FR, RL, RR.
The walk frame: the start pose is the origin, the robot faces the goal direction (as every simulated walk starts).
"""
import json
import math
import time

import numpy as np

DT = 0.02                                                    # policy period (Playground ctrl_dt)
KP, KD, ACTION_SCALE = 35.0, 0.5, 0.5
PG_LEGS = ('FR', 'FL', 'RR', 'RL')                             # network / MJCF order
CAN_LEGS = ('FL', 'FR', 'RL', 'RR')                            # go1_bridge joint_states / joint_cmd order
PG_FROM_CAN = np.array([3 * CAN_LEGS.index(l) + j for l in PG_LEGS for j in range(3)])   # x_pg = x_can[PG_FROM_CAN]
CAN_FROM_PG = np.array([3 * PG_LEGS.index(l) + j for l in CAN_LEGS for j in range(3)])   # x_can = x_pg[CAN_FROM_PG]
FOOT_PG_FROM_CAN = np.array([CAN_LEGS.index(l) for l in PG_LEGS])
HOME = np.array([0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8])        # keyframe "home" (PG order)
HOME_Z = 0.278
IMU_POS = np.array([-0.01592, -0.06659, -0.00617])           # the MJCF's imu site in the trunk frame (velocimeter / gyro)
TAU_MAX = np.array([23.7, 23.7, 35.55] * 4)
Q_LO = np.array([-0.863, -0.686, -2.818] * 4)
Q_HI = np.array([0.863, 4.501, -0.888] * 4)

CMD_MAX = np.array([1.5, 0.5, 1.0])
K_POS, K_YAW = 1.5, 2.0
V_REF, A_REF, T0 = 0.8, 1.0, 0.5
STAND_DELAY = 0.5
SETTLE = 0.3                                                 # s logged after the hand-over (goal_zs --settle)


# ---------------------------------------------------------------- policy
class Policy:
    """npz from hw/export_policy.py: mean, std (48), W0..W3, b0..b3."""

    def __init__(self, path):
        d = np.load(path)
        self.path = path
        self.mean, self.std = d['mean'].astype(np.float64), d['std'].astype(np.float64)
        self.W = [d[f'W{i}'].astype(np.float64) for i in range(4)]
        self.b = [d[f'b{i}'].astype(np.float64) for i in range(4)]

    def __call__(self, obs):
        x = (np.asarray(obs, np.float64) - self.mean) / self.std
        for i in range(3):
            x = x @ self.W[i] + self.b[i]
            x = x / (1.0 + np.exp(-x))                          # swish (brax MLP default)
        x = x @ self.W[3] + self.b[3]
        return np.tanh(x[:12])


# ---------------------------------------------------------------- reference and command (walkenv.reference / command)
def reference(t, goal):
    goal = np.asarray(goal, float)
    vmax = goal[2] if goal.shape[0] == 3 else V_REF
    g = goal[:2]
    L = np.linalg.norm(g)
    u = g / L
    tt = max(t - T0, 0.0)
    t_acc = vmax / A_REF
    s = 0.5 * A_REF * min(tt, t_acc) ** 2 + vmax * max(tt - t_acc, 0.0)
    v = A_REF * tt if tt < t_acc else vmax
    rem = L - s
    if rem < 0.5 * vmax ** 2 / A_REF:
        v = math.sqrt(max(2 * A_REF * rem, 0.0))
    s = min(s, L)
    if s >= L:
        v = 0.0
    return s * u, v * u, math.atan2(u[1], u[0])


def ref_end_time(goal):
    goal = np.asarray(goal, float)
    vmax = goal[2] if goal.shape[0] == 3 else V_REF
    return T0 + np.linalg.norm(goal[:2]) / vmax + vmax / (2 * A_REF)


def command(p, yaw, goal, t):
    pr, vr, head = reference(t, goal)
    d = vr + K_POS * (pr - np.asarray(p)[:2])
    c, s = math.cos(yaw), math.sin(yaw)
    db = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    v = np.clip(db, -CMD_MAX[:2], CMD_MAX[:2])
    eh = math.atan2(math.sin(head - yaw), math.cos(head - yaw))
    wz = float(np.clip(K_YAW * eh, -CMD_MAX[2], CMD_MAX[2]))
    return np.array([v[0], v[1], wz])


def handover_time(goal):
    return float(ref_end_time(goal)) + STAND_DELAY


def walk_steps(goal):
    """Control steps of one logged walk: to the hand-over plus SETTLE (goal_zs.py's n)."""
    return int((handover_time(goal) + SETTLE) / DT)


# ---------------------------------------------------------------- rotations
def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def yaw_of_R(R):
    return math.atan2(R[1, 0], R[0, 0])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class WalkFrame:
    """World (mocap) -> walk frame: origin at the start position (ground level kept: z stays the mocap's), the robot's start
    heading turned to the goal direction th, as every simulated walk starts (origin, at rest, facing the goal)."""

    def __init__(self, p0_world, yaw0_world, th):
        self.p0 = np.array([p0_world[0], p0_world[1], 0.0])
        self.Rf = Rz(th - yaw0_world)                         # walk = Rf (world - p0)

    def pos(self, p_world):
        return self.Rf @ (np.asarray(p_world) - self.p0)

    def rot(self, R_world):
        return self.Rf @ R_world

    def vec(self, v_world):
        return self.Rf @ np.asarray(v_world)

    def to_world(self, p_walk):
        return self.Rf.T @ np.asarray(p_walk) + self.p0


# ---------------------------------------------------------------- observation
def build_obs(v_origin_body, gyro_body, R, q_pg, dq_pg, last_act, cmd):
    """The 48-vector of Go1JoystickFlatTerrain._get_obs (noise off). v_origin_body: the trunk origin's velocity in the trunk frame;
    the velocimeter sits at the imu site, so v_site = v_origin + w x r_imu (both in the trunk frame; the site is not rotated)."""
    w = np.asarray(gyro_body, float)
    lin = np.asarray(v_origin_body, float) + np.cross(w, IMU_POS)
    grav = R.T @ np.array([0.0, 0.0, -1.0])
    return np.concatenate([lin, w, grav, np.asarray(q_pg) - HOME, np.asarray(dq_pg), np.asarray(last_act), np.asarray(cmd)])


def motor_targets(a):
    return HOME + ACTION_SCALE * np.asarray(a)


# ---------------------------------------------------------------- safety
def tilt_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))


# ---------------------------------------------------------------- walk log -> the simulated rollout's fields
class WalkLog:
    """One walk, one row per policy step. save() writes the fields stage/metrics read from a simulated rollout
    (walkenv.Walker._rollout's `out`): p, quat (w x y z), qj, v (world linear of the trunk origin, body angular), qd, cmd, a,
    contact, obs_state, tau (measured, after the step), q_after (joints after the step); plus raw timing and signals."""

    KEYS = ('t', 'p', 'quat', 'qj', 'v', 'qd', 'cmd', 'a', 'contact', 'obs_state', 'tau_now', 'foot_force', 'pose_age', 'p_world')

    def __init__(self, goal, frame, meta):
        self.goal, self.frame, self.meta = np.asarray(goal, float), frame, dict(meta)
        self.rows = {k: [] for k in self.KEYS}
        self.t_wall0 = time.time()

    def add(self, **kw):
        for k in self.KEYS:
            self.rows[k].append(np.asarray(kw[k], float))
        for k, v in kw.items():                                   # extra signals (e.g. w_mocap, imu_quat): logged as given
            if k not in self.KEYS:
                self.rows.setdefault(k, []).append(np.asarray(v, float))

    def arrays(self, final_q=None, final_tau=None):
        r = {k: np.stack(v) for k, v in self.rows.items() if v}
        n = len(r['t'])
        # the simulated tau / q_after are read after env.step: here the next row's measurement (the last row: the final reading)
        r['tau'] = np.concatenate([r['tau_now'][1:], [final_tau if final_tau is not None else r['tau_now'][-1]]])
        r['q_after'] = np.concatenate([r['qj'][1:], [final_q if final_q is not None else r['qj'][-1]]])
        assert len(r['tau']) == n
        return r

    def save(self, path, **final):
        r = self.arrays(**final)
        np.savez(path, goal=self.goal, frame_p0=self.frame.p0, frame_R=self.frame.Rf,
                 meta=json.dumps(dict(self.meta, t_wall0=self.t_wall0, saved=time.time())), **r)
        return r


# ---------------------------------------------------------------- one walk, step by step (shared by walker_node.py and the sim tests)
class WalkController:
    """One goal walk. Call step() once per policy period with the latest measurements (all in the network's joint order):
        p_w, R_w        trunk origin position / rotation in the mocap world (after the rigid-body offset and body-yaw calibration)
        v_body          trunk origin velocity in the trunk frame (mocap, filtered)
        gyro            trunk angular velocity in the trunk frame (Go1 IMU)
        q, dq, tau      joints (encoders) and measured motor torque (tauEst)
        foot_force      raw foot forces (logged only)
        pose_age        s since the last mocap pose
    It returns (q_target, status): q_target for the motor PD (KP, KD), status 'walk' | 'stand' (after the hand-over) | 'done'
    | 'abort: <reason>' (the caller damps). The walk frame is fixed at the first step: the start pose turned to face the goal."""

    def __init__(self, policy, goal, max_tilt=45.0, max_drop=0.12, max_pose_age=0.1, bounds=None, meta=None):
        self.pi, self.goal = policy, np.asarray(goal, float)
        self.th = math.atan2(self.goal[1], self.goal[0])
        self.n = walk_steps(self.goal)
        self.t_sw = handover_time(self.goal)
        # max_drop: the trunk more than this below its height at the walk's start (relative: foam / mats change the floor level)
        self.max_tilt, self.max_drop, self.max_pose_age, self.bounds = max_tilt, max_drop, max_pose_age, bounds
        self.z0 = None
        self.k = 0
        self.last_a = np.zeros(12)
        self.frame = None
        self.log = None
        self.meta = dict(meta or {}, policy=getattr(policy, 'path', ''), goal=self.goal.tolist(), n=self.n, t_switch=self.t_sw)

    def step(self, p_w, R_w, v_body, gyro, q, dq, tau, foot_force, pose_age, **extra):
        if self.frame is None:
            self.frame = WalkFrame(p_w, yaw_of_R(R_w), self.th)
            self.log = WalkLog(self.goal, self.frame, self.meta)
            self.z0 = float(p_w[2])
        if self.k >= self.n:
            return motor_targets(np.zeros(12)), 'done'
        p, R = self.frame.pos(p_w), self.frame.rot(R_w)
        t = self.k * DT
        reason = None
        if pose_age > self.max_pose_age:
            reason = f'mocap stale {pose_age:.3f} s'
        elif tilt_deg(R) > self.max_tilt:
            reason = f'tilt {tilt_deg(R):.0f} deg'
        elif self.z0 - p_w[2] > self.max_drop:
            reason = f'trunk {100 * (self.z0 - p_w[2]):.0f} cm below its start height'
        elif self.bounds is not None and not (self.bounds[0] <= p_w[0] <= self.bounds[1] and self.bounds[2] <= p_w[1] <= self.bounds[3]):
            reason = f'left the area at ({p_w[0]:.2f}, {p_w[1]:.2f})'
        cmd = command(p, yaw_of_R(R), self.goal, t)
        obs = build_obs(v_body, gyro, R, q, dq, self.last_a, cmd)
        a = self.pi(obs) if t < self.t_sw else np.zeros(12)
        self.log.add(t=t, p=p, quat=R_to_quat(R), qj=q, v=np.r_[R @ np.asarray(v_body), gyro], qd=dq, cmd=cmd, a=a,
                     contact=(np.asarray(foot_force) > 20.0).astype(float), obs_state=obs, tau_now=tau, foot_force=foot_force,
                     pose_age=pose_age, p_world=p_w, **extra)
        if reason:
            self.log.meta['abort'] = dict(step=self.k, reason=reason)
            return motor_targets(np.zeros(12)), 'abort: ' + reason
        self.last_a = a
        self.k += 1
        return motor_targets(a), ('walk' if t < self.t_sw else 'stand') if self.k < self.n else 'done'
