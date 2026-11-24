"""Go1 GOAL-CONDITIONED walker on hardware (2026-11-24): the network itself walks to the goal and turns; no command law.

Mirrors ilc_go1lla/goalwalk.py (the training env):
  obs (48)    walker_core.build_obs with the last 3 entries = the goal in the body (yaw) frame: [gx_b, gy_b] clipped to OBS_R m,
              and the bearing atan2(gy_b, gx_b) (0 within ARRIVE m of the goal)
  walk frame  the start pose, NOT turned: x = the robot's start heading, so a goal may lie in any direction (behind = (-d, 0))
  end         the policy runs until it has held within ARRIVE of the goal for HOLD_S (trained: then a new goal comes), or until
              T_MAX = d / 0.8 + T_EXTRA; the network keeps standing POST_S s more (vel2 learned to stay), then the standing
              controller (a = 0) for SETTLE s, logged, then 'done'
Everything else (policy, joint orders, PD, safety checks, logs) is walker_core's.
"""
import math

import numpy as np

import walker_core_m1121 as C

ARRIVE, HOLD_S, OBS_R, VMAX = 0.2, 1.0, 3.0, 0.8        # goalwalk.py
T_EXTRA = 4.0                                           # s beyond the straight-line time at VMAX (turning, starting, stopping)
POST_S = 2.0                                            # s the NETWORK keeps standing at the goal after arrival (vel2 trains 1-6 s)
SETTLE = C.SETTLE


def goal_obs(p, yaw, goal):
    """goalwalk.GoalWalker._goal_obs: p (walk frame xy), yaw (walk frame), goal (walk frame xy)."""
    dw = np.asarray(goal, float)[:2] - np.asarray(p, float)[:2]
    c, s = math.cos(yaw), math.sin(yaw)
    gb = np.array([c * dw[0] + s * dw[1], -s * dw[0] + c * dw[1]])
    d = float(np.linalg.norm(gb))
    gbc = gb * min(1.0, OBS_R / max(d, 1e-6))
    return np.array([gbc[0], gbc[1], math.atan2(gb[1], gb[0]) if d > ARRIVE else 0.0])


def t_max(goal):
    return float(np.linalg.norm(np.asarray(goal, float)[:2])) / VMAX + T_EXTRA


class GoalWalkController(C.WalkController):
    """C.WalkController's interface (step() once per policy period; returns (q_target, status)), with the goal observation,
    an unturned walk frame, and the end at arrival-and-hold (or T_MAX)."""

    def __init__(self, policy, goal, **kw):
        super().__init__(policy, goal, **kw)
        self.th = None                                   # set at the first step: the frame keeps the start heading
        self.n_max = int(round((t_max(self.goal) + SETTLE) / C.DT))
        self.n = self.n_max
        self.t_sw = t_max(self.goal)
        self.hold = int(round(HOLD_S / C.DT))
        self.inside = 0
        self.meta.update(kind='goal', t_max=self.t_sw, arrive=ARRIVE, hold_s=HOLD_S)

    def step(self, p_w, R_w, v_body, gyro, q, dq, tau, foot_force, pose_age, **extra):
        if self.frame is None:
            self.th = C.yaw_of_R(R_w)                    # WalkFrame(p0, yaw0, th = yaw0): rotation 0 -> the start heading is +x
            self.frame = C.WalkFrame(p_w, C.yaw_of_R(R_w), 0.0)
            self.log = C.WalkLog(self.goal, self.frame, self.meta)
            self.z0 = float(p_w[2])
        if self.k >= self.n:
            return C.motor_targets(np.zeros(12)), 'done'
        p, R = self.frame.pos(p_w), self.frame.rot(R_w)
        t = self.k * C.DT
        reason = None
        if pose_age > self.max_pose_age:
            reason = f'mocap stale {pose_age:.3f} s'
        elif C.tilt_deg(R) > self.max_tilt:
            reason = f'tilt {C.tilt_deg(R):.0f} deg'
        elif self.z0 - p_w[2] > self.max_drop:
            reason = f'trunk {100 * (self.z0 - p_w[2]):.0f} cm below its start height'
        elif self.bounds is not None and not (self.bounds[0] <= p_w[0] <= self.bounds[1] and self.bounds[2] <= p_w[1] <= self.bounds[3]):
            reason = f'left the area at ({p_w[0]:.2f}, {p_w[1]:.2f})'
        g = goal_obs(p, C.yaw_of_R(R), self.goal)
        walking = t < self.t_sw
        obs = C.build_obs(v_body, gyro, R, q, dq, self.last_a, g)
        a = self.pi(obs) if walking else np.zeros(12)
        self.log.add(t=t, p=p, quat=C.R_to_quat(R), qj=q, v=np.r_[R @ np.asarray(v_body), gyro], qd=dq, cmd=g, a=a,
                     contact=(np.asarray(foot_force) > 20.0).astype(float), obs_state=obs, tau_now=tau, foot_force=foot_force,
                     pose_age=pose_age, p_world=p_w, **extra)
        if reason:
            self.log.meta['abort'] = dict(step=self.k, reason=reason)
            return C.motor_targets(np.zeros(12)), 'abort: ' + reason
        if walking and 'arrived' not in self.log.meta:   # arrival: held within ARRIVE for HOLD_S -> stand, log SETTLE more
            self.inside = self.inside + 1 if np.linalg.norm(self.goal[:2] - p[:2]) < ARRIVE else 0
            if self.inside >= self.hold and 'arrived' not in self.log.meta:
                self.t_sw = t + POST_S
                self.n = self.k + 1 + int(round((POST_S + SETTLE) / C.DT))
                self.log.meta['arrived'] = dict(step=self.k, t=round(t, 3))
        self.last_a = a
        self.k += 1
        return C.motor_targets(a), ('walk' if walking else 'stand') if self.k < self.n else 'done'
