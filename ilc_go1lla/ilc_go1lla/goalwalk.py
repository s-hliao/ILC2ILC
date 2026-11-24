"""Goal-conditioned Go1 walker (2026-11-24): the NETWORK walks to goals and turns, no hand-written command law.

Subclass of Playground's Go1 Joystick env.
- The observation keeps its 48-dim layout; the last 3 entries (the joystick command) become the goal in the body frame:
  [gx_b, gy_b] clipped to a 3 m radius, and the bearing atan2(gy_b, gx_b) (zeroed within ARRIVE m).
- Goals: 1-4 m away in ANY direction from random start headings (Playground's reset already randomizes yaw +-pi).
- After the robot holds within ARRIVE of the goal for HOLD_S seconds (variant vel2: a random 1-6 s per goal, so it learns to
  STAY), a new goal is drawn from where it stands
  (goal-to-goal, so turns of any size are trained).
- Rewards: Playground's gait / regularization terms unchanged. The velocity targets those terms use are derived from the
  goal (reward shaping only; the policy never sees them):
    v* = VMAX * min(1, d / SLOW_D) toward the goal (body frame), wz* = clip(K_TURN * bearing) while d > ARRIVE.
  Optional terms (scale 0 = off): progress (d_prev - d) / dt, heading cos(bearing) while far, arrive (stand still at goal).
"""
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco_playground._src.locomotion.go1 import joystick

VMAX, SLOW_D, ARRIVE, HOLD_S, K_TURN, WZ_MAX = 0.8, 0.6, 0.2, 1.0, 1.5, 1.0
R_MIN, R_MAX, OBS_R = 1.0, 4.0, 3.0


def default_config(variant='vel'):
    cfg = joystick.default_config()
    cfg.goal_hold = [HOLD_S, HOLD_S]
    if variant == 'vel2':                       # (11-24 07:50) hold a random 1-6 s at each goal: v1 (1 s) walked off after arriving
        cfg.goal_hold = [1.0, 6.0]
    sc = cfg.reward_config.scales
    sc.progress, sc.heading, sc.arrive = 0.0, 0.0, 0.5
    if variant == 'prog':                       # task reward instead of velocity shaping
        sc.tracking_lin_vel, sc.tracking_ang_vel = 0.0, 0.0
        sc.progress, sc.heading, sc.arrive = 2.0, 0.5, 1.0
    return cfg


def _yaw(q):
    return jp.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


class GoalWalker(joystick.Joystick):

    def _rel(self, data, goal):
        p, yaw = data.qpos[:2], _yaw(data.qpos[3:7])
        dw = goal - p
        c, s = jp.cos(yaw), jp.sin(yaw)
        gb = jp.array([c * dw[0] + s * dw[1], -s * dw[0] + c * dw[1]])
        d = jp.linalg.norm(gb)
        return gb, d, jp.arctan2(gb[1], gb[0])

    def _goal_obs(self, data, info):
        gb, d, b = self._rel(data, info['goal'])
        gb = gb * jp.minimum(1.0, OBS_R / jp.maximum(d, 1e-6))
        return jp.array([gb[0], gb[1], jp.where(d > ARRIVE, b, 0.0)])

    def _desired(self, data, info):
        gb, d, b = self._rel(data, info['goal'])
        far = d > ARRIVE
        sp = VMAX * jp.minimum(1.0, d / SLOW_D) * far
        wz = jp.clip(K_TURN * b, -WZ_MAX, WZ_MAX) * (d > 2 * ARRIVE)
        return jp.array([sp * jp.cos(b), sp * jp.sin(b), wz])

    def _new_goal(self, rng, p):
        k1, k2 = jax.random.split(rng)
        r = jax.random.uniform(k1, minval=R_MIN, maxval=R_MAX)
        th = jax.random.uniform(k2, minval=-jp.pi, maxval=jp.pi)
        return p + r * jp.array([jp.cos(th), jp.sin(th)])

    def _hold_n(self, rng):
        lo, hi = self._config.goal_hold
        return jp.round(jax.random.uniform(rng, minval=lo, maxval=hi) / self.dt).astype(jp.int32)

    def reset(self, rng):
        rng, k = jax.random.split(rng)
        st = super().reset(rng)
        info = st.info
        info['goal'] = self._new_goal(k, st.data.qpos[:2])
        info['at_goal'] = jp.zeros((), jp.int32)
        info['hold_n'] = self._hold_n(jax.random.fold_in(k, 3))
        info['d_prev'] = jp.linalg.norm(info['goal'] - st.data.qpos[:2])
        info['command'] = self._desired(st.data, info)
        st.metrics['dist'] = info['d_prev']
        st.metrics['goals_reached'] = jp.zeros(())
        return st.replace(obs=self._get_obs(st.data, info), info=info)

    def _get_obs(self, data, info):
        if 'goal' not in info:                  # inside Joystick.reset, before the goal exists
            return super()._get_obs(data, info)
        return super()._get_obs(data, dict(info, command=self._goal_obs(data, info)))

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        inf = dict(info, command=self._desired(data, info))
        r = super()._get_reward(data, action, inf, metrics, done, first_contact, contact)
        gb, d, b = self._rel(data, info['goal'])
        v = self.get_local_linvel(data)[:2]
        r['progress'] = jp.clip((info['d_prev'] - d) / self.dt, -1.0, 1.0)
        r['heading'] = jp.cos(b) * (d > 2 * ARRIVE)
        r['arrive'] = (d < ARRIVE) * jp.exp(-jp.sum(v ** 2) / 0.05)
        return r

    def step(self, state, action):
        st = super().step(state, action)
        info = st.info
        p = st.data.qpos[:2]
        d = jp.linalg.norm(info['goal'] - p)
        at = jp.where(d < ARRIVE, info['at_goal'] + 1, 0)
        hit = at >= info['hold_n']
        info['rng'], k, kh = jax.random.split(info['rng'], 3)
        info['hold_n'] = jp.where(hit, self._hold_n(kh), info['hold_n'])
        info['goal'] = jp.where(hit, self._new_goal(k, p), info['goal'])
        info['at_goal'] = jp.where(hit, 0, at)
        info['d_prev'] = jp.linalg.norm(info['goal'] - p)
        info['command'] = self._desired(st.data, info)
        st.metrics['dist'] = d
        st.metrics['goals_reached'] = hit.astype(jp.float32)
        return st.replace(info=info)
