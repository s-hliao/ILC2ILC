"""Batched Go1 + ball task: walk through a sequence of goals in the mocap room carrying the ball; the next goal starts
the period after the trunk is within REACH of the current one (switch on reach, drone soccer's protocol), or after
TIMEOUT. The ball should stay near the box centre and never leave the box.

Observation (12): the goal in the body frame / 2 m (clipped to +-1.5), trunk velocities vx, vy, wz, the ball offset
/ travel (2), the ball velocity / 0.5 m/s (2), the last action (3). (No clock: the policy is stationary.)
Escape: the ball leaves the box when it hits a wall hard enough to vault the 7 cm rim (penetration of the padded wall
> ESC_PEN, i.e. an impact faster than ~1 m/s); the operator puts it back (evaluation) or the lane ends (training).
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ilc_dogball import model as m

NOBS = 12
REACH = 0.2
TIMEOUT = 10.0
ESC_PEN = 0.018
V_STOP = 0.08            # stop mode: at rest below this speed (m/s) ...
V_MOVED = 0.3            # ... after having moved faster than this during the goal
STOP_ZONE = 0.6          # ... within this distance of the goal (else it is a pause, not the stop)
ROOM = np.array([[-2.0, 2.0], [-1.5, 1.5]])          # usable mocap-room area (m)
MARGIN = 0.3


def params(base=None, n=None):
    base = m.NOMINAL if base is None else base
    d = {k: jnp.asarray(float(v)) for k, v in base.items()}
    if n is not None:
        d = {k: jnp.full((n,), v) for k, v in d.items()}
    return d


def goal_body(x, g):
    d = g - x[:2]
    c, s = jnp.cos(x[2]), jnp.sin(x[2])
    return jnp.stack([c * d[0] + s * d[1], -s * d[0] + c * d[1]])


def observe(x, g):
    gb = jnp.clip(goal_body(x, g) / 2.0, -1.5, 1.5)
    return jnp.concatenate([gb, x[3:6], x[6:8] / m.TRAVEL, x[8:10] / 0.5, x[13:16]])


def escaped(x):
    return jnp.any(jnp.abs(x[6:8]) - m.TRAVEL > ESC_PEN)


def wall_contact(x, tol=0.005):
    return jnp.any(jnp.abs(x[6:8]) > m.TRAVEL - tol)


def sample_goals(rng, n, n_goals, step=(1.0, 2.5)):
    """Start states (n, 16) at rest, ball centred, random pose in the room; goal sequences (n, n_goals, 2)."""
    lo, hi = ROOM[:, 0] + MARGIN, ROOM[:, 1] - MARGIN
    x0 = np.zeros((n, m.NX))
    x0[:, :2] = rng.uniform(lo, hi, (n, 2))
    x0[:, 2] = rng.uniform(-np.pi, np.pi, n)
    x0[:, 13:16] = (0 - 0.5 * (m.CMD_HI + m.CMD_LO)) / (0.5 * (m.CMD_HI - m.CMD_LO))   # a_prev = zero command
    goals = np.zeros((n, n_goals, 2))
    prev = x0[:, :2].copy()
    for k in range(n_goals):
        for i in range(n):
            for _ in range(100):
                ang = rng.uniform(-np.pi, np.pi)
                r = rng.uniform(*step)
                g = prev[i] + r * np.array([np.cos(ang), np.sin(ang)])
                if np.all(g >= lo) and np.all(g <= hi):
                    break
            else:
                g = rng.uniform(lo, hi)
            goals[i, k] = g
        prev = goals[:, k]
    return x0, goals


def zero_action():
    return (0 - 0.5 * (m.CMD_HI + m.CMD_LO)) / (0.5 * (m.CMD_HI - m.CMD_LO))


@partial(jax.jit, static_argnums=(4,), static_argnames=("reset_ball", "stop_mode"))
def rollout(policy_params, x0, goals, p, policy_fn, noise, reset_ball=False, stop_mode=False):
    """Fly lanes for noise.shape[0] periods through their goal sequences.

    Returns a dict of per-period arrays (T, n, ...): x (state after the period; x0 prepended as 'x0'), obs, act,
    goal_idx (the goal in force during the period), reached (the period ended within REACH: the goal switches),
    escaped (the ball left the box during the period), t_goal (time on the goal at the period's start).
    After all goals are done the lane holds its last goal. reset_ball: an escaped ball is put back at the centre
    (evaluation); otherwise an escape ends the lane (alive = 0 afterwards).
    """
    n_goals = goals.shape[1]

    def one(carry, inp):
        x, gi, tg, alive, t, moved = carry
        k, e = inp
        g = jax.vmap(lambda gg, i: gg[jnp.minimum(i, n_goals - 1)])(goals, gi)
        o = jax.vmap(observe)(x, g)
        a = policy_fn(policy_params, o) + e
        x1 = jax.vmap(m.step, (0, 0, None, 0))(x, a, t, p)
        esc = jax.vmap(escaped)(x1)
        if reset_ball:
            x1 = jnp.where(esc[:, None], x1.at[:, 6:10].set(0.0), x1)
        dist = jnp.linalg.norm(x1[:, :2] - g, axis=-1)
        speed = jnp.hypot(x1[:, 3], x1[:, 4])
        moved1 = moved | (speed > V_MOVED)
        if stop_mode:      # the dog has stopped: the goal is done wherever it stopped (the stop error is the metric)
            reached = moved1 & (speed < V_STOP) & (dist < STOP_ZONE) & (gi < n_goals)
        else:
            reached = (dist < REACH) & (gi < n_goals)
        timeout = (tg + m.DT >= TIMEOUT) & (gi < n_goals)
        adv = reached | timeout
        out = dict(x=x1, obs=o, act=a, goal_idx=gi, reached=reached, timeout=timeout, escaped=esc, t_goal=tg,
                   alive=alive, goal=g, dist=dist)
        gi1 = gi + adv.astype(gi.dtype)
        tg1 = jnp.where(adv, 0.0, tg + m.DT)
        alive1 = alive if reset_ball else alive * (1.0 - esc.astype(alive.dtype))
        moved2 = jnp.where(adv, False, moved1)
        return (x1, gi1, tg1, alive1, t + m.DT, moved2), out
    n = x0.shape[0]
    T = noise.shape[0]
    carry0 = (x0, jnp.zeros(n, jnp.int32), jnp.zeros(n), jnp.ones(n), jnp.asarray(0.0), jnp.zeros(n, bool))
    _, out = jax.lax.scan(one, carry0, (jnp.arange(T), noise))
    return out
