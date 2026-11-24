"""The DistILC stage for the Go1 ball task (drone soccer's stage, ported). One trial = one goal segment of a flight.

Rows at every period t of a segment (scaled; all relative to the measured state):
  ball      sign(b) (|b_t| - MARGIN_FRAC travel) / BALL_SCALE per axis while the ball is outside its safe zone (the
            task error: the ball approaching a wall; zero inside, so the rows do not ask the dog to stop accelerating)
            -- masked while the ball touches a wall (the padded-wall contact is hybrid: no authority), after an escape
  approach  the lateral miss of the trunk's velocity line w.r.t. the goal centre, (v x (g - p)) / |v| / MISS_SCALE,
            while |v| >= 0.2 m/s and the heading of v is within 45 deg of the goal (drone soccer's aiming row)
G: the closed-loop lifted Jacobian of the rows w.r.t. action offsets in the NOMINAL model along the measured states
(the command-filter state, unmeasurable on hardware, is replaced by the last command), the policy's own feedback
through the observation. Step: du = -beta (G'G + lam I)^-1 G'e, lam = delta tr(G'G) / n_cols, rms cap; actions at the
box limits masked. Targets y = pi(o) + du.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ilc_dogball import model as m
from ilc_dogball import sim

BALL_SCALE = 0.03
MISS_SCALE = 0.3
V_AIM = 0.2
AIM_COS = np.cos(np.radians(45.0))
V_STAR = None                     # speed row target (m/s): the initial policy's median speed at reach in the NOMINAL sim
STOP_SCALE = 0.05                 # stop rows: the measured stop error (m) at the stop tick (parking: terminal rows converge)
SPEED_SCALE = 0.5                 # (set by stage_run; drone soccer's pass-speed row, 2026-11-16 option 1)
MASK_CONTACT = True               # drop ball rows while the ball touches a wall (set by stage_run --no-contact-mask)


def measured(xs, acts_prev):
    """What hardware measures: the state with the command filter replaced by the last command."""
    xs = np.array(xs, float).copy()
    xs[:, 10:13] = np.asarray(m.action_to_cmd(jnp.asarray(acts_prev)))
    return xs


MARGIN = 0.6 * m.TRAVEL                 # = ppo.MARGIN_FRAC * travel


def _ball_rows(x):
    b = x[6:8]
    return jnp.sign(b) * (jnp.abs(b) - MARGIN) / BALL_SCALE


def _miss_row(x, g):
    c, s = jnp.cos(x[2]), jnp.sin(x[2])
    v = jnp.stack([x[3] * c - x[4] * s, x[3] * s + x[4] * c])
    d = g - x[:2]
    return (v[0] * d[1] - v[1] * d[0]) / (jnp.linalg.norm(v) + 1e-6) / MISS_SCALE


@partial(jax.jit, static_argnums=(5,))
def linearize(params, xs, acts, g, t0, actor_fn, p):
    """A (T, 16, 16), B (T, 16, 3), K (T, 3, 16) along the segment's states xs[:-1], actions acts."""
    def pi(x):
        return actor_fn(params, sim.observe(x, g)[None])[0]
    ts = t0 + jnp.arange(acts.shape[0]) * m.DT

    def one(x, a, t):
        A, B = jax.jacfwd(m.step, argnums=(0, 1))(x, a, t, p)
        return A, B, jax.jacfwd(pi)(x)
    return jax.vmap(one)(xs[:-1], acts, ts)


@jax.jit
def _row_jacobians(xs, g):
    eb = jax.vmap(_ball_rows)(xs)
    Cb = jax.vmap(jax.jacfwd(_ball_rows))(xs)
    em = jax.vmap(_miss_row, (0, None))(xs, g)
    Cm = jax.vmap(jax.grad(_miss_row), (0, None))(xs, g)
    return eb, Cb, em, Cm


def _speed(x):
    return jnp.sqrt(x[3] ** 2 + x[4] ** 2 + 1e-6)


def rows(xs, g, ball=True, approach=True, with_kinds=False, speed_at=None, stop_at=None):
    """Rows of a segment's states xs (T+1, 16) (rows at ticks 1..T): e (R,), C (R, 16), ticks (R,)
    [, kinds (R,): 0 / 1 ball x / y, 2 approach]."""
    eb, Cb, em, Cm = (np.asarray(z) for z in _row_jacobians(jnp.asarray(xs), jnp.asarray(g)))
    b = xs[:, 6:8]
    free = ~np.any(np.abs(b) > m.TRAVEL - 0.005, -1)
    esc = np.cumsum(np.any(np.abs(b) - m.TRAVEL > sim.ESC_PEN, -1)) > 0
    c, s = np.cos(xs[:, 2]), np.sin(xs[:, 2])
    v = np.stack([xs[:, 3] * c - xs[:, 4] * s, xs[:, 3] * s + xs[:, 4] * c], -1)
    d = g[None] - xs[:, :2]
    sp = np.linalg.norm(v, axis=-1)
    aim = (sp >= V_AIM) & (np.sum(v * d, -1) >= AIM_COS * sp * np.linalg.norm(d, axis=-1))
    e, C, tk, kd = [], [], [], []
    for t in range(1, len(xs)):
        if esc[t]:
            break
        if ball and (free[t] or not MASK_CONTACT):
            for i in range(2):
                if abs(b[t, i]) > MARGIN[i]:
                    e.append(eb[t, i]); C.append(Cb[t, i]); tk.append(t); kd.append(i)
        if approach and aim[t]:
            e.append(em[t]); C.append(Cm[t]); tk.append(t); kd.append(2)
    if stop_at is not None:                            # the stop error (kinds 4, 5): x, y of the trunk at rest minus the goal
        for i in range(2):
            e.append(float((xs[stop_at, i] - g[i]) / STOP_SCALE))
            row = np.zeros(xs.shape[1]); row[i] = 1.0 / STOP_SCALE
            C.append(row); tk.append(int(stop_at)); kd.append(4 + i)
    if speed_at is not None and V_STAR is not None:       # the speed row at the reach tick (kind 3)
        x = jnp.asarray(xs[speed_at])
        e.append(float((_speed(x) - V_STAR) / SPEED_SCALE)); C.append(np.asarray(jax.grad(_speed)(x)) / SPEED_SCALE)
        tk.append(int(speed_at)); kd.append(3)
    out = (np.asarray(e).reshape(-1), np.asarray(C).reshape(-1, xs.shape[1]), np.asarray(tk, int), np.asarray(kd, int))
    return out if with_kinds else out[:3]


def lifted(A, B, K, masks, C, ticks):
    """G (R, T, na): row r = C_r dx_{t_r} / d du_j (closed loop), j < t_r."""
    T, nx, na = B.shape
    G = np.zeros((len(ticks), T, na))
    Acl = A + np.einsum('tij,tjk->tik', B, K)
    S = np.zeros((T, nx, na))
    for t in range(1, T + 1):
        S[:t - 1] = np.einsum('ik,jkl->jil', Acl[t - 1], S[:t - 1])
        S[t - 1] = B[t - 1] * masks[t - 1][None]
        sel = ticks == t
        if sel.any():
            G[sel, :t] = np.einsum('rk,jkl->rjl', C[sel], S[:t])
    return G


def gn_step(G, e, beta=0.5, delta=0.5, cap_rms=0.3):
    flat = G.reshape(len(e), -1)
    free = np.abs(flat).sum(0) > 0
    Gf = flat[:, free]
    normal = Gf.T @ Gf
    n = normal.shape[0]
    lam = delta * np.trace(normal) / max(n, 1) + 1e-9
    stepf = -beta * np.linalg.solve(normal + lam * np.eye(n), Gf.T @ e)
    rms = np.sqrt(np.mean(stepf ** 2)) if n else 0.0
    if rms > cap_rms:
        stepf *= cap_rms / rms
    step = np.zeros(flat.shape[1])
    step[free] = stepf
    return step.reshape(G.shape[1:])


def segment_step(params, actor_fn, xs, acts, acts_prev, g, t0, p_nom, ball=True, approach=True, reached=False, stop=False, **kw):
    """The GN step (T, 3) for one measured segment: xs (T+1, 16) states, acts (T, 3) applied actions."""
    xm = measured(xs, np.concatenate([acts_prev[None], acts]))
    e, C, tk = rows(xm, g, ball, approach, speed_at=len(xm) - 1 if (reached and not stop) else None,
                    stop_at=len(xm) - 1 if (reached and stop) else None)
    if len(e) == 0:
        return np.zeros_like(acts), e
    A, B, K = (np.asarray(z) for z in linearize(params, jnp.asarray(xm), jnp.asarray(acts), jnp.asarray(g),
                                                 float(t0), actor_fn, p_nom))
    mean = np.asarray(actor_fn(params, jax.vmap(sim.observe, (0, None))(jnp.asarray(xm[:-1]), jnp.asarray(g))))
    masks = (np.abs(mean) < 0.98).astype(float)
    G = lifted(A, B, K, masks, C, tk)
    return gn_step(G, e, **kw), e
