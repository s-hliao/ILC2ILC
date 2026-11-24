"""The DistILC hardware stage for drift parking (drone soccer's stage, ported): one park = one trial.

Per park, from the MEASURED trajectory (states x_t, actions a_t; the slot):
  rows   predicted stop pose: at every tick while the car moves, where the NOMINAL model says the car stops if the current
         policy flies on from the measured x_t (slot frame: along, lateral / 0.15 m, heading / 10 deg) -- the analog of
         drone soccer's predicted lateral miss: time-local, correctable while the car is still sliding;
         final pose: the measured stop (same scales);
         optional sideslip floor (--beta-row): hinge(beta_min - |beta|) / 5 deg while V > v_drift (keeps the entry a drift).
  G      the closed-loop lifted Jacobian of the rows w.r.t. action offsets: nominal A_t, B_t along the measured states,
         the policy's own feedback K_t = d pi / d x through the observation; row r = C_r Phi(t_r, j + 1) B_j.
  step   du = -beta (G'G + lam I)^-1 G'e, lam = delta tr(G'G) / n_cols, rms-capped (pogo / car / drone form); action
         components at the box limits masked.
  targets y_t = pi(o_t) + du_t, replayed with age decay; refit with the anchored loss (nominal-sim anchor pool,
         weight rho N0 / (N0 + parks)).
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ilc_park import sim

ROW_SCALE = np.array([0.15, 0.15, np.radians(10.0)])          # along, lateral (m), heading (rad)
V_MOVING = 0.1


def make_policy(actor_fn):
    def pi_of_state(params, x, slot, t):
        return actor_fn(params, sim.observe(x, slot, t)[None])[0]
    return pi_of_state


@partial(jax.jit, static_argnums=(4,))
def linearize(params, xs, as_, slot, actor_fn, p):
    """A (T, 9, 9), B (T, 9, 2), K (T, 2, 9) of the nominal car along a measured park (one lane)."""
    pi = make_policy(actor_fn)
    T = as_.shape[0]
    ts = jnp.arange(T) * sim.DT

    def one(x, a, t):
        A, B = jax.jacfwd(sim.step, argnums=(0, 1))(x, a, p)
        K = jax.jacfwd(pi, argnums=1)(params, x, slot, t)
        return A, B, K
    return jax.vmap(one)(xs[:-1], as_, ts)


def _predicted_stop(params, x, slot, t, actor_fn, horizon, p):
    """The nominal closed-loop flight from x at time t for `horizon` periods: the final pose in the slot frame
    (along, lateral, heading) -- the predicted stop."""
    pi = make_policy(actor_fn)

    def body(xx, k):
        a = pi(params, xx, slot, t + k * sim.DT)
        return sim.step(xx, a, p), None
    xe, _ = jax.lax.scan(body, x, jnp.arange(horizon))
    pos, dpsi = sim.slot_frame(xe, slot)
    return jnp.concatenate([pos, dpsi[None]])


@partial(jax.jit, static_argnums=(4, 5))
def predicted_stops(params, xs, slot, ts, actor_fn, horizon, p):
    """Predicted stops (n, 3) and their Jacobians w.r.t. the state (n, 3, 9) from n measured states at times ts."""
    f = lambda x, t: _predicted_stop(params, x, slot, t, actor_fn, horizon, p)          # noqa: E731
    return jax.vmap(f)(xs, ts), jax.vmap(jax.jacfwd(f))(xs, ts)


def _final_pose(x, slot):
    pos, dpsi = sim.slot_frame(x, slot)
    return jnp.concatenate([pos, dpsi[None]])


A_STOP = 2.6          # m/s^2: the nominal policy's braking deceleration in the NOMINAL sim (median over held-out parks from
                      # 1.5 m/s to rest, IQR 2.40-2.73; scripts measured 2026-11-16) -- a nominal-model quantity


def _kinematic_stop(x, slot):
    """Where the car stops if it decelerates at A_STOP along its current velocity line, and its heading then (yaw rate
    decaying linearly over the stop): drone soccer's velocity-line miss, extended to a stop. Local in x, no rollout."""
    c, s = jnp.cos(x[2]), jnp.sin(x[2])
    vw = jnp.stack([x[3] * c - x[4] * s, x[3] * s + x[4] * c])
    V = jnp.sqrt(x[3] ** 2 + x[4] ** 2 + 1e-6)
    t = V / A_STOP
    xe = jnp.concatenate([x[:2] + vw * t / 2, jnp.stack([x[2] + x[5] * t / 2]), x[3:]])
    return _final_pose(xe, slot)


def rows_and_outputs(params, xs, slot, actor_fn, p, t_stop, horizon, beta_row=False, beta_min_deg=10.0, v_drift=1.2,
                     ticks_p=None, pred='rollout'):
    """Scaled error rows, their output Jacobians C (R, 9) w.r.t. the state at their tick, and the ticks (R,).

    ``ticks_p``: evaluate the predicted-stop rows at these ticks (the positivity check re-evaluates a re-flown park at the
    first flight's ticks); default: every tick before the stop at which the car moves."""
    xs = jnp.asarray(xs)
    V = np.asarray(sim.speed(xs))
    if ticks_p is None:
        ticks_p = np.array([t for t in range(1, t_stop) if V[t] >= V_MOVING], int)
    errs, outs, ticks = [], [], []
    if pred == 'none':
        ticks_p = ticks_p[:0]
    if len(ticks_p) and pred == 'kinematic':
        e = jax.vmap(_kinematic_stop, (0, None))(xs[ticks_p], slot)
        C = jax.vmap(jax.jacfwd(_kinematic_stop), (0, None))(xs[ticks_p], slot)
    elif len(ticks_p):
        e, C = predicted_stops(params, xs[ticks_p], slot, jnp.asarray(ticks_p * sim.DT), actor_fn, horizon, p)
    if len(ticks_p):
        errs.append((np.asarray(e) / ROW_SCALE).ravel())
        outs.append((np.asarray(C) / ROW_SCALE[None, :, None]).reshape(-1, xs.shape[1]))
        ticks.append(np.repeat(ticks_p, 3))
    errs.append(np.asarray(_final_pose(xs[t_stop], slot)) / ROW_SCALE)            # the measured stop
    outs.append(np.asarray(jax.jacfwd(_final_pose)(xs[t_stop], slot)) / ROW_SCALE[:, None])
    ticks.append(np.full(3, t_stop))
    if beta_row:
        bmin = np.radians(beta_min_deg)
        beta = np.asarray(sim.sideslip(xs))
        tb = np.array([t for t in range(1, t_stop) if V[t] > v_drift and abs(beta[t]) < bmin], int)
        if len(tb):
            g = np.asarray(jax.vmap(jax.grad(sim.sideslip))(xs[tb])) * (-np.sign(beta[tb]))[:, None]
            errs.append((bmin - np.abs(beta[tb])) / np.radians(5.0))
            outs.append(g / np.radians(5.0))
            ticks.append(tb)
    return np.concatenate(errs), np.concatenate(outs), np.concatenate(ticks)


def lifted_rows_jacobian(A, B, K, masks, outputs, row_ticks):
    """G (R, T, 2): row r = C_r Phi_cl(t_r, j + 1) B_j for j < t_r (forward sensitivities); masked columns zero."""
    T = B.shape[0]
    R = len(row_ticks)
    G = np.zeros((R, T, B.shape[2]))
    Acl = A + np.einsum('tij,tjk->tik', B, K)
    for j in range(T):
        S = B[j] * masks[j][None, :]               # d x_{j+1} / d du_j
        for t in range(j + 1, int(row_ticks.max()) + 1):
            sel = row_ticks == t
            if sel.any():
                G[sel, j] = outputs[sel] @ S
            if t < T:
                S = Acl[t] @ S
    return G


def gn_step(G, e, beta, delta, cap_rms):
    flat = G.reshape(len(e), -1)
    normal = flat.T @ flat
    n = normal.shape[0]
    lam = delta * np.trace(normal) / n + 1e-9
    step = -beta * np.linalg.solve(normal + lam * np.eye(n), flat.T @ e)
    free = np.abs(flat).sum(0) > 0
    rms = np.sqrt(np.mean(step[free] ** 2)) if free.any() else 0.0
    if rms > cap_rms:
        step *= cap_rms / rms
    return step.reshape(G.shape[1:])


def stop_tick(xs, hold=4):
    """The first tick from which the car stays below V_MOVING for `hold` periods (else the last tick)."""
    V = np.asarray(sim.speed(xs))
    slow = V < V_MOVING
    for t in range(1, len(V) - hold):
        if slow[t:t + hold].all():
            return t
    return len(V) - 1
