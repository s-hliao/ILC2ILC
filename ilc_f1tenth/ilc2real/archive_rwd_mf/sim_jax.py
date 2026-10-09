"""
The nominal GPU sim (the quadruped's MJX analog): car_model.dynamics in JAX, RK4 at 0.25 ms inside each 25 ms
control period, batched over lanes; the policy's observation (path-relative state, deviation from the plan at the
car's own arc length, preview of the plan ahead), the ILC error, and exact one-step Jacobians by autodiff.

Plans (trajopt_drift.dense_plan + plan_lqr gains) are stacked into padded per-track arrays; each lane carries its
plan id, its projection index on the fine s-grid (local search: the figure-eight crosses itself), its cumulative
progress (laps), and per-lane model parameters (nominal, or domain-randomized for the DR variants / baselines),
including actuation offsets and command delays.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import car_model as cm

O = cm.jax_ops()
DT = 0.025
SUB = 100
from consts import PREVIEW, DEV_SCALE, ERR_SCALE, FAIL_EY, FAIL_EPSI, WIN   # noqa: E402  (shared with sim_jax_np)
NOBS = 8 + 5 + 7 * len(PREVIEW)
PKEYS = ['m', 'Iz', 'mu_x', 'mu_y', 'K_y', 'k_t', 'tau_s', 'd_off', 'd_delay', 'i_delay']


def wrap(a):
    return jnp.arctan2(jnp.sin(a), jnp.cos(a))


class Plans:
    """Padded stack of plans. plan dicts as saved by trajopt_drift (+ 'K' dense gains, optional)."""

    def __init__(self, plans, names):
        self.names = names
        M = max(len(p['s']) for p in plans)
        self.M = jnp.array([len(p['s']) for p in plans])
        self.ds = jnp.array([float(p['ds']) for p in plans])
        self.L = jnp.array([float(p['length']) for p in plans])
        self.lap_time = np.array([float(p['lap_time']) for p in plans])

        def pad(a):
            out = np.zeros((M,) + a.shape[1:])
            out[:len(a)] = a
            return out
        self.xy = jnp.array(np.stack([pad(p['xy']) for p in plans]))
        self.theta = jnp.array(np.stack([pad(p['theta']) for p in plans]))
        self.kap = jnp.array(np.stack([pad(p['kappa_s']) for p in plans]))
        self.z = jnp.array(np.stack([pad(p['z']) for p in plans]))
        self.u = jnp.array(np.stack([pad(p['u']) for p in plans]))
        self.K = jnp.array(np.stack([pad(p['K']) if 'K' in p else np.zeros((len(p['s']), 2, 9)) for p in plans]))
        self.xg = [np.asarray(p['x']) for p in plans]           # global plan states (resets), host side
        self.s_host = [np.asarray(p['s']) for p in plans]


def nominal_params(n, **over):
    base = dict(cm.NOMINAL, T_se=0.0)
    d = {k: jnp.full((n,), float(v)) for k, v in base.items()}
    d['d_off'] = jnp.zeros(n)
    d['d_delay'] = jnp.zeros(n, jnp.int32)
    d['i_delay'] = jnp.zeros(n, jnp.int32)
    for k, v in over.items():
        d[k] = jnp.asarray(v)
    return d


def dr_params(key, n, scale=1.0):
    """Domain randomization family (the DR variants / baselines): mass +-25 %, friction +-20 %, lateral tire
    stiffness +-20 %, motor constant +-20 %, servo time constant 0.03-0.08 s, steering offset +-0.03 rad, steering /
    current delay 0-1 control periods."""
    k = jax.random.split(key, 8)
    u = lambda kk, lo, hi: lo + (hi - lo) * jax.random.uniform(kk, (n,))
    m = u(k[0], 1 - 0.25 * scale, 1 + 0.25 * scale)
    mu = u(k[1], 1 - 0.2 * scale, 1 + 0.2 * scale)
    p = nominal_params(n)
    p['m'] = p['m'] * m
    p['Iz'] = p['Iz'] * m
    p['mu_x'] = p['mu_x'] * mu
    p['mu_y'] = p['mu_y'] * mu
    p['K_y'] = p['K_y'] * u(k[2], 1 - 0.2 * scale, 1 + 0.2 * scale)
    p['k_t'] = p['k_t'] * u(k[3], 1 - 0.2 * scale, 1 + 0.2 * scale)
    p['tau_s'] = u(k[4], 0.04 - 0.01 * scale, 0.04 + 0.04 * scale)
    p['d_off'] = u(k[5], -0.03 * scale, 0.03 * scale)
    p['d_delay'] = (jax.random.uniform(k[6], (n,)) < 0.3 * scale).astype(jnp.int32)
    p['i_delay'] = (jax.random.uniform(k[7], (n,)) < 0.3 * scale).astype(jnp.int32)
    return p


def _rk4_period(x, a, p):
    h = DT / SUB

    def body(_, x):
        k1 = cm.dynamics(x, a, p, O)
        k2 = cm.dynamics(x + h / 2 * k1, a, p, O)
        k3 = cm.dynamics(x + h / 2 * k2, a, p, O)
        k4 = cm.dynamics(x + h * k3, a, p, O)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x.at[6:8].set(jnp.maximum(x[6:8], 0.0))
    return jax.lax.fori_loop(0, SUB, body, x)


def step(x, a_applied, p):
    """One control period for one lane; a_applied = [delta_cmd, I_cmd] (physical units, after delays / offsets)."""
    return _rk4_period(x, a_applied, p)


def project(P, pid, idx, pos):
    M = P.M[pid]
    ii = jnp.mod(idx + WIN, M)
    d2 = jnp.sum((P.xy[pid, ii] - pos) ** 2, -1)
    return ii[jnp.argmin(d2)]


def ref_at(P, pid, i, off, arr):
    """arr (plan-indexed, padded) at fine-grid index i plus off meters (linear, periodic)."""
    M = P.M[pid]
    f = off / P.ds[pid]
    fl = jnp.floor(f)
    w = f - fl
    i0 = jnp.mod(i + fl.astype(jnp.int32), M)
    i1 = jnp.mod(i0 + 1, M)
    a0, a1 = arr[pid, i0], arr[pid, i1]
    w = w.reshape(w.shape + (1,) * (a0.ndim - w.ndim))
    return (1 - w) * a0 + w * a1


def path_coords(P, pid, i, x):
    """(s_off along the tangent from grid point i, e_y, e_psi) of the car at x."""
    th = P.theta[pid, i]
    d = x[:2] - P.xy[pid, i]
    s_off = d[0] * jnp.cos(th) + d[1] * jnp.sin(th)
    e_y = -d[0] * jnp.sin(th) + d[1] * jnp.cos(th)
    e_psi = wrap(x[2] - th - s_off * P.kap[pid, i])
    return s_off, e_y, e_psi


def features(P, pid, i, x):
    """Policy observation and ILC error (unscaled 5-vector) of one lane at state x, projection index i."""
    s_off, e_y, e_psi = path_coords(P, pid, i, x)
    zr = ref_at(P, pid, i, s_off, P.z)
    Rw = cm.NOMINAL['R_w']
    dev = jnp.stack([e_y - zr[0], wrap(e_psi - zr[1]), x[3] - zr[2], x[4] - zr[3], x[5] - zr[4],
                     (x[6] - zr[5]) * Rw, (x[7] - zr[6]) * Rw, x[8] - zr[7]])
    raw = jnp.stack([x[3] / 3.0, x[4], x[5] / 3.0, x[8] / 0.4, x[7] * Rw / 4.0])
    pv = []
    for d in PREVIEW:
        z = ref_at(P, pid, i, s_off + d, P.z)
        u = ref_at(P, pid, i, s_off + d, P.u)
        k = ref_at(P, pid, i, s_off + d, P.kap)
        pv.append(jnp.stack([k, z[1], z[2] / 3.0, z[3], z[4] / 3.0, u[0] / 0.4, u[1] / 20.0]))
    obs = jnp.concatenate([dev / DEV_SCALE, raw, jnp.concatenate(pv)])
    return obs, dev[:5]


def expert_action(P, pid, i, x):
    """The plan's periodic LQR: u*(s) - K(s) dz (dFz unmeasured: no feedback on it). Physical units."""
    s_off, e_y, e_psi = path_coords(P, pid, i, x)
    zr = ref_at(P, pid, i, s_off, P.z)
    ur = ref_at(P, pid, i, s_off, P.u)
    K = ref_at(P, pid, i, s_off, P.K)
    dz = jnp.stack([e_y - zr[0], wrap(e_psi - zr[1]), x[3] - zr[2], x[4] - zr[3], x[5] - zr[4],
                    x[6] - zr[5], x[7] - zr[6], x[8] - zr[7], 0.0])
    return ur - K @ dz


# ---------------------------------------------------------------------------------------------------------------
# the network (numpy-compatible: policy weights as a list of (W, b))
def init_mlp(key, sizes):
    ps = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k = jax.random.split(key)
        scale = (1.0 / a) ** 0.5 * (0.01 if i == len(sizes) - 2 else 1.0)
        ps.append((jax.random.normal(k, (a, b)) * scale, jnp.zeros(b)))
    return ps


def mlp(ps, o):
    h = o
    for W, b in ps[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = ps[-1]
    return h @ W + b                      # normalized action (x A_SCALE: physical)


A_SCALE = jnp.array(cm.A_SCALE)


# ---------------------------------------------------------------------------------------------------------------
def rollout(P, policy, T, x0, idx0, pid, p, noise=None, expert_mix=0.0):
    """Fly T control periods on every lane. policy(obs) -> normalized action (or None: the expert flies).
    noise: (n, T, 2) normalized action offsets (exploration; a disturbance the ILC targets ignore).
    Returns dict of (n, T+1, .) states, (n, T, .) obs / mean actions / applied actions / errors, indices,
    progress (m) and failure flags. A lane that fails (|e_y| > 0.6 m, heading error > 1.2 rad, vx < 0.3) holds."""
    n = x0.shape[0]
    if noise is None:
        noise = jnp.zeros((n, T, 2))

    def lane(x0, i0, pid, p, nz):
        def body(c, nz_t):
            x, i, prog, failed, buf = c
            i_new = project(P, pid, i, x[:2])
            s_off_old = 0.0
            dprog = jnp.mod((i_new - i) + P.M[pid] // 2, P.M[pid]) - P.M[pid] // 2
            obs, err = features(P, pid, i_new, x)
            mu = policy(obs) if policy is not None else expert_action(P, pid, i_new, x) / A_SCALE
            a = mu + nz_t
            a_phys = a * A_SCALE
            buf_new = jnp.stack([a_phys, buf[0]])                 # [now, previous]
            d_cmd = jnp.where(p['d_delay'] > 0, buf[0][0], a_phys[0]) + p['d_off']
            i_cmd = jnp.where(p['i_delay'] > 0, buf[0][1], a_phys[1])
            x_new = step(x, jnp.stack([d_cmd, i_cmd]), p)
            s_off, e_y, e_psi = path_coords(P, pid, i_new, x)
            bad = (jnp.abs(err[0]) > FAIL_EY) | (jnp.abs(err[1]) > FAIL_EPSI) | (x[3] < 0.3) | ~jnp.all(jnp.isfinite(x_new))
            failed_new = failed | bad
            x_out = jnp.where(failed_new, x, x_new)
            c_new = (x_out, i_new, prog + dprog * P.ds[pid], failed_new, buf_new)
            return c_new, (x, obs, mu, a, err, i_new, failed_new, jnp.stack([d_cmd, i_cmd]))
        c0 = (x0, i0, 0.0, False, jnp.zeros((2, 2)))
        cT, out = jax.lax.scan(body, c0, nz)
        xs, obs, mu, a, err, idx, failed, applied = out
        return dict(x=jnp.concatenate([xs, cT[0][None]]), obs=obs, mu=mu, a=a, err=err, idx=idx, failed=failed,
                    applied=applied, progress=cT[2], x_end=cT[0], idx_end=cT[1], failed_end=cT[3])
    return jax.vmap(lane)(x0, idx0, pid, p, noise)


def start_states(P, key, pid, s0, pert=0.0):
    """Lanes at the plan's state at arc length s0 (host arrays), with a perturbation of relative size pert:
    lateral +-0.1 m, heading +-0.1 rad, speed +-0.3 m/s, sideslip +-0.3 m/s, yaw rate +-0.5 rad/s (x pert)."""
    xs, idx = [], []
    for k, (pi, s) in enumerate(zip(np.asarray(pid), np.asarray(s0))):
        sg = P.s_host[pi]
        i = int(np.searchsorted(sg, np.mod(s, sg[-1] + (sg[1] - sg[0])))) % len(sg)
        xs.append(P.xg[pi][i])
        idx.append(i)
    x = np.array(xs)
    if pert > 0:
        r = np.asarray(jax.random.uniform(key, (len(x), 5), minval=-1, maxval=1)) * pert
        th = x[:, 2] - np.asarray([P.xg[pi][i][2] - 0 for pi, i in zip(np.asarray(pid), idx)]) * 0
        x[:, 0] += -np.sin(x[:, 2]) * 0.1 * r[:, 0]
        x[:, 1] += np.cos(x[:, 2]) * 0.1 * r[:, 0]
        x[:, 2] += 0.1 * r[:, 1]
        x[:, 3] += 0.3 * r[:, 2]
        x[:, 4] += 0.3 * r[:, 3]
        x[:, 5] += 0.5 * r[:, 4]
        x[:, 7] = np.maximum(x[:, 7] + 0.3 * r[:, 2] / cm.NOMINAL['R_w'], 0)
    return jnp.array(x), jnp.array(idx, dtype=jnp.int32)


# ---------------------------------------------------------------------------------------------------------------
# Jacobians along flown lanes, for the closed-loop GN ILC step
def lane_jacobians(P, policy_params, policy_fn, xs, idx, pid, p, use_policy_fb=True):
    """Per period t: A_t = d x_{t+1} / d x_t, B_t = d x_{t+1} / d a_t (normalized action, at the applied action),
    the closed-loop feedback K_t = d pi / d x_t (through the observation), and C_t = d err_t / d x_t.
    xs (T+1, 10), idx (T,), applied (T, 2). Delays are not modelled in the Jacobians (nominal: none)."""
    def one(x, i, a_phys):
        A = jax.jacfwd(lambda xx: step(xx, a_phys, p))(x)
        B = jax.jacfwd(lambda aa: step(x, aa * A_SCALE, p))(a_phys / A_SCALE)
        C = jax.jacfwd(lambda xx: features(P, pid, i, xx)[1])(x)
        if use_policy_fb:
            K = jax.jacfwd(lambda xx: policy_fn(policy_params, features(P, pid, i, xx)[0]))(x)
        else:
            K = jnp.zeros((2, 10))
        return A, B, C, K
    return one
