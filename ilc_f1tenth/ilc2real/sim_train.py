#!/usr/bin/env python3
"""
sim_train.py --plans P1 P2 ... --out DIR: the sim stage of ILC2Real for the F1TENTH (plane_train.py's analog) --
one network, conditioned on the plan through its preview, that drives every plan of the bank (two tracks x target
sideslips 0..25 deg: grip to full drift) lap after lap, trained in the NOMINAL GPU sim (sim_jax) by deep ILC.

  0. warm start (--bc-iters): the network cloned onto the plans' periodic LQR (DAgger: the network's own rollouts,
     relabelled by the expert at the states they visit)
  every iteration:
  1. a batch of lanes over the plans: from random arc lengths with perturbed states (explore: also smooth action
     offsets of rms --a-off, a disturbance, not DR) or clean; --chain of the lanes continue from the last batch's end
     states (lap after lap: the lap transitions are trained on, as the infinite drift needs)
  2. the batch flown by the network; per lane the one-step Jacobians A_t, B_t, the network's own feedback K_t =
     d pi / d x (through the observation), and the error map C_t = d err_t / d x_t, all exact (autodiff)
  3. per lane the Gauss-Newton ILC step on its arc-length-indexed tracking error (lateral offset, heading,
     vx, vy, yaw rate against the plan at the car's own s; scaled by ERR_SCALE) through the CLOSED-loop lifted
     sensitivity G[t, j] = C_t Phi_cl(t, j+1) B_j: step = -beta (G'G + delta tr(G'G)/n I)^-1 G'e, rms capped
  4. targets a*_t = mu_t + step_t at the lane's observations (mu the network's mean); rows past a failure or with
     |e_y| > --max-ey are dropped
  5. the network regresses onto the targets of the last --replay iterations (Adam), optionally held near its
     previous outputs on the pool (--anchor-w: trust region)
--dr: every lane on its own randomized model (sim_jax.dr_params), Jacobians from that lane's model (the DR variant).
Snapshots: DIR/policy_itNNN.npz (+ eval metrics in DIR/log.jsonl).
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--plans', nargs='+', required=True)
ap.add_argument('--eval-plans', nargs='+', default=None)
ap.add_argument('--out', required=True)
ap.add_argument('--init', default=None, help='start from this policy npz (skip the warm start)')
ap.add_argument('--bc-iters', type=int, default=6)
ap.add_argument('--iters', type=int, default=60)
ap.add_argument('--lanes', type=int, default=192)
ap.add_argument('--T', type=int, default=160, help='control periods per lane (4 s)')
ap.add_argument('--explore-frac', type=float, default=0.5)
ap.add_argument('--pert', type=float, default=1.0, help='start-state perturbation of the exploring lanes')
ap.add_argument('--a-off', type=float, default=0.05)
ap.add_argument('--chain', type=float, default=0.25)
ap.add_argument('--beta', type=float, default=0.5)
ap.add_argument('--delta', type=float, default=0.02)
ap.add_argument('--cap', type=float, default=0.08, help='rms cap of a lane step (normalized actions)')
ap.add_argument('--max-ey', type=float, default=0.3)
ap.add_argument('--err-scale', type=float, nargs=6, default=None, metavar=('EY', 'EPSI', 'VX', 'VY', 'R', 'V'),
                help='the ILC error rows\' scales (default sim_jax.ERR_SCALE: 0.05 0.1 0.3 0.3 0.5)')
ap.add_argument('--no-feedback-jac', action='store_true', help='ablation: open-loop G (K_t = 0)')
ap.add_argument('--replay', type=int, default=4)
ap.add_argument('--steps', type=int, default=300)
ap.add_argument('--mb', type=int, default=4096)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--lr-final', type=float, default=1e-4)
ap.add_argument('--anchor-w', type=float, default=0.0)
ap.add_argument('--hidden', type=int, default=256)
ap.add_argument('--dr', type=float, default=0.0, help='domain randomization scale (0: nominal, 1: the DR family)')
ap.add_argument('--eval-every', type=int, default=10)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--gpu', default=None)
a = ap.parse_args()
if a.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax                                       # noqa: E402
import jax.numpy as jnp                          # noqa: E402
import numpy as np                               # noqa: E402
import optax                                     # noqa: E402

import bank                                      # noqa: E402
import sim_jax as sj                             # noqa: E402

os.makedirs(a.out, exist_ok=True)
json.dump(vars(a), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)
LOG = open(os.path.join(a.out, 'log.jsonl'), 'a')
plans = bank.load_bank(a.plans)
P = sj.Plans(plans, a.plans)
NP = len(plans)
eval_names = a.eval_plans or a.plans
eval_idx = [a.plans.index(n) for n in eval_names]
key = jax.random.PRNGKey(a.seed)
rng = np.random.default_rng(a.seed)
SIM_LAPS = 0.0                                    # sample count, laps of sim driving
T0 = time.time()


def log(**kw):
    kw.update(t=round(time.time() - T0, 1), sim_laps=round(SIM_LAPS, 1))
    print(json.dumps(kw), flush=True)
    LOG.write(json.dumps(kw) + '\n')
    LOG.flush()


def policy_fn(params, o):
    return sj.mlp(params, o)


@jax.jit
def fly_policy(params, x0, i0, pid, p, noise):
    return sj.rollout(P, lambda o: policy_fn(params, o), a.T, x0, i0, pid, p, noise)


@jax.jit
def fly_expert(x0, i0, pid, p, noise):
    return sj.rollout(P, None, a.T, x0, i0, pid, p, noise)


@jax.jit
def expert_labels(xs, idx, pid):
    f = lambda x, i, q: (sj.expert_action(P, q, i, x) - sj.A_OFF) / sj.A_SCALE
    return jax.vmap(lambda X, I, q: jax.vmap(lambda x, i: f(x, i, q))(X, I))(xs, idx, pid)


def ou_noise(key, n, T, rms):
    rho = 0.9
    e = jax.random.normal(key, (T, n, 2)) * rms * np.sqrt(1 - rho ** 2)

    def body(c, et):
        c = rho * c + et
        return c, c
    _, out = jax.lax.scan(body, jax.random.normal(key, (n, 2)) * rms, e)
    return jnp.transpose(out, (1, 0, 2))


@jax.jit
def gn_steps(params, xs, idx, applied, pid, p, failed, err):
    """Per lane the closed-loop GN ILC step (T, 2) and a row mask."""
    T = a.T

    def lane(X, I, Ap, q, pp, F, E):
        one = sj.lane_jacobians(P, params, policy_fn, None, None, q, pp, use_policy_fb=not a.no_feedback_jac)
        A, B, C, K = jax.vmap(one)(X[:-1], I, Ap)              # (T, ...)
        Acl = A + jnp.einsum('tij,tjk->tik', B, K)
        # columns: V_j (10x2) propagated; rows err_t (t >= 1) = C_t V
        def body(V, t):
            # V: (T, NX, 2), column j holds d x_t / d a_j for j < t (zero for j >= t)
            rows = jnp.einsum('ij,kjl->kil', C[t], V)          # (T, 5, 2): d err_t / d a_j
            Vn = jnp.einsum('ij,kjl->kil', Acl[t], V)
            Vn = Vn.at[t].set(B[t])
            return Vn, rows
        V0 = jnp.zeros((T, sj.NXS, 2))
        _, rows = jax.lax.scan(body, V0, jnp.arange(T))      # rows[t] = d err_t / d a_j, (T, T, 5, 2)
        valid = (~F) & (jnp.abs(E[:, 0]) < a.max_ey)
        valid = jnp.cumprod(valid.astype(jnp.float32)) > 0     # rows until the first invalid one
        w = 1.0 / jnp.array(sj.ERR_SCALE if a.err_scale is None else a.err_scale)
        G = rows * w[None, None, :, None] * valid[:, None, None, None]
        G = jnp.transpose(G, (0, 2, 1, 3)).reshape(T * sj.NE, T * 2)
        e = (E * w[None, :] * valid[:, None]).reshape(-1)
        H = G.T @ G
        n = H.shape[0]
        lam = a.delta * jnp.trace(H) / n + 1e-9
        step = -a.beta * jnp.linalg.solve(H + lam * jnp.eye(n), G.T @ e)
        rms = jnp.sqrt(jnp.mean(step ** 2))
        step = step * jnp.minimum(1.0, a.cap / (rms + 1e-12))
        e_pred = e + G @ step
        return step.reshape(T, 2), valid, 0.5 * jnp.sum(e ** 2), 0.5 * jnp.sum(e_pred ** 2)
    return jax.vmap(lane)(xs, idx, applied, pid, p, failed, err)


# ---------------------------------------------------------------------------------------------------------------
key, k0 = jax.random.split(key)
params = sj.init_mlp(k0, [sj.NOBS, a.hidden, a.hidden, 2])
opt = optax.adam(a.lr)


@jax.jit
def fit_step(params, opt_state, o, y, wgt, o_anc, y_anc, lr_scale):
    def loss(ps):
        l = jnp.sum(wgt[:, None] * (policy_fn(ps, o) - y) ** 2) / (jnp.sum(wgt) + 1e-9)
        if a.anchor_w > 0:
            l = l + a.anchor_w * jnp.mean((policy_fn(ps, o_anc) - y_anc) ** 2)
        return l
    l, g = jax.value_and_grad(loss)(params)
    upd, opt_state = opt.update(g, opt_state, params)
    upd = jax.tree_util.tree_map(lambda u: u * lr_scale, upd)
    return optax.apply_updates(params, upd), opt_state, l


def fit(params, O, Y, Wt, steps, lr_scale=1.0):
    opt_state = opt.init(params)
    n = len(O)
    O, Y, Wt = jnp.asarray(O), jnp.asarray(Y), jnp.asarray(Wt)
    Y_anc = policy_fn(params, O) if a.anchor_w > 0 else Y
    for s in range(steps):
        b = jnp.asarray(rng.integers(0, n, min(a.mb, n)))
        params, opt_state, l = fit_step(params, opt_state, O[b], Y[b], Wt[b], O[b], Y_anc[b], lr_scale)
    return params, float(l)


def save(params, tag):
    np.savez(os.path.join(a.out, f'policy_{tag}.npz'), **{f'W{i}': np.asarray(W) for i, (W, b) in enumerate(params)},
             **{f'b{i}': np.asarray(b) for i, (W, b) in enumerate(params)}, plans=np.array(a.plans))


def lanes_params(n, key):
    return sj.dr_params(key, n, a.dr) if a.dr > 0 else sj.nominal_params(n)


def evaluate(params, it):
    """Nominal sim: 3 chained laps from each eval plan's start, clean and from 4 perturbed starts."""
    global SIM_LAPS
    T_eval = int(3 * max(P.lap_time[eval_idx]) / sj.DT)
    pid = np.repeat(eval_idx, 5)
    pert = np.tile([0.0, 1.0, 1.0, 1.0, 1.0], len(eval_idx))
    xs, ids = [], []
    for k, (q, pe) in enumerate(zip(pid, pert)):
        x0, i0 = sj.start_states(P, jax.random.PRNGKey(1000 + k), [q], [0.0], pert=pe)
        xs.append(x0[0]); ids.append(i0[0])
    x0, i0 = jnp.stack(xs), jnp.stack(ids)
    out = jax.jit(lambda ps, x0, i0, pid, p: sj.rollout(P, lambda o: policy_fn(ps, o), T_eval, x0, i0, pid, p))(
        params, x0, i0, jnp.array(pid), sj.nominal_params(len(pid)))
    err = np.asarray(out['err'])
    fail = np.asarray(out['failed'][:, -1])
    prog = np.asarray(out['progress']) / np.asarray(P.L)[pid]
    res = {}
    for q in eval_idx:
        m = pid == q
        e = err[m]
        res[a.plans[q]] = dict(fail=float(fail[m].mean()), laps=float(prog[m].mean()),
                               rms_ey=float(np.sqrt(np.mean(e[~fail[m]][:, :, 0] ** 2))) if (~fail[m]).any() else None,
                               rms_vy=float(np.sqrt(np.mean(e[~fail[m]][:, :, 3] ** 2))) if (~fail[m]).any() else None)
    log(event='eval', it=it, res=res)
    return res


# ---------------------------------------------------------------------------------------------------------------
def sample_lanes(key, n, chain_state=None):
    pid = rng.integers(0, NP, n)
    s0 = rng.uniform(0, 1, n) * np.asarray(P.L)[pid]
    explore = rng.uniform(size=n) < a.explore_frac
    pert = np.where(explore, a.pert, 0.2)
    xs, ids = [], []
    keys = jax.random.split(key, n)
    for k in range(n):
        x0, i0 = sj.start_states(P, keys[k], [pid[k]], [s0[k]], pert=pert[k])
        xs.append(x0[0]); ids.append(i0[0])
    x0, i0 = jnp.stack(xs), jnp.stack(ids)
    if chain_state is not None:
        cx, ci, cp, cf = chain_state
        ok = np.where(~np.asarray(cf))[0]
        m = min(int(a.chain * n), len(ok))
        if m > 0:
            pick = rng.choice(ok, m, replace=False)
            x0 = x0.at[:m].set(cx[pick]); i0 = i0.at[:m].set(ci[pick])
            pid[:m] = np.asarray(cp)[pick]
            explore[:m] = False
    return x0, i0, jnp.array(pid), explore


if a.init:
    d = np.load(a.init)
    params = [(jnp.array(d[f'W{i}']), jnp.array(d[f'b{i}'])) for i in range(len([k for k in d.files if k[0] == 'W']))]
else:
    # warm start: DAgger onto the plans' LQR
    O_all, Y_all = [], []
    for it in range(a.bc_iters):
        key, k1, k2, k3 = jax.random.split(key, 4)
        x0, i0, pid, explore = sample_lanes(k1, a.lanes)
        noise = ou_noise(k2, a.lanes, a.T, a.a_off) * jnp.asarray(explore)[:, None, None]
        p = lanes_params(a.lanes, k3)
        out = fly_expert(x0, i0, pid, p, noise) if it == 0 else fly_policy(params, x0, i0, pid, p, noise)
        SIM_LAPS += float(np.sum(np.asarray(out['progress']) / np.asarray(P.L)[np.asarray(pid)]))
        lab = expert_labels(out['x'][:, :-1], out['idx'], pid)
        ok = ~np.asarray(out['failed'])
        O_all.append(np.asarray(out['obs'])[ok]); Y_all.append(np.asarray(lab)[ok])
        O, Y = np.concatenate(O_all), np.concatenate(Y_all)
        params, l = fit(params, O, Y, np.ones(len(O)), 1500 if it == 0 else 600)
        log(event='bc', it=it, loss=l, n=len(O), fail=float(np.asarray(out['failed'][:, -1]).mean()))
    save(params, 'bc')
    evaluate(params, -1)

pool = []
chain = None
for it in range(a.iters):
    key, k1, k2, k3 = jax.random.split(key, 4)
    x0, i0, pid, explore = sample_lanes(k1, a.lanes, chain)
    noise = ou_noise(k2, a.lanes, a.T, a.a_off) * jnp.asarray(explore)[:, None, None]
    p = lanes_params(a.lanes, k3)
    out = fly_policy(params, x0, i0, pid, p, noise)
    SIM_LAPS += float(np.sum(np.asarray(out['progress']) / np.asarray(P.L)[np.asarray(pid)]))
    chain = (out['x_end'], out['idx_end'], pid, out['failed_end'])
    step, valid, J, Jp = gn_steps(params, out['x'], out['idx'], (out['applied'] - sj.A_OFF) / sj.A_SCALE, pid, p,
                                  out['failed'], out['err'])
    step, J, Jp = np.asarray(step), np.asarray(J), np.asarray(Jp)
    finite = np.isfinite(step).all(axis=(1, 2)) & np.isfinite(J) & np.isfinite(np.asarray(out['obs'])).all(axis=(1, 2))
    valid = np.asarray(valid) & finite[:, None]           # non-finite lanes (a diverged randomized model) dropped
    step = np.nan_to_num(step)
    tgt = np.asarray(out['mu']) + step
    O = np.asarray(out['obs'])[valid]
    Y = tgt[valid]
    pool.append((O, Y))
    pool = pool[-a.replay:]
    Oa = np.concatenate([q[0] for q in pool])
    Ya = np.concatenate([q[1] for q in pool])
    Wa = np.concatenate([np.full(len(q[0]), 0.5 ** (len(pool) - 1 - k)) for k, q in enumerate(pool)])
    frac = it / max(1, a.iters - 1)
    lr_scale = (1 - frac) + frac * a.lr_final / a.lr
    prev = params
    params, l = fit(params, Oa, Ya, Wa, a.steps, lr_scale)
    if not np.isfinite(l) or not all(np.isfinite(np.asarray(W)).all() for W, b in params):
        params = prev                                       # a non-finite fit is discarded
    J, Jp = J[finite], Jp[finite]
    log(event='ilc', it=it, loss=l, J=float(np.median(J)), J_pred=float(np.median(Jp)),
        fail=float(np.asarray(out['failed'][:, -1]).mean()), rows=int(valid.sum()),
        step_rms=float(np.sqrt(np.mean(np.asarray(step) ** 2))))
    if (it + 1) % a.eval_every == 0 or it == a.iters - 1:
        save(params, f'it{it + 1:03d}')
        evaluate(params, it + 1)
save(params, 'final')
log(event='done', seconds=time.time() - T0)
