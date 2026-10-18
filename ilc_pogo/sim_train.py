#!/usr/bin/env python3
"""
sim_train.py --out DIR: the sim stage of DistILC on the pogo stick (the car's sim_train.py analog) -- ONE network,
conditioned on the goal (v*, h*), trained in the NOMINAL GPU sim (pogo_jax) by deep ILC.

  0. warm start (--bc-iters): the network cloned onto the Raibert controller (DAgger: its own rollouts relabelled
     by the expert at the states they visit)
  every iteration:
  1. a batch of lanes, goals uniform in the goal box: drops at the goal's apex, perturbed (explore: also smooth action
     offsets of rms --a-off; a disturbance, not DR) or clean; --chain of the lanes continue from the last batch's end
     states under a NEW goal (the goal transitions and the stationary gait are trained on)
  2. per lane the one-period Jacobians A_t, B_t, the network's own feedback K_t = d pi / d x (through its
     observation), the error map C_t (autodiff of the nominal model)
  3. per lane the closed-loop Gauss-Newton ILC step on its task errors (flight: predicted apex - h*, vx - v*),
     G[t, j] = C_t Phi_cl(t, j+1) B_j, step = -beta (G'G + delta tr(G'G)/n I)^-1 G'e, rms capped
  4. targets a*_t = mu_t + step_t at the lane's observations (rows after a fall dropped)
  5. the network regresses onto the targets of the last --replay iterations (Adam), held near its previous outputs
     (--anchor-w: trust region)
--dr S: every lane on its own randomized model (pogo_jax.dr_params), Jacobians from that lane's model (DR variant).
--no-feedback-jac: open-loop G (K = 0) ablation. Snapshots DIR/policy_itNNN.npz, metrics DIR/log.jsonl.
--accept 1: every update is kept only if it lowers the measured task cost of the same lanes (re-flown): a regression
onto model-predicted corrections can drift where the model's prediction is poor (the speed loop through the touchdown
angle); the check makes the sim stage a descent method on the measured cost, as the hardware stage's rollback does.
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--init', default=None, help='start from this policy npz (skip the warm start)')
ap.add_argument('--bc-iters', type=int, default=8)
ap.add_argument('--iters', type=int, default=40)
ap.add_argument('--lanes', type=int, default=256)
ap.add_argument('--T', type=int, default=300, help='control periods per lane (3 s, ~6 hops)')
ap.add_argument('--explore-frac', type=float, default=0.7)
ap.add_argument('--pert', type=float, default=1.0)
ap.add_argument('--a-off', type=float, default=0.15)
ap.add_argument('--chain', type=float, default=0.3)
ap.add_argument('--beta', type=float, default=0.5)
ap.add_argument('--delta', type=float, default=0.02)
ap.add_argument('--cap', type=float, default=0.08)
ap.add_argument('--no-feedback-jac', action='store_true')
ap.add_argument('--fbk-refresh', type=int, default=0, help='> 0: the network feedback inside the sensitivity comes from a '
                'snapshot refreshed every N iterations (a target network), not from the network being fitted')
ap.add_argument('--fb-jac', default='policy', choices=('policy', 'expert', 'none'), help='the feedback inside the '
                'closed-loop sensitivity: the network\'s own (default), the Raibert controller\'s (fixed: the sensitivity '
                'cannot co-evolve with the network\'s gains), or none (= --no-feedback-jac)')
ap.add_argument('--replay', type=int, default=4)
ap.add_argument('--steps', type=int, default=400)
ap.add_argument('--mb', type=int, default=4096)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--lr-final', type=float, default=1e-4)
ap.add_argument('--anchor-w', type=float, default=1.0)
ap.add_argument('--accept', type=int, default=0, help='1: the measured-improvement check (the hardware stage\'s rollback, '
                'in sim): after each fit the SAME lanes (starts, models, noise) are re-flown; if their task cost did not fall '
                'the update is discarded and the step (beta, cap) halved; an accepted step grows it back (x1.25, to the set value). '
                '2: a backtracking line search on that measured cost along the regression direction, theta + alpha (theta_fit - '
                'theta), alpha = 1, 1/2, 1/4, 1/8; the best alpha that lowers the cost is kept, else the update is discarded '
                '(the GN gain is not cut)')
ap.add_argument('--hidden', type=int, default=128)
ap.add_argument('--dr', type=float, default=0.0)
ap.add_argument('--eval-every', type=int, default=5)
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

import common as C                               # noqa: E402
import pogo_jax as pj                            # noqa: E402

os.makedirs(a.out, exist_ok=True)
json.dump(vars(a), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)
LOG = open(os.path.join(a.out, 'log.jsonl'), 'a')
key = jax.random.PRNGKey(a.seed)
rng = np.random.default_rng(a.seed)
SIM_SECONDS = 0.0
T0 = time.time()
VAL_GOALS = C.goals_from_seed(8, C.VAL_SEED)


def log(**kw):
    kw.update(t=round(time.time() - T0, 1), sim_hours=round(SIM_SECONDS / 3600, 3))
    print(json.dumps(kw), flush=True)
    LOG.write(json.dumps(kw) + '\n')
    LOG.flush()


policy_fn = pj.mlp


@jax.jit
def fly_policy(params, x0, g, p, noise):
    return pj.rollout(lambda o: policy_fn(params, o), a.T, x0, g, p, noise)


@jax.jit
def fly_expert(x0, g, p, noise):
    return pj.rollout(None, a.T, x0, g, p, noise)


@jax.jit
def expert_labels(xs, g):
    return jax.vmap(lambda X, gg: jax.vmap(lambda x: pj.expert(x, gg))(X))(xs, g)


def ou_noise(key, n, T, rms, rho=0.9):
    e = jax.random.normal(key, (T, n, 2)) * rms * np.sqrt(1 - rho ** 2)

    def body(c, et):
        c = rho * c + et
        return c, c
    _, out = jax.lax.scan(body, jax.random.normal(key, (n, 2)) * rms, e)
    return jnp.transpose(out, (1, 0, 2))


@jax.jit
def gn_steps(params, xs, applied, g, p, failed, err, beta, cap):
    def lane(X, Ap, gg, pp, F, E):
        one = pj.lane_jacobians(policy_fn, params, gg, pp, 'none' if a.no_feedback_jac else a.fb_jac)
        A, B, Cm, K = jax.vmap(one)(X[:-1], Ap)
        valid = ~F
        step, J, Jp = pj.gn_lane(A, B, Cm, K, E, valid, beta, a.delta, cap)
        return step, valid, J, Jp
    return jax.vmap(lane)(xs, applied, g, p, failed, err)


key, k0 = jax.random.split(key)
params = pj.init_mlp(k0, [C.NOBS, a.hidden, a.hidden, 2])
opt = optax.adam(a.lr)


@jax.jit
def fit_step(params, opt_state, o, y, wgt, y_anc, lr_scale, aw):
    def loss(ps):
        out = policy_fn(ps, o)
        l = jnp.sum(wgt[:, None] * (out - y) ** 2) / (jnp.sum(wgt) + 1e-9)
        return l + aw * jnp.mean((out - y_anc) ** 2)
    l, gr = jax.value_and_grad(loss)(params)
    upd, opt_state = opt.update(gr, opt_state, params)
    upd = jax.tree_util.tree_map(lambda u: u * lr_scale, upd)
    return optax.apply_updates(params, upd), opt_state, l


def fit(params, O, Y, Wt, steps, lr_scale=1.0, aw=0.0):
    opt_state = opt.init(params)
    n = len(O)
    O, Y, Wt = jnp.asarray(O), jnp.asarray(Y), jnp.asarray(Wt)
    Y_anc = jnp.concatenate([policy_fn(params, O[i:i + 262144]) for i in range(0, n, 262144)])
    for s in range(steps):
        b = jnp.asarray(rng.integers(0, n, min(a.mb, n)))
        params, opt_state, l = fit_step(params, opt_state, O[b], Y[b], Wt[b], Y_anc[b], lr_scale, aw)
    return params, float(l)


def save(params, tag):
    C.save_policy(os.path.join(a.out, f'policy_{tag}.npz'), params)


def lanes_params(n, key):
    return pj.dr_params(key, n, a.dr) if a.dr > 0 else pj.nominal_params(n)


def evaluate(params, it):
    """Nominal sim, the validation goals (seed 301): 4 perturbed drops each, 10 s without a reset."""
    global SIM_SECONDS
    G = np.repeat(VAL_GOALS, 4, 0)
    x0 = pj.start_states(jax.random.PRNGKey(4242), G, 1.0)
    out = jax.jit(lambda ps, x0, g, p: pj.rollout(lambda o: policy_fn(ps, o), 1000, x0, g, p))(
        params, x0, jnp.asarray(G), pj.nominal_params(len(G)))
    SIM_SECONDS += 10.0 * len(G)
    xs, fe = np.asarray(out['x']), np.asarray(out['failed_end'])
    ms = [C.hop_metrics(xs[i], G[i], fe[i]) for i in range(len(G))]
    ok = [m for m in ms if not m['failed']]
    res = dict(success=float(np.mean([m['success'] for m in ms])), fail=float(np.mean([m['failed'] for m in ms])),
               rms_apex=float(np.mean([m['rms_apex'] for m in ok if m['rms_apex'] is not None])) if ok else None,
               v_err=float(np.mean([abs(m['v_err']) for m in ok if m['v_err'] is not None])) if ok else None)
    log(event='eval', it=it, **res)
    return res


def sample_lanes(key, n, chain_state=None):
    G = C.GOAL_LO + (C.GOAL_HI - C.GOAL_LO) * rng.uniform(size=(n, 2))
    explore = rng.uniform(size=n) < a.explore_frac
    pert = np.where(explore, a.pert, 0.2)
    x0 = pj.start_states(key, G, pert)
    if chain_state is not None:
        cx, cf = chain_state
        ok = np.where(~np.asarray(cf))[0]
        m = min(int(a.chain * n), len(ok))
        if m > 0:
            pick = rng.choice(ok, m, replace=False)
            x0 = x0.at[:m].set(cx[pick])
            explore[:m] = rng.uniform(size=m) < 0.5
    return x0, jnp.asarray(G), explore


if a.init:
    params = [(jnp.array(W), jnp.array(b)) for W, b in C.load_policy(a.init)]
else:
    O_all, Y_all = [], []
    for it in range(a.bc_iters):
        key, k1, k2, k3 = jax.random.split(key, 4)
        x0, G, explore = sample_lanes(k1, a.lanes)
        noise = ou_noise(k2, a.lanes, a.T, a.a_off) * jnp.asarray(explore)[:, None, None]
        p = lanes_params(a.lanes, k3)
        out = fly_expert(x0, G, p, noise) if it == 0 else fly_policy(params, x0, G, p, noise)
        SIM_SECONDS += a.lanes * a.T * C.DT
        lab = expert_labels(out['x'][:, :-1], G)
        ok = ~np.asarray(out['failed'])
        O_all.append(np.asarray(out['obs'])[ok]); Y_all.append(np.asarray(lab)[ok])
        O, Y = np.concatenate(O_all), np.concatenate(Y_all)
        params, l = fit(params, O, Y, np.ones(len(O)), 2000 if it == 0 else 800)
        log(event='bc', it=it, loss=l, n=len(O), fail=float(np.asarray(out['failed_end']).mean()))
    save(params, 'bc')
    evaluate(params, -1)

def lane_cost(out):
    """Task cost per lane (scaled squared errors over the rows before a fall; a fall costs its remaining rows at 10)."""
    e = np.asarray(out['err']) / C.ERR_SCALE
    f = np.asarray(out['failed'])
    return 0.5 * np.sum(np.where(f[..., None], 10.0, e ** 2), axis=(1, 2))


pool = []
chain = None
beta_now, cap_now = a.beta, a.cap
params_K = params
for it in range(a.iters):
    key, k1, k2, k3 = jax.random.split(key, 4)
    x0, G, explore = sample_lanes(k1, a.lanes, chain)
    noise = ou_noise(k2, a.lanes, a.T, a.a_off) * jnp.asarray(explore)[:, None, None]
    p = lanes_params(a.lanes, k3)
    out = fly_policy(params, x0, G, p, noise)
    SIM_SECONDS += a.lanes * a.T * C.DT
    chain = (out['x_end'], out['failed_end'])
    CH = max(16, int(256 * (300 / a.T) ** 2))                # the GN step in lane chunks (GPU memory: G is (2T)x(2T) per lane)
    if a.fbk_refresh and it % a.fbk_refresh == 0:
        params_K = params
    pK = params_K if a.fbk_refresh else params
    parts = [gn_steps(pK, out['x'][i:i + CH], out['a'][i:i + CH], G[i:i + CH],
                      jax.tree_util.tree_map(lambda v: v[i:i + CH], p), out['failed'][i:i + CH], out['err'][i:i + CH],
                      beta_now, cap_now) for i in range(0, a.lanes, CH)]
    step, valid, J, Jp = (jnp.concatenate([q[j] for q in parts]) for j in range(4))
    step, J, Jp = np.asarray(step), np.asarray(J), np.asarray(Jp)
    finite = np.isfinite(step).all(axis=(1, 2)) & np.isfinite(J)
    valid = np.asarray(valid) & finite[:, None]
    step = np.nan_to_num(step)
    tgt = np.asarray(out['mu']) + step
    pool.append((np.asarray(out['obs'])[valid], tgt[valid]))
    pool = pool[-a.replay:]
    Oa = np.concatenate([q[0] for q in pool])
    Ya = np.concatenate([q[1] for q in pool])
    Wa = np.concatenate([np.full(len(q[0]), 0.5 ** (len(pool) - 1 - k)) for k, q in enumerate(pool)])
    frac = it / max(1, a.iters - 1)
    prev = params
    params, l = fit(params, Oa, Ya, Wa, a.steps, (1 - frac) + frac * a.lr_final / a.lr, a.anchor_w)
    if not np.isfinite(l) or not all(np.isfinite(np.asarray(W)).all() for W, b in params):
        params = prev
    acc = None
    if a.accept == 2:
        c0 = np.sum(lane_cost(out))
        best, best_c = None, c0
        for alpha in (1.0, 0.5, 0.25, 0.125):
            cand = jax.tree_util.tree_map(lambda new, old: old + alpha * (new - old), params, prev)
            c1 = np.sum(lane_cost(fly_policy(cand, x0, G, p, noise)))
            SIM_SECONDS += a.lanes * a.T * C.DT
            if c1 < best_c:
                best, best_c = (alpha, cand), c1
            if best is not None and alpha < 1.0:
                break
        acc = best[0] if best is not None else False
        if best is None:
            params = prev
            pool = pool[:-1]
        else:
            params = best[1]
    elif a.accept:
        c0 = lane_cost(out)
        out1 = fly_policy(params, x0, G, p, noise)
        SIM_SECONDS += a.lanes * a.T * C.DT
        c1 = lane_cost(out1)
        acc = bool(np.sum(c1) < np.sum(c0))
        if acc:
            beta_now, cap_now = min(a.beta, 1.25 * beta_now), min(a.cap, 1.25 * cap_now)
        else:
            params = prev
            pool = pool[:-1]
            beta_now, cap_now = 0.5 * beta_now, 0.5 * cap_now
    log(event='ilc', it=it, accepted=acc, beta=beta_now, J=float(np.median(J[finite])), J_pred=float(np.median(Jp[finite])),
        loss=l,
        fail=float(np.asarray(out['failed_end']).mean()), rows=int(valid.sum()),
        step_rms=float(np.sqrt(np.mean(step ** 2))))
    if (it + 1) % a.eval_every == 0 or it == a.iters - 1:
        save(params, f'it{it + 1:03d}')
        evaluate(params, it + 1)
save(params, 'final')
log(event='done', seconds=time.time() - T0)
