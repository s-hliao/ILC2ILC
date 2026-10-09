#!/usr/bin/env python3
"""
ppo_dr.py --plans ... --out DIR: the RL baselines in the same GPU sim (sim_jax), same observation, same network
size and the same warm start as ours (--init: the LQR-cloned network), trained over the domain-randomization family
(sim_jax.dr_params, --dr 1):
  --mode ppo  PPO+DR: the policy sees the observation only
  --mode rma  RMA: a teacher PPO policy also sees the (normalized) randomized parameters z; then an adaptation module
              phi(history of the last H observation deviations and actions) -> z_hat is regressed onto z on the
              teacher's own rollouts (Kumar et al. 2021, phase 2); deployed as pi(o, phi(history))
Reward per period: 1 - 0.1 min(|err / ERR_SCALE|^2, 10) while on track (the ILC error; a spin-out / departure ends
the lane). Exported as policy npz in the same format; RMA additionally exports the adaptation module (rma_adapt.npz)
and real_car.run handles it via policy dicts.
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--plans', nargs='+', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--init', default=None)
ap.add_argument('--mode', default='ppo', choices=('ppo', 'rma'))
ap.add_argument('--dr', type=float, default=1.0)
ap.add_argument('--iters', type=int, default=300)
ap.add_argument('--lanes', type=int, default=1024)
ap.add_argument('--T', type=int, default=160)
ap.add_argument('--epochs', type=int, default=4)
ap.add_argument('--mb', type=int, default=16384)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--log-std', type=float, default=-2.5)
ap.add_argument('--gamma', type=float, default=0.98)
ap.add_argument('--lam', type=float, default=0.95)
ap.add_argument('--clip', type=float, default=0.2)
ap.add_argument('--hist', type=int, default=10)
ap.add_argument('--adapt-iters', type=int, default=40)
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
rng = np.random.default_rng(a.seed)
key = jax.random.PRNGKey(a.seed)
T0 = time.time()
SIM_LAPS = 0.0
NZ = 8                                            # privileged parameters (RMA)
ZKEYS = ['m', 'muf', 'Cf', 'kt', 'tau_s', 'd_off', 'd_delay', 'i_delay']
ZNOM = np.array([4.6, 0.6, 250.0, 0.0258, 0.055, 0.0, 0.5, 0.5])
ZSC = np.array([0.6, 0.12, 50.0, 0.005, 0.025, 0.03, 0.5, 0.5])


def log(**kw):
    kw.update(t=round(time.time() - T0, 1), sim_laps=round(SIM_LAPS, 1))
    print(json.dumps(kw), flush=True)
    LOG.write(json.dumps(kw) + '\n')
    LOG.flush()


def zvec(p):
    return (jnp.stack([p[k].astype(jnp.float32) for k in ZKEYS], -1) - ZNOM) / ZSC


n_in = sj.NOBS + (NZ if a.mode == 'rma' else 0)
key, k0, k1 = jax.random.split(key, 3)
actor = sj.init_mlp(k0, [n_in, 256, 256, 2])
if a.init:
    d = np.load(a.init)
    W0 = np.asarray(d['W0'])
    W0 = np.concatenate([W0, np.zeros((n_in - W0.shape[0], W0.shape[1]))]) if n_in > W0.shape[0] else W0
    actor = [(jnp.array(W0), jnp.array(d['b0']))] + [(jnp.array(d[f'W{i}']), jnp.array(d[f'b{i}'])) for i in (1, 2)]
critic = sj.init_mlp(k1, [n_in, 256, 256, 1])
log_std = jnp.full(2, a.log_std)
params = dict(actor=actor, critic=critic, log_std=log_std)
opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr))
opt_state = opt.init(params)


def sample_lanes(n):
    pid = rng.integers(0, NP, n)
    s0 = rng.uniform(0, 1, n) * np.asarray(P.L)[pid]
    xs, ids = [], []
    for k in range(n):
        x0, i0 = sj.start_states(P, jax.random.PRNGKey(int(rng.integers(1 << 30))), [pid[k]], [s0[k]],
                                 pert=float(rng.uniform(0, 1)))
        xs.append(x0[0]); ids.append(i0[0])
    return jnp.stack(xs), jnp.stack(ids), jnp.array(pid)


@jax.jit
def fly(params, x0, i0, pid, p, noise):
    z = zvec(p)

    def run(x0, i0, q, pp, nz, zz):
        pol = (lambda o: sj.mlp(params['actor'], jnp.concatenate([o, zz]))) if a.mode == 'rma' else \
            (lambda o: sj.mlp(params['actor'], o))
        return sj.rollout(P, pol, a.T, x0[None], i0[None], q[None], jax.tree_util.tree_map(lambda v: v[None], pp),
                          nz[None])
    out = jax.vmap(run)(x0, i0, pid, p, noise, z)
    return jax.tree_util.tree_map(lambda v: v[:, 0], out), z


def rewards(out):
    e = out['err'] / jnp.array(sj.ERR_SCALE)
    r = 1.0 - 0.1 * jnp.minimum(jnp.sum(e ** 2, -1), 10.0)
    alive = ~out['failed']
    return r * alive, alive


@jax.jit
def gae(params, obs_in, r, alive):
    v = sj.mlp(params['critic'], obs_in)[..., 0]
    v_next = jnp.concatenate([v[:, 1:], v[:, -1:]], 1) * jnp.concatenate([alive[:, 1:], alive[:, -1:]], 1)

    def body(adv, t):
        d = r[:, t] + a.gamma * v_next[:, t] - v[:, t]
        adv = d + a.gamma * a.lam * adv * alive[:, t]
        return adv, adv
    _, advs = jax.lax.scan(body, jnp.zeros(r.shape[0]), jnp.arange(r.shape[1])[::-1])
    adv = advs[::-1].T
    return adv, adv + v


@jax.jit
def update(params, opt_state, o, act, logp_old, adv, ret, w):
    def loss(ps):
        mu = sj.mlp(ps['actor'], o)
        std = jnp.exp(ps['log_std'])
        logp = jnp.sum(-0.5 * ((act - mu) / std) ** 2 - ps['log_std'], -1)
        ratio = jnp.exp(logp - logp_old)
        an = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)
        pl = -jnp.minimum(ratio * an, jnp.clip(ratio, 1 - a.clip, 1 + a.clip) * an)
        v = sj.mlp(ps['critic'], o)[:, 0]
        vl = (v - ret) ** 2
        return jnp.sum(w * (pl + 0.5 * vl)) / jnp.sum(w)
    l, g = jax.value_and_grad(loss)(params)
    u, opt_state = opt.update(g, opt_state, params)
    return optax.apply_updates(params, u), opt_state, l


def save(tag):
    act = params['actor']
    np.savez(os.path.join(a.out, f'policy_{tag}.npz'), **{f'W{i}': np.asarray(W) for i, (W, b) in enumerate(act)},
             **{f'b{i}': np.asarray(b) for i, (W, b) in enumerate(act)}, mode=a.mode)


for it in range(a.iters):
    x0, i0, pid = sample_lanes(a.lanes)
    key, kn, kp = jax.random.split(key, 3)
    p = sj.dr_params(kp, a.lanes, a.dr) if a.dr > 0 else sj.nominal_params(a.lanes)
    noise = jax.random.normal(kn, (a.lanes, a.T, 2)) * jnp.exp(params['log_std'])
    out, z = fly(params, x0, i0, pid, p, noise)
    SIM_LAPS += float(np.sum(np.asarray(out['progress']) / np.asarray(P.L)[np.asarray(pid)]))
    r, alive = rewards(out)
    o_in = jnp.concatenate([out['obs'], jnp.broadcast_to(z[:, None], out['obs'].shape[:2] + (NZ,))], -1) \
        if a.mode == 'rma' else out['obs']
    adv, ret = gae(params, o_in, r, alive.astype(jnp.float32))
    std = jnp.exp(params['log_std'])
    logp = jnp.sum(-0.5 * ((out['a'] - out['mu']) / std) ** 2 - params['log_std'], -1)
    flat = lambda v: v.reshape((-1,) + v.shape[2:])
    O, A_, LP, AD, RT, W = map(flat, (o_in, out['a'], logp, adv, ret, alive.astype(jnp.float32)))
    n = O.shape[0]
    for ep in range(a.epochs):
        perm = jnp.array(rng.permutation(n))
        for s in range(0, n, a.mb):
            b = perm[s:s + a.mb]
            params, opt_state, l = update(params, opt_state, O[b], A_[b], LP[b], AD[b], RT[b], W[b])
    if it % 10 == 0 or it == a.iters - 1:
        log(event='ppo', it=it, reward=float(jnp.sum(r) / a.lanes), fail=float(np.asarray(out['failed'][:, -1]).mean()),
            ey=float(jnp.sqrt(jnp.sum(alive * out['err'][..., 0] ** 2) / jnp.sum(alive))), loss=float(l),
            std=np.asarray(jnp.exp(params['log_std'])).tolist())
    if (it + 1) % 100 == 0:
        save(f'it{it + 1:03d}')
save('final')

if a.mode == 'rma':
    # phase 2: the adaptation module, history of (obs deviations, actions) -> z, on the teacher's rollouts
    H = a.hist
    nh = H * (8 + 2)
    key, k2 = jax.random.split(key)
    adapt = sj.init_mlp(k2, [nh, 256, 128, NZ])
    opt2 = optax.adam(3e-4)
    st2 = opt2.init(adapt)

    @jax.jit
    def adapt_step(ad, st, x, y):
        l, g = jax.value_and_grad(lambda q: jnp.mean((sj.mlp(q, x) - y) ** 2))(ad)
        u, st = opt2.update(g, st, ad)
        return optax.apply_updates(ad, u), st, l

    def histories(out):
        f = jnp.concatenate([out['obs'][..., :8], out['a']], -1)            # (n, T, 10)
        f = jnp.pad(f, ((0, 0), (H - 1, 0), (0, 0)))
        idx = jnp.arange(a.T)[:, None] + jnp.arange(H)[None]
        return f[:, idx].reshape(f.shape[0], a.T, nh)

    for it in range(a.adapt_iters):
        x0, i0, pid = sample_lanes(a.lanes // 2)
        key, kn, kp = jax.random.split(key, 3)
        p = sj.dr_params(kp, a.lanes // 2, a.dr)
        out, z = fly(params, x0, i0, pid, p, jnp.zeros((a.lanes // 2, a.T, 2)))
        SIM_LAPS += float(np.sum(np.asarray(out['progress']) / np.asarray(P.L)[np.asarray(pid)]))
        X = histories(out).reshape(-1, nh)
        Y = jnp.broadcast_to(z[:, None], (z.shape[0], a.T, NZ)).reshape(-1, NZ)
        ok = (~out['failed']).reshape(-1)
        X, Y = X[ok], Y[ok]
        for s in range(200):
            b = jnp.array(rng.integers(0, X.shape[0], 8192))
            adapt, st2, l = adapt_step(adapt, st2, X[b], Y[b])
        if it % 5 == 0:
            log(event='adapt', it=it, loss=float(l), var=float(jnp.mean(jnp.var(Y, 0))))
    np.savez(os.path.join(a.out, 'rma_adapt.npz'), **{f'W{i}': np.asarray(W) for i, (W, b) in enumerate(adapt)},
             **{f'b{i}': np.asarray(b) for i, (W, b) in enumerate(adapt)}, hist=H)
log(event='done')
