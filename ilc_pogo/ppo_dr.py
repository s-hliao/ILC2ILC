#!/usr/bin/env python3
"""
ppo_dr.py --out DIR: the RL baselines on the pogo stick (the car's ppo_dr.py analog), in the same GPU sim (pogo_jax),
same observation, same network size and the same warm start as ours (--init: the Raibert-cloned network), trained over
the domain-randomization family (pogo_jax.dr_params, --dr 1):
  --mode ppo  PPO+DR: the policy sees the observation only
  --mode rma  RMA (Kumar et al. 2021): a teacher PPO policy also sees the normalized randomized parameters z
              (m, k, b, gth, phi_off, l0); then (phase 2) an adaptation module phi(history of the last H (obs_t,
              a_{t-1})) -> z_hat is regressed onto z on the teacher's own rollouts; deployed as pi(o, phi(history))
Lanes: goals uniform in the goal box, drops at the goal's apex with a perturbation of size U(0, 1) (sim_train's).
Reward per period: 1 - 0.1 min(|e / ERR_SCALE|^2, 10) while alive, e = [predicted apex - h*, vx - v*] in EVERY phase
(reward_errors: the apex from the vertical energy incl. the spring's in stance; the ILC's flight-only rows let PPO
stop hopping for a perfect score); a fall ends the lane. Exported in our policy npz format (kind 'mlp' / 'rma'; RMA also rma_adapt.npz),
loaded on the real sticks by policies.load_any; agent_<tag>.npz holds the critic and exploration std (on-robot
fine-tuning, data matching).
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--init', default=None)
ap.add_argument('--mode', default='ppo', choices=('ppo', 'rma'))
ap.add_argument('--dr', type=float, default=1.0)
ap.add_argument('--iters', type=int, default=500)
ap.add_argument('--lanes', type=int, default=1024)
ap.add_argument('--T', type=int, default=300)
ap.add_argument('--epochs', type=int, default=4)
ap.add_argument('--mb', type=int, default=16384)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--log-std', type=float, default=-2.5)
ap.add_argument('--gamma', type=float, default=0.98)
ap.add_argument('--lam', type=float, default=0.95)
ap.add_argument('--clip', type=float, default=0.2)
ap.add_argument('--hidden', type=int, default=128)
ap.add_argument('--hist', type=int, default=50, help='RMA history length (control periods; 50 = ~one hop)')
ap.add_argument('--adapt-iters', type=int, default=60)
ap.add_argument('--adapt-only', action='store_true', help='RMA: skip PPO, load the teacher from DIR/policy_final.npz and refit phase 2')
ap.add_argument('--teacher', default=None, help='RMA --adapt-only: the teacher policy npz (default DIR/policy_final.npz); '
                'copied into DIR')
ap.add_argument('--p2-onpolicy', type=int, default=0, help='RMA phase 2 on the STUDENT\'s rollouts pi(o, phi(h)) '
                '(DAgger: iteration 0 flies the teacher, then the student; data aggregated over --p2-keep iterations)')
ap.add_argument('--p2-am', type=float, default=0.0, help='RMA phase 2 action-matching weight: + w |pi(o, z_hat) - '
                'pi(o, z)|^2 through the frozen actor (0: z-MSE only)')
ap.add_argument('--p2-dr', type=float, default=None, help='RMA phase 2 DR scale (default --dr)')
ap.add_argument('--p2-steps', type=int, default=200, help='RMA phase 2 gradient steps per iteration')
ap.add_argument('--p2-keep', type=int, default=8, help='RMA phase 2 on-policy: iterations kept in the buffer')
ap.add_argument('--p2-sub', type=int, default=6, help='RMA phase 2 on-policy: keep every p2-sub-th period')
ap.add_argument('--z-noise', type=float, default=0.0, help='RMA teacher: gaussian noise on z during PPO (robustness '
                'to the estimate)')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--save-at', type=int, nargs='*', default=[], help='also snapshot after these iterations (a sim-data curve)')
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
rng = np.random.default_rng(a.seed)
key = jax.random.PRNGKey(a.seed)
T0 = time.time()
SIM_SECONDS = 0.0
ZKEYS = ['m', 'k', 'b', 'gth', 'phi_off', 'l0']
ZNOM = np.array([10.0, 4000.0, 15.0, 1.0, 0.0, 0.5])
ZSC = np.array([2.5, 800.0, 15.0, 0.2, 0.03, 0.02])
NZ = len(ZKEYS)


def log(**kw):
    kw.update(t=round(time.time() - T0, 1), sim_hours=round(SIM_SECONDS / 3600, 3))
    print(json.dumps(kw), flush=True)
    LOG.write(json.dumps(kw) + '\n')
    LOG.flush()


def zvec(p):
    return (jnp.stack([p[k].astype(jnp.float32) for k in ZKEYS], -1) - ZNOM) / ZSC


n_in = C.NOBS + (NZ if a.mode == 'rma' else 0)
key, k0, k1 = jax.random.split(key, 3)
actor = pj.init_mlp(k0, [n_in, a.hidden, a.hidden, 2])
if a.init:
    init = C.load_policy(a.init)
    W0 = np.asarray(init[0][0])
    if n_in > W0.shape[0]:
        W0 = np.concatenate([W0, np.zeros((n_in - W0.shape[0], W0.shape[1]))])
    actor = [(jnp.array(W0), jnp.array(init[0][1]))] + [(jnp.array(W), jnp.array(b)) for W, b in init[1:]]
critic = pj.init_mlp(k1, [n_in, a.hidden, a.hidden, 1])
params = dict(actor=actor, critic=critic, log_std=jnp.full(2, a.log_std))
opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr))
opt_state = opt.init(params)


def sample_lanes(n, key):
    G = C.GOAL_LO + (C.GOAL_HI - C.GOAL_LO) * rng.uniform(size=(n, 2))
    x0 = pj.start_states(key, G, rng.uniform(0, 1, n))
    return x0, jnp.asarray(G)


@jax.jit
def fly(params, x0, G, p, noise, zn=None):
    z = zvec(p)
    z = z if zn is None else z + zn

    def run(x0, gg, pp, nz, zz):
        pol = (lambda o: pj.mlp(params['actor'], jnp.concatenate([o, zz]))) if a.mode == 'rma' else \
            (lambda o: pj.mlp(params['actor'], o))
        return pj.rollout(pol, a.T, x0[None], gg[None], jax.tree_util.tree_map(lambda v: v[None], pp), nz[None])
    out = jax.vmap(run)(x0, G, p, noise, z)
    return jax.tree_util.tree_map(lambda v: v[:, 0], out), z


def reward_errors(x, G, p):
    """The reward's error rows, defined in every phase (the ILC rows are flight-only, which PPO games by not hopping):
    the predicted apex from the vertical mechanical energy incl. the spring's, z + vz^2 / 2g + k (l0 - L)^2 / 2mg in
    stance (the lane's own parameters: privileged, sim-only), and vx - v*."""
    st = x[..., 5]
    L = C.leg_length(x, jnp)
    c = jnp.maximum(p['l0'][:, None] - L, 0.0) * st
    apex = x[..., 1] + 0.5 * x[..., 3] ** 2 / C.GRAV + 0.5 * p['k'][:, None] * c ** 2 / (p['m'][:, None] * C.GRAV)
    return jnp.stack([apex - G[:, None, 1], x[..., 2] - G[:, None, 0]], -1)


def rewards(out, G, p):
    e = reward_errors(out['x'][:, :-1], G, p) / jnp.array(C.ERR_SCALE)
    alive = ~out['failed']
    return (1.0 - 0.1 * jnp.minimum(jnp.sum(e ** 2, -1), 10.0)) * alive, alive


@jax.jit
def gae(params, obs_in, r, alive):
    v = pj.mlp(params['critic'], obs_in)[..., 0]
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
        mu = pj.mlp(ps['actor'], o)
        logp = jnp.sum(-0.5 * ((act - mu) / jnp.exp(ps['log_std'])) ** 2 - ps['log_std'], -1)
        ratio = jnp.exp(logp - logp_old)
        an = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)
        pl = -jnp.minimum(ratio * an, jnp.clip(ratio, 1 - a.clip, 1 + a.clip) * an)
        vl = (pj.mlp(ps['critic'], o)[:, 0] - ret) ** 2
        return jnp.sum(w * (pl + 0.5 * vl)) / jnp.sum(w)
    l, g = jax.value_and_grad(loss)(params)
    u, opt_state = opt.update(g, opt_state, params)
    return optax.apply_updates(params, u), opt_state, l


def save(tag):
    extra = dict(kind=a.mode if a.mode == 'rma' else 'mlp')
    if a.mode == 'rma':
        extra.update(zkeys=np.array(ZKEYS), znom=ZNOM, zsc=ZSC, nz=NZ)
    C.save_policy(os.path.join(a.out, f'policy_{tag}.npz'), params['actor'], **extra)
    cr = params['critic']
    np.savez(os.path.join(a.out, f'agent_{tag}.npz'), log_std=np.asarray(params['log_std']), mode=a.mode, nz=NZ,
             **{f'cW{i}': np.asarray(W) for i, (W, b) in enumerate(cr)}, **{f'cb{i}': np.asarray(b) for i, (W, b) in enumerate(cr)})


if a.adapt_only:
    tp = a.teacher or os.path.join(a.out, 'policy_final.npz')
    params['actor'] = [(jnp.array(W), jnp.array(b)) for W, b in C.load_policy(tp)]
    if a.teacher:
        save('final')
for it in range(0 if a.adapt_only else a.iters):
    key, kl, kn, kp = jax.random.split(key, 4)
    x0, G = sample_lanes(a.lanes, kl)
    p = pj.dr_params(kp, a.lanes, a.dr) if a.dr > 0 else pj.nominal_params(a.lanes)
    noise = jax.random.normal(kn, (a.lanes, a.T, 2)) * jnp.exp(params['log_std'])
    zn = a.z_noise * jax.random.normal(jax.random.fold_in(kn, 1), (a.lanes, NZ)) if a.mode == 'rma' and a.z_noise > 0 else None
    out, z = fly(params, x0, G, p, noise, zn)
    SIM_SECONDS += a.lanes * a.T * C.DT
    r, alive = rewards(out, G, p)
    o_in = jnp.concatenate([out['obs'], jnp.broadcast_to(z[:, None], out['obs'].shape[:2] + (NZ,))], -1) \
        if a.mode == 'rma' else out['obs']
    adv, ret = gae(params, o_in, r, alive.astype(jnp.float32))
    logp = jnp.sum(-0.5 * ((out['a'] - out['mu']) / jnp.exp(params['log_std'])) ** 2 - params['log_std'], -1)
    flat = lambda v: v.reshape((-1,) + v.shape[2:])
    O, A_, LP, AD, RT, W = map(flat, (o_in, out['a'], logp, adv, ret, alive.astype(jnp.float32)))
    n = O.shape[0]
    for ep in range(a.epochs):
        perm = jnp.array(rng.permutation(n))
        for s in range(0, n, a.mb):
            b = perm[s:s + a.mb]
            params, opt_state, l = update(params, opt_state, O[b], A_[b], LP[b], AD[b], RT[b], W[b])
    if it % 10 == 0 or it == a.iters - 1:
        e = out['err']
        fl = 1.0 - out['x'][:, :-1, 5]
        nfl = jnp.maximum(jnp.sum(alive * fl), 1.0)
        log(event='ppo', it=it, reward=float(jnp.sum(r) / a.lanes), fail=float(np.asarray(out['failed_end']).mean()),
            apex=float(jnp.sqrt(jnp.sum(alive * e[..., 0] ** 2) / nfl)), v=float(jnp.sqrt(jnp.sum(alive * e[..., 1] ** 2) / nfl)),
            flight=float(jnp.sum(alive * fl) / jnp.maximum(jnp.sum(alive), 1.0)),
            loss=float(l), std=np.asarray(jnp.exp(params['log_std'])).tolist())
    if (it + 1) % 100 == 0 or (it + 1) in a.save_at:
        save(f'it{it + 1:03d}')
if not a.adapt_only:
    save('final')

if a.mode == 'rma':
    # phase 2: the adaptation module, history of (obs_t, a_{t-1}) over the last H periods -> z, teacher rollouts
    H = a.hist
    nf = C.NOBS + 2
    key, k2 = jax.random.split(key)
    adapt = pj.init_mlp(k2, [H * nf, 256, 128, NZ])
    opt2 = optax.adam(3e-4)
    st2 = opt2.init(adapt)

    @jax.jit
    def adapt_step(ad, st, x, y):
        l, g = jax.value_and_grad(lambda q: jnp.mean((pj.mlp(q, x) - y) ** 2))(ad)
        u, st = opt2.update(g, st, ad)
        return optax.apply_updates(ad, u), st, l

    @jax.jit
    def histories(obs, act):
        a_prev = jnp.concatenate([jnp.zeros_like(act[:, :1]), act[:, :-1]], 1)
        f = jnp.concatenate([obs, a_prev], -1)                         # (n, T, nf)
        f = jnp.pad(f, ((0, 0), (H - 1, 0), (0, 0)))
        idx = jnp.arange(a.T)[:, None] + jnp.arange(H)[None]
        return f[:, idx].reshape(f.shape[0], a.T, H * nf)

    nl = a.lanes // 2
    p2dr = a.dr if a.p2_dr is None else a.p2_dr
    actor = params['actor']

    @jax.jit
    def adapt_loss_step(ad, st, x, y, o):
        def loss(q):
            zh = pj.mlp(q, x)
            l = jnp.mean((zh - y) ** 2)
            if a.p2_am > 0:
                at = pj.mlp(actor, jnp.concatenate([o, y], -1))
                ah = pj.mlp(actor, jnp.concatenate([o, zh], -1))
                l = l + a.p2_am * jnp.mean(jnp.sum((ah - at) ** 2, -1))
            return l
        l, g = jax.value_and_grad(loss)(ad)
        u, st = opt2.update(g, st, ad)
        return optax.apply_updates(ad, u), st, l

    @jax.jit
    def student_fly(ad, x0, G, p, teach):
        """The deployed RMA loop in sim: history buffer of (obs_t, a_{t-1}) as rma_np.RMAPolicy, z_hat = phi(buffer),
        a = pi(o, z_hat); `teach` lanes use the true z instead (iteration 0)."""
        z = zvec(p)

        def lane(x0, gg, pp, zz, tl):
            def body(c, t):
                x, f, buf, ap_ = c
                o = pj.obs(x, gg)
                buf = jnp.concatenate([buf[1:], jnp.concatenate([o, ap_])[None]], 0)
                zh = pj.mlp(ad, buf.reshape(-1))
                zu = jnp.where(tl, zz, zh)
                act = pj.mlp(actor, jnp.concatenate([o, zu]))
                xn = pj.step(x, act, pp)
                fn = f | C.failed(xn, jnp)
                xn = jnp.where(fn, x, xn)
                return (xn, fn, buf, act), dict(h=buf.reshape(-1), o=o, f=f, zh=zh, err=pj.err(x, gg))
            c0 = (x0, jnp.array(False), jnp.zeros((H, nf)), jnp.zeros(2))
            (xe, fe, _, _), out = jax.lax.scan(body, c0, jnp.arange(a.T))
            out['failed_end'] = fe
            return out
        out = jax.vmap(lane)(x0, G, p, z, teach)
        return out, z

    buf = []
    for it in range(a.adapt_iters):
        key, kl, kp = jax.random.split(key, 3)
        x0, G = sample_lanes(nl, kl)
        p = pj.dr_params(kp, nl, p2dr)
        if a.p2_onpolicy:
            teach = jnp.full((nl,), it == 0)
            out, z = student_fly(adapt, x0, G, p, teach)
            sel = jnp.arange(0, a.T, a.p2_sub)                       # the buffer stays on the device
            X = out['h'][:, sel].reshape(-1, H * nf)
            O = out['o'][:, sel].reshape(-1, C.NOBS)
            Y = jnp.broadcast_to(z[:, None], (nl, len(sel), NZ)).reshape(-1, NZ)
            ok = jnp.asarray(np.where(~np.asarray(out['f'][:, sel]).reshape(-1))[0])
            buf.append((X[ok], Y[ok], O[ok]))
            buf = buf[-a.p2_keep:]
            Xa, Ya, Oa = [jnp.concatenate([b_[i] for b_ in buf]) for i in range(3)]
            fl = np.asarray(out['failed_end'])
            zmse = float(jnp.mean((out['zh'] - z[:, None]) ** 2))
            e = np.asarray(out['err']); act = (np.abs(e).sum(-1) > 0) & ~np.asarray(out['f'])
            apex = float(np.sqrt(np.mean(e[..., 0][act] ** 2))) if act.any() else None
            dv = float(np.sqrt(np.mean(e[..., 1][act] ** 2))) if act.any() else None
        else:
            out, z = fly(params, x0, G, p, jnp.zeros((nl, a.T, 2)))
            Xa = histories(out['obs'], out['a']).reshape(-1, H * nf)
            Ya = jnp.broadcast_to(z[:, None], (nl, a.T, NZ)).reshape(-1, NZ)
            Oa = out['obs'].reshape(-1, C.NOBS)
            ok = np.asarray(~out['failed']).reshape(-1)
            Xa, Ya, Oa = Xa[ok], Ya[ok], Oa[ok]
            fl, zmse, apex, dv = np.asarray(out['failed_end']), None, None, None
        SIM_SECONDS += nl * a.T * C.DT
        for s_ in range(a.p2_steps):
            b = jnp.array(rng.integers(0, Xa.shape[0], 8192))
            adapt, st2, l = adapt_loss_step(adapt, st2, Xa[b], Ya[b], Oa[b])
        if it % 5 == 0 or it == a.adapt_iters - 1:
            log(event='adapt', it=it, loss=float(l), var=float(jnp.mean(jnp.var(Ya, 0))), fail=float(fl.mean()),
                zmse_student=zmse, apex_student=apex, dv_student=dv, n=int(Xa.shape[0]))
    np.savez(os.path.join(a.out, 'rma_adapt.npz'), **{f'W{i}': np.asarray(W) for i, (W, b) in enumerate(adapt)},
             **{f'b{i}': np.asarray(b) for i, (W, b) in enumerate(adapt)}, hist=H)
log(event='done', seconds=round(time.time() - T0, 1))
