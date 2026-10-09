#!/usr/bin/env python3
"""
ppo_real_car.py --init NPZ --arm NAME: the RL baselines trained ON THE REAL CARS with privileged information (the
data-matching study, user 2026-10-09: how much real data does a baseline need to match our best model after the
hardware stage?). Starts from the baseline's sim-trained network and runs PPO on each multi-body car itself:

  episodes  one lap (1.05) on one of --goals (the hardware stage's 6), from a random arc length with a small start
            perturbation, the policy's mean plus Gaussian exploration (std: the sim run's final, --std), under the
            room's safety envelope (F1T_SAFETY_EY: a crash ends the episode)
  reward    per period 1 - 0.1 min(|err_true / TASK_SCALE|^2, 10) while on the track: the same task as our hardware
            stage (path e_y and pace, grip or drift), from the car's TRUE state (privileged: no sensing noise)
  critic    asymmetric: the observation plus the car's true planar state and its true dynamics vector z (privileged)
  PPO       clipped surrogate, GAE; per car its own actor / critic (as the hardware stage adapts per car)

--init a PPO+DR policy: the actor reads the observation (PPO+DR fine-tuned on the real car).
--init an RMA teacher (ppo_dr.py --mode rma: obs + z): each car's TRUE z is folded into the first layer -- the
teacher with the oracle latent, the best RMA's adaptation module could ever estimate -- then fine-tuned the same way.
Checkpoints at --checkpoints episodes per car -> runs/hw/<arm>_ep<K>/ (args.json, <car>_policy.npz, summary.json:
real laps, crashes), evaluated by reeval.py like every other arm; <arm>_ep0 is the starting network.
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--init', required=True)
ap.add_argument('--arm', required=True)
ap.add_argument('--cars', nargs='+', default=['real_nom', 'real_mass', 'real_mu', 'real_act', 'real_lag'])
ap.add_argument('--goals', nargs='+', default=[f'mocap_{t}_b{b}' for t in ('square2fast', 'figfast') for b in (10, 18, 25)])
ap.add_argument('--per-iter', type=int, default=24, help='episodes per car per PPO iteration')
ap.add_argument('--checkpoints', type=int, nargs='+', default=[24, 48, 96, 192, 384, 768, 1536, 2016])
ap.add_argument('--std', type=float, nargs=2, default=[0.049, 0.046], help='exploration std (normalized actions): '
                'the sim PPO run\'s final (logs/ppo_dr_s0.log)')
ap.add_argument('--epochs', type=int, default=8)
ap.add_argument('--mb', type=int, default=1024)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--lr-critic', type=float, default=1e-3)
ap.add_argument('--critic-warmup', type=int, default=300, help='critic-only steps on the first batch')
ap.add_argument('--gamma', type=float, default=0.98)
ap.add_argument('--lam', type=float, default=0.95)
ap.add_argument('--clip', type=float, default=0.2)
ap.add_argument('--seed', type=int, default=4242)
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
import car_model as cm                           # noqa: E402
import mb_car as mc                              # noqa: E402
import real_car as rc                            # noqa: E402
from consts import TASK_SCALE                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# ppo_dr.py's privileged vector z (RMA): its keys, nominal and scale -- the real cars' true values mapped the same way
ZNOM = np.array([4.6, 0.6, 250.0, 0.0258, 0.055, 0.0, 0.5, 0.5])
ZSC = np.array([0.6, 0.12, 50.0, 0.005, 0.025, 0.03, 0.5, 0.5])
XSC = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 2.0, 50.0, 10.0, 0.2])   # true planar state scale for the critic


def true_z(car):
    c = mc.REAL_CARS[car]
    mb, N = c['mb'], cm.NOMINAL
    v = np.array([N['m'] * mb.get('mass', 1.0), N['muf'] * mb.get('mu', 1.0), N['Cf'] * mb.get('ky', 1.0),
                  N['kt'] * c['kt'], c['tau'], c['d_off'], c['d_delay'], c['i_delay']])
    return ((v - ZNOM) / ZSC).astype(np.float32)


def load_actor(path, car):
    d = np.load(path)
    W = [np.asarray(d[f'W{i}'], np.float32) for i in range(3)]
    b = [np.asarray(d[f'b{i}'], np.float32) for i in range(3)]
    nobs = W[0].shape[0]
    if 'mode' in d.files and str(d['mode']) == 'rma':  # the teacher: fold the car's true z into the first layer
        nz = len(ZNOM)
        b[0] = b[0] + true_z(car) @ W[0][nobs - nz:]
        W[0] = W[0][:nobs - nz]
    return [(jnp.array(w), jnp.array(bb)) for w, bb in zip(W, b)]


def mlp(ps, x):
    for W, b in ps[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = ps[-1]
    return x @ W + b


def init_mlp(key, sizes):
    ps = []
    for k, (i, o) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
        ps.append((jax.random.normal(k, (i, o)) * np.sqrt(1.0 / i), jnp.zeros(o)))
    return ps


def save_policy(f, ps):
    np.savez(f, **{f'W{i}': np.asarray(W) for i, (W, b) in enumerate(ps)}, **{f'b{i}': np.asarray(b) for i, (W, b) in enumerate(ps)})


def ckpt_dir(k):
    d = os.path.join(HERE, 'runs/hw', f'{a.arm}_ep{k}')
    os.makedirs(d, exist_ok=True)
    return d


def write_ckpt(k, actors, laps, crashes, eps, t):
    d = ckpt_dir(k)
    for c in a.cars:
        save_policy(os.path.join(d, f'{c}_policy.npz'), actors[c])
    args = dict(vars(a), policy=a.init, iters=k, trial='ppo_real_car', episodes_per_car=k)
    json.dump(args, open(os.path.join(d, 'args.json'), 'w'), indent=1)
    json.dump(dict(args=args, real_laps=dict(laps), crashes={c: crashes[c] for c in a.cars}, episodes=dict(eps),
                   seconds=t, safety_ey=float(os.environ.get('F1T_SAFETY_EY') or 0.6)),
              open(os.path.join(d, 'summary.json'), 'w'), indent=1)


def main():
    rng = np.random.default_rng(a.seed)
    plans = {p['name']: p for p in bank.load_bank(a.goals)}
    key = jax.random.PRNGKey(a.seed)
    actors = {c: load_actor(a.init, c) for c in a.cars}
    nobs = actors[a.cars[0]][0][0].shape[0]
    nin_c = nobs + 9 + len(ZNOM)
    critics = {}
    for c in a.cars:
        key, k = jax.random.split(key)
        critics[c] = init_mlp(k, [nin_c, 256, 256, 1])
    log_std = {c: jnp.log(jnp.array(a.std, jnp.float32)) for c in a.cars}
    opt_a, opt_c = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr)), optax.adam(a.lr_critic)
    st_a = {c: opt_a.init((actors[c], log_std[c])) for c in a.cars}
    st_c = {c: opt_c.init(critics[c]) for c in a.cars}

    @jax.jit
    def critic_step(cp, st, x, ret):
        l, g = jax.value_and_grad(lambda q: jnp.mean((mlp(q, x)[:, 0] - ret) ** 2))(cp)
        u, st = opt_c.update(g, st, cp)
        return optax.apply_updates(cp, u), st, l

    @jax.jit
    def actor_step(ap_, st, o, act, lp_old, adv):
        def loss(q):
            ps, ls = q
            mu = mlp(ps, o)
            lp = jnp.sum(-0.5 * ((act - mu) / jnp.exp(ls)) ** 2 - ls, -1)
            ratio = jnp.exp(lp - lp_old)
            return -jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - a.clip, 1 + a.clip) * adv))
        l, g = jax.value_and_grad(loss)(ap_)
        u, st = opt_a.update(g, st, ap_)
        return optax.apply_updates(ap_, u), st, l

    laps = {c: 0.0 for c in a.cars}
    eps = {c: 0 for c in a.cars}
    crashes = {c: 0 for c in a.cars}
    zs = {c: true_z(c) for c in a.cars}
    t0 = time.time()
    write_ckpt(0, actors, laps, crashes, eps, 0.0)
    done_ck = {0}
    it = 0
    while min(eps.values()) < max(a.checkpoints):
        jobs = []
        for c in a.cars:
            pol = [(np.asarray(W), np.asarray(b)) for W, b in actors[c]]
            std = np.exp(np.asarray(log_std[c]))
            for k in range(a.per_iter):
                pl = plans[a.goals[(it * a.per_iter + k) % len(a.goals)]]
                T = int(np.ceil(1.05 * pl['lap_time'] / rc.DT))
                jobs.append(dict(car_name=c, plan=pl, policy=pol, laps=1.05, s0=rng.uniform() * float(pl['length']),
                                 pert=rng.uniform(-0.1, 0.1, 5), seed=int(rng.integers(1 << 30)),
                                 action_noise=rng.normal(size=(T, 2)) * std))
        res = rc.run_many(jobs)
        for c in a.cars:
            rr = [r for r, j in zip(res, jobs) if j['car_name'] == c]
            O, X, A, LP, R, D = [], [], [], [], [], []
            for r in rr:
                n = len(r['obs'])
                if n < 2:
                    continue
                e = np.asarray(r['err_true'])[:, :len(TASK_SCALE)] / TASK_SCALE
                rew = 1.0 - 0.1 * np.minimum(np.sum(e ** 2, -1), 10.0)
                O.append(r['obs']); A.append(r['a'])
                X.append(np.c_[np.asarray(r['x_true'])[:, :9] / XSC, np.broadcast_to(zs[c], (n, len(ZNOM)))])
                R.append(rew); D.append(bool(r['failed']))
                laps[c] += max(float(r['laps']), 0.0)
                crashes[c] += int(bool(r['failed']))
            eps[c] += len(rr)
            # GAE per episode (the critic on obs + privileged state / z); a crash is terminal, a completed lap
            # bootstraps from the last state's value
            advs, rets, obs_all, act_all, cin_all = [], [], [], [], []
            for o, x, act, rew, dead in zip(O, X, A, R, D):
                cin = np.c_[o, x].astype(np.float32)
                v = np.asarray(mlp(critics[c], jnp.array(cin))[:, 0])
                v_next = np.r_[v[1:], 0.0 if dead else v[-1]]
                delta = rew + a.gamma * v_next - v
                adv = np.zeros_like(delta)
                g = 0.0
                for t in range(len(delta) - 1, -1, -1):
                    g = delta[t] + a.gamma * a.lam * g
                    adv[t] = g
                advs.append(adv); rets.append(adv + v); obs_all.append(o); act_all.append(act); cin_all.append(cin)
            Oc, Ac, Cc = (jnp.array(np.concatenate(z), jnp.float32) for z in (obs_all, act_all, cin_all))
            ADV, RET = np.concatenate(advs), np.concatenate(rets)
            ADV = jnp.array((ADV - ADV.mean()) / (ADV.std() + 1e-8), jnp.float32)
            RET = jnp.array(RET, jnp.float32)
            if it == 0:                               # critic warm-up on the first batch (policy untouched)
                for s in range(a.critic_warmup):
                    b = jnp.array(rng.integers(0, len(RET), min(a.mb, len(RET))))
                    critics[c], st_c[c], lc = critic_step(critics[c], st_c[c], Cc[b], RET[b])
                continue
            mu_old = mlp(actors[c], Oc)
            LPo = jnp.sum(-0.5 * ((Ac - mu_old) / jnp.exp(log_std[c])) ** 2 - log_std[c], -1)
            n = len(RET)
            for ep in range(a.epochs):
                perm = rng.permutation(n)
                for s in range(0, n, a.mb):
                    b = jnp.array(perm[s:s + a.mb])
                    (actors[c], log_std[c]), st_a[c], la = actor_step((actors[c], log_std[c]), st_a[c], Oc[b], Ac[b], LPo[b], ADV[b])
                    critics[c], st_c[c], lc = critic_step(critics[c], st_c[c], Cc[b], RET[b])
        it += 1
        print(json.dumps(dict(it=it, eps=eps, laps={c: round(v, 1) for c, v in laps.items()}, crashes=crashes,
                              ey={c: round(float(np.mean([rc.lap_metrics(r, j['plan']).get('rms_ey') or np.nan
                                                          for r, j in zip(res, jobs) if j['car_name'] == c])), 3)
                                  for c in a.cars},
                              t=round(time.time() - t0, 1))), flush=True)
        for k in a.checkpoints:
            if k not in done_ck and min(eps.values()) >= k:
                write_ckpt(k, actors, laps, crashes, eps, time.time() - t0)
                done_ck.add(k)


if __name__ == '__main__':
    main()
