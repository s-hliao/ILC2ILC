#!/usr/bin/env python3
"""
rmaD_train.py --init runs/rmaCov_s<s>/policy_final.npz --out runs/rmaD_s<s> (2026-11-07, user: the RMA oracle-latent reference "has to
cover all deployment dynamics"): the RMA TEACHER trained in the DEPLOYMENT simulator over every axis the deployment cars vary
(rmaD_latent.py: 14-d privileged latent, a box containing all 50 deployment cars), reading [obs, z]. Not deployable; the reference.

  simulator  mb_car via real_car.run (the multi-body car, its sensing, its tire wear), the evaluation's settings (F1T_DEV_TIME=1,
             F1T_SPIN_RHO=3, F1T_MU=0.6, plans_nom06); one random car per episode (rmaD_latent.sample_car)
  episodes   --laps (5.05: the evaluation's length, so wear plays out) of one of rmaN's 6 training plans from a random arc length,
             start perturbation N(0, 0.5) (the evaluation's), the policy mean + Gaussian exploration
  reward     ppo_real_car.py's: per period 1 - 0.1 min(|err_true / TASK_SCALE|^2, 10)
  PPO        ppo_real_car.py's (clipped surrogate, GAE, Adam + clip 1.0, critic warm-up), ONE shared actor / critic over the random cars;
             critic asymmetric: obs + true planar state + z
  init       the planar covering teacher rmaCov (obs + its 8 z keys = z[:8]); the 6 new z inputs at zero weight; log std its final
Saves OUT/policy_it<k>.npz every --save-every and OUT/policy_final.npz (W*/b*, mode 'rmaD', nz 14) + log.jsonl.
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--init', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--plans', nargs='+', default=['mocap_square2fast_b0_free', 'mocap_square2fast_b0_free_v85', 'mocap_square2fast_b0_free_v70',
                                               'mocap_figfast_b0_free', 'mocap_figfast_b0_free_v85', 'mocap_figfast_b0_free_v70'])
ap.add_argument('--iters', type=int, default=200)
ap.add_argument('--episodes', type=int, default=128, help='episodes (random cars) per iteration')
ap.add_argument('--laps', type=float, default=5.05)
ap.add_argument('--epochs', type=int, default=4)
ap.add_argument('--mb', type=int, default=16384)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--lr-critic', type=float, default=1e-3)
ap.add_argument('--critic-warmup', type=int, default=300)
ap.add_argument('--gamma', type=float, default=0.98)
ap.add_argument('--lam', type=float, default=0.95)
ap.add_argument('--clip', type=float, default=0.2)
ap.add_argument('--save-every', type=int, default=25)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--car', default='', help='(2026-11-07, the per-car PPO CEILING) every episode on this one car (its true z) instead of a random car')
a = ap.parse_args()
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax                                       # noqa: E402
import jax.numpy as jnp                          # noqa: E402
import numpy as np                               # noqa: E402
import optax                                     # noqa: E402

import bank                                      # noqa: E402
import real_car as rc                            # noqa: E402
import rmaD_latent as L                          # noqa: E402
from consts import TASK_SCALE                    # noqa: E402

XSC = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 2.0, 50.0, 10.0, 0.2])   # ppo_real_car.py's true planar state scale for the critic


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


def save(f, ps, ls):
    np.savez(f, **{f'W{i}': np.asarray(W) for i, (W, b) in enumerate(ps)}, **{f'b{i}': np.asarray(b) for i, (W, b) in enumerate(ps)},
             mode='rmaD', nz=L.NZ, log_std=np.asarray(ls))


def main():
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)
    log = open(os.path.join(a.out, 'log.jsonl'), 'a')
    rng = np.random.default_rng(a.seed)
    key = jax.random.PRNGKey(a.seed)
    plans = bank.load_bank(a.plans)
    d = np.load(a.init)
    W = [np.asarray(d[f'W{i}'], np.float32) for i in range(3)]
    b = [np.asarray(d[f'b{i}'], np.float32) for i in range(3)]
    nobs = 55                                                                       # sim_jax.NOBS (real_car's observation)
    if W[0].shape[0] == nobs + 8:                                                   # an 8-z teacher (rmaCov): the 6 new z at 0
        W[0] = np.concatenate([W[0], np.zeros((L.NZ - 8, W[0].shape[1]), np.float32)])
    assert W[0].shape[0] == nobs + L.NZ, W[0].shape                                 # else an rmaD teacher (obs + 14 z)
    actor = [(jnp.array(w), jnp.array(bb)) for w, bb in zip(W, b)]
    ag = os.path.join(os.path.dirname(a.init), 'agent_final.npz')
    log_std = jnp.array(np.load(ag)['log_std'], jnp.float32) if os.path.exists(ag) else \
        (jnp.array(d['log_std'], jnp.float32) if 'log_std' in d.files else jnp.full(2, -2.5))
    key, k = jax.random.split(key)
    critic = init_mlp(k, [nobs + 9 + L.NZ, 256, 256, 1])
    opt_a, opt_c = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr)), optax.adam(a.lr_critic)
    st_a, st_c = opt_a.init((actor, log_std)), opt_c.init(critic)

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

    t0, sim_laps = time.time(), 0.0
    for it in range(a.iters + 1):
        pol_np = [(np.asarray(Wt), np.asarray(bt)) for Wt, bt in actor]
        std = np.exp(np.asarray(log_std))
        jobs, zs = [], []
        for e in range(a.episodes):
            car = a.car or L.sample_car(rng)
            z = L.z_of(car)
            pl = plans[int(rng.integers(len(plans)))]
            T = int(np.ceil(a.laps * pl['lap_time'] / rc.DT))
            jobs.append(dict(car_name=car, plan=pl, policy=dict(rma_oracle=True, actor=pol_np, z=z), laps=a.laps,
                             s0=rng.uniform() * float(pl['length']), pert=rng.normal(size=5) * 0.5, seed=int(rng.integers(1 << 30)),
                             action_noise=rng.normal(size=(T, 2)) * std))
            zs.append(z)
        res = rc.run_many(jobs)
        O, A, C, ADV, RET, fails, eys, laps, rews = [], [], [], [], [], 0, [], 0.0, []
        for r, z, j in zip(res, zs, jobs):
            n = len(r['obs'])
            if n < 2:
                continue
            e = np.asarray(r['err_true'])[:, :len(TASK_SCALE)] / TASK_SCALE
            rew = 1.0 - 0.1 * np.minimum(np.sum(e ** 2, -1), 10.0)
            o = np.asarray(r['obs'], np.float32)
            cin = np.c_[o, np.asarray(r['x_true'])[:, :9] / XSC, np.broadcast_to(z, (n, L.NZ))].astype(np.float32)
            v = np.asarray(mlp(critic, jnp.array(cin))[:, 0])
            dead = bool(r['failed'])
            v_next = np.r_[v[1:], 0.0 if dead else v[-1]]
            delta = rew + a.gamma * v_next - v
            adv = np.zeros_like(delta); g = 0.0
            for t in range(len(delta) - 1, -1, -1):
                g = delta[t] + a.gamma * a.lam * g
                adv[t] = g
            O.append(np.c_[o, np.broadcast_to(z, (n, L.NZ))]); A.append(np.asarray(r['a'], np.float32)); C.append(cin)
            ADV.append(adv); RET.append(adv + v)
            fails += int(dead); laps += max(float(r['laps']), 0.0); rews.append(float(rew.mean()))
            eys.append(float(np.sqrt(np.mean(np.asarray(r['err_true'])[:, 0] ** 2))))
        sim_laps += laps
        Oc, Ac, Cc = (jnp.array(np.concatenate(x), jnp.float32) for x in (O, A, C))
        adv_all, ret_all = np.concatenate(ADV), np.concatenate(RET)
        ADVj = jnp.array((adv_all - adv_all.mean()) / (adv_all.std() + 1e-8), jnp.float32)
        RETj = jnp.array(ret_all, jnp.float32)
        n = len(ret_all)
        if it == 0:
            for s in range(a.critic_warmup):
                bb = jnp.array(rng.integers(0, n, min(a.mb, n)))
                critic, st_c, lc = critic_step(critic, st_c, Cc[bb], RETj[bb])
        else:
            mu_old = mlp(actor, Oc)
            LPo = jnp.sum(-0.5 * ((Ac - mu_old) / jnp.exp(log_std)) ** 2 - log_std, -1)
            for ep in range(a.epochs):
                perm = rng.permutation(n)
                for s in range(0, n, a.mb):
                    bb = jnp.array(perm[s:s + a.mb])
                    (actor, log_std), st_a, la = actor_step((actor, log_std), st_a, Oc[bb], Ac[bb], LPo[bb], ADVj[bb])
                    critic, st_c, lc = critic_step(critic, st_c, Cc[bb], RETj[bb])
        rec = dict(it=it, fail=fails / len(jobs), ey=float(np.mean(eys)), reward=float(np.mean(rews)), samples=int(n),
                   std=np.exp(np.asarray(log_std)).round(4).tolist(), sim_laps=round(sim_laps, 1), t=round(time.time() - t0, 1))
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
        if it % a.save_every == 0:
            save(os.path.join(a.out, f'policy_it{it}.npz'), actor, log_std)
    save(os.path.join(a.out, 'policy_final.npz'), actor, log_std)
    print(json.dumps(dict(event='done', sim_laps=round(sim_laps, 1), t=round(time.time() - t0, 1))), flush=True)


if __name__ == '__main__':
    main()
