#!/usr/bin/env python3
"""
fada_real_car.py --idm runs/hw/fada_hw24/idm.npz --arm NAME: FADA trained ON THE REAL CARS with privileged data (the
data-matching study, user 2026-10-09). Starts from FADA's sim-trained IDM (fada_car.py: IDM(o_t, d_{t+1}) -> a_t
over the DR family) and adapts it on each multi-body car the way FADA does -- every real transition relabelled in
hindsight (o_t, the deviation d_{t+1} actually reached) -> a_t as flown, a LoRA of rank --rank on each IDM layer
fitted to all of the car's transitions so far -- but with up to ~2000 episodes per car and:

  --privileged 1  the relabelling uses the car's TRUE state (observation features without sensing noise)
  planner         FADA's planner asks the IDM for the next deviation. The assumption under test:
                  --lam L         d* = L d_t: the deviation from the nominal plan decays (FADA's, L = 0.8): assumes the
                                  nominal plan is what the real car should fly
                  --oracle-arm A  d* = the deviation that arm's per-car networks (our best model after the hardware
                                  stage) actually reach on this car at the same point of the lap: a planner that
                                  already knows what this car can fly (privileged; its laps are counted separately,
                                  oracle_laps)
episodes   one lap (1.05) on one of --goals from a random arc length, small start perturbation, under the room's
           safety envelope (F1T_SAFETY_EY); per car --per-iter episodes per adaptation round
Checkpoints at --checkpoints episodes per car -> runs/hw/<arm>_ep<K>/ (reeval.py-compatible: <car>_policy.npz holds
the merged IDM, lam and the oracle tables).
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--idm', required=True)
ap.add_argument('--arm', required=True)
ap.add_argument('--cars', nargs='+', default=['real_nom', 'real_mass', 'real_mu', 'real_act', 'real_lag'])
ap.add_argument('--goals', nargs='+', default=[f'mocap_{t}_b{b}' for t in ('square2fast', 'figfast') for b in (10, 18, 25)])
ap.add_argument('--eval-plans', nargs='+', default=[f'mocap_{t}_b{b}' for t in ('square2fast', 'figfast') for b in (25, 21, 14, 0)])
ap.add_argument('--per-iter', type=int, default=24)
ap.add_argument('--checkpoints', type=int, nargs='+', default=[24, 48, 96, 192, 384, 768, 1536, 2016])
ap.add_argument('--lam', type=float, default=0.8)
ap.add_argument('--oracle-arm', default='')
ap.add_argument('--oracle-laps', type=float, default=3.0, help='chained laps per plan the oracle flies on each car')
ap.add_argument('--privileged', type=int, default=1)
ap.add_argument('--rank', type=int, default=4)
ap.add_argument('--adapt-steps', type=int, default=500)
ap.add_argument('--seed', type=int, default=5151)
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
import real_car as rc                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ND = 8                                            # the observation's plan-deviation entries (fada_car.py)


def mlp(ps, x):
    for W, b in ps[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = ps[-1]
    return x @ W + b


def load_mlp(path):
    d = np.load(path)
    n = len([k for k in d.files if k.startswith('W')])
    return [(np.asarray(d[f'W{i}'], np.float32), np.asarray(d[f'b{i}'], np.float32)) for i in range(n)]


def oracle_tables(car, plans):
    """Per plan: the mean (true) deviation the oracle arm's network for this car reaches at each plan index."""
    pol = load_mlp(os.path.join(HERE, 'runs/hw', a.oracle_arm, f'{car}_policy.npz'))
    jobs = [dict(car_name=car, plan=pl, policy=pol, laps=a.oracle_laps, s0=0.0, pert=np.zeros(5), seed=11 + k)
            for k, pl in enumerate(plans.values())]
    res = rc.run_many(jobs)
    tabs, laps = {}, 0.0
    for (name, pl), r in zip(plans.items(), res):
        laps += max(float(r['laps']), 0.0)
        M = len(pl['x'])
        d = np.asarray(r['obs_true'])[:, :ND]
        idx = np.asarray(r['idx']) % M
        s, n = np.zeros((M, ND)), np.zeros(M)
        np.add.at(s, idx, d)
        np.add.at(n, idx, 1)
        have = n > 0
        tab = np.zeros((M, ND), np.float32)
        if have.sum() >= 2:                     # unvisited indices (after a crash): periodic interpolation
            grid = np.arange(M)
            for j in range(ND):
                tab[:, j] = np.interp(grid, grid[have], s[have, j] / n[have], period=M)
        tabs[name] = tab
    return tabs, laps


def main():
    rng = np.random.default_rng(a.seed)
    names = list(dict.fromkeys(a.goals + a.eval_plans))
    plans = {p['name']: p for p in bank.load_bank(names)}
    base = [(jnp.array(W), jnp.array(b)) for W, b in load_mlp(a.idm)]
    opt = optax.adam(1e-3)

    def merged(lo):
        return [(W + A @ B, b) for (W, b), (A, B) in zip(base, lo)]

    def lora_init(k):
        out = []
        for W, b in base:
            k, kk = jax.random.split(k)
            out.append((jax.random.normal(kk, (W.shape[0], a.rank)) * 0.01, jnp.zeros((a.rank, W.shape[1]))))
        return out

    @jax.jit
    def astep(lo, st, x, y):
        l, g = jax.value_and_grad(lambda q: jnp.mean((mlp(merged(q), x) - y) ** 2))(lo)
        u, st = opt.update(g, st, lo)
        return optax.apply_updates(lo, u), st, l

    oracle, oracle_laps = {}, {}
    if a.oracle_arm:
        for c in a.cars:
            oracle[c], oracle_laps[c] = oracle_tables(c, plans)
    lora = {c: lora_init(jax.random.PRNGKey(7)) for c in a.cars}
    st = {c: opt.init(lora[c]) for c in a.cars}
    data = {c: ([], []) for c in a.cars}
    laps = {c: 0.0 for c in a.cars}
    eps = {c: 0 for c in a.cars}
    crashes = {c: 0 for c in a.cars}

    def policy(c):
        return dict(fada=[(np.asarray(W), np.asarray(b)) for W, b in merged(lora[c])], lam=a.lam,
                    oracle=oracle.get(c) or None)

    def write_ckpt(k, t):
        d = os.path.join(HERE, 'runs/hw', f'{a.arm}_ep{k}')
        os.makedirs(d, exist_ok=True)
        for c in a.cars:
            p = policy(c)
            np.savez(os.path.join(d, f'{c}_policy.npz'), **{f'W{i}': W for i, (W, b) in enumerate(p['fada'])},
                     **{f'b{i}': b for i, (W, b) in enumerate(p['fada'])}, lam=a.lam, mode='fada',
                     **{f'oracle__{n}': v for n, v in (oracle.get(c) or {}).items()})
        args = dict(vars(a), policy=a.idm, episodes_per_car=k)
        json.dump(args, open(os.path.join(d, 'args.json'), 'w'), indent=1)
        json.dump(dict(args=args, real_laps=dict(laps), oracle_laps=oracle_laps, crashes=crashes, episodes=eps,
                       seconds=t, safety_ey=float(os.environ.get('F1T_SAFETY_EY') or 0.6)),
                  open(os.path.join(d, 'summary.json'), 'w'), indent=1)

    t0 = time.time()
    write_ckpt(0, 0.0)
    done = {0}
    it = 0
    while min(eps.values()) < max(a.checkpoints):
        jobs = []
        for c in a.cars:
            pol = policy(c)
            for k in range(a.per_iter):
                pl = plans[a.goals[(it * a.per_iter + k) % len(a.goals)]]
                jobs.append(dict(car_name=c, plan=pl, policy=pol, laps=1.05, s0=rng.uniform() * float(pl['length']),
                                 pert=rng.uniform(-0.1, 0.1, 5), seed=int(rng.integers(1 << 30))))
        res = rc.run_many(jobs)
        for c in a.cars:
            rr = [r for r, j in zip(res, jobs) if j['car_name'] == c]
            for r in rr:
                laps[c] += max(float(r['laps']), 0.0)
                crashes[c] += int(bool(r['failed']))
                o = np.asarray(r['obs_true'] if a.privileged else r['obs'])
                if len(o) > 1:                    # hindsight relabelling: (o_t, d_{t+1} reached) -> a_t as flown
                    data[c][0].append(np.concatenate([o[:-1], o[1:, :ND]], -1))
                    data[c][1].append(np.asarray(r['a'])[:-1])
            eps[c] += len(rr)
            X, Y = jnp.array(np.concatenate(data[c][0]), jnp.float32), jnp.array(np.concatenate(data[c][1]), jnp.float32)
            for s in range(a.adapt_steps):
                b = jnp.array(rng.integers(0, X.shape[0], min(2048, X.shape[0])))
                lora[c], st[c], l = astep(lora[c], st[c], X[b], Y[b])
        it += 1
        print(json.dumps(dict(it=it, eps=eps, laps={c: round(v, 1) for c, v in laps.items()}, crashes=crashes,
                              t=round(time.time() - t0, 1))), flush=True)
        for k in a.checkpoints:
            if k not in done and min(eps.values()) >= k:
                write_ckpt(k, time.time() - t0)
                done.add(k)


if __name__ == '__main__':
    main()
