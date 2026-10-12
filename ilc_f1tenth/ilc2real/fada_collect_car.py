#!/usr/bin/env python3
"""
fada_collect_car.py --out DIR: FADA's inverse dynamics model trained on EACH multi-body car's own transitions (the
exact-dynamics oracle, log/dilc/plane/match/EXACT_ORACLES_RUNBOOK.md C2; user 2026-10-11: "make true dynamics the true robot
for rma and fada"). fada_car.py fits one IDM over the single-track DR family in the JAX sim; here every transition is flown on
the car itself through real_car.run_many (the function and cars every evaluation uses; real_car.exact_check asserts the
tight-track setup), one IDM per car:
  behaviour   the warm-start network (--init) plus smooth AR(1) action noise (rho 0.85, std --a-off: fada_car.py's), from
              random arc lengths of the goal plans (with --rungs-from: each car's rung variants, the idealized reference)
  data        privileged hindsight pairs (o_t, d_{t+H}) -> a_t from the TRUE observation (obs_true), d = the trunk entries
              --dims (0-4: e_y, e_psi, vx, vy, r; the actuator-state target made fada_car's IDM near open-loop replay)
  fit         fada_car.py's IDM (MLP [NOBS + ND, 256, 256, 2], Adam 3e-4, --train-steps x 8192)
-> DIR/idm_<car>.npz (W/b, lam, dims, horizon), DIR/collect.json. fada_real_car.py --idm DIR/idm_{car}.npz adapts them.
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--init', default='runs/bcB/policy_final.npz')
ap.add_argument('--cars', nargs='+', default=['real_nom', 'real_mass', 'real_mu', 'real_act', 'real_lag'])
ap.add_argument('--goals', nargs='+', default=[f'mocap_{t}_b{b}' for t in ('square2fast', 'figfast') for b in (0, 10, 18, 25)])
ap.add_argument('--rungs-from', default='')
ap.add_argument('--episodes', type=int, default=6000, help='per car')
ap.add_argument('--batch', type=int, default=600)
ap.add_argument('--T', type=int, default=160, help='control periods per episode')
ap.add_argument('--a-off', type=float, default=0.2)
ap.add_argument('--lam', type=float, default=0.8)
ap.add_argument('--dims', type=int, nargs='+', default=[0, 1, 2, 3, 4])
ap.add_argument('--horizon', type=int, default=4)
ap.add_argument('--train-steps', type=int, default=30000)
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
import real_car as rc                            # noqa: E402
import sim_jax as sj                             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DIMS, H = np.array(a.dims), a.horizon
RUNGS = json.load(open(os.path.join(HERE, 'runs/hw', a.rungs_from, 'summary.json')))['rungs'] if a.rungs_from else {}
track = lambda n: n.split('_b')[0]


def goal_for(c, g):
    sfx = RUNGS.get(c, {}).get(track(g), '')
    return g + sfx if sfx and os.path.exists(os.path.join(bank.PLAN_DIR, g + sfx + '.npz')) else g


def main():
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)
    rc.exact_check(a.cars, os.path.join(a.out, 'exact_check.json'))
    rng = np.random.default_rng(a.seed)
    names = list(dict.fromkeys(goal_for(c, g) for c in a.cars for g in a.goals))
    plans = {p['name']: p for p in bank.load_bank(names)}
    d = np.load(a.init)
    beh = [(np.asarray(d[f'W{i}']), np.asarray(d[f'b{i}'])) for i in range(3)]
    stats = dict(cars={}, rungs={c: RUNGS.get(c, {}) for c in a.cars})
    t0 = time.time()
    for c in a.cars:
        X, Y, flown, crashes, laps = [], [], 0, 0, 0.0
        while flown < a.episodes:
            jobs = []
            for k in range(min(a.batch, a.episodes - flown)):
                pl = plans[goal_for(c, a.goals[(flown + k) % len(a.goals)])]
                e = rng.normal(size=(a.T, 2)) * a.a_off * np.sqrt(1 - 0.85 ** 2)
                nz = np.zeros_like(e)
                for t in range(1, a.T):
                    nz[t] = 0.85 * nz[t - 1] + e[t]
                jobs.append(dict(car_name=c, plan=pl, policy=beh, laps=1.05, s0=rng.uniform() * float(pl['length']),
                                 pert=rng.uniform(-1, 1, 5), seed=int(rng.integers(1 << 30)), T_max=a.T, action_noise=nz))
            res = rc.run_many(jobs)
            flown += len(jobs)
            for r in res:
                laps += max(float(r['laps']), 0.0)
                crashes += int(bool(r['failed']))
                o = np.asarray(r['obs_true'])
                if len(o) > H:
                    X.append(np.concatenate([o[:-H], o[H:][:, DIMS]], -1))
                    Y.append(np.asarray(r['a'])[:-H])
            print(json.dumps(dict(car=c, flown=flown, crashes=crashes, laps=round(laps, 1), t=round(time.time() - t0))),
                  flush=True)
        Xj, Yj = jnp.array(np.concatenate(X), jnp.float32), jnp.array(np.concatenate(Y), jnp.float32)
        key = jax.random.PRNGKey(a.seed)
        idm = sj.init_mlp(key, [Xj.shape[1], 256, 256, 2])
        opt = optax.adam(3e-4)
        st = opt.init(idm)

        @jax.jit
        def step(ps, st, x, y):
            l, g = jax.value_and_grad(lambda q: jnp.mean((sj.mlp(q, x) - y) ** 2))(ps)
            u, st = opt.update(g, st, ps)
            return optax.apply_updates(ps, u), st, l
        for s in range(a.train_steps):
            b = jnp.array(rng.integers(0, Xj.shape[0], 8192))
            idm, st, l = step(idm, st, Xj[b], Yj[b])
        base = [(np.asarray(W), np.asarray(b)) for W, b in idm]
        np.savez(os.path.join(a.out, f'idm_{c}.npz'), **{f'W{i}': W for i, (W, b) in enumerate(base)},
                 **{f'b{i}': b for i, (W, b) in enumerate(base)}, lam=a.lam ** H, dims=DIMS, horizon=H)
        stats['cars'][c] = dict(episodes=flown, crashes=crashes, laps=laps, samples=int(Xj.shape[0]), idm_loss=float(l))
        print(json.dumps(dict(car=c, idm_loss=float(l), samples=int(Xj.shape[0]), t=round(time.time() - t0))), flush=True)
    stats['seconds'] = time.time() - t0
    json.dump(stats, open(os.path.join(a.out, 'collect.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
