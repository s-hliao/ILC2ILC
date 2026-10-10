#!/usr/bin/env python3
"""
fada_car.py --out DIR: the FADA-style baseline (the quadruped's fada_train.py / fada_adapt.py) on the car.
  train   an inverse dynamics model IDM(o_t, d*_{t+1}) -> a_t over the domain-randomization family: behaviour = the
          warm-start network plus smooth action noise (coverage), every transition (o_t, deviation reached d_{t+1}, a_t);
          the "planner" asks for d*_{t+1} = lam d_t (deviation from the plan decaying), d = the observation's first 8
          (plan-deviation) entries.
  adapt   on each real car, the same 6 goals x 4 laps as hw_stage: the IDM flies, every real transition is relabelled
          in hindsight (o_t, d_{t+1} as reached) -> a_t as flown -- reward-free, the tracking error is never used --
          and a LoRA of rank --rank on each IDM layer (base frozen) is fitted to all the car's transitions so far.
  then hw_stage's evaluation protocol (reeval.py-compatible: <car>_policy.npz holds the merged IDM + lam).
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--init', default='runs/bcB/policy_final.npz', help='behaviour network for the IDM data')
ap.add_argument('--plans', nargs='+', default=[f'mocap_{t}_b{b}' for t in ('square2fast', 'figfast') for b in (0, 10, 18, 25)])
ap.add_argument('--goals', nargs='+', default=[f'mocap_{t}_b{b}' for t in ('square2fast', 'figfast') for b in (10, 18, 25)])
ap.add_argument('--cars', nargs='+', default=['real_nom', 'real_mass', 'real_mu', 'real_act', 'real_lag'])
ap.add_argument('--data-iters', type=int, default=60)
ap.add_argument('--lanes', type=int, default=512)
ap.add_argument('--T', type=int, default=160)
ap.add_argument('--a-off', type=float, default=0.2)
ap.add_argument('--lam', type=float, default=0.8)
ap.add_argument('--dims', type=int, nargs='+', default=list(range(8)), help='the deviation entries the IDM targets (0-4: '
                'e_y, e_psi, vx, vy, r; 5-7: wheel speed, current, steering -- the actuator states, which make the one-step '
                'inverse trivial: the IDM then reads the action off them and ignores the path)')
ap.add_argument('--horizon', type=int, default=1, help='the IDM targets the deviation H control periods ahead (planner '
                'd* = lam^H d)')
ap.add_argument('--train-steps', type=int, default=30000)
ap.add_argument('--iters', type=int, default=4)
ap.add_argument('--rank', type=int, default=4)
ap.add_argument('--adapt-steps', type=int, default=500)
ap.add_argument('--eval-runs', type=int, default=4)
ap.add_argument('--eval-laps', type=float, default=5.0)
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

ND = len(a.dims)
DIMS = np.array(a.dims)
H = a.horizon
LAM_EFF = a.lam ** H


def idm_in(obs, d_next):
    return jnp.concatenate([obs, d_next], -1)


def main():
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(vars(a), policy=os.path.join(a.out, 'idm.npz')), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)
    zs = a.out.rstrip('/') + '_zs'                       # the zero-shot arm (the IDM before adaptation)
    os.makedirs(zs, exist_ok=True)
    json.dump(dict(cars=a.cars, policy=os.path.join(a.out, 'idm.npz')), open(os.path.join(zs, 'args.json'), 'w'))
    rng = np.random.default_rng(a.seed)
    key = jax.random.PRNGKey(a.seed)
    plans = bank.load_bank(list(dict.fromkeys(a.plans + a.goals)))
    by_name = {p['name']: p for p in plans}
    P = sj.Plans([by_name[n] for n in a.plans], a.plans)
    d = np.load(a.init)
    beh = [(jnp.array(d[f'W{i}']), jnp.array(d[f'b{i}'])) for i in range(3)]
    t0 = time.time()
    sim_laps = 0.0

    # ---- IDM data over the DR family
    fly = jax.jit(lambda x0, i0, pid, p, nz: sj.rollout(P, lambda o: sj.mlp(beh, o), a.T, x0, i0, pid, p, nz))
    X, Y = [], []
    for it in range(a.data_iters):
        pid = rng.integers(0, len(a.plans), a.lanes)
        s0 = rng.uniform(0, 1, a.lanes) * np.asarray(P.L)[pid]
        xs, ids = [], []
        for k in range(a.lanes):
            x0, i0 = sj.start_states(P, jax.random.PRNGKey(int(rng.integers(1 << 30))), [pid[k]], [s0[k]], pert=1.0)
            xs.append(x0[0]); ids.append(i0[0])
        key, k1, k2 = jax.random.split(key, 3)
        p = sj.dr_params(k1, a.lanes, 1.0)
        rho = 0.85
        e = np.asarray(jax.random.normal(k2, (a.T, a.lanes, 2))) * a.a_off * np.sqrt(1 - rho ** 2)
        nz = np.zeros_like(e)
        for t in range(1, a.T):
            nz[t] = rho * nz[t - 1] + e[t]
        out = fly(jnp.stack(xs), jnp.stack(ids), jnp.array(pid), p, jnp.array(nz.transpose(1, 0, 2)))
        sim_laps += float(np.sum(np.asarray(out['progress']) / np.asarray(P.L)[pid]))
        obs = np.asarray(out['obs'])
        act = np.asarray(out['a'])
        al = ~np.asarray(out['failed'])
        ok = al[:, :-H] & al[:, H:]                    # both ends of the (o_t, d_{t+H}) pair on track
        X.append(np.concatenate([obs[:, :-H], obs[:, H:][..., DIMS]], -1)[ok])
        Y.append(act[:, :-H][ok])
    X, Y = np.concatenate(X), np.concatenate(Y)
    print(json.dumps(dict(event='data', n=len(X), sim_laps=sim_laps, t=time.time() - t0)), flush=True)

    key, k0 = jax.random.split(key)
    idm = sj.init_mlp(k0, [sj.NOBS + ND, 256, 256, 2])
    opt = optax.adam(3e-4)
    st = opt.init(idm)
    Xj, Yj = jnp.array(X), jnp.array(Y)

    @jax.jit
    def step(ps, st, x, y):
        l, g = jax.value_and_grad(lambda q: jnp.mean((sj.mlp(q, x) - y) ** 2))(ps)
        u, st = opt.update(g, st, ps)
        return optax.apply_updates(ps, u), st, l
    for s in range(a.train_steps):
        b = jnp.array(rng.integers(0, len(X), 8192))
        idm, st, l = step(idm, st, Xj[b], Yj[b])
    print(json.dumps(dict(event='idm', loss=float(l), t=time.time() - t0)), flush=True)
    base = [(np.asarray(W), np.asarray(b)) for W, b in idm]
    np.savez(os.path.join(a.out, 'idm.npz'), **{f'W{i}': W for i, (W, b) in enumerate(base)},
             **{f'b{i}': b for i, (W, b) in enumerate(base)}, lam=LAM_EFF, dims=DIMS, sim_laps=sim_laps)

    # ---- adaptation on each car: LoRA on every layer, hindsight relabelling
    def merged(lora):
        return [(W + A @ B, b) for (W, b), (A, B) in zip(base, lora)]

    def lora_init(k):
        out = []
        for i, (W, b) in enumerate(base):
            k, kk = jax.random.split(k)
            out.append((jax.random.normal(kk, (W.shape[0], a.rank)) * 0.01, jnp.zeros((a.rank, W.shape[1]))))
        return out
    opt2 = optax.adam(1e-3)

    @jax.jit
    def astep(lo, st, x, y):
        def loss(lo):
            ps = [(jnp.array(W) + A @ B, jnp.array(b)) for (W, b), (A, B) in zip(base, lo)]
            return jnp.mean((sj.mlp(ps, x) - y) ** 2)
        l, g = jax.value_and_grad(loss)(lo)
        u, st = opt2.update(g, st, lo)
        return optax.apply_updates(lo, u), st, l

    def as_policy(lora):
        return dict(fada=[(np.asarray(W), np.asarray(b)) for W, b in merged(lora)], lam=LAM_EFF, dims=DIMS)

    summary = dict(args=vars(a), real_laps={}, trials={})
    pols = {}
    for c in a.cars:
        lora = lora_init(jax.random.PRNGKey(7))
        st2 = opt2.init(lora)
        data_x, data_y = [], []
        summary['real_laps'][c] = 0.0
        summary['trials'][c] = []
        for it in range(a.iters):
            jobs = [dict(car_name=c, plan=by_name[g], policy=as_policy(lora), laps=1.05, s0=0.0,
                         pert=rng.uniform(-0.1, 0.1, 5), seed=9001 + 100 * it + gi) for gi, g in enumerate(a.goals)]
            res = rc.run_many(jobs)
            for j, r in zip(jobs, res):
                summary['real_laps'][c] += max(r['laps'], 0.0)
                m = rc.lap_metrics(r, j['plan'])
                summary['trials'][c].append(dict(it=it, goal=j['plan']['name'], failed=m.get('failed'), rms_ey=m.get('rms_ey')))
                if len(r['obs']) > 1:
                    data_x.append(np.concatenate([r['obs'][:-H], r['obs'][H:][:, DIMS]], -1))
                    data_y.append(r['a'][:-H])
            Xa, Ya = jnp.array(np.concatenate(data_x)), jnp.array(np.concatenate(data_y))
            for s in range(a.adapt_steps):
                b = jnp.array(rng.integers(0, Xa.shape[0], min(2048, Xa.shape[0])))
                lora, st2, l = astep(lora, st2, Xa[b], Ya[b])
            print(json.dumps(dict(car=c, it=it, loss=float(l), fails=int(sum(r['failed'] for r in res)))), flush=True)
        pols[c] = as_policy(lora)
        np.savez(os.path.join(a.out, f'{c}_policy.npz'), **{f'W{i}': W for i, (W, b) in enumerate(pols[c]['fada'])},
                 **{f'b{i}': b for i, (W, b) in enumerate(pols[c]['fada'])}, lam=LAM_EFF, dims=DIMS, mode='fada')
    summary['sim_laps'] = sim_laps
    summary['seconds'] = time.time() - t0
    json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    json.dump(dict(real_laps={}), open(os.path.join(zs, 'summary.json'), 'w'))
    print(json.dumps(dict(event='done', t=time.time() - t0)), flush=True)


if __name__ == '__main__':
    main()
