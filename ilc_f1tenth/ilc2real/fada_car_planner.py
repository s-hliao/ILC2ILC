#!/usr/bin/env python3
"""
fada_car_planner.py --idm runs/fig1/fada_d60_s<s>/idm.npz --out runs/fig1/fada_d60_s<s>/planner.npz (2026-11-07, FADA audit,
runs/screen/FADA_AUDIT_1107.md): FADA's LEARNED planner (Xie et al., arXiv 2606.28476, Eq. 4.3) for the car baseline, replacing the
fixed deviation decay d* = lam^H d that fada_car.py used (the audit measured that decay as unreachable: the deviation asked for
4 periods ahead is 41 % of the current one; the car reaches ~100 %, gap 0.6 of the deviation on every car, before and after LoRA).

  L_P(phi) = E || IDM_frozen(o_t, P_phi(o_t)) - a*_t ||^2,   a*_t = the oracle's action at o_t

The oracle = the network the IDM's data came from (fada_car.py --init, freeBCN_s0), noise-free; the states = fada_car.py's IDM data
distribution exactly (its seed, plans, DR family, 512 lanes x 60 iterations x 160 periods, smooth action noise a_off), so the planner
sees the states the IDM was trained on (FADA's DAgger states are its student's rollouts; this is the documented deviation). The IDM is
frozen (the paper: stop-gradient). Architecture as the IDM (MLP 256 x 256, tanh), Adam 3e-4, 30000 steps, batch 8192 (the IDM's).
Reported: the first-action loss of the old decay planner vs the learned one on held-out states.
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--idm', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--steps', type=int, default=30000)
a = ap.parse_args()
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax                                       # noqa: E402
import jax.numpy as jnp                          # noqa: E402
import numpy as np                               # noqa: E402
import optax                                     # noqa: E402

import bank                                      # noqa: E402
import sim_jax as sj                             # noqa: E402


def main():
    src = json.load(open(os.path.join(os.path.dirname(a.idm), 'args.json')))     # fada_car.py's own settings
    dims = np.asarray(src['dims']); H = int(src['horizon']); lam_eff = float(src['lam']) ** H
    rng = np.random.default_rng(src['seed'] + 777)                              # a fresh draw of the same distribution
    key = jax.random.PRNGKey(src['seed'] + 777)
    plans = bank.load_bank(list(dict.fromkeys(src['plans'])))
    by_name = {p['name']: p for p in plans}
    P = sj.Plans([by_name[n] for n in src['plans']], src['plans'])
    d = np.load(src['init'])
    beh = [(jnp.array(d[f'W{i}']), jnp.array(d[f'b{i}'])) for i in range(3)]
    e = np.load(a.idm)
    idm = [(jnp.array(e[f'W{i}']), jnp.array(e[f'b{i}'])) for i in range(len([k for k in e.files if k.startswith('W')]))]
    t0 = time.time()
    fly = jax.jit(lambda x0, i0, pid, p, nz: sj.rollout(P, lambda o: sj.mlp(beh, o), src['T'], x0, i0, pid, p, nz))
    O = []
    for it in range(src['data_iters']):
        lanes = src['lanes']
        pid = rng.integers(0, len(src['plans']), lanes)
        s0 = rng.uniform(0, 1, lanes) * np.asarray(P.L)[pid]
        xs, ids = [], []
        for k in range(lanes):
            x0, i0 = sj.start_states(P, jax.random.PRNGKey(int(rng.integers(1 << 30))), [pid[k]], [s0[k]], pert=1.0)
            xs.append(x0[0]); ids.append(i0[0])
        key, k1, k2 = jax.random.split(key, 3)
        p = sj.dr_params(k1, lanes, 1.0)
        rho = 0.85
        ee = np.asarray(jax.random.normal(k2, (src['T'], lanes, 2))) * src['a_off'] * np.sqrt(1 - rho ** 2)
        nz = np.zeros_like(ee)
        for t in range(1, src['T']):
            nz[t] = rho * nz[t - 1] + ee[t]
        out = fly(jnp.stack(xs), jnp.stack(ids), jnp.array(pid), p, jnp.array(nz.transpose(1, 0, 2)))
        ok = ~np.asarray(out['failed'])
        O.append(np.asarray(out['obs'])[ok])
    O = np.concatenate(O).astype(np.float32)
    perm = rng.permutation(len(O)); n_te = min(50000, len(O) // 10)
    Ote, Otr = jnp.array(O[perm[:n_te]]), jnp.array(O[perm[n_te:]])
    astar = lambda o: sj.mlp(beh, o)                                             # the oracle's noise-free action
    idm_a = lambda o, dn: sj.mlp(idm, jnp.concatenate([o, dn], -1))
    print(json.dumps(dict(event='data', n=int(len(O)), t=round(time.time() - t0, 1))), flush=True)

    key, k0 = jax.random.split(key)
    pl = sj.init_mlp(k0, [sj.NOBS, 256, 256, len(dims)])
    opt = optax.adam(3e-4)
    st = opt.init(pl)

    def loss_fn(q, o):
        return jnp.mean((idm_a(o, sj.mlp(q, o)) - jax.lax.stop_gradient(astar(o))) ** 2)

    @jax.jit
    def step(q, st, o):
        l, g = jax.value_and_grad(loss_fn)(q, o)
        u, st = opt.update(g, st, q)
        return optax.apply_updates(q, u), st, l
    for s in range(a.steps):
        b = jnp.array(rng.integers(0, Otr.shape[0], 8192))
        pl, st, l = step(pl, st, Otr[b])
    te_new = float(loss_fn(pl, Ote))
    te_old = float(jnp.mean((idm_a(Ote, lam_eff * Ote[:, dims]) - astar(Ote)) ** 2))          # the decay planner
    te_hold = float(jnp.mean((idm_a(Ote, Ote[:, dims]) - astar(Ote)) ** 2))                   # "keep the deviation"
    dn = np.asarray(sj.mlp(pl, Ote)); dc = np.asarray(Ote[:, dims])
    ratio = (np.sqrt((dn ** 2).mean(0)) / (np.sqrt((dc ** 2).mean(0)) + 1e-9)).round(3).tolist()
    res = dict(event='planner', train_loss=float(l), heldout_first_action_mse=dict(learned=te_new, decay_lam=te_old, hold=te_hold),
               target_rms_over_current_rms=ratio, lam_eff=lam_eff, t=round(time.time() - t0, 1))
    print(json.dumps(res), flush=True)
    base = [(np.asarray(W), np.asarray(b)) for W, b in pl]
    np.savez(a.out, **{f'P_W{i}': W for i, (W, b) in enumerate(base)}, **{f'P_b{i}': b for i, (W, b) in enumerate(base)})
    json.dump(res, open(a.out.replace('.npz', '.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
