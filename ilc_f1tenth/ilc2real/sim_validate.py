#!/usr/bin/env python3
"""
sim_validate.py --runs runs/A runs/B --out runs/stopping/sim_val.json: the SIM-ONLY validation of every sim-stage
snapshot (the stopping-rule study; the quadruped's snapsel.py for the car). No real-car data: the nominal Fiala sim
(JAX) on the training plans and the HELD-OUT plans (beta 14 / 21: never trained on), under conditions the sim stage
did not train under:

  nominal    nominal dynamics, the usual start perturbation (0.5)
  pstart     perturbed starts (perturbation 2.0: offsets, heading, speed)
  delay      one-period steering and current delays
  dist       a disturbance: N(0, 0.1) normalized action offsets every period
  dr         dynamics drawn from the DR family (sim_jax.dr_params, scale 1)
  drwide     the DR family widened (scale 1.5)

Per snapshot and condition: the success rate (no failure, RMS e_y <= 10 cm, pace >= 90 %, as the real cars' metric),
the failure rate and the RMS lateral error, on the training plans and on the held-out ones, over --lanes lanes per
plan and 2 chained laps. -> {run: {iteration: {condition: {train|heldout: {success, fail, rms_ey}}}}}
"""
import argparse
import glob
import json
import os
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--runs', nargs='+', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--lanes', type=int, default=24)
ap.add_argument('--laps', type=float, default=2.0)
ap.add_argument('--seed', type=int, default=4040)
ap.add_argument('--gpu', default=None)
a = ap.parse_args()
if a.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax                                       # noqa: E402
import jax.numpy as jnp                          # noqa: E402
import numpy as np                               # noqa: E402

import bank                                      # noqa: E402
import sim_jax as sj                             # noqa: E402

TRAIN = [f'mocap_{t}_b{b}' for t in ('square2fast', 'figfast') for b in (0, 10, 18, 25)]
HELD = [f'mocap_{t}_b{b}' for t in ('square2fast', 'figfast') for b in (14, 21)]
NAMES = TRAIN + HELD
COND = dict(nominal=dict(pert=0.5), pstart=dict(pert=2.0), delay=dict(pert=0.5, delay=True),
            dist=dict(pert=0.5, noise=0.1), dr=dict(pert=0.5, dr=1.0), drwide=dict(pert=0.5, dr=1.5))


def snapshots(run):
    out = {}
    bc = os.path.join(run, 'policy_bc.npz')
    if os.path.exists(bc):
        out[0] = bc
    for f in glob.glob(os.path.join(run, 'policy_it*.npz')):
        out[int(re.findall(r'it(\d+)', f)[0])] = f
    return dict(sorted(out.items()))


def load(f):
    d = np.load(f)
    n = len([k for k in d.files if k.startswith('W')])
    return [(jnp.array(d[f'W{i}']), jnp.array(d[f'b{i}'])) for i in range(n)]


def main():
    P = sj.Plans(bank.load_bank(NAMES), NAMES)
    T = int(np.ceil(a.laps * max(P.lap_time) / sj.DT))
    n = a.lanes * len(NAMES)
    pid = np.repeat(np.arange(len(NAMES)), a.lanes)
    rng = np.random.default_rng(a.seed)
    s0 = rng.uniform(0, 1, n) * np.asarray(P.L)[pid]
    key = jax.random.PRNGKey(a.seed)
    starts = {}
    for cname, c in COND.items():                  # the same starts / draws for every snapshot (common random numbers)
        xs, ids = [], []
        for k in range(n):
            x0, i0 = sj.start_states(P, jax.random.PRNGKey(a.seed + k), [pid[k]], [s0[k]], pert=c['pert'])
            xs.append(x0[0]); ids.append(i0[0])
        key, kp, kn = jax.random.split(key, 3)
        p = sj.dr_params(kp, n, c['dr']) if c.get('dr') else sj.nominal_params(n)
        if c.get('delay'):
            p['d_delay'] = jnp.ones(n, jnp.int32)
            p['i_delay'] = jnp.ones(n, jnp.int32)
        noise = jax.random.normal(kn, (n, T, 2)) * c.get('noise', 0.0)
        starts[cname] = (jnp.stack(xs), jnp.stack(ids), p, noise)
    Vp = np.hypot(np.asarray(P.z)[..., 2], np.asarray(P.z)[..., 3])      # the plans' speed per index
    res = {}
    for run in a.runs:
        name = os.path.basename(run.rstrip('/'))
        res[name] = {}
        for it, f in snapshots(run).items():
            ps = load(f)
            res[name][it] = {}
            for cname, (x0, i0, p, noise) in starts.items():
                out = sj.rollout(P, lambda o: sj.mlp(ps, o), T, x0, i0, jnp.array(pid), p, noise)
                fail = np.asarray(out['failed'][:, -1])
                e = np.asarray(out['err'])[..., 0]
                idx = np.asarray(out['idx'])
                X = np.asarray(out['x'])[:, :-1]
                alive = ~np.asarray(out['failed'])
                ey = np.sqrt(np.sum(alive * e ** 2, 1) / np.maximum(alive.sum(1), 1))
                pace = np.sum(alive * np.hypot(X[..., 3], X[..., 4]), 1) / np.maximum(
                    np.sum(alive * Vp[pid[:, None], idx], 1), 1e-6)
                ok = (~fail) & (ey <= 0.10) & (pace >= 0.9)
                held = np.isin(pid, [NAMES.index(h) for h in HELD])
                res[name][it][cname] = {grp: dict(success=float(ok[m].mean()), fail=float(fail[m].mean()),
                                                  rms_ey=float(np.mean(ey[m & ~fail])) if (m & ~fail).any() else None)
                                        for grp, m in (('train', ~held), ('heldout', held))}
            print(json.dumps(dict(run=name, it=it, **{c: round(v['heldout']['success'], 3) for c, v in res[name][it].items()})),
                  flush=True)
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
