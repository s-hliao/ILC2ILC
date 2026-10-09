#!/usr/bin/env python3
"""
track_ilc.py --out DIR: the per-track ILC baseline (the quadruped's per-goal JumpILC analog): on each real car and
each evaluation plan separately, from scratch, the plan's periodic LQR stabilizer plus a feedforward ff(s) on the
plan's arc-length grid (periodic: it repeats lap after lap) learned by the same closed-loop GN ILC step (ilc_core,
the nominal model's Jacobians at the measured states, the LQR's feedback in the closed loop); one lap per trial
from the plan's start. Evaluated after --budgets laps per plan with hw_stage's protocol (multi-lap runs from
spread starts, reserved seeds).
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--cars', nargs='+', default=['real_nom', 'real_mass', 'real_mu', 'real_act', 'real_lag'])
ap.add_argument('--plans', nargs='+', default=[f'mocap_{t}_b{b}' for t in ('square2fast', 'figfast') for b in (25, 21, 14, 0)])
ap.add_argument('--budgets', type=int, nargs='+', default=[4, 12])
ap.add_argument('--beta', type=float, default=0.5)
ap.add_argument('--delta', type=float, default=0.05)
ap.add_argument('--cap', type=float, default=0.05)
ap.add_argument('--objective', default='task', choices=('task', 'plan'))
ap.add_argument('--smooth', type=float, default=0.05, help='m: Gaussian smoothing of the feedforward along s')
ap.add_argument('--eval-runs', type=int, default=4)
ap.add_argument('--eval-laps', type=float, default=5.0)
ap.add_argument('--eval-seed', type=int, default=701)
ap.add_argument('--seed', type=int, default=9001)
ap.add_argument('--gpu', default=None)
a = ap.parse_args()
if a.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax.numpy as jnp                          # noqa: E402
import numpy as np                               # noqa: E402
from scipy.ndimage import gaussian_filter1d      # noqa: E402

import bank                                      # noqa: E402
import ilc_core                                  # noqa: E402
import real_car as rc                            # noqa: E402
import sim_jax as sj                             # noqa: E402


def to_grid(step, idx, valid, M, ds, smooth):
    """A time-indexed step onto the s-grid: each period's step over the grid points the car crossed in it."""
    g = np.zeros((M, 2))
    cnt = np.zeros(M)
    T = len(idx)
    for t in range(T):
        if not valid[t]:
            continue
        i0 = idx[t]
        i1 = idx[t + 1] if t + 1 < T else i0 + 1
        n = (i1 - i0) % M
        n = max(1, min(n, M // 4))
        ii = (i0 + np.arange(n)) % M
        g[ii] += step[t]
        cnt[ii] += 1
    g = np.where(cnt[:, None] > 0, g / np.maximum(cnt, 1)[:, None], 0.0)
    return gaussian_filter1d(g, smooth / ds, axis=0, mode='wrap')


def main():
    os.makedirs(a.out, exist_ok=True)
    plans = bank.load_bank(a.plans)
    P = sj.Plans(plans, a.plans)
    T = int(np.ceil(1.05 * max(P.lap_time) / sj.DT))
    gn = ilc_core.make_gn(P, T, lambda ps, o: o[:2] * 0, a.beta, a.delta, a.cap, 'closed', feedback='expert',
                          err_scale=sj.TASK_SCALE if a.objective == 'task' else sj.ERR_SCALE)
    rs = np.random.default_rng(a.seed)
    ff = {(c, k): np.zeros((len(pl['s']), 2)) for c in a.cars for k, pl in enumerate(plans)}
    results = dict(args=vars(a), budgets={})
    t0 = time.time()
    done = 0
    for budget in sorted(a.budgets):
        for it in range(done, budget):
            jobs = [dict(car_name=c, plan=pl, policy=None, laps=1.05, s0=0.0, pert=rs.uniform(-0.1, 0.1, 5),
                         seed=a.seed + 100 * it + k, ff=ff[(c, k)]) for c in a.cars for k, pl in enumerate(plans)]
            res = rc.run_many(jobs)
            keys = [(c, k) for c in a.cars for k in range(len(plans))]
            xs, idx, acts, err, valid = ilc_core.pad_runs(res, T)
            pid = np.array([k for c, k in keys])
            step, J, Jp = gn(None, jnp.array(xs), jnp.array(idx), jnp.array(acts), jnp.array(pid), jnp.array(valid),
                             jnp.array(err))
            step = np.nan_to_num(np.asarray(step))
            for n, (key, r) in enumerate(zip(keys, res)):
                pl = plans[key[1]]
                m = min(len(r['idx']), T)
                ff[key] = ff[key] + to_grid(step[n, :m], r['idx'][:m], valid[n, :m], len(pl['s']), float(pl['ds']),
                                            a.smooth)
            fails = np.mean([r['failed'] for r in res])
            print(json.dumps(dict(it=it, fails=float(fails), J=float(np.mean(J)), t=round(time.time() - t0, 1))),
                  flush=True)
        done = budget
        # evaluation at this budget
        jobs, keys = [], []
        for c in a.cars:
            for k, pl in enumerate(plans):
                r0 = np.random.default_rng(a.eval_seed + sum(map(ord, pl['name'])))
                for j in range(a.eval_runs):
                    jobs.append(dict(car_name=c, plan=pl, policy=None, laps=a.eval_laps, s0=(j / a.eval_runs) * float(pl['length']),
                                     pert=r0.uniform(-0.5, 0.5, 5), seed=a.eval_seed * 7 + j, ff=ff[(c, k)]))
                    keys.append((c, pl['name']))
        res = rc.run_many(jobs)
        ev = {}
        for (c, name), r, j in zip(keys, res, jobs):
            ev.setdefault(c, {}).setdefault(name, []).append(rc.lap_metrics(r, j['plan']))
        summ = {c: {n: dict(success=float(np.mean([m['success'] for m in ms])),
                            fail=float(np.mean([m['failed'] for m in ms])),
                            rms_ey=float(np.mean([m['rms_ey'] for m in ms if not m['failed']])) if any(not m['failed'] for m in ms) else None,
                            rms_dbeta=float(np.mean([m['rms_dbeta'] for m in ms if not m['failed']])) if any(not m['failed'] for m in ms) else None,
                            pace=float(np.mean([m['pace'] for m in ms if not m['failed']])) if any(not m['failed'] for m in ms) else None,
                            mean_abs_beta=float(np.mean([m['mean_abs_beta'] for m in ms if not m['failed']])) if any(not m['failed'] for m in ms) else None)
                    for n, ms in d.items()} for c, d in ev.items()}
        results['budgets'][str(budget)] = summ
        json.dump(results, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
        print('BUDGET', budget, json.dumps({c: {n[6:]: round(v['success'], 2) for n, v in d.items()} for c, d in summ.items()}),
              flush=True)


if __name__ == '__main__':
    main()
