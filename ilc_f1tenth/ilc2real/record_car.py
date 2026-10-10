#!/usr/bin/env python3
"""
record_car.py --root DIR --eval NAME=ARM ... --hwstage NAME=ARM ...: recorded car trajectories for the videos and
trajectory figures (the quadruped's export_trajectories.py analog), -> DIR/trajectories/{eval,hwstage}_<NAME>.npz + .json.

  --eval NAME=ARM      the arm's per-car networks (runs/hw/ARM/<car>_policy.npz; a zero-shot arm: its starting policy;
                       lqr_zs: the plans' LQR) flown on the 8 evaluation plans on each car: one run of --laps chained
                       laps from the plan's start (seed 701, as animate_drift.py), the same for every method
  --hwstage NAME=ARM   the arm's hardware stage flown again with its own arguments (same seeds) and every trial lap
                       recorded (hw_stage.py --record), into DIR/runs/rec/ARM (the arm itself is left as it is)

Each npz: x (n, T, 6) the TRUE planar state X Y psi vx vy r, idx (n, T) the plan index, a (n, T, 2) the normalized
action, n_steps (n,), meta (JSON list: car, plan, it (-1 evaluation), failed, crash, laps, rms_ey, pace, mean |beta|),
and per plan its geometry: plan__<name>__xy (M, 2), plan__<name>__beta (M,) the planned sideslip (deg),
plan__<name>__lap_time. The environment (F1T_MU, F1T_PLANS, F1T_TRACKS, F1T_SAFETY_EY) must be the run's.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CARS = ['real_nom', 'real_mass', 'real_mu', 'real_act', 'real_lag']
EVAL = [f'mocap_{t}_b{b}' for t in ('square2fast', 'figfast') for b in (25, 21, 14, 0)]


def save(path, recs, plans):
    """recs: [(meta dict, real_car.run result)], plans: the plan dicts they flew."""
    import real_car as rc
    by = {p['name']: p for p in plans}
    T = max(len(r['x_true']) for _, r in recs)
    n = len(recs)
    x = np.full((n, T, 6), np.nan, np.float32)
    idx = np.zeros((n, T), np.int32)
    act = np.zeros((n, T, 2), np.float32)
    ns = np.zeros(n, np.int32)
    meta = []
    for k, (m, r) in enumerate(recs):
        L = len(r['x_true'])
        ns[k] = L
        x[k, :L] = np.asarray(r['x_true'])[:, :6]
        x[k, L:] = x[k, L - 1]
        idx[k, :L] = r['idx']
        idx[k, L:] = r['idx'][-1]
        act[k, :L] = r['a']
        mm = rc.lap_metrics(r, by[m['plan']])
        meta.append(dict(m, rms_ey=mm.get('rms_ey'), pace=mm.get('pace'), mean_abs_beta=mm.get('mean_abs_beta'),
                         success=mm.get('success'), peak_ey=r.get('peak_ey'), n_steps=int(L)))
    geo = {}
    for name, p in by.items():
        geo[f'plan__{name}__xy'] = np.asarray(p['xy'], np.float32)
        geo[f'plan__{name}__beta'] = np.degrees(np.arctan2(p['z'][:, 3], p['z'][:, 2])).astype(np.float32)
        geo[f'plan__{name}__lap_time'] = np.float32(p['lap_time'])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, x=x, idx=idx, a=act, n_steps=ns, dt=np.float32(rc.DT), meta=np.array(json.dumps(meta)), **geo)
    json.dump(dict(file=os.path.basename(path), n=n, T=int(T), dt=rc.DT, plans=list(by),
                   env={k: os.environ.get(k) for k in ('F1T_MU', 'F1T_PLANS', 'F1T_TRACKS', 'F1T_SAFETY_EY')},
                   runs=meta), open(os.path.splitext(path)[0] + '.json', 'w'), indent=1)
    print(f'{path}: {n} runs x {T} steps ({os.path.getsize(path) / 1e6:.1f} MB)', flush=True)


def load(path):
    z = np.load(path)
    T = {k: z[k] for k in ('x', 'idx', 'a', 'n_steps')}
    T['dt'] = float(z['dt'])
    T['meta'] = json.loads(str(z['meta']))
    T['plans'] = {}
    for k in z.files:
        if k.startswith('plan__') and k.endswith('__xy'):
            n = k[len('plan__'):-len('__xy')]
            T['plans'][n] = dict(xy=z[k], beta=z[f'plan__{n}__beta'], lap_time=float(z[f'plan__{n}__lap_time']))
    T['name'] = os.path.splitext(os.path.basename(path))[0]
    return T


def record_eval(root, name, arm, laps):
    import bank
    import real_car as rc
    import reeval
    d = os.path.join(root, 'runs/hw', arm)
    args = json.load(open(os.path.join(d, 'args.json')))
    plans = {p['name']: p for p in bank.load_bank(EVAL)}
    jobs, metas = [], []
    for car in args.get('cars', CARS):
        f = os.path.join(d, f'{car}_policy.npz')
        pol_path = f if os.path.exists(f) else (args['policy'] if args['policy'] == 'lqr' else
                                                os.path.join(root, args['policy']))
        pol = reeval.load(pol_path)
        for pn, pl in plans.items():
            jobs.append(dict(car_name=car, plan=pl, policy=pol, laps=laps, s0=0.0, pert=np.zeros(5), seed=701))
            metas.append(dict(car=car, plan=pn, it=-1, arm=arm))
    res = rc.run_many(jobs)
    recs = [(dict(m, failed=bool(r['failed']), crash=r.get('crash', ''), laps=float(r['laps'])), r)
            for m, r in zip(metas, res)]
    save(os.path.join(root, 'trajectories', f'eval_{name}.npz'), recs, list(plans.values()))


def record_hwstage(root, name, arm):
    d = os.path.join(root, 'runs/hw', arm)
    args = json.load(open(os.path.join(d, 'args.json')))
    out = os.path.join(root, 'runs/rec', arm)
    cmd = [sys.executable, '-u', os.path.join(HERE, 'hw_stage.py'), '--policy', os.path.join(root, args['policy']),
           '--out', out, '--record', os.path.join(root, 'trajectories', f'hwstage_{name}.npz'),
           '--iters', str(args['iters']), '--goals', *args['goals'], '--eval-runs', '1', '--eval-laps', '1',
           '--crash-rollback', str(args.get('crash_rollback', 0)), '--trial-laps', str(args.get('trial_laps', 1))]
    for k in ('beta', 'delta', 'cap', 'anchor_w', 'anchor_n0', 'age_decay', 'steps', 'lr', 'rollback', 'seed',
              'train_starts', 'objective'):
        if k in args and args[k] is not None:
            cmd += [f'--{k.replace("_", "-")}', str(args[k])]
    if args.get('jac'):
        cmd += ['--jac', args['jac']]
    print(' '.join(cmd[2:6]), '...', flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=HERE)
    ap.add_argument('--eval', nargs='*', default=[], metavar='NAME=ARM')
    ap.add_argument('--hwstage', nargs='*', default=[], metavar='NAME=ARM')
    ap.add_argument('--laps', type=float, default=3.0)
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    for e in a.eval:
        n, arm = e.split('=')
        record_eval(root, n, arm, a.laps)
    for e in a.hwstage:
        n, arm = e.split('=')
        record_hwstage(root, n, arm)
