#!/usr/bin/env python3
"""
eval_axes.py --tag T --cars C ... --plans P ... --net NAME=SPEC ...: the car's perturbation suite (the quadruped's
per-axis evaluation): networks on perturbed cars / starts / tracks, reeval.py's protocol (spread starts with a
perturbation, chained laps, reserved seeds) -> runs/axes/T.json {net: {car: {plan: metrics}}}.

  cars   real cars, single-axis cars "ax:key=value" (mb_car.car_config) and sensing suffixes "+mocapbad" etc.
  SPEC   zs:<policy npz>      a sim network (zero-shot); rma_s0's teacher via reeval.load (its adaptation module)
         arm:<runs/hw/ARM>    the arm's network adapted on the test car's BASE car (its name before "+"; "ax:" cars:
                              real_nom's) -- the adapted network under a perturbation it was not adapted under
         file:<policy npz>    one fixed network for every car (a network adapted under a perturbation: the trained x
                              tested matrix)
         lqr                  the plans' LQR
  --launch v0 v_hand current  a slow rolling start (real_car.run launch) from s = 0, scored from the 2nd lap on
"""
import argparse
import json
import os
import sys

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np                               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bank                                      # noqa: E402
import real_car as rc                            # noqa: E402
import reeval                                    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True)
ap.add_argument('--cars', nargs='+', required=True)
ap.add_argument('--plans', nargs='+', default=[f'mocap_{t}_b{b}' for t in ('square2fast', 'figfast') for b in (25, 21, 14, 0)])
ap.add_argument('--net', nargs='+', required=True, metavar='NAME=SPEC')
ap.add_argument('--runs', type=int, default=6)
ap.add_argument('--laps', type=float, default=5.0)
ap.add_argument('--seed', type=int, default=1701)
ap.add_argument('--launch', type=float, nargs=3, default=None, metavar=('V0', 'V_HAND', 'CURRENT'))
a = ap.parse_args()


def base_car(car):
    b = car.split('+')[0]
    return 'real_nom' if b.startswith('ax:') else b


def policy_for(spec, car):
    if spec == 'lqr':
        return None
    kind, path = spec.split(':', 1)
    path = path if os.path.isabs(path) else os.path.join(HERE, path)
    if kind == 'arm':
        return reeval.load(os.path.join(path, f'{base_car(car)}_policy.npz'))
    return reeval.load(path)


def tail(r, k0):
    """The run from step k0 on (a rolling start's 2nd lap on), for lap_metrics."""
    if len(r['x_true']) <= k0 + 2:
        return r
    out = dict(r)
    for k in ('x', 'x_true', 'err', 'err_true', 'idx', 'obs', 'mu', 'a'):
        out[k] = np.asarray(r[k])[k0:]
    return out


def main():
    plans = {p['name']: p for p in bank.load_bank(a.plans)}
    nets = dict(n.split('=', 1) for n in a.net)
    launch = dict(v0=a.launch[0], v_hand=a.launch[1], current=a.launch[2]) if a.launch else None
    jobs, keys = [], []
    for name, spec in nets.items():
        for car in a.cars:
            pol = policy_for(spec, car)
            for pn, pl in plans.items():
                rs = np.random.default_rng(a.seed + a.plans.index(pn))
                for k in range(a.runs):
                    j = dict(car_name=car, plan=pl, policy=pol, laps=a.laps, seed=a.seed * 3 + k)
                    if launch:
                        j.update(s0=(k / a.runs) * float(pl['length']), pert=np.zeros(5), launch=launch)
                    else:
                        j.update(s0=(k / a.runs) * float(pl['length']), pert=rs.uniform(-0.5, 0.5, 5))
                    jobs.append(j)
                    keys.append((name, car, pn))
    res = rc.run_many(jobs)
    out = {}
    for (name, car, pn), r, j in zip(keys, res, jobs):
        k0 = int(round(float(j['plan']['lap_time']) / rc.DT)) if launch else 0
        m = rc.lap_metrics(tail(r, k0) if launch else r, j['plan'])
        m['crash'] = r.get('crash', '')
        out.setdefault(name, {}).setdefault(car, {}).setdefault(pn, []).append(m)
    summ = {}
    for name, d in out.items():
        for car, dd in d.items():
            for pn, ms in dd.items():
                ok = [m for m in ms if not m['failed']]
                mean = lambda k: float(np.mean([m[k] for m in ok])) if ok else None
                summ.setdefault(name, {}).setdefault(car, {})[pn] = dict(
                    success=float(np.mean([m['success'] for m in ms])), fail=float(np.mean([m['failed'] for m in ms])),
                    wall=float(np.mean([m.get('crash') == 'wall' for m in ms])), rms_ey=mean('rms_ey'), pace=mean('pace'),
                    mean_abs_beta=mean('mean_abs_beta'), n=len(ms))
    os.makedirs(os.path.join(HERE, 'runs/axes'), exist_ok=True)
    json.dump(dict(args=vars(a), eval=summ), open(os.path.join(HERE, 'runs/axes', f'{a.tag}.json'), 'w'), indent=1)
    for name, d in summ.items():
        B = [p for p in a.plans if p.endswith('_b25')] or a.plans
        print(a.tag, name, {c: round(100 * np.mean([v[p]['success'] for p in B if p in v]), 0) for c, v in d.items()}, flush=True)


if __name__ == '__main__':
    main()
