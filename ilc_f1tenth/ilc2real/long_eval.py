#!/usr/bin/env python3
"""
long_eval.py ARM [--laps 20]: the "infinite drift" check -- each car's network from a hardware-stage arm
(runs/hw/ARM/<car>_policy.npz; zero-shot arms: the arm's starting policy) flies --laps consecutive laps of the
full-drift plans without a reset, from the plan's start, with sensing noise. Reports per car and plan: laps completed,
RMS lateral / sideslip error per lap (first, middle, last), and whether the error grows over laps (slope of the
per-lap RMS e_y, cm per 10 laps). -> runs/hw/ARM/long_eval.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bank                                      # noqa: E402
import real_car as rc                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    if path == 'lqr':
        return None
    d = np.load(path)
    n = len([k for k in d.files if k.startswith('W')])
    return [(d[f'W{i}'], d[f'b{i}']) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arm')
    ap.add_argument('--laps', type=float, default=20.0)
    ap.add_argument('--plans', nargs='+', default=['mocap_square2fast_b25', 'mocap_figfast_b25'])
    ap.add_argument('--seed', type=int, default=777)
    a = ap.parse_args()
    d = os.path.join(HERE, 'runs/hw', a.arm)
    args = json.load(open(os.path.join(d, 'args.json')))
    plans = bank.load_bank(a.plans)
    jobs = []
    for car in args['cars']:
        f = os.path.join(d, f'{car}_policy.npz')
        pol = load(f) if os.path.exists(f) else load(args['policy'])
        for pl in plans:
            jobs.append(dict(car_name=car, plan=pl, policy=pol, laps=a.laps, s0=0.0, pert=np.zeros(5), seed=a.seed))
    res = rc.run_many(jobs)
    out = {}
    for j, r in zip(jobs, res):
        m = rc.lap_metrics(r, j['plan'])
        pe = np.array(m.get('per_lap_ey') or [np.nan]) * 100
        slope = float(np.polyfit(np.arange(len(pe)), pe, 1)[0] * 10) if len(pe) > 2 else float('nan')
        out.setdefault(j['car_name'], {})[j['plan']['name']] = dict(
            laps=round(m['laps'], 2), failed=m['failed'], rms_ey_cm=round(100 * m.get('rms_ey', np.nan), 2),
            rms_dbeta=round(m.get('rms_dbeta', np.nan), 2), mean_abs_beta=round(m.get('mean_abs_beta', np.nan), 1),
            lap_ey_first_mid_last=[round(float(pe[0]), 2), round(float(pe[len(pe) // 2]), 2), round(float(pe[-1]), 2)],
            ey_slope_cm_per_10laps=round(slope, 3))
    json.dump(out, open(os.path.join(d, 'long_eval.json'), 'w'), indent=1)
    for car, v in out.items():
        for p, m in v.items():
            print(f'{car:10s} {p:24s}', m)


if __name__ == '__main__':
    main()
