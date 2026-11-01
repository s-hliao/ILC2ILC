#!/usr/bin/env python3
"""scale_plan.py NAME --k K --tag TAG (2026-10-31; user: "it's ok to plan faster for rubber, just not in a friction-aware
sense"): the same plan flown K times faster -- a uniform time scaling of a solved plan, no new solve and no friction model.
Path, e_y, e_psi and the steering stay; vx, vy, r and omega_w scale by K, time by 1 / K (lateral acceleration by K^2). The
motor current scales by K (an approximation: it is only the plan's feed-forward, not the networks' reference). The plan may
exceed the tire's grip: the car does what it can, and pace is scored against this plan.
    python scale_plan.py mocap_square2fast_b0_free --k 1.3 --tag _x130   -> $F1T_PLANS/mocap_square2fast_b0_free_x130.npz
"""
import argparse
import json
import os

import numpy as np

PLAN_DIR = os.environ.get('F1T_PLANS') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plans')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--k', type=float, required=True)
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    k = a.k
    d = dict(np.load(os.path.join(PLAN_DIR, f'{a.name}.npz'), allow_pickle=True))
    for f in ('z', 'x', 'Z', 'Zc', 'U', 'u', 't'):
        d[f] = np.array(d[f], float)
    d['z'][:, 2:7] *= k                       # vx, vy, r, omega_w, I
    d['x'][:, 3:8] *= k
    d['Z'][2:7] *= k
    d['Zc'][:, 2:7] *= k
    d['U'][1] *= k
    d['u'][:, 1] *= k
    d['t'] /= k
    d['lap_time'] = float(d['lap_time']) / k
    info = json.loads(str(d['info']))
    d['info'] = json.dumps(dict(info, time_scale=k, scaled_from=a.name, lap_time=d['lap_time']))
    out = os.path.join(PLAN_DIR, f'{a.name}{a.tag}.npz')
    np.savez(out, **d)
    v = np.hypot(d['z'][:, 2], d['z'][:, 3])
    print(f'{out}: lap {d["lap_time"]:.2f} s, v {v.min():.2f}..{v.max():.2f} m/s')


if __name__ == '__main__':
    main()
