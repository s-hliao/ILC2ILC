#!/usr/bin/env python3
"""scale_plan_var.py NAME --k0 K0 --dk DK --tag TAG (2026-11-01): scale_plan.py with a time scale that varies smoothly along the lap,
k(s) = K0 + DK sin(2 pi s / L) (one plan, one start line; the speed varies along the lap instead of between plans). The same
approximation as scale_plan.py: path, e_y, e_psi and the steering stay; vx, vy, r, omega_w and the current scale by k(s), the time
step by 1 / k(s); no new solve and no friction model."""
import argparse, json, os
import numpy as np
PLAN_DIR = os.environ.get('F1T_PLANS') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plans')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name'); ap.add_argument('--k0', type=float, required=True); ap.add_argument('--dk', type=float, required=True)
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    d = dict(np.load(os.path.join(PLAN_DIR, f'{a.name}.npz'), allow_pickle=True))
    for f in ('z', 'x', 'Z', 'Zc', 'U', 'u', 't', 's'):
        d[f] = np.array(d[f], float)
    M, N = len(d['s']), int(d['N'])
    L = float(d['length'])
    kf = a.k0 + a.dk * np.sin(2 * np.pi * d['s'] / L)             # fine grid
    assert M % N == 0, (M, N)
    kc = kf[::M // N][:N]                                          # coarse nodes (M / N fine samples each)
    d['z'][:, 2:7] *= kf[:, None]
    d['x'][:, 3:8] *= kf[:, None]
    d['u'][:, 1] *= kf
    d['Z'][2:7] *= kc[None, :]
    d['Zc'][:, 2:7] *= kc[:, None, None]
    d['U'][1] *= kc
    dt = np.diff(np.r_[d['t'], float(d['lap_time'])])              # the old step to the next sample (the last one closes the lap)
    dtn = dt / kf
    d['t'] = np.r_[0.0, np.cumsum(dtn)[:-1]]
    d['lap_time'] = float(np.sum(dtn))
    info = json.loads(str(d['info']))
    d['info'] = json.dumps(dict(info, time_scale=f'{a.k0}+{a.dk}sin', scaled_from=a.name, lap_time=d['lap_time']))
    out = os.path.join(PLAN_DIR, f'{a.name}{a.tag}.npz')
    np.savez(out, **d)
    v = np.hypot(d['z'][:, 2], d['z'][:, 3])
    print(f'{out}: lap {d["lap_time"]:.2f} s, v {v.min():.2f}..{v.max():.2f} m/s, k {kf.min():.3f}..{kf.max():.3f}')


if __name__ == '__main__':
    main()
