"""Which slots can the nominal car drift into? A trajectory-optimization sweep over slot offsets from the release state.

Slots (dx, dy, dpsi) relative to the release state on the left-turning drift circle (R0, beta0); each solved by
ilc_park.trajopt.solve. Writes the table (success, maneuver time, peak sideslip, braking) to --out (json), which
defines the training slot distribution and the held-out slots.

    F1T_MU=0.6 python scripts/feasibility.py --out results/feasibility_R1.5_b15.json
"""
import argparse
import itertools
import json
import multiprocessing
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def one(args):
    R0, beta0, dx, dy, dpsi = args
    from ilc_park import car_model as cm
    from ilc_park.trajopt import solve, start_state
    p = dict(cm.NOMINAL)
    x0, _ = start_state(R0, beta0, p)
    _, info = solve(x0, (dx, dy, np.radians(dpsi)), p, N=80, T_max=4.0)
    return dict(R0=R0, beta0=beta0, dx=dx, dy=dy, dpsi=dpsi, **{k: info[k] for k in
                ('success', 'T', 'beta_max', 'beta_fast_min', 'end_err_xy', 'end_v', 'I_range', 'seconds')})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--R0', type=float, default=1.5)
    ap.add_argument('--beta0', type=float, default=-15.0)
    ap.add_argument('--dx', type=float, nargs='+', default=[-0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument('--dy', type=float, nargs='+', default=[-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ap.add_argument('--dpsi', type=float, nargs='+', default=[0, 45, 90, 135, 180, -45, -90])
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    jobs = [(a.R0, a.beta0, dx, dy, dpsi) for dx, dy, dpsi in itertools.product(a.dx, a.dy, a.dpsi)]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    rows = []
    with multiprocessing.get_context('spawn').Pool(a.workers) as pool:
        for row in pool.imap_unordered(one, jobs):          # each solve capped at 120 s CPU (trajopt max_cpu_s)
            rows.append(row)
            json.dump(rows, open(a.out, 'w'), indent=1)    # partial results survive an interruption
            print(f'{len(rows)}/{len(jobs)}', flush=True) if len(rows) % 25 == 0 else None
    ok = [r for r in rows if r['success']]
    print(f'{len(ok)}/{len(rows)} slots feasible; maneuver time {np.median([r["T"] for r in ok]):.2f} s median')
    for dpsi in a.dpsi:
        grid = np.full((len(a.dy), len(a.dx)), '.', dtype=object)
        for r in rows:
            if r['dpsi'] == dpsi:
                grid[a.dy.index(r['dy']), a.dx.index(r['dx'])] = f"{r['T']:.1f}" if r['success'] else 'x'
        print(f'dpsi {dpsi:+.0f} deg (rows dy {a.dy[::-1]} top->bottom, cols dx {a.dx}):')
        for row in grid[::-1]:
            print('   ' + ' '.join(f'{c:>4s}' for c in row))


if __name__ == '__main__':
    main()
