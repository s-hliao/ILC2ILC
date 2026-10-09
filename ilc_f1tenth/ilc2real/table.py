#!/usr/bin/env python3
"""table.py [ARM ...]: the results table (success = on track, RMS e_y <= 10 cm, pace >= 90 % of the plan's -- grip or
drift alike; |beta| the mean sideslip actually flown) over hardware-stage / evaluation arms (runs/hw/<arm>/summary.json, and
runs/trackilc for the per-track ILC baseline). Columns: success rate (%) and failure (spin-out / departure) rate over
cars x runs on the full-drift plans (beta 25, both tracks: the end product), on the held-out drift plans (beta 14,
21: never trained on, in sim or on the cars), and on grip (beta 0); mean RMS lateral error (cm) and sideslip error
(deg) of the completed drift runs; real laps used per car."""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = dict(drift=['mocap_square2fast_b25', 'mocap_figfast_b25'],
              heldout=['mocap_square2fast_b21', 'mocap_figfast_b21', 'mocap_square2fast_b14', 'mocap_figfast_b14'],
              grip=['mocap_square2fast_b0', 'mocap_figfast_b0'])


def agg(ev, names, cars=None):
    s, f, ey, pc, bt = [], [], [], [], []
    for car, d in ev.items():
        if cars and car not in cars:
            continue
        for n in names:
            if n not in d:
                continue
            v = d[n]
            s.append(v['success']); f.append(v['fail'])
            if v.get('rms_ey') is not None:
                ey.append(v['rms_ey'])
                if v.get('pace') is not None:
                    pc.append(v['pace'])
                if v.get('mean_abs_beta') is not None:
                    bt.append(v['mean_abs_beta'])
    m = lambda x: float(np.mean(x)) if x else float('nan')
    return m(s) * 100, m(f) * 100, m(ey) * 100, m(pc) * 100, m(bt)


def row(name, ev, laps=None):
    out = [name]
    for g in ('drift', 'heldout', 'grip'):
        s, f, ey, pc, bt = agg(ev, GROUPS[g])
        out += [f'{s:5.1f}', f'{f:5.1f}'] + ([f'{ey:5.1f}', f'{pc:5.1f}'] if g == 'drift' else []) + [f'{bt:5.1f}']
    out.append('' if laps is None else f'{laps:.0f}')
    return out


BIG = '--big' in sys.argv
if BIG:
    sys.argv.remove('--big')


def main():
    arms = sys.argv[1:] or sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(HERE, 'runs/hw/*/summary.json')))
    hdr = ['arm', 'b25 ok%', 'fail%', 'e_y cm', 'pace%', '|beta|', 'held-out ok%', 'fail%', '|beta|', 'grip ok%', 'fail%',
           '|beta|', 'real laps/car']
    rows = []
    for arm in arms:
        f = os.path.join(HERE, 'runs/hw', arm, 'summary.json')
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        laps = np.mean(list(d['real_laps'].values())) if d.get('real_laps') else 0.0
        big = os.path.join(HERE, 'runs/hw', arm, 'eval_big.json')
        ada = os.path.join(HERE, 'runs/hw', arm, 'eval_big_adaptive.json')
        if BIG and os.path.exists(ada):
            rows.append(row(arm + ' [x12, adaptive goals]', json.load(open(ada)), laps))
        elif BIG and os.path.exists(big):
            rows.append(row(arm + ' [x12]', json.load(open(big)), laps))
        elif 'eval' in d:
            rows.append(row(arm, d['eval'], laps))
    f = os.path.join(HERE, 'runs/trackilc_big/summary.json' if BIG else 'runs/trackilc/summary.json')
    if os.path.exists(f):
        d = json.load(open(f))
        for b, ev in d['budgets'].items():
            rows.append(row(f'track_ilc ({b} laps per plan)', ev, int(b) * 8))
    w = [max(len(r[i]) for r in rows + [hdr]) for i in range(len(hdr))]
    line = lambda r: ' | '.join(x.rjust(w[i]) if i else x.ljust(w[i]) for i, x in enumerate(r))
    print(line(hdr))
    print('-+-'.join('-' * x for x in w))
    for r in rows:
        print(line(r))


if __name__ == '__main__':
    main()
