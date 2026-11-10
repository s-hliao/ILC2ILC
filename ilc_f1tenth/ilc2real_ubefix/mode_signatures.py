#!/usr/bin/env python3
"""
mode_signatures.py REC.npz [...]: the mode-cell analysis of a recorded car hardware stage (src/paper/theory: the
margin lemma), from hw_stage --record files that hold the full true state (record_car.save: x9, applied).
The car has no contact events; its pieces are the Fiala tire regimes and the actuator clips. Per control period:
  front / rear axle SLIDING: |tan alpha| >= tan alpha_sl = 3 mu Fz / C (the nominal Fiala model's sliding limit, static
                             axle loads), alpha from the planar state (vx, vy, r, steering)
  steering / current at their limits (the applied commands)
The signature of a trial is the regime sequence along the lap: the flags binned by plan arc length (NB bins per lap,
majority per bin) and run-length encoded. Consecutive trials of a car at the same goal (plan) are compared:
  topological change  a different run-length-encoded sequence of regimes (a slide that appears / disappears)
  piece change        any bin's flags differ
and the margin: |1 - |tan alpha| / tan alpha_sl| per axle (0 = on the sliding limit), its median and the fraction of
periods within 10 % of the limit. -> REC_modes.json; prints the rates.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import car_model as cm                                  # noqa: E402

NB = 100
N0 = cm.NOMINAL
G = 9.81


def regimes(x9, app):
    m, lf, lr = N0['m'], N0['lf'], N0['lr']
    fzf, fzr = m * G * lr / (lf + lr), m * G * lf / (lf + lr)
    tsl_f = 3 * N0['muf'] * fzf / N0['Cf']
    tsl_r = 3 * N0['mur'] * fzr / N0['Cr']
    vx = np.maximum(x9[:, 3], 0.1)
    vy, r, d = x9[:, 4], x9[:, 5], x9[:, 8]
    taf = np.abs(np.tan(d - np.arctan2(vy + lf * r, vx)))
    tar = np.abs(np.tan(np.arctan2(vy - lr * r, vx)))
    flags = np.stack([taf >= tsl_f, tar >= tsl_r, np.abs(app[:, 0]) >= N0['s_max'] - 1e-3,
                      (app[:, 1] <= N0['I_min'] + 1e-3) | (app[:, 1] >= N0['I_max'] - 1e-3)], 1)
    margin = np.stack([np.abs(1 - taf / tsl_f), np.abs(1 - tar / tsl_r)], 1)
    return flags, margin


def signature(flags, idx, M):
    b = (np.asarray(idx) * NB // M).astype(int) % NB
    lap = np.cumsum(np.r_[0, np.diff(b) < -NB // 2])                 # lap counter (the index wraps)
    key = lap * NB + b
    ks = np.unique(key)
    binned = np.array([flags[key == k].mean(0) >= 0.5 for k in ks])
    codes = binned @ (1 << np.arange(flags.shape[1]))
    rle = [int(c) for i, c in enumerate(codes) if i == 0 or c != codes[i - 1]]
    return ks, codes, rle


def main():
    for path in sys.argv[1:]:
        z = np.load(path, allow_pickle=True)
        assert 'x9' in z.files, 'record without the full state: rerun hw_stage --record after the record_car patch'
        meta = json.loads(str(z['meta']))
        x9, app, idx, ns = z['x9'], z['applied'], z['idx'], z['n_steps']
        M = {k[len('plan__'):-len('__xy')]: len(z[k]) for k in z.files if k.startswith('plan__') and k.endswith('__xy')}
        sig, rows, marg, slide = {}, [], [], []
        for i, m in enumerate(meta):
            L = int(ns[i])
            fl, mg = regimes(x9[i, :L], app[i, :L])
            marg.append(mg)
            slide.append(fl[:, :2].mean(0))
            ks, codes, rle = signature(fl, idx[i, :L], M[m['plan']])
            key = (m['car'], m.get('goal', m['plan']))
            if key in sig:
                k0, c0, r0, f0 = sig[key]
                common = np.intersect1d(k0, ks)
                same_bins = bool(np.array_equal(c0[np.isin(k0, common)], codes[np.isin(ks, common)]))
                topo = r0 == rle and f0 == bool(m.get('failed'))
                rows.append(dict(car=m['car'], plan=m['plan'], it=m.get('it'), topo_change=not topo,
                                 piece_change=not (topo and same_bins), n_regimes=len(rle), n_regimes_prev=len(r0),
                                 failed=bool(m.get('failed'))))
            sig[key] = (ks, codes, rle, bool(m.get('failed')))
        mg = np.concatenate(marg)
        res = dict(file=path, trials=len(meta), pairs=len(rows),
                   topo_change=float(np.mean([r['topo_change'] for r in rows])) if rows else None,
                   piece_change=float(np.mean([r['piece_change'] for r in rows])) if rows else None,
                   regimes_per_trial_median=float(np.median([r['n_regimes'] for r in rows])) if rows else None,
                   slide_frac_front=float(np.mean([s[0] for s in slide])), slide_frac_rear=float(np.mean([s[1] for s in slide])),
                   margin_median_front=float(np.median(mg[:, 0])), margin_median_rear=float(np.median(mg[:, 1])),
                   near_limit_frac_front=float(np.mean(mg[:, 0] < 0.1)), near_limit_frac_rear=float(np.mean(mg[:, 1] < 0.1)))
        json.dump(dict(summary=res, rows=rows), open(os.path.splitext(path)[0] + '_modes.json', 'w'), indent=1)
        print(json.dumps(res))


if __name__ == '__main__':
    main()
