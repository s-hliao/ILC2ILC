#!/usr/bin/env python3
"""
mode_signatures.py ARM_DIR [...]: the mode-cell analysis of a quadruped hardware stage (src/paper/theory: the
margin lemma), from the saved training jumps ARM/<robot>/it*/real/*.npz (the CPU robots' true contact flags per tick).
Per jump the mode signature:
  events   per foot, every contact change (tick -> its control sample; 1 touchdown / 0 lift-off), in order
  landing  whether the first touchdown after the push-off comes before the landing sample N (the model's ballistic
           flight to N assumes after)
  sat      the samples where a leg pair has no torque headroom (the Go1 envelope, as deploy.headroom)
  fell
Consecutive jumps of a robot at the same goal (iteration k vs k+1) are compared:
  topological change  a different sequence of event types per foot, or a fall changed
  piece change        also any event in a different control sample, or a different saturation pattern
and the time margin of each jump: the smallest distance of an event from a control-sample boundary (in samples).
Prints per arm the rates and writes ARM/mode_signatures.json.
"""
import glob
import json
import os
import re
import sys

import numpy as np

GO1_TAU = np.tile([23.7, 23.7, 35.55], 4)
GO1_W = np.tile([30.1, 30.1, 20.06], 4)
# planar leg pairs (front, rear) in the canonical joint order (FR FL RR RL x hip thigh calf)
PAIRS = [list(range(0, 6)), list(range(6, 12))]
DEB = 3                                                              # ticks
DT = 0.01                                                            # the control sample (the bank's dt)


def signature(path):
    z = np.load(path, allow_pickle=True)
    c = np.asarray(z['rec_true_contacts'], float) > 0.5           # (ticks, 4)
    for f in range(c.shape[1]):                                      # debounce: a contact state held < DEB ticks is flicker
        run_start = 0
        for k in range(1, len(c) + 1):
            if k == len(c) or c[k, f] != c[run_start, f]:
                if k - run_start < DEB and run_start > 0:
                    c[run_start:k, f] = c[run_start - 1, f]
                run_start = k
    N = len(z['logX']) - 1                                           # the landing sample
    Nc = len(z['act'])                                               # the policy's contact samples
    tick = float(np.median(np.diff(np.asarray(z['rec_t'], float))))
    tps = int(round(DT / tick))                                      # ticks per control sample (the record runs past N)
    ev = []
    for f in range(c.shape[1]):
        d = np.where(c[1:, f] != c[:-1, f])[0] + 1
        ev += [(int(k), f, int(c[k, f])) for k in d]
    ev.sort()
    events = [(int(k // tps), f, s) for k, f, s in ev]
    margins = [min((k % tps) / tps, 1.0 - (k % tps) / tps) for k, _, _ in ev if k < Nc * tps]
    push = [e for e in events if e[0] < Nc]                           # the policy-controlled push-off
    land = [e for e in events if Nc <= e[0] <= N]                     # flight and landing up to the error's sample
    # the first touchdown after every foot has left the ground (the landing)
    air = np.where(~c.any(1))[0]
    td = None
    if len(air):
        after = np.where(c[air[0]:].any(1))[0]
        td = (air[0] + after[0]) / tps if len(after) else None
    tau, dq = np.asarray(z['rec_tau_total'], float), np.asarray(z['rec_dq'], float)
    drop = np.clip((GO1_W - np.abs(dq)) / (0.5 * GO1_W), 0.0, 1.0)
    avail = np.where(tau * dq > 0, GO1_TAU * drop, GO1_TAU)
    ratio = np.abs(tau) / np.maximum(avail, 1e-3)
    k = (np.arange(len(ratio)) / tps).astype(int)
    sat = np.zeros((Nc, 2), bool)
    for kk in range(Nc):
        sel = k == kk
        if sel.any():
            for p in range(2):
                sat[kk, p] = np.median(ratio[sel][:, PAIRS[p]].max(1)) >= 1.0
    info = np.asarray(z['info'], float)
    return dict(events=events, push=push, land=land, types_push=[(f, s) for _, f, s in push],
                types_land=[(f, s) for _, f, s in land], types=[(f, s) for _, f, s in events], fell=bool(info[3] > 0.5),
                landing_early=bool(td is not None and td < N), td=td, N=N, sat=sat,
                t_margin=float(min(margins)) if margins else 0.5, n_events=len(events))


def analyse(arm):
    rows = []
    for rdir in sorted(glob.glob(os.path.join(arm, 'real_*'))):
        if not os.path.isdir(rdir):
            continue
        robot = os.path.basename(rdir)
        by_goal = {}
        for f in glob.glob(os.path.join(rdir, 'it*', 'real', '*.npz')):
            it = int(re.search(r'/it(\d+)/', f).group(1))
            g = re.search(r'_g([0-9.]+_[0-9.]+)_', os.path.basename(f)).group(1)
            by_goal.setdefault(g, []).append((it, f))
        for g, lst in by_goal.items():
            lst.sort()
            sigs = [(it, signature(f)) for it, f in lst]
            for (i0, s0), (i1, s1) in zip(sigs[:-1], sigs[1:]):
                topo = s0['types'] == s1['types'] and s0['fell'] == s1['fell']
                piece = topo and s0['events'] == s1['events'] and np.array_equal(s0['sat'], s1['sat'])
                pref = lambda a, b: a[:min(len(a), len(b))] == b[:min(len(a), len(b))]   # common prefix (as the pogo)
                tp = s0['types_push'] == s1['types_push']
                tl_pref = pref(s0['types_land'], s1['types_land']) and s0['fell'] == s1['fell']
                pp = tp and s0['push'] == s1['push'] and np.array_equal(s0['sat'], s1['sat'])
                tl = s0['types_land'] == s1['types_land'] and s0['fell'] == s1['fell']
                pl = tl and s0['land'] == s1['land']
                rows.append(dict(robot=robot, goal=g, it=i1, topo_change=not topo, piece_change=not piece,
                                 push_topo=not tp, push_piece=not pp, land_topo=not tl, land_piece=not pl,
                                 land_topo_prefix=not tl_pref, fell_change=s0['fell'] != s1['fell'],
                                 n_push=len(s1['push']), n_land=len(s1['land']),
                                 landing_switch=s0['landing_early'] != s1['landing_early'],
                                 landing_early=s1['landing_early'], fell=s1['fell'], t_margin=s1['t_margin'],
                                 sat_frac=float(s1['sat'].mean()), n_events=s1['n_events'],
                                 n_events_prev=s0['n_events']))
    return rows


def main():
    for arm in sys.argv[1:]:
        rows = analyse(arm)
        if not rows:
            print(arm, 'no jump pairs')
            continue
        m = lambda k: float(np.mean([r[k] for r in rows]))
        res = dict(arm=arm, pairs=len(rows), push_topo=m('push_topo'), push_piece=m('push_piece'),
                   land_topo=m('land_topo'), land_piece=m('land_piece'), land_topo_prefix=m('land_topo_prefix'),
                   fell_change=m('fell_change'),
                   n_push_median=float(np.median([r['n_push'] for r in rows])),
                   n_land_median=float(np.median([r['n_land'] for r in rows])),
                   topo_change=m('topo_change'), piece_change=m('piece_change'),
                   landing_switch=m('landing_switch'), landing_early=m('landing_early'), fell=m('fell'),
                   t_margin_median=float(np.median([r['t_margin'] for r in rows])), sat_frac=m('sat_frac'),
                   n_events_median=float(np.median([r['n_events'] for r in rows])))
        json.dump(dict(summary=res, rows=rows), open(os.path.join(arm, 'mode_signatures.json'), 'w'), indent=1)
        print(json.dumps(res))


if __name__ == '__main__':
    main()
