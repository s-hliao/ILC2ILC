#!/usr/bin/env python3
"""offset_score_1122.py RES.npz [RES.npz ...] (2026-11-22, written from matcha; RUNBOOK_1122.md): the CONSISTENT lateral offset of a
5-lap run, as the 11-22 sim screens scored it (laps 1-4 by plan progress; lap 0 = the launch lap):
  repeatable = rms over 8 track segments of the lap-averaged offset; worst = largest |lap-averaged segment offset|;
  scatter = mean lap-to-lap std per segment (small vs repeatable = consistent); peak = max |e_y|; mean = signed mean offset;
  pace = mean |V| / mean |V_plan| after the hand-over; PACE = the pace rounded DOWN to 0.05 (the stage's --pace-target, as screened).
Uses the plan file of each run (plans_nom06). Prints one block per run.
    python3 ~/car_runs/offset_score_1122.py /home/kathy/car_sessions/1122_rubber/res_00002.npz
"""
import json, os, sys
import numpy as np

PLANS = os.path.expanduser('~/ros2_ws/src/ilc_f1tenth/ilc2real/plans_nom06')


def score(f):
    z = np.load(f, allow_pickle=True)
    plan = str(z['plan']); P = np.load(os.path.join(PLANS, plan + '.npz'), allow_pickle=True)
    M = len(P['x']); idx = np.asarray(z['idx']).astype(int)
    ey = np.asarray(z['err_true'])[:, 0]; x = np.asarray(z['x_true'])
    v = np.hypot(x[:, 3], x[:, 4]); zz = np.asarray(P['z'])[idx]; vp = np.hypot(zz[:, 2], zz[:, 3])
    st = np.asarray(z['applied'])[:, 0]
    kh = int(z['handover']) if np.asarray(z['handover']).shape == () and z['handover'] is not None else 0
    prog = np.r_[0, np.cumsum(((np.diff(idx) + M // 2) % M - M // 2) / M)]
    lap = np.floor(prog + 1e-9).astype(int)
    L = [k for k in range(1, 5) if np.any(lap == k)]
    m = (lap >= 1) & (lap <= 4)
    b = idx[m] * 8 // M
    B = np.full((len(L), 8), np.nan)
    for i, k in enumerate(L):
        for j in range(8):
            s = (lap[m] == k) & (b == j)
            if s.any():
                B[i, j] = 100 * ey[m][s].mean()
    seg = np.nanmean(B, 0)
    post = np.arange(len(ey)) >= kh
    pace = float(np.mean(v[post]) / np.mean(vp[post]))
    return dict(run=f, plan=plan, laps=float(prog[-1]), lap_rms=[round(100 * float(np.sqrt(np.mean(ey[lap == k] ** 2))), 1) for k in L],
                lap_mean=[round(100 * float(np.mean(ey[lap == k])), 1) for k in L], seg_mean=[round(float(s), 1) for s in seg],
                repeatable=round(float(np.sqrt(np.nanmean(seg ** 2))), 1), worst=round(float(np.nanmax(np.abs(seg))), 1),
                scatter=round(float(np.nanmean(np.nanstd(B, 0))), 1), peak=round(100 * float(np.max(np.abs(ey))), 1),
                mean=round(100 * float(np.mean(ey[m])), 1), pace=round(pace, 3), PACE=float(np.floor(pace * 20 + 1e-9) / 20),
                steer_sat=round(float(np.mean(np.abs(st[m]) >= 0.335)), 2))


for f in sys.argv[1:]:
    r = score(f)
    print(f"{os.path.basename(f)}  {r['plan']}  laps {r['laps']:.2f}")
    print(f"  CONSISTENT offset: repeatable {r['repeatable']} cm, worst segment {r['worst']} cm, lap scatter {r['scatter']} cm"
          f" | peak {r['peak']} cm | mean {r['mean']:+} cm | pace {r['pace']} -> PACE={r['PACE']:g} | steering at limit {r['steer_sat']}")
    print(f"  per lap rms {r['lap_rms']}  mean {r['lap_mean']}")
    print(f"  segments (lap-averaged, cm): {r['seg_mean']}")
    print('  ' + json.dumps({k: r[k] for k in ('repeatable', 'worst', 'scatter', 'peak', 'mean', 'pace', 'PACE')}))
