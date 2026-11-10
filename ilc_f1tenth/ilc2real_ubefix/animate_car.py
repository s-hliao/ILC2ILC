#!/usr/bin/env python3
"""
animate_car.py MODE TRAJ.npz ... --car CAR --out FILE.mp4: top-down videos of recorded car runs (record_car.py), the
quadruped's animate_hw_stage.py for the car. The track's plan in grey, the car's path coloured by |sideslip| (0..35
deg), the car (heading) with its velocity arrow (the angle between them is the sideslip).

  hwstage  HW.npz --car C               the hardware stage iteration by iteration: one panel per training plan, the
                                        earlier iterations' paths faded (light = earliest)
  eval     EVAL.npz --car C [--label L] one method on the 8 evaluation plans at once (3 chained laps from the start)
  compare  A.npz B.npz ... --car C --plan P --labels ...   the methods side by side on one plan, with the sideslip
                                        against the plan's underneath
--stride control periods (25 ms) per frame (2: real time at 20 fps). MP4 (H.264, pausable; imageio-ffmpeg).
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                  # noqa: E402
from matplotlib.patches import Polygon           # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import record_car                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=('hwstage', 'eval', 'compare'))
ap.add_argument('traj', nargs='+')
ap.add_argument('--car', required=True)
ap.add_argument('--plan', default='mocap_figfast_b25')
ap.add_argument('--labels', nargs='*')
ap.add_argument('--label', default='')
ap.add_argument('--stride', type=int, default=2)
ap.add_argument('--dpi', type=int, default=90)
ap.add_argument('--out', required=True)
a = ap.parse_args()

SURF, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
SEQ = ['#9ec5f4', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']
CAR_LAB = {'real_nom': 'nominal car', 'real_mass': 'mass +25 %', 'real_mu': 'friction x0.8', 'real_act': 'actuators',
           'real_lag': 'tire / steering lag'}
Ts = [record_car.load(p) for p in a.traj]
DT = Ts[0]['dt']


def car_poly(x, y, psi, L=0.40, W=0.22):
    c, s = np.cos(psi), np.sin(psi)
    pts = np.array([[L / 2, W / 2], [L / 2, -W / 2], [-L / 2, -W / 2], [-L / 2, W / 2]])
    return np.c_[x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]]


def beta_of(X):
    return np.degrees(np.arctan2(X[:, 4], X[:, 3]))


def plan_lim(xy, pad=0.45):
    return xy[:, 0].min() - pad, xy[:, 0].max() + pad, xy[:, 1].min() - pad, xy[:, 1].max() + pad


def draw_track(ax, T, name):
    xy = T['plans'][name]['xy']
    ax.plot(xy[:, 0], xy[:, 1], color='0.82', lw=7, solid_capstyle='round', zorder=0)
    ax.plot(xy[:, 0], xy[:, 1], color='0.55', lw=0.8, ls='--', zorder=1)
    l = plan_lim(xy)
    ax.set_xlim(l[0], l[1]); ax.set_ylim(l[2], l[3]); ax.set_aspect('equal')
    ax.tick_params(labelsize=7, colors=INK2)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


def draw_run(ax, T, i, k, trail=True):
    n = T['n_steps'][i]
    kk = min(k, n - 1)
    X = T['x'][i, :kk + 1]
    b = beta_of(X)
    if trail:
        ax.scatter(X[:, 0], X[:, 1], c=np.abs(b), cmap='magma_r', vmin=0, vmax=35, s=3, zorder=2, lw=0)
    x, y, psi, vx, vy = X[-1, :5]
    ax.add_patch(Polygon(car_poly(x, y, psi), closed=True, fc='#0072B2', ec='k', lw=0.7, zorder=4))
    g = psi + np.arctan2(vy, vx)
    sp = np.hypot(vx, vy)
    ax.arrow(x, y, 0.22 * sp * np.cos(g), 0.22 * sp * np.sin(g), width=0.018, color='#D55E00', zorder=5)
    m = T['meta'][i]
    done = k >= n - 1
    st = (('CRASHED' if m['crash'] == 'wall' else 'SPUN OUT') if m['failed'] else
          f"done, e_y {100 * (m['rms_ey'] or 0):.1f} cm") if done else f"|beta| {abs(b[-1]):.0f} deg"
    return st


def segments():
    """[(suptitle, [(panel title, T, run index, [(T, i) faded earlier runs])])]"""
    segs = []
    if a.mode == 'hwstage':
        T = Ts[0]
        idx = [i for i, m in enumerate(T['meta']) if m['car'] == a.car]
        plans = list(dict.fromkeys(T['meta'][i]['plan'] for i in idx))
        its = sorted({T['meta'][i]['it'] for i in idx})
        for it in its:
            panels = []
            for p in plans:
                cur = [i for i in idx if T['meta'][i]['it'] == it and T['meta'][i]['plan'] == p]
                prev = [(T, i) for i in idx if T['meta'][i]['it'] < it and T['meta'][i]['plan'] == p]
                if cur:
                    panels.append((p.replace('mocap_', ''), T, cur[0], prev))
            segs.append((f"{CAR_LAB.get(a.car, a.car)}: hardware stage ({T['name'].replace('hwstage_', '')}) -- "
                         f"iteration {it + 1} of {len(its)}: the network after {it} update{'s' if it != 1 else ''}", panels))
    elif a.mode == 'eval':
        T = Ts[0]
        idx = [i for i, m in enumerate(T['meta']) if m['car'] == a.car]
        panels = [(T['meta'][i]['plan'].replace('mocap_', ''), T, i, []) for i in idx]
        segs.append((f"{a.label or T['name'].replace('eval_', '')}: {CAR_LAB.get(a.car, a.car)}, the 8 evaluation plans "
                     f"(beta 25 / 21 / 14 / 0; 21 and 14 never trained on), 3 laps without reset", panels))
    else:
        labels = a.labels or [T['name'].replace('eval_', '') for T in Ts]
        panels = []
        for T, lab in zip(Ts, labels):
            i = [i for i, m in enumerate(T['meta']) if m['car'] == a.car and m['plan'] == a.plan]
            if i:
                panels.append((lab, T, i[0], []))
        segs.append((f"{a.plan.replace('mocap_', '')} on the {CAR_LAB.get(a.car, a.car)}: every method, 3 laps without reset",
                     panels))
    return segs


SEGS = segments()
npan = max(len(p) for _, p in SEGS)
ncol = 4 if a.mode == 'eval' else (3 if npan > 4 else npan)
nrow = int(np.ceil(npan / ncol))
strip = a.mode == 'compare'
fig = plt.figure(figsize=(4.2 * ncol, (4.4 + (1.3 if strip else 0)) * nrow + 0.6), facecolor=SURF)
frames = []
for s, (_, panels) in enumerate(SEGS):
    n = max(p[1]['n_steps'][p[2]] for p in panels)
    frames += [(s, k) for k in range(0, n, a.stride)] + [(s, n - 1)] * 30


def draw(f):
    s, k = frames[f]
    title, panels = SEGS[s]
    fig.clf()
    fig.suptitle(title, x=0.01, ha='left', fontsize=11, color=INK)
    gs = fig.add_gridspec(nrow * (2 if strip else 1), ncol, height_ratios=([4, 1.2] * nrow) if strip else None)
    for j, (pt, T, i, prev) in enumerate(panels):
        r, c = divmod(j, ncol)
        ax = fig.add_subplot(gs[r * (2 if strip else 1), c])
        ax.set_facecolor(SURF)
        plan = T['meta'][i]['plan']
        draw_track(ax, T, plan)
        for q, (Tp, ip) in enumerate(prev):           # earlier iterations, faded
            Xp = Tp['x'][ip, :Tp['n_steps'][ip]]
            ax.plot(Xp[:, 0], Xp[:, 1], color=SEQ[min(q, len(SEQ) - 1)], lw=0.8, alpha=0.6, zorder=1)
        st = draw_run(ax, T, i, k)
        ax.set_title(f'{pt}\n{st}', fontsize=9, loc='left', color=INK)
        if strip:
            bx = fig.add_subplot(gs[r * 2 + 1, c])
            kk = min(k, T['n_steps'][i] - 1)
            X = T['x'][i, :kk + 1]
            t = np.arange(kk + 1) * T['dt']
            bx.plot(t, T['plans'][plan]['beta'][T['idx'][i, :kk + 1]], color=INK2, lw=1, ls='--', label='plan')
            bx.plot(t, beta_of(X), color='#D55E00', lw=1, label='car')
            bx.set_xlim(0, max(T['n_steps']) * T['dt']); bx.set_ylim(-40, 40)
            bx.set_ylabel('sideslip (deg)', fontsize=8); bx.tick_params(labelsize=7)
            if j == 0:
                bx.legend(fontsize=7, loc='upper right', frameon=False)
    fig.text(0.01, 0.005, 'grey: the plan; path coloured by |sideslip| (light 0 -> dark 35 deg); blue: car heading; '
             'red: velocity' + ('; faded blue: earlier iterations' if a.mode == 'hwstage' else ''), fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))


import imageio_ffmpeg                            # noqa: E402
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
fig.set_dpi(a.dpi)
draw(0)
fig.canvas.draw()
w, h = fig.canvas.get_width_height()
wr = imageio_ffmpeg.write_frames(a.out, (w - w % 2, h - h % 2), fps=1 / (DT * a.stride), codec='libx264', quality=None,
                                 output_params=['-pix_fmt', 'yuv420p', '-crf', '24', '-movflags', '+faststart'])
wr.send(None)
for f in range(len(frames)):
    draw(f)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:h - h % 2, :w - w % 2, :3]
    wr.send(np.ascontiguousarray(img))
wr.close()
print(f'{len(frames)} frames -> {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)', flush=True)
