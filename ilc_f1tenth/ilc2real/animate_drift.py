#!/usr/bin/env python3
"""
animate_drift.py --policy NPZ|lqr --car real_nom --plan mocap_figfast_b25 --laps 4 --out drift.gif: a continuous
multi-lap run on a real car, animated: the track (plan path), the car's footprint (heading) and its velocity vector
(the angle between them is the sideslip), a trail coloured by |beta|, and a strip of beta against the plan's beta*
over time. Several (policy, car) panels side by side with --compare. GIF: dpi 100, global 48-color palette (as the
quadruped's animate_hw_stage.py).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib                                # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                  # noqa: E402
from matplotlib.patches import Polygon           # noqa: E402
from PIL import Image                            # noqa: E402

import bank                                      # noqa: E402
import real_car as rc                            # noqa: E402


def load(path):
    if path == 'lqr':
        return None
    d = np.load(path)
    n = len([k for k in d.files if k.startswith('W')])
    return [(d[f'W{i}'], d[f'b{i}']) for i in range(n)]


def car_poly(x, y, psi, L=0.40, W=0.22):
    c, s = np.cos(psi), np.sin(psi)
    pts = np.array([[L / 2, W / 2], [L / 2, -W / 2], [-L / 2, -W / 2], [-L / 2, W / 2], [L / 2 + 0.06, 0]])[[0, 4, 1, 2, 3]]
    return np.c_[x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compare', nargs='+', required=True, help='LABEL=POLICY@CAR entries (POLICY: npz or lqr)')
    ap.add_argument('--plan', default='mocap_figfast_b25')
    ap.add_argument('--laps', type=float, default=3.0)
    ap.add_argument('--stride', type=int, default=2, help='control periods per frame (40 Hz / stride fps x speed)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--colors', type=int, default=48)
    ap.add_argument('--seed', type=int, default=701)
    a = ap.parse_args()
    plan = bank.load_plan(a.plan)
    runs = []
    for e in a.compare:
        label, rest = e.split('=', 1)
        pol, car = rest.split('@')
        r = rc.run(car, plan, policy=load(pol), laps=a.laps, s0=0.0, pert=np.zeros(5), seed=a.seed)
        m = rc.lap_metrics(r, plan)
        runs.append((label, car, r, m))
        print(label, car, {k: v for k, v in m.items() if k != 'per_lap_ey'})
    n = len(runs)
    T = max(len(r['x_true']) for _, _, r, _ in runs)
    fig = plt.figure(figsize=(4.6 * n, 6.2))
    gs = fig.add_gridspec(2, n, height_ratios=[4, 1.3])
    xy = plan['xy']
    pad = 0.6
    lim = (xy[:, 0].min() - pad, xy[:, 0].max() + pad, xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    beta_ref_all = np.degrees(np.arctan2(plan['z'][:, 3], plan['z'][:, 2]))
    frames = []
    for t in range(0, T, a.stride):
        fig.clf()
        gs = fig.add_gridspec(2, n, height_ratios=[4, 1.3])
        for k, (label, car, r, m) in enumerate(runs):
            ax = fig.add_subplot(gs[0, k])
            bx = fig.add_subplot(gs[1, k])
            X = r['x_true']
            tt = min(t, len(X) - 1)
            ax.plot(xy[:, 0], xy[:, 1], color='0.75', lw=6, solid_capstyle='round', zorder=0)
            ax.plot(xy[:, 0], xy[:, 1], color='0.5', lw=0.8, ls='--', zorder=1)
            beta = np.degrees(np.arctan2(X[:tt + 1, 4], X[:tt + 1, 3]))
            ax.scatter(X[:tt + 1, 0], X[:tt + 1, 1], c=np.abs(beta), cmap='magma_r', vmin=0, vmax=35, s=4, zorder=2)
            x, y, psi, vx, vy = X[tt, :5]
            ax.add_patch(Polygon(car_poly(x, y, psi), closed=True, fc='#2b6cb0', ec='k', lw=0.8, zorder=4))
            g = psi + np.arctan2(vy, vx)
            ax.arrow(x, y, 0.25 * np.hypot(vx, vy) * np.cos(g), 0.25 * np.hypot(vx, vy) * np.sin(g), width=0.02,
                     color='#c53030', zorder=5)
            done = tt >= len(X) - 1
            status = ('SPUN OUT' if r['failed'] else 'done') if done else f'lap {min(tt * 0.025 / float(plan["lap_time"]), a.laps):.1f}'
            ax.set_title(f'{label} on {car}\n|beta| {abs(beta[-1]):.0f} deg   {status}', fontsize=11)
            ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3]); ax.set_aspect('equal')
            ax.tick_params(labelsize=9)
            br = beta_ref_all[r['idx'][:tt + 1]]
            ts = np.arange(tt + 1) * 0.025
            bx.plot(ts, br, color='0.4', lw=1.2, ls='--', label='plan')
            bx.plot(ts, beta, color='#c53030', lw=1.2, label='car')
            bx.set_xlim(0, T * 0.025); bx.set_ylim(-40, 40)
            bx.set_ylabel('sideslip (deg)', fontsize=9); bx.set_xlabel('time (s)', fontsize=9)
            bx.tick_params(labelsize=9)
            if k == 0:
                bx.legend(fontsize=8, loc='upper right')
        fig.suptitle(f'{plan["name"].replace("mocap_", "")}: continuous drift, {a.laps:g} laps without reset '
                     '(blue: heading, red: velocity)', fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.canvas.draw()
        frames.append(Image.frombuffer('RGBA', fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert('RGB'))
    pal = frames[len(frames) // 2].quantize(colors=a.colors, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(a.out, save_all=True, append_images=q[1:], duration=int(1000 * 0.025 * a.stride), loop=0, optimize=True)
    print('saved', a.out, len(q), 'frames', os.path.getsize(a.out) // 1024, 'KB')


if __name__ == '__main__':
    main()
