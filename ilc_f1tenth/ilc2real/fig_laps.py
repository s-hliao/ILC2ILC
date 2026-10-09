#!/usr/bin/env python3
"""fig_laps.py: the lap-budget ablation (scheduler_laps.py) -> figs/lap_budget.png.
Full-drift plans (beta 25, both tracks), 5 cars x 12 runs x 5 chained laps per point (runs/hw/<arm>/eval_big.json):
  (a) success (on track, RMS e_y <= 10 cm, pace >= 90 %) with Wilson 95 % intervals, (b) RMS lateral error of the
  completed runs, (c) mean |beta| actually flown -- against the real laps per car spent on the two beta-25 plans:
  0 = <net>_zs, 2..10 = <net>_lapN, 24 / 48 / 96 = <net>_hw24 / hw48 / hw96. The per-track ILC (runs/trackilc_laps)
  spends b laps on each plan, so 2b on the two beta-25 plans (and the same on the 6 others)."""
import json
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DRIFT = ['mocap_square2fast_b25', 'mocap_figfast_b25']
SURF, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'


def ev(arm):
    f = os.path.join(HERE, 'runs/hw', arm, 'eval_big.json')
    return json.load(open(f)) if os.path.exists(f) else None


def stats(e, runs=12):
    s = [e[c][n]['success'] for c in e for n in DRIFT if n in e[c]]
    y = [e[c][n]['rms_ey'] for c in e for n in DRIFT if n in e[c] and e[c][n].get('rms_ey') is not None]
    b = [e[c][n]['mean_abs_beta'] for c in e for n in DRIFT if n in e[c] and e[c][n].get('mean_abs_beta') is not None]
    n = len(s) * runs
    p = float(np.mean(s))
    z = 1.96                                      # Wilson 95 % interval over the n runs
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return dict(s=100 * p, lo=100 * (c - h), hi=100 * (c + h), ey=100 * np.mean(y), beta=np.mean(b))


def series(net, extra=()):
    pts = [(0, f'{net}_zs')] + [(k, f'{net}_lap{k}') for k in (2, 4, 6, 8, 10)] + list(extra)
    return [(x, stats(ev(a))) for x, a in pts if ev(a)]


LINES = [
    ('ours (nominal Fiala sim, seed 0), then our hardware stage', '#0072B2', 'o', '-',
     series('v3nom_s0', [(24, 'v3nom_s0_hw24'), (48, 'v3nom_s0_hw48'), (96, 'v3nom_s0_hw96')])),
    ('ours (seed 1)', '#56B4E9', 'v', '-', series('v3nom_s1', [(24, 'v3nom_s1_hw24')])),
    ('PPO + DR, then our hardware stage', '#009E73', 'D', '-',
     series('ppo_dr', [(24, 'ppo_dr_hw24'), (48, 'ppo_dr_hw48')])),
]
tl = json.load(open(os.path.join(HERE, 'runs/trackilc_laps/summary.json')))['budgets']
LINES.append(('per-track ILC (plan LQR + feedforward), b laps per plan', '#E69F00', 's', '--',
              [(0, stats(ev('lqr_zs')))] + sorted((2 * int(b), stats(e)) for b, e in tl.items())))
VAR = [('6 goals x 1 iteration', 'v3nom_s0_lap6g6', 6), ('larger steps, 6 laps', 'v3nom_s0_lap6big', 6),
       ('larger steps, 10 laps', 'v3nom_s0_lap10big', 10)]

TICKS = [0, 2, 4, 6, 8, 10, 24, 48, 96]
X = lambda b: TICKS.index(b) if b in TICKS else float(np.interp(b, TICKS, range(len(TICKS))))   # evenly spaced budgets
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2), facecolor=SURF)
keys = [('s', 'full-drift success (%)'), ('ey', 'RMS lateral error, completed runs (cm)'), ('beta', 'mean |sideslip| flown (deg)')]
for ax, (k, lab) in zip(axs, keys):
    ax.set_facecolor(SURF)
    for name, col, mk, ls, pts in LINES:
        x = [X(p[0]) for p in pts]
        y = [p[1][k] for p in pts]
        ax.plot(x, y, ls=ls, marker=mk, color=col, lw=2, ms=6, mec=SURF, label=name)
        if k == 's':
            ax.vlines(x, [p[1]['lo'] for p in pts], [p[1]['hi'] for p in pts], color=col, lw=1, alpha=0.6)
    for j, (name, arm, x) in enumerate(VAR):
        e = ev(arm)
        if e:
            ax.plot(X(x) + 0.15, stats(e)[k], marker='^>x'[j], ls='none', color=INK2, ms=6, label=f'ours (seed 0): {name}')
    ax.set_xticks(range(len(TICKS)))
    ax.set_xticklabels([str(t) for t in TICKS])
    ax.axvspan(5.35, 5.65, color=SURF, zorder=3)                 # the scale changes after 10 laps
    ax.text(5.5, 0.0, '//', transform=ax.get_xaxis_transform(), ha='center', va='center', color=INK2, zorder=4)
    ax.set_xlabel('real laps per car (beta-25 plans)')
    ax.set_ylabel(lab)
    ax.grid(alpha=0.25)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
axs[0].axhline(80, color=INK2, lw=0.8, ls=':')
axs[0].text(0.1, 33, 'dotted: 80 % = all of the other 4 cars;\nreal_mu (mu x 0.8) never succeeds for the ILC-trained networks',
            fontsize=7.5, color=INK2)
axs[0].set_ylim(30, 102)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=8.5)
fig.suptitle('Lap budget at mu 0.2: full-drift plans (beta 25, both tracks), 5 cars x 12 runs x 5 laps per point; laps axis not to scale after 10',
             x=0.01, ha='left', fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0.13, 1, 0.95))
os.makedirs(os.path.join(HERE, 'figs'), exist_ok=True)
out = os.path.join(HERE, 'figs', 'lap_budget.png')
fig.savefig(out, dpi=130, facecolor=SURF)
print('->', out)
