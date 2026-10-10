#!/usr/bin/env python3
"""
car_traj_figs.py [--root DIR] [--tag TEXT]: static figures of the recorded car trajectories (record_car.py) ->
DIR/figs/traj_*.png.

  traj_paths_<plan>     every car x the key methods: the path flown (3 laps, no reset) against the plan, coloured by
                        |sideslip|, with the RMS lateral error and the outcome
  traj_sideslip_<plan>  per car: the sideslip flown over time against the plan's (the first two laps)
  traj_hwstage_<car>    the hardware stage on that car (ours + DR (B)): each training plan's lap, iteration by iteration
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import record_car                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--root', default=HERE)
ap.add_argument('--tag', default='')
a = ap.parse_args()
ROOT = os.path.abspath(a.root)
TR, OUT = os.path.join(ROOT, 'trajectories'), os.path.join(ROOT, 'figs')
os.makedirs(OUT, exist_ok=True)
TAG = f' -- {a.tag}' if a.tag else ''
SURF, INK2 = '#fcfcfb', '#52514e'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', pink='#CC79A7', sky='#56B4E9', red='#D55E00', grey='#7f7f7f')
SEQ = ['#9ec5f4', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']
CARS = ['real_nom', 'real_mass', 'real_mu', 'real_act', 'real_lag']
CAR_LAB = {'real_nom': 'nominal', 'real_mass': 'mass +25 %', 'real_mu': 'friction x0.8', 'real_act': 'actuators',
           'real_lag': 'tire / steering lag'}
KEY = [('eval_ours_24laps', 'ours + 24', C['blue']), ('eval_ours_dr_b_24laps', 'ours+DR (B) + 24', C['green']),
       ('eval_ppo_dr_zeroshot', 'PPO+DR, zero-shot', C['pink']), ('eval_ppo_dr_2laps', 'PPO+DR + 2 (our stage)', C['pink']),
       ('eval_ppo_dr_4laps', 'PPO+DR + 4 (our stage)', C['pink']),
       ('eval_rma_zeroshot', 'RMA, zero-shot', C['red']), ('eval_fada_24laps', 'FADA + 24', C['orange'])]
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'savefig.dpi': 110,
                     'savefig.bbox': 'tight', 'font.size': 9})
L = {n: record_car.load(os.path.join(TR, n + '.npz')) for n, _, _ in KEY if os.path.exists(os.path.join(TR, n + '.npz'))}
KEY = [k for k in KEY if k[0] in L]
beta = lambda X: np.degrees(np.arctan2(X[:, 4], X[:, 3]))


def run_of(T, car, plan):
    i = [i for i, m in enumerate(T['meta']) if m['car'] == car and m['plan'] == plan]
    return i[0] if i else None


def paths(plan):
    fig, axs = plt.subplots(len(CARS), len(KEY), figsize=(2.3 * len(KEY), 2.7 * len(CARS)), squeeze=False)
    for r, car in enumerate(CARS):
        for c, (n, lab, col) in enumerate(KEY):
            ax = axs[r, c]
            T = L[n]
            xy = T['plans'][plan]['xy']
            ax.plot(xy[:, 0], xy[:, 1], color='0.82', lw=5, solid_capstyle='round', zorder=0)
            i = run_of(T, car, plan)
            if i is not None:
                X = T['x'][i, :T['n_steps'][i]]
                ax.scatter(X[:, 0], X[:, 1], c=np.abs(beta(X)), cmap='magma_r', vmin=0, vmax=35, s=0.6, lw=0, zorder=2)
                m = T['meta'][i]
                out = ('crashed' if m['crash'] == 'wall' else 'spun out') if m['failed'] else \
                      f"e_y {100 * (m['rms_ey'] or 0):.1f} cm, |b| {m['mean_abs_beta'] or 0:.0f}"
                ax.set_title(f"{lab if r == 0 else ''}{chr(10) if r == 0 else ''}{out}", fontsize=7.5,
                             color='#b00' if m['failed'] else '#222')
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c == 0:
                ax.set_ylabel(CAR_LAB[car], fontsize=9)
    fig.suptitle(f"{plan.replace('mocap_', '')}: 3 laps without reset per car (path coloured by |sideslip|, light 0 -> dark "
                 f"35 deg){TAG}", fontsize=10)
    fig.savefig(os.path.join(OUT, f"traj_paths_{plan.replace('mocap_', '')}.png"))
    plt.close(fig)


def sideslip(plan):
    fig, axs = plt.subplots(len(CARS), 1, figsize=(11, 2.0 * len(CARS)), sharex=True)
    for ax, car in zip(axs, CARS):
        drawn_plan = False
        for n, lab, col in KEY:
            T = L[n]
            i = run_of(T, car, plan)
            if i is None:
                continue
            lt = T['plans'][plan]['lap_time']
            k = min(T['n_steps'][i], int(2 * lt / T['dt']))
            X = T['x'][i, :k]
            t = np.arange(k) * T['dt']
            if not drawn_plan:
                ax.plot(t, T['plans'][plan]['beta'][T['idx'][i, :k]], color=INK2, lw=1.4, ls='--', label='plan')
                drawn_plan = True
            ax.plot(t, beta(X), color=col, lw=0.9, ls=':' if 'zero-shot' in lab and 'PPO' in lab else '-', label=lab)
        ax.set_ylabel(f'{CAR_LAB[car]}\nsideslip (deg)', fontsize=8)
        ax.set_ylim(-45, 45)
        ax.grid(alpha=0.25)
    axs[0].legend(fontsize=7, ncol=4, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.45))
    axs[-1].set_xlabel('time (s), the first two laps')
    fig.suptitle(f"{plan.replace('mocap_', '')}: sideslip flown against the plan's{TAG}", fontsize=10, y=1.02)
    fig.savefig(os.path.join(OUT, f"traj_sideslip_{plan.replace('mocap_', '')}.png"))
    plt.close(fig)


def hwstage(car, name='hwstage_ours_dr_b'):
    fp = os.path.join(TR, name + '.npz')
    if not os.path.exists(fp):
        return
    T = record_car.load(fp)
    idx = [i for i, m in enumerate(T['meta']) if m['car'] == car]
    plans = list(dict.fromkeys(T['meta'][i]['plan'] for i in idx))
    its = sorted({T['meta'][i]['it'] for i in idx})
    fig, axs = plt.subplots(1, len(plans), figsize=(2.6 * len(plans), 3.6), squeeze=False)
    for ax, p in zip(axs[0], plans):
        xy = T['plans'][p]['xy']
        ax.plot(xy[:, 0], xy[:, 1], color='0.82', lw=5, solid_capstyle='round', zorder=0)
        for it in its:
            for i in [i for i in idx if T['meta'][i]['plan'] == p and T['meta'][i]['it'] == it]:
                X = T['x'][i, :T['n_steps'][i]]
                ax.plot(X[:, 0], X[:, 1], color=SEQ[min(it, len(SEQ) - 1)], lw=1.1, label=f'iteration {it + 1}')
        ax.set_title(p.replace('mocap_', ''), fontsize=8)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, ncol=len(its), loc='lower center', bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"our learner + DR (B): the hardware stage's laps on the {CAR_LAB[car]} car (light = earliest){TAG}",
                 fontsize=10)
    fig.savefig(os.path.join(OUT, f'traj_hwstage_{car}.png'))
    plt.close(fig)


if __name__ == '__main__':
    for p in ('mocap_square2fast_b25', 'mocap_figfast_b25'):
        paths(p)
        sideslip(p)
    for c in ('real_nom', 'real_mu'):
        hwstage(c)
    print('->', OUT)
