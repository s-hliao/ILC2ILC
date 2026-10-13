#!/usr/bin/env python3
"""
figures.py: the pogo-stick study's figures -> src/paper/pogo/figures/ (PNG). Every method keeps one color across
figures (the dataviz skill's validated categorical order, assigned by method, never by rank); lines also carry
markers and a legend so identity is never color alone.
  fig1_methods       success zero-shot vs after 24 real trials, per method; inside / outside the DR family
  fig2_budget        success and apex error vs real trials per robot
  fig3_coverage      success along each perturbation axis, DR range shaded
  fig4_positivity    the hardware stage's positivity (cos predicted vs measured error change) along the axes
  fig5_controls      sim-stage and hardware-stage ablations
  fig6_cost          sim hours of training vs real trials
  fig7_goals         success per test goal on the goal plane (v*, h*)
  fig8_matching      data matching: success vs real trials per robot (log), ours vs RL trained ON the robots
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib                                   # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402

import common as C                                  # noqa: E402
import scheduler_axes as SA                         # noqa: E402
from table import load, IN_FAM, OUT_FAM, ROB        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'paper', 'pogo', 'figures')
COL = dict(ours='#2a78d6', oursdr='#eb6834', ppo='#1baf7a', rma='#eda100', fada='#e87ba4', ilc='#008300',
           raibert='#4a3aa7', ctrl='#e34948')
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False})


def succ(arm, robots=ROB):
    s = load(arm)
    if s is None:
        return None
    ms = [m for r in robots if r in s['runs'] for m in s['runs'][r]]
    return 100 * np.mean([m['success'] for m in ms]) if ms else None


def ci(p, n):
    """Wilson 95 % half-widths (lo, hi) for a percentage p over n runs."""
    if p is None or n == 0:
        return (0, 0)
    q, z = p / 100, 1.96
    c = (q + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(q * (1 - q) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (100 * (q - (c - h)), 100 * (c + h - q))


def nruns(arm, robots=ROB):
    s = load(arm)
    return sum(len(s['runs'][r]) for r in robots if r in s['runs']) if s else 0


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(name)


def fig1():
    rows = [('Raibert (classical)', 'expert_zs', None, 'raibert'), ('per-goal ILC (classical)', None, 'goalilc_b4', 'ilc'),
            ('ours, nominal sim', 'nom_s0_zs', 'nom_s0_hw24', 'ours'), ('ours + DR (A)', 'drA_s0_zs', 'drA_s0_hw24', 'oursdr'),
            ('ours + DR (B)', 'drB_s0_zs', 'drB_s0_hw24', 'oursdr'), ('PPO+DR (+ our stage)', 'ppo_dr_zs', 'ppo_dr_hw24', 'ppo'),
            ('RMA', 'rma_zs', None, 'rma'), ('FADA', 'fada_zs', 'fada_hw24', 'fada')]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, (title, robots) in zip(axs, (('all 6 robots', ROB), ('inside the DR family', IN_FAM), ('outside the DR family', OUT_FAM))):
        y = np.arange(len(rows))[::-1]
        for yi, (lab, zs, hw, c) in zip(y, rows):
            for arm, off, alpha in ((zs, 0.18, 0.45), (hw, -0.18, 1.0)):
                v = succ(arm, robots) if arm else None
                if v is None:
                    continue
                e = ci(v, nruns(arm, robots))
                ax.barh(yi + off, v, height=0.34, color=COL[c], alpha=alpha, xerr=[[e[0]], [e[1]]],
                        error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
                ax.text(v + e[1] + 1.5, yi + off, f'{v:.0f}', va='center', fontsize=7, color=INK2)
        ax.set_yticks(y, [r[0] for r in rows])
        ax.set_xlim(0, 112)
        ax.set_title(title, loc='left', color=INK)
        ax.set_xlabel('success % (test goals)')
    from matplotlib.patches import Patch
    axs[0].legend(handles=[Patch(color='0.75', label='zero-shot (light)'), Patch(color='0.35', label='+24 real trials per robot (dark)')],
                  loc='lower right', fontsize=7)
    save(fig, 'fig1_methods')


def fig2():
    lines = [('ours, nominal sim', 'ours', [(0, 'nom_s0_zs'), (6, 'nom_s0_hw6'), (12, 'nom_s0_hw12'), (24, 'nom_s0_hw24'), (48, 'nom_s0_hw48')]),
             ('ours + DR (B)', 'oursdr', [(0, 'drB_s0_zs'), (6, 'drB_s0_hw6'), (12, 'drB_s0_hw12'), (24, 'drB_s0_hw24'), (48, 'drB_s0_hw48')]),
             ('PPO+DR + our stage', 'ppo', [(0, 'ppo_dr_zs'), (6, 'ppo_dr_hw6'), (12, 'ppo_dr_hw12'), (24, 'ppo_dr_hw24'), (48, 'ppo_dr_hw48')]),
             ('FADA', 'fada', [(0, 'fada_zs'), (24, 'fada_hw24')]),
             ('per-goal ILC (own goals)', 'ilc', [(6, 'goalilc_b1'), (12, 'goalilc_b2'), (24, 'goalilc_b4'), (48, 'goalilc_b8')])]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for lab, c, pts in lines:
        xs, ys, es, ap = [], [], [], []
        for b, arm in pts:
            s = load(arm)
            if s is None:
                continue
            ok = [m for r in ROB if r in s['runs'] for m in s['runs'][r] if not m['failed'] and m['rms_apex'] is not None]
            xs.append(b); ys.append(succ(arm)); es.append(ci(ys[-1], nruns(arm)))
            ap.append(100 * np.mean([m['rms_apex'] for m in ok]) if ok else np.nan)
        if not xs:
            continue
        es = np.array(es).T
        axs[0].errorbar(xs, ys, yerr=es, color=COL[c], marker='o', ms=5, lw=2, capsize=2, label=lab)
        axs[1].plot(xs, ap, color=COL[c], marker='o', ms=5, lw=2, label=lab)
    rz = succ('expert_zs')
    if rz is not None:
        axs[0].axhline(rz, color=COL['raibert'], ls=':', lw=1.2, label='Raibert')
    axs[0].set_ylabel('success %'); axs[1].set_ylabel('RMS apex error (cm)')
    axs[1].axhline(2.0, color=INK2, ls='--', lw=0.8)
    axs[1].text(47, 2.05, 'success band', ha='right', fontsize=7, color=INK2)
    for ax in axs:
        ax.set_xlabel('real trials per robot (3 s each)')
        ax.set_xticks([0, 6, 12, 24, 48])
    axs[0].legend(fontsize=7)
    save(fig, 'fig2_budget')


def axis_curve(arm, axis):
    s = load(f'axes/{arm}_{axis}')
    if s is None:
        return None, None
    rob = SA.robots(axis)
    return np.array(SA.AXES[axis]), np.array([100 * s['eval'][r]['success'] if r in s['eval'] else np.nan for r in rob])


def fig3():
    arms = [('Raibert', 'expert_zs', 'raibert', ':'), ('ours zero-shot', 'nom_s0_zs', 'ours', '--'),
            ('ours +24', 'nom_s0_hw24', 'ours', '-'), ('ours + DR (B) zero-shot', 'drB_s0_zs', 'oursdr', '--'),
            ('ours + DR (B) +24', 'drB_s0_hw24', 'oursdr', '-'), ('PPO+DR zero-shot', 'ppo_dr_zs', 'ppo', '--'),
            ('PPO+DR +24 (our stage)', 'ppo_dr_hw24', 'ppo', '-'), ('RMA', 'rma_zs', 'rma', '-')]
    names = dict(k='spring stiffness (x nominal)', m='mass (x nominal)', gth='thrust gain (x nominal)',
                 phi_off='leg-angle offset (rad)', kp='progressive spring kp', b='leg damping (N s / m)')
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.2), sharey=True)
    for ax, axis in zip(axs.flat, SA.AXES):
        lo, hi = SA.DR_RANGE[axis]
        if hi > lo:
            ax.axvspan(lo, hi, color='0.88', lw=0, zorder=0)
        else:
            ax.axvline(lo, color='0.75', lw=5, zorder=0)
        for lab, arm, c, ls in arms:
            x, y = axis_curve(arm, axis)
            if x is not None:
                ax.plot(x, y, color=COL[c], ls=ls, marker='o', ms=4, lw=1.8, label=lab)
        ax.set_xlabel(names[axis]); ax.set_ylim(-3, 103)
    axs[0, 0].set_ylabel('success %'); axs[1, 0].set_ylabel('success %')
    axs[0, 0].legend(fontsize=7, loc='lower left')
    fig.suptitle('success along single perturbation axes (on real_nom); grey band = the DR family\'s range', fontsize=9, x=0.01, ha='left')
    save(fig, 'fig3_coverage')


def fig4():
    names = dict(k='spring x', m='mass x', gth='thrust gain x', phi_off='leg offset rad', kp='kp', b='damping')
    fig, axs = plt.subplots(2, 3, figsize=(12, 5.6), sharey=True)
    for ax, axis in zip(axs.flat, SA.AXES):
        for arm, c in (('nom_s0_hw24', 'ours'), ('drB_s0_hw24', 'oursdr')):
            s = load(f'axes/{arm}_{axis}')
            if s is None:
                continue
            xs, med, q1, q3 = [], [], [], []
            for v, r in zip(SA.AXES[axis], SA.robots(axis)):
                cs = [q['cos'] for q in s.get('positivity', {}).get(r, [])]
                if cs:
                    xs.append(v); med.append(np.median(cs)); q1.append(np.percentile(cs, 25)); q3.append(np.percentile(cs, 75))
            ax.fill_between(xs, q1, q3, color=COL[c], alpha=0.18, lw=0)
            ax.plot(xs, med, color=COL[c], marker='o', ms=4, lw=1.8, label='ours' if c == 'ours' else 'ours + DR (B)')
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel(names[axis]); ax.set_ylim(-1, 1)
    axs[0, 0].set_ylabel('cos(predicted, measured)'); axs[1, 0].set_ylabel('cos(predicted, measured)')
    axs[0, 0].legend(fontsize=7)
    save(fig, 'fig4_positivity')


def fig5():
    rows = [('ours: sim stage + 24 trials', 'nom_s0_hw24', 'ours'), ('zero-shot', 'nom_s0_zs', 'ours'),
            ('sim stage w/o exploration, +24', 'noexp_s0_hw24', 'ctrl'), ('open-loop G in the sim stage, +24', 'olG_s0_hw24', 'ctrl'),
            ('BC only (no deep ILC), zero-shot', 'bc_zs', 'ctrl'), ('hardware stage: open-loop G', 'nom_s0_hw24_openG', 'ctrl'),
            ('hardware stage: flipped G (180 deg)', 'nom_s0_hw24_flip', 'ctrl'), ('hardware stage: no trust region', 'nom_s0_hw24_noTR', 'ctrl'),
            ('hardware stage: no rollback', 'nom_s0_hw24_noRB', 'ctrl'), ('hardware stage: perturbed training drops', 'nom_s0_hw24_rep2', 'ctrl')]
    rows = [r for r in rows if load(r[1]) is not None]
    fig, ax = plt.subplots(figsize=(7, 0.36 * len(rows) + 0.8))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, arm, c) in zip(y, rows):
        v = succ(arm)
        e = ci(v, nruns(arm))
        ax.barh(yi, v, height=0.6, color=COL[c], xerr=[[e[0]], [e[1]]], error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
        ax.text(v + e[1] + 1.5, yi, f'{v:.0f}', va='center', fontsize=7, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows]); ax.set_xlim(0, 110); ax.set_xlabel('success % (6 robots, test goals)')
    save(fig, 'fig5_controls')


def sim_hours(run):
    f = os.path.join(HERE, 'runs', run, 'log.jsonl')
    if not os.path.exists(f):
        return None
    h = [json.loads(l).get('sim_hours') for l in open(f) if 'sim_hours' in l]
    h = [x for x in h if x is not None]
    return h[-1] if h else None


def fig6():
    pts = [('ours', 'nom_s0', 'nom_s0_hw24', 24, 'ours'), ('ours + DR (B)', 'drB_s0', 'drB_s0_hw24', 24, 'oursdr'),
           ('PPO+DR', 'ppo_dr_s0', 'ppo_dr_zs', 0, 'ppo'), ('RMA', 'rma_s0', 'rma_zs', 0, 'rma')]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for lab, run, arm, trials, c in pts:
        h, v = sim_hours(run), succ(arm)
        if h is None or v is None:
            continue
        if run == 'nom_s0':
            h += sim_hours('bc') or 1.8
        ax.scatter(h, v, s=60, color=COL[c], edgecolor='white', linewidth=2, zorder=3)
        ax.annotate(f'{lab}\n{trials} real trials', (h, v), textcoords='offset points', xytext=(6, 4), fontsize=7, color=INK2)
    ax.set_xscale('log'); ax.set_xlabel('simulated hours of training'); ax.set_ylabel('success %')
    save(fig, 'fig6_cost')


def fig7():
    arms = [('Raibert', 'expert_zs'), ('ours zero-shot', 'nom_s0_zs'), ('ours +24', 'nom_s0_hw24'), ('ours + DR (B) +24', 'drB_s0_hw24'),
            ('PPO+DR', 'ppo_dr_zs'), ('RMA', 'rma_zs')]
    arms = [a for a in arms if load(a[1]) is not None]
    fig, axs = plt.subplots(1, len(arms), figsize=(2.6 * len(arms), 2.8), sharey=True, squeeze=False)
    for ax, (lab, arm) in zip(axs[0], arms):
        s = load(arm)
        G = np.array(s['goals_eval'])
        rate = []
        for g in G:
            ms = [m for r in ROB if r in s['runs'] for m in s['runs'][r] if np.allclose(m['goal'], g)]
            rate.append(100 * np.mean([m['success'] for m in ms]) if ms else np.nan)
        sc = ax.scatter(G[:, 0], G[:, 1], c=rate, cmap='Blues', vmin=0, vmax=100, s=110, edgecolor=INK2, linewidth=0.6)
        ax.scatter(C.TRAIN_GOALS[:, 0], C.TRAIN_GOALS[:, 1], marker='x', color=INK2, s=18, linewidth=0.8)
        ax.set_title(lab, loc='left', fontsize=8); ax.set_xlabel('v* (m/s)')
    axs[0, 0].set_ylabel('h* (m)')
    fig.colorbar(sc, ax=axs[0].tolist(), shrink=0.8, label='success % (6 robots)')
    save(fig, 'fig7_goals')


def fig8():
    lines = [('ours, nominal sim', 'ours', '-', [(0, 'nom_s0_zs'), (6, 'nom_s0_hw6'), (12, 'nom_s0_hw12'), (24, 'nom_s0_hw24'), (48, 'nom_s0_hw48')]),
             ('PPO+DR + our stage', 'ppo', '-', [(0, 'ppo_dr_zs'), (6, 'ppo_dr_hw6'), (12, 'ppo_dr_hw12'), (24, 'ppo_dr_hw24'), (48, 'ppo_dr_hw48')]),
             ('PPO trained on the robot (privileged critic)', 'ppo', '--', [(b, f'match/ppo_ep{b}') for b in (0, 24, 96, 384, 768, 1536, 2000)]),
             ('RMA, oracle latent, PPO on the robot', 'rma', '--', [(b, f'match/rma_oracle_ep{b}') for b in (0, 24, 96, 384, 768, 1536, 2000)]),
             ('per-goal ILC (own goals)', 'ilc', ':', [(6, 'goalilc_b1'), (12, 'goalilc_b2'), (24, 'goalilc_b4'), (48, 'goalilc_b8')])]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for lab, c, ls, pts in lines:
        xs, ys = [], []
        for b, arm in pts:
            v = succ(arm)
            if v is not None:
                xs.append(max(b, 1)); ys.append(v)
        if xs:
            ax.plot(xs, ys, color=COL[c], ls=ls, marker='o', ms=5, lw=2, label=lab)
    ax.set_xscale('log'); ax.set_xlabel('real trials per robot (3 s each; 0 plotted at 1)'); ax.set_ylabel('success % (6 robots)')
    ax.set_ylim(-3, 103)
    ax.legend(fontsize=7, loc='lower right')
    save(fig, 'fig8_matching')


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        try:
            f()
        except Exception as e:                       # a figure whose arms are missing is skipped
            print(f.__name__, 'skipped:', repr(e))
