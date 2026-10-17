#!/usr/bin/env python3
"""figures_v3.py: the 2026-10-17 figures from the frozen test pass (summary_test.json, 3 seeds, cluster-bootstrap CIs)
-> src/paper/pogo/figures/fig1_methods_test.png (methods, zero-shot vs adapted) and fig2_budget_test.png (success vs
real trials: test pass at 12 / 24 / 36 / 48 trials, the validation curve 6..96 behind it)."""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                 # noqa: E402
import numpy as np                              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget                                   # noqa: E402
import ci                                       # noqa: E402

R = 'runs/v3'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'paper', 'pogo', 'figures')
os.makedirs(OUT, exist_ok=True)


def load(pat, seeds=(0, 1, 2), f='summary_test.json', root=R):
    out = []
    for s in seeds:
        p = os.path.join(root, pat.format(s=s), f)
        if os.path.exists(p):
            out.append(json.load(open(p)))
    return out


def bar(ax, x, ss, color, label=None):
    m, lo, hi = ci.cluster_ci(ss)
    ax.bar(x, 100 * m, 0.38, color=color, label=label)
    ax.errorbar(x, 100 * m, [[100 * (m - lo)], [100 * (hi - m)]], color='k', capsize=3, lw=1)


def fig1():
    rows = [('Raibert', 'expert_zs', None, (0,)), ('per-goal ILC', None, 'goalilc_b8', None),
            ('PPO+DR', 'ppo_dr_s{s}_zs', 'ppo_s{s}_ep48', None), ('RMA (fixed)', 'rmaR1_s{s}_zs', None, None),
            ('FADA (fixed)', 'fadaF2_s{s}_zs', 'fadaF2_s{s}_hw24', None),
            ('ours, nominal sim', 'nom_s{s}_zs', 'nom_s{s}_curve/it6', None),
            ('ours + DR', 'drLK20_s{s}_zs', 'drLK20_s{s}_curve/it6', None),
            ('PPO+DR + our stage', 'ppo_dr_s{s}_zs', 'ppo_dr_s{s}_curve/it6', None)]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, (name, zs, ad, seeds) in enumerate(rows):
        if zs:
            bar(ax, i - 0.2, load(zs, seeds or (0, 1, 2)), '#9ab', 'zero-shot' if i == 0 else None)
        if ad:
            ss = load(ad, (None,), root='runs/v3test') if 'goalilc' in ad else load(ad)
            bar(ax, i + 0.2, ss, '#d62', 'after real trials' if i == 1 else None)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('success (%)')
    ax.set_ylim(0, 105)
    ax.set_title('Test set (seed 1301): 6 sticks x 8 goals x 4 drops, 3 seeds, 95 % cluster CI\n'
                 'adapted: ours 36 trials, FADA 24, PPO on the robot 48, per-goal ILC 48', fontsize=8)
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig1_methods_test.png'), dpi=160)


def fig2():
    fig, ax = plt.subplots(figsize=(6, 3.6))
    val = budget.analyse(budget.curves())
    for arm, c, name in (('nom', '#d62', 'ours, nominal sim'), ('drLK20', '#e9a', 'ours + DR'),
                         ('ppo_dr', '#36b', 'PPO+DR + our stage')):
        b = sorted(val[arm]['rows'])
        ax.plot(b, [100 * val[arm]['rows'][k]['success'] for k in b], ':', color=c, lw=1)
        xs, ms, los, his = [0], [], [], []
        m, lo, hi = ci.cluster_ci(load({'nom': 'nom_s{s}_zs', 'drLK20': 'drLK20_s{s}_zs', 'ppo_dr': 'ppo_dr_s{s}_zs'}[arm]))
        ms.append(m); los.append(lo); his.append(hi)
        for it in (2, 4, 6, 8):
            m, lo, hi = ci.cluster_ci(load(f'{arm}_s{{s}}_curve/it{it}'))
            xs.append(6 * it); ms.append(m); los.append(lo); his.append(hi)
        ax.plot(xs, 100 * np.array(ms), 'o-', color=c, label=name)
        ax.fill_between(xs, 100 * np.array(los), 100 * np.array(his), color=c, alpha=0.15)
    ppo = [0, 24, 48, 96, 192, 384]
    pm = [ci.cluster_ci(load('ppo_dr_s{s}_zs' if n == 0 else f'ppo_s{{s}}_ep{n}'))[0] for n in ppo]
    ax.plot(ppo, 100 * np.array(pm), 's--', color='0.4', label='PPO on the robot (privileged critic)')
    m = ci.cluster_ci(load('rmaR1_s{s}_zs'))[0]
    ax.axhline(100 * m, color='g', lw=1, ls='-.', label='RMA (fixed), zero-shot')
    ax.axvline(36, color='k', lw=0.6, ls=':')
    ax.set_xscale('symlog', linthresh=6)
    ax.set_xticks([0, 6, 12, 24, 36, 48, 96, 192, 384], ['0', '6', '12', '24', '36', '', '96', '192', '384'])
    ax.set_xlim(-0.5, 450)
    ax.set_xlabel('real trials per stick (3 s each)')
    ax.set_ylabel('success (%)')
    ax.set_ylim(0, 102)
    ax.set_title('Budget: test pass (solid, 95 % CI) and validation curve (dotted)', fontsize=8)
    ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig2_budget_test.png'), dpi=160)


if __name__ == '__main__':
    fig2()
    if os.path.exists('runs/v3test/goalilc_b8/summary_test.json'):
        fig1()
    else:
        print('goalilc test pass not done yet: fig1 skipped')
