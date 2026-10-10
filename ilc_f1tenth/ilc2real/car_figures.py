#!/usr/bin/env python3
"""
car_figures.py [--root DIR] [--out DIR] [--tag TEXT]: the F1TENTH figures at the car's REAL budget, 0 / 2 / 10 real
laps per car (user, 2026-10-10: "replace the 24 lap graphs"; the 24-lap set is in src/paper/car/figures/old_24lap/). Every hardware
stage: the two beta-25 plans alternated, one chained 2-lap trial per iteration. Evaluation: runs/hw/<arm>/eval_big.json,
5 multi-body cars x 8 plans x n runs x 5 chained laps. Two scores, 95 % Wilson intervals over cars x plans x runs:
  success     on track all 5 laps, RMS e_y <= 10 cm, pace >= 90 % of the plan's (grip or drift)
  completion  all 5 chained laps without a wall crash (0.3 m envelope) or a spin-out

  fig1_methods      both scores on the beta-25 plans, zero-shot / + 2 / + 10 real laps, per method
  fig2_per_car      both scores per car, + 10 real laps
  fig3_budget       both scores and the RMS lateral error against real laps (0 .. 10)
  fig5_accuracy     RMS lateral error and the sideslip actually flown, + 10 real laps
  fig6_multilap     20-lap chains without a reset of the 10-lap networks (long_eval.py)
  fig7_plans        held-out drift plans (beta 14 / 21, never trained on) and grip (beta 0), both scores, + 10
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--root', default=HERE)
ap.add_argument('--out', default=None)
ap.add_argument('--tag', default='', help='appended to the titles')
a = ap.parse_args()
ROOT = os.path.abspath(a.root)
OUT = a.out or (os.path.normpath(os.path.join(HERE, '..', '..', 'paper', 'car', 'figures')) if ROOT == HERE else os.path.join(ROOT, 'figs'))   # src/paper
os.makedirs(OUT, exist_ok=True)
TAG = f' -- {a.tag}' if a.tag else ''
CARS = ['real_nom', 'real_mass', 'real_mu', 'real_act', 'real_lag']
CAR_LAB = {'real_nom': 'nominal', 'real_mass': 'mass +25 %', 'real_mu': 'friction x0.8', 'real_act': 'actuators',
           'real_lag': 'tire / steering lag'}
G = dict(drift=['mocap_square2fast_b25', 'mocap_figfast_b25'],
         heldout=['mocap_square2fast_b21', 'mocap_figfast_b21', 'mocap_square2fast_b14', 'mocap_figfast_b14'],
         grip=['mocap_square2fast_b0', 'mocap_figfast_b0'])
RUNS = 12                                        # runs per car x plan when an eval has no 'n'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', pink='#CC79A7', sky='#56B4E9', red='#D55E00',
         yellow='#F0E442', grey='#7f7f7f', black='#222222')
SURF = '#fcfcfb'
plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 130, 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'savefig.bbox': 'tight',
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
BUDGETS = (0, 2, 10)
SCORES = (('success', 'success (%)'), ('complete', 'completes all 5 laps (%)'))


def ev(arm):
    f = os.path.join(ROOT, 'runs/hw', arm, 'eval_big.json')
    return json.load(open(f)) if os.path.exists(f) else None


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    return p, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)


def score(m, key):
    return 1.0 - m['fail'] if key == 'complete' else m['success']


def rate(arms, group='drift', cars=None, key='success'):
    """Pooled over the arms (seeds) x cars x plans x runs -> (%, CI half-width %, n)."""
    k = n = 0
    for arm in arms:
        e = ev(arm)
        if not e:
            continue
        for c in cars or CARS:
            for p in G[group]:
                if c in e and p in e[c]:
                    nn = e[c][p].get('n', RUNS)
                    k += score(e[c][p], key) * nn
                    n += nn
    p, h = wilson(round(k), n)
    return 100 * p, 100 * h, n


def mean_of(arms, key, group='drift', scale=1.0):
    v = []
    for arm in arms:
        e = ev(arm) or {}
        v += [e[c][p][key] * scale for c in CARS if c in e for p in G[group] if p in e[c] and e[c][p].get(key) is not None]
    return float(np.mean(v)) if v else np.nan


def have(arms):
    return [x for x in arms if ev(x)]


def at(nets, L):
    """The arms of these sim networks at L real laps (0: zero-shot)."""
    return have([f'{n}_zs' if L == 0 else f'{n}_lap{L}' for n in nets])


# (label, colour, marker, sim networks (seeds pooled), hardware-stage seed variants)
METHODS = [
    ('ours + DR (task objective)', C['blue'], '*', ['v3drBt_s0'], []),
    ('ours + DR (B)', C['green'], '^', ['v3drB_s0', 'v3drB_s1'], []),
    ('ours + DR (A)', C['sky'], 'v', ['v3drA_s0', 'v3drA_s1'], []),
    ('ours, nominal sim', C['black'], 'o', ['v3nom_s0', 'v3nom_s1', 'v3nom_s2'], []),
    ('PPO+DR + our hardware stage', C['pink'], 's', ['ppo_dr'], ['_s2']),
    ('RMA (zero-shot)', C['red'], 'D', ['rma'], []),
    ('FADA (zero-shot)', C['orange'], 'P', ['fada_hw24'], []),
    ('plan LQR', C['grey'], 'x', ['lqr'], []),
]


def method_arms(nets, var, L):
    arms = at(nets, L)
    if L:
        arms += have([f'{n}_lap{L}{v}' for n in nets for v in var])
    return arms


def trackilc(key, group='drift'):
    """Per-plan ILC (no network): k laps on each plan -> {laps on the group's plans per car: (p, h, n)}."""
    f = os.path.join(ROOT, 'runs/trackilc_laps/summary.json')
    if not os.path.exists(f):
        return {}
    out = {}
    for k, e in json.load(open(f))['budgets'].items():
        kk = nn = 0
        for c in CARS:
            for p in G[group]:
                if c in e and p in e[c]:
                    n = e[c][p].get('n', RUNS)
                    kk += score(e[c][p], key) * n
                    nn += n
        pp, hh = wilson(round(kk), nn)
        out[len(G[group]) * int(k)] = (100 * pp, 100 * hh, nn)
    return out


def fig1():
    fig, axs = plt.subplots(2, 1, figsize=(13.5, 8.2), sharex=True)
    for ax, (key, yl) in zip(axs, SCORES):
        rows = []
        for lab, col, mk, nets, var in METHODS:
            for L in BUDGETS:
                arms = method_arms(nets, var, L)
                if arms:
                    rows.append((lab, L, col, rate(arms, key=key)))
        ti = trackilc(key)
        for L in (2, 10):
            if L in ti:
                rows.append(('per-plan ILC (no network)', L, C['grey'], ti[L]))
        labs = list(dict.fromkeys(r[0] for r in rows))
        w = 0.26
        for r_lab, L, col, (p, h, n) in rows:
            i = labs.index(r_lab)
            j = BUDGETS.index(L)
            x = i + (j - 1) * w
            hatch, alpha = ('//', 1.0) if L == 0 else (None, 0.45 if L == 2 else 1.0)
            ax.bar(x, p, w * 0.92, color=SURF if L == 0 else col, alpha=alpha if L else 1, edgecolor=col, hatch=hatch,
                   lw=1.1)
            ax.errorbar(x, p, yerr=h, color='#333', capsize=1.5, lw=0.7)
            ax.text(x, min(p + h + 1.5, 103), f'{p:.0f}', ha='center', fontsize=6.5, color='#333')
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=15, ha='right', fontsize=8.5)
        ax.set_ylabel(yl)
        ax.set_ylim(0, 110)
        ax.grid(axis='x', alpha=0)
    from matplotlib.patches import Patch
    axs[0].legend([Patch(facecolor=SURF, edgecolor='#555', hatch='//'), Patch(facecolor='#555', alpha=0.45),
                   Patch(facecolor='#555')], ['zero-shot', '+ 2 real laps', '+ 10 real laps'], fontsize=8,
                  frameon=False, ncol=3, loc='upper right')
    axs[0].set_title(f'Full-drift (beta 25) plans, 5 cars x 2 tracks x 5-lap chained runs: success (top) and full '
                     f'completion (bottom){TAG}', fontsize=10)
    fig.savefig(os.path.join(OUT, 'fig1_methods.png'))
    plt.close(fig)


def fig2():
    sel = [m for m in METHODS if m[0] in ('ours + DR (task objective)', 'ours + DR (B)', 'ours, nominal sim',
                                         'PPO+DR + our hardware stage', 'RMA (zero-shot)', 'plan LQR')]
    fig, axs = plt.subplots(2, 1, figsize=(12.5, 7.6), sharex=True)
    for ax, (key, yl) in zip(axs, SCORES):
        rows = [(lab + ('' if L else ', zero-shot') + (f' + {L}' if L else ''), col, L, method_arms(nets, var, L))
                for lab, col, mk, nets, var in sel for L in (0, 10) if method_arms(nets, var, L)]
        w = 0.85 / len(rows)
        for j, (lab, col, L, arms) in enumerate(rows):
            for i, c in enumerate(CARS):
                p, h, n = rate(arms, cars=[c], key=key)
                x = i + (j - (len(rows) - 1) / 2) * w
                ax.bar(x, p, w * 0.92, color=col if L else SURF, edgecolor=col, hatch=None if L else '//', lw=1.0,
                       label=lab if i == 0 else None)
                ax.errorbar(x, p, yerr=h, color='#333', capsize=1.2, lw=0.6)
        ax.set_ylabel(yl)
        ax.set_ylim(0, 108)
        ax.grid(axis='x', alpha=0)
    axs[1].set_xticks(range(len(CARS)))
    axs[1].set_xticklabels([CAR_LAB[c] for c in CARS])
    axs[1].legend(fontsize=7.5, frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    axs[0].set_title(f'Per car, zero-shot (hatched) and + 10 real laps (beta 25 plans){TAG}', fontsize=10)
    fig.savefig(os.path.join(OUT, 'fig2_per_car.png'))
    plt.close(fig)


def fig3():
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    Ls = (0, 2, 4, 6, 8, 10)
    for lab, col, mk, nets, var in METHODS:
        pts = [(L, method_arms(nets, var, L)) for L in Ls]
        pts = [(L, aa) for L, aa in pts if aa]
        if len(pts) < 2:
            continue
        for ax, (key, _) in zip(axs[:2], SCORES):
            r = [rate(aa, key=key) for _, aa in pts]
            ax.errorbar([L for L, _ in pts], [q[0] for q in r], yerr=[q[1] for q in r], color=col, marker=mk,
                        ms=9 if mk == '*' else 6, lw=2.2 if 'ours' in lab else 1.4, capsize=2, label=lab, mec='#333',
                        mew=0.4, zorder=5 if 'ours' in lab else 3)
        axs[2].plot([L for L, _ in pts], [mean_of(aa, 'rms_ey', scale=100) for _, aa in pts], color=col, marker=mk,
                    ms=9 if mk == '*' else 6, lw=2.2 if 'ours' in lab else 1.4, mec='#333', mew=0.4)
    for ax, (key, _) in zip(axs[:2], SCORES):
        ti = trackilc(key)
        if ti:
            xs = sorted(ti)
            ax.errorbar(xs, [ti[x][0] for x in xs], yerr=[ti[x][1] for x in xs], color=C['grey'], marker='d', ms=5,
                        lw=1.2, ls='--', capsize=2, label='per-plan ILC (no network; laps on the two beta-25 plans)')
    for ax, (key, yl) in zip(axs, SCORES + (('', 'RMS lateral error, completed runs (cm)'),)):
        ax.set_xticks(Ls)
        ax.set_xlabel('real laps per car')
        ax.set_ylabel(yl if key != 'success' else 'full-drift success (%)')
    for ax in axs[:2]:
        ax.set_ylim(0, 105)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(f'Real-lap budget, 0-10 laps per car (beta 25 plans){TAG}', fontsize=11)
    fig.savefig(os.path.join(OUT, 'fig3_budget.png'))
    plt.close(fig)


def fig5():
    rows = [(lab + (', zero-shot' if L == 0 else f' + {L}'), col, '//' if L == 0 else None, method_arms(nets, var, L))
            for lab, col, mk, nets, var in METHODS for L in (0, 10) if method_arms(nets, var, L)]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, key, sc, yl in ((axs[0], 'rms_ey', 100, 'RMS lateral error, completed runs (cm)'),
                            (axs[1], 'mean_abs_beta', 1, 'mean |sideslip| flown (deg)')):
        for i, (lab, col, hatch, aa) in enumerate(rows):
            v = mean_of(aa, key, scale=sc)
            if np.isnan(v):
                continue
            ax.bar(i, v, 0.7, color=col if not hatch else SURF, edgecolor=col, hatch=hatch, lw=1.1)
            ax.text(i, v + 0.3, f'{v:.1f}', ha='center', fontsize=7, color='#333')
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r[0] for r in rows], rotation=40, ha='right', fontsize=8)
        ax.set_ylabel(yl)
        ax.grid(axis='x', alpha=0)
    axs[0].axhline(10, color=C['grey'], ls=':', lw=1)
    axs[0].text(0, 10.3, 'the success band: 10 cm', fontsize=8, color=C['grey'])
    axs[1].axhline(25, color=C['grey'], ls=':', lw=1)
    axs[1].text(0, 25.5, 'the plans ask for 25 deg', fontsize=8, color=C['grey'])
    fig.suptitle(f'Accuracy and the sideslip actually flown, zero-shot and + 10 real laps (beta 25 plans){TAG}',
                 fontsize=10)
    fig.savefig(os.path.join(OUT, 'fig5_accuracy.png'))
    plt.close(fig)


def fig6():
    arms = [('ours + DR (task objective), zero-shot', 'v3drBt_s0_zs', C['blue'], '//'),
            ('ours + DR (task objective) + 10', 'v3drBt_s0_lap10', C['blue'], None),
            ('ours + DR (B) + 10', 'v3drB_s0_lap10', C['green'], None),
            ('ours, nominal sim + 10', 'v3nom_s0_lap10', C['black'], None),
            ('PPO+DR + 10 (our stage)', 'ppo_dr_lap10', C['pink'], None)]
    arms = [x for x in arms if os.path.exists(os.path.join(ROOT, 'runs/hw', x[1], 'long_eval.json'))]
    if not arms:
        print('fig6: no 10-lap long_eval.json yet (scheduler_lap10.py long_l10)')
        return
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.2))
    w = 0.8 / len(arms)
    for j, (lab, arm, col, hatch) in enumerate(arms):
        L = json.load(open(os.path.join(ROOT, 'runs/hw', arm, 'long_eval.json')))
        for i, c in enumerate(CARS):
            v = [L[c][p] for p in G['drift'] if c in L and p in L[c]]
            x = i + (j - (len(arms) - 1) / 2) * w
            axs[0].bar(x, np.mean([q['laps'] for q in v]), w * 0.92, color=col if not hatch else SURF, edgecolor=col,
                       hatch=hatch, lw=1.1, label=lab if i == 0 else None)
            fl = [q['lap_ey_first_mid_last'] for q in v if not q['failed'] and q.get('lap_ey_first_mid_last')]
            if fl:
                m = np.nanmean(np.array(fl, float), 0)
                axs[1].plot([x - w * 0.3, x, x + w * 0.3], m, color=col, marker='o', ms=3, lw=1.2)
    for ax in axs:
        ax.set_xticks(range(len(CARS)))
        ax.set_xticklabels([CAR_LAB[c] for c in CARS], fontsize=8.5)
        ax.grid(axis='x', alpha=0)
    axs[0].axhline(20, color=C['grey'], ls=':', lw=1)
    axs[0].set_ylabel('laps completed of 20 (no reset)')
    axs[1].set_ylabel('RMS e_y of the first / middle / last lap (cm)')
    axs[0].legend(fontsize=8, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(1.1, -0.1))
    fig.suptitle(f'Continuous driving: 20-lap chains on the beta-25 plans after 10 real laps (a flat line = no drift-off)'
                 f'{TAG}', fontsize=10)
    fig.savefig(os.path.join(OUT, 'fig6_multilap.png'))
    plt.close(fig)


def fig7():
    fig, axs = plt.subplots(2, 2, figsize=(14, 7.6), sharey=True)
    for jj, (grp, t) in enumerate((('heldout', 'held-out drift plans (beta 14 / 21: never trained on)'),
                                   ('grip', 'grip plans (beta 0)'))):
        for ii, (key, yl) in enumerate(SCORES):
            ax = axs[ii, jj]
            rows = [(lab + (', 0-shot' if L == 0 else f' + {L}'), col, L, method_arms(nets, var, L))
                    for lab, col, mk, nets, var in METHODS for L in (0, 10) if method_arms(nets, var, L)]
            for i, (lab, col, L, arms) in enumerate(rows):
                p, h, n = rate(arms, grp, key=key)
                ax.bar(i, p, 0.7, color=col if L else SURF, edgecolor=col, hatch=None if L else '//', lw=1.1)
                ax.errorbar(i, p, yerr=h, color='#333', capsize=1.5, lw=0.7)
            ax.set_xticks(range(len(rows)))
            ax.set_xticklabels([r[0] for r in rows] if ii == 1 else [''] * len(rows), rotation=45, ha='right',
                               fontsize=7.5)
            ax.set_ylim(0, 105)
            ax.grid(axis='x', alpha=0)
            if jj == 0:
                ax.set_ylabel(yl)
            if ii == 0:
                ax.set_title(t, fontsize=10)
    fig.suptitle(f'Beyond the training plans: zero-shot (hatched) and + 10 real laps{TAG}', fontsize=11)
    fig.savefig(os.path.join(OUT, 'fig7_plans.png'))
    plt.close(fig)


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig5, fig6, fig7):
        try:
            f()
            print('ok', f.__name__)
        except Exception as e:                       # one missing input must not stop the rest
            print('FAILED', f.__name__, repr(e))
    print('->', OUT)
