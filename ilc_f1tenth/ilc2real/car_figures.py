#!/usr/bin/env python3
"""
car_figures.py [--root DIR] [--out DIR] [--tag TEXT]: the F1TENTH ILC2Real figures (the quadruped's quad_figures.py
set, for the car), from a results root (runs/hw/<arm>/eval_big.json: 5 multi-body cars x 8 plans x n runs (6 on the tight tracks) x 5 chained
laps; long_eval.json: 20-lap chains). Success = on track, RMS e_y <= 10 cm, pace >= 90 % (grip or drift); 95 % Wilson
intervals over cars x plans x runs. --root archive_mu02_wide (the original tracks) until the tight-track rerun is in;
then the default root (.) regenerates them.

  fig1_methods      full-drift (beta 25) success, zero-shot and after 24 real laps, per method
  fig2_per_car      full-drift success per car, our learner + DR and the baselines
  fig3_budget       success and RMS lateral error against real laps per car (0 .. 96)
  fig4_lap_budget   the small-budget ablation (fig_laps.py's figure, from the same root)
  fig5_accuracy     RMS lateral error and the sideslip actually flown (the plans ask for 25 deg)
  fig6_multilap     20-lap chains without a reset: laps completed and the first / middle / last lap's e_y
  fig7_plans        success on the held-out drift plans (14 / 21 deg, never trained on) and on grip (0 deg)
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--root', default=HERE)
ap.add_argument('--out', default=None)
ap.add_argument('--tag', default='', help='appended to the titles (e.g. "original tracks")')
a = ap.parse_args()
ROOT = os.path.abspath(a.root)
OUT = a.out or os.path.join(ROOT, 'figs')
os.makedirs(OUT, exist_ok=True)
TAG = f' -- {a.tag}' if a.tag else ''
CARS = ['real_nom', 'real_mass', 'real_mu', 'real_act', 'real_lag']
CAR_LAB = {'real_nom': 'nominal', 'real_mass': 'mass +25 %', 'real_mu': 'friction x0.8', 'real_act': 'actuators',
           'real_lag': 'tire / steering lag'}
G = dict(drift=['mocap_square2fast_b25', 'mocap_figfast_b25'],
         heldout=['mocap_square2fast_b21', 'mocap_figfast_b21', 'mocap_square2fast_b14', 'mocap_figfast_b14'],
         grip=['mocap_square2fast_b0', 'mocap_figfast_b0'])
RUNS = 12                                        # runs per car x plan when an eval has no 'n'
# the paper palette (the quadruped's quad_figures.py)
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', pink='#CC79A7', sky='#56B4E9', red='#D55E00',
         yellow='#F0E442', grey='#7f7f7f', black='#222222')
SURF = '#fcfcfb'
plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 130, 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'savefig.bbox': 'tight',
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})


def ev(arm):
    d = os.path.join(ROOT, 'runs/hw', arm)
    for f in ('eval_big_adaptive.json', 'eval_big.json'):
        if os.path.exists(os.path.join(d, f)):
            return json.load(open(os.path.join(d, f)))
    s = os.path.join(d, 'summary.json')
    return json.load(open(s)).get('eval') if os.path.exists(s) else None


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    return p, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d


def rate(arms, group='drift', cars=None):
    """Pooled success over the arms (seeds) x cars x plans x runs -> (%, CI half-width %, n)."""
    k = n = 0
    for arm in arms if isinstance(arms, (list, tuple)) else [arms]:
        e = ev(arm)
        if not e:
            continue
        for c in cars or CARS:
            for p in G[group]:
                if c in e and p in e[c]:
                    k += e[c][p]['success'] * e[c][p].get('n', RUNS)
                    n += e[c][p].get('n', RUNS)
    p, h = wilson(round(k), n)
    return 100 * p, 100 * h, n


def mean_of(arms, key, group='drift', scale=1.0):
    v = []
    for arm in arms if isinstance(arms, (list, tuple)) else [arms]:
        e = ev(arm) or {}
        v += [e[c][p][key] * scale for c in CARS if c in e for p in G[group] if p in e[c] and e[c][p].get(key) is not None]
    return float(np.mean(v)) if v else np.nan


def have(arms):
    return [x for x in arms if ev(x)]


# the methods: (label, colour, zero-shot arms, + 24 real laps arms) -- seeds pooled
METHODS = [
    ('ours: nominal-sim learner', C['blue'], ['v3nom_s0_zs', 'v3nom_s1_zs', 'v3nom_s2_zs'],
     ['v3nom_s0_hw24', 'v3nom_s1_hw24', 'v3nom_s2_hw24']),
    ('ours: our learner + DR (A)', C['sky'], ['v3drA_s0_zs', 'v3drA_s1_zs'], ['v3drA_s0_hw24', 'v3drA_s1_hw24']),
    ('ours: our learner + DR (B)', C['green'], ['v3drB_s0_zs', 'v3drB_s1_zs'], ['v3drB_s0_hw24', 'v3drB_s1_hw24']),
    ('PPO+DR (+ our hardware stage)', C['pink'], ['ppo_dr_zs'], ['ppo_dr_hw24', 'ppo_dr_hw24_s2']),
    ('RMA', C['red'], ['rma_zs'], []),
    ('FADA (its own adaptation)', C['orange'], ['fada_hw24_zs'], ['fada_hw24']),
    ('plan LQR', C['grey'], ['lqr_zs'], []),
]

SHORT = {'ours: nominal-sim learner': 'ours', 'ours: our learner + DR (A)': 'ours+DR (A)',
         'ours: our learner + DR (B)': 'ours+DR (B)', 'PPO+DR (+ our hardware stage)': 'PPO+DR',
                  'RMA': 'RMA', 'FADA (its own adaptation)': 'FADA', 'plan LQR': 'plan LQR'}


def bars(ax, rows, ylab, ylim=(0, 105)):
    """rows: (label, colour, hatch, (p, h, n)) -> horizontal-free vertical bars with intervals and values."""
    x = np.arange(len(rows))
    for i, (lab, col, hatch, (p, h, n)) in enumerate(rows):
        if np.isnan(p):
            continue
        ax.bar(i, p, 0.7, color=col if not hatch else SURF, edgecolor=col, hatch=hatch, lw=1.2)
        ax.errorbar(i, p, yerr=h, color='#333', capsize=3, lw=1)
        ax.text(i, min(p + h + 1.5, ylim[1] - 4), f'{p:.0f}', ha='center', fontsize=8, color='#333')
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=30, ha='right', fontsize=8.5)
    ax.set_ylabel(ylab)
    ax.set_ylim(*ylim)
    ax.grid(axis='x', alpha=0)


def fig1():
    rows = []
    for lab, col, zs, hw in METHODS:
        if have(zs):
            rows.append((f'{lab}\nzero-shot', col, '//', rate(have(zs))))
        if have(hw):
            rows.append((f'{lab}\n+ 24 real laps', col, None, rate(have(hw))))
    tl = os.path.join(ROOT, 'runs/trackilc_big/summary.json')
    if os.path.exists(tl):                         # per-track ILC: its own budgets (per plan, 8 plans)
        b = json.load(open(tl))['budgets']
        for k in sorted(b, key=int):
            e = b[k]
            kk = sum(e[c][p]['success'] * e[c][p].get('n', RUNS) for c in CARS if c in e for p in G['drift'] if p in e[c])
            nn = sum(e[c][p].get('n', RUNS) for c in CARS if c in e for p in G['drift'] if p in e[c])
            pp, hh = wilson(round(kk), nn)
            rows.append((f'per-track ILC\n{k} laps / plan ({8 * int(k)})', C['black'], None, (100 * pp, 100 * hh, nn)))
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    bars(ax, rows, 'full-drift success (%)')
    ax.set_title(f'Zero-shot (hatched) and after 24 real laps per car: beta 25 plans, 5 cars x 2 tracks x 12 '
                 f'five-lap runs{TAG}', fontsize=10)
    fig.savefig(os.path.join(OUT, 'fig1_methods.png'))
    plt.close(fig)


def fig2():
    arms = [('ours, zero-shot', C['blue'], '//', ['v3nom_s0_zs', 'v3nom_s1_zs', 'v3nom_s2_zs']),
            ('ours + 24', C['blue'], None, ['v3nom_s0_hw24', 'v3nom_s1_hw24', 'v3nom_s2_hw24']),
            ('our learner + DR (B), zero-shot', C['green'], '//', ['v3drB_s0_zs', 'v3drB_s1_zs']),
            ('our learner + DR (B) + 24', C['green'], None, ['v3drB_s0_hw24', 'v3drB_s1_hw24']),
            ('PPO+DR, zero-shot', C['pink'], '//', ['ppo_dr_zs']),
            ('PPO+DR + 2 laps of our stage', C['pink'], None, ['ppo_dr_lap2']),
            ('RMA, zero-shot', C['red'], '//', ['rma_zs'])]
    arms = [x for x in arms if have(x[3])]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    w = 0.8 / len(arms)
    for j, (lab, col, hatch, aa) in enumerate(arms):
        for i, c in enumerate(CARS):
            p, h, n = rate(have(aa), cars=[c])
            x = i + (j - (len(arms) - 1) / 2) * w
            ax.bar(x, p, w * 0.92, color=col if not hatch else SURF, edgecolor=col, hatch=hatch, lw=1.1,
                   label=lab if i == 0 else None)
            ax.errorbar(x, p, yerr=h, color='#333', capsize=1.5, lw=0.7)
    ax.set_xticks(range(len(CARS)))
    ax.set_xticklabels([CAR_LAB[c] for c in CARS])
    ax.set_ylabel('full-drift success (%)')
    ax.set_ylim(0, 108)
    ax.grid(axis='x', alpha=0)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    ax.set_title(f'Per car (multi-body "real" cars; the GPU sim stays nominal){TAG}', fontsize=10)
    fig.savefig(os.path.join(OUT, 'fig2_per_car.png'))
    plt.close(fig)


def fig3():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    series = [('ours: nominal-sim learner + our hardware stage', C['blue'], 'o',
               [(0, ['v3nom_s0_zs', 'v3nom_s1_zs', 'v3nom_s2_zs']), (24, ['v3nom_s0_hw24', 'v3nom_s1_hw24', 'v3nom_s2_hw24']),
                (48, ['v3nom_s0_hw48']), (96, ['v3nom_s0_hw96'])]),
              ('ours: our learner + DR (A) + our stage', C['sky'], 'v', [(0, ['v3drA_s0_zs', 'v3drA_s1_zs']),
                                                                        (24, ['v3drA_s0_hw24', 'v3drA_s1_hw24'])]),
              ('ours: our learner + DR (B) + our stage', C['green'], '^', [(0, ['v3drB_s0_zs', 'v3drB_s1_zs']),
                                                                         (24, ['v3drB_s0_hw24', 'v3drB_s1_hw24'])]),
              ('PPO+DR + our hardware stage', C['pink'], 's', [(0, ['ppo_dr_zs']), (2, ['ppo_dr_lap2']), (4, ['ppo_dr_lap4']),
                                                               (10, ['ppo_dr_lap10']), (24, ['ppo_dr_hw24', 'ppo_dr_hw24_s2']),
                                                               (48, ['ppo_dr_hw48', 'ppo_dr_hw48_s2', 'ppo_dr_hw48_s3'])])]
    for lab, col, mk, pts in series:
        pts = [(x, have(aa)) for x, aa in pts if have(aa)]
        if not pts:
            continue
        r = [(x, *rate(aa)) for x, aa in pts]
        axs[0].errorbar([q[0] for q in r], [q[1] for q in r], yerr=[q[2] for q in r], color=col, marker=mk, ms=6,
                        lw=1.8, capsize=3, label=lab)
        axs[1].plot([x for x, _ in pts], [mean_of(aa, 'rms_ey', scale=100) for _, aa in pts], color=col, marker=mk,
                    ms=6, lw=1.8, label=lab)
    tl = os.path.join(ROOT, 'runs/trackilc_big/summary.json')
    if os.path.exists(tl):
        b = json.load(open(tl))['budgets']
        r = []
        for k in sorted(b, key=int):
            e = b[k]
            kk = sum(e[c][p]['success'] * e[c][p].get('n', RUNS) for c in CARS if c in e for p in G['drift'] if p in e[c])
            nn = sum(e[c][p].get('n', RUNS) for c in CARS if c in e for p in G['drift'] if p in e[c])
            pp, hh = wilson(round(kk), nn)
            r.append((2 * int(k), 100 * pp, 100 * hh))
        axs[0].errorbar([q[0] for q in r], [q[1] for q in r], yerr=[q[2] for q in r], color=C['black'], marker='D',
                        ms=5, lw=1.2, ls='--', capsize=3, label='per-track ILC (laps on the two beta-25 plans)')
    for ax in axs:
        ax.set_xscale('symlog', linthresh=10, linscale=0.8)
        ax.set_xticks([0, 2, 4, 10, 24, 48, 96])
        ax.set_xticklabels(['0', '2', '4', '10', '24', '48', '96'])
        ax.set_xlabel('real laps per car')
    axs[0].set_ylabel('full-drift success (%)')
    axs[0].set_ylim(0, 105)
    axs[1].set_ylabel('RMS lateral error, completed runs (cm)')
    axs[0].legend(fontsize=7.5, frameon=False, loc='lower right')
    fig.suptitle(f'Real-lap budget{TAG}', fontsize=11)
    fig.savefig(os.path.join(OUT, 'fig3_budget.png'))
    plt.close(fig)


def fig4():
    subprocess.run([sys.executable, os.path.join(HERE, 'fig_laps.py'), '--root', ROOT, '--out',
                    os.path.join(OUT, 'fig4_lap_budget.png')], check=True, stdout=subprocess.DEVNULL)


def fig5():
    rows = [(SHORT[lab] + (' + 24' if hw else ', zero-shot'), col, None if hw else '//', aa)
            for lab, col, zs, hwa in METHODS for hw, aa in ((False, zs), (True, hwa)) if have(aa)]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, key, sc, yl in ((axs[0], 'rms_ey', 100, 'RMS lateral error, completed runs (cm)'),
                            (axs[1], 'mean_abs_beta', 1, 'mean |sideslip| flown (deg)')):
        for i, (lab, col, hatch, aa) in enumerate(rows):
            v = mean_of(have(aa), key, scale=sc)
            ax.bar(i, v, 0.7, color=col if not hatch else SURF, edgecolor=col, hatch=hatch, lw=1.1)
            ax.text(i, v + 0.3, f'{v:.1f}', ha='center', fontsize=7.5, color='#333')
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r[0] for r in rows], rotation=35, ha='right', fontsize=8.5)
        ax.set_ylabel(yl)
        ax.grid(axis='x', alpha=0)
    axs[1].axhline(25, color=C['grey'], ls=':', lw=1)
    axs[1].text(0, 25.5, 'the plans ask for 25 deg', fontsize=8, color=C['grey'])
    fig.suptitle(f'Accuracy and the sideslip actually flown (beta-25 plans; grip or drift is the car\'s choice){TAG}',
                 fontsize=10)
    fig.savefig(os.path.join(OUT, 'fig5_accuracy.png'))
    plt.close(fig)


def fig6():
    arms = [('ours, zero-shot', 'v3nom_s0_zs', C['blue'], '//'), ('ours + 24', 'v3nom_s0_hw24', C['blue'], None),
            ('PPO+DR + 48 (our stage)', 'ppo_dr_hw48', C['pink'], None),
            ('ours + goal fallback + 24', 'v6nom_s1_hw24fb', C['blue'], 'xx')]
    arms = [x for x in arms if os.path.exists(os.path.join(ROOT, 'runs/hw', x[1], 'long_eval.json'))]
    if not arms:
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
    axs[0].legend(fontsize=8, frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(1.1, -0.1))
    fig.suptitle(f'Continuous driving: 20-lap chains on the beta-25 plans (completed chains; a flat line = no drift-off){TAG}',
                 fontsize=10)
    fig.savefig(os.path.join(OUT, 'fig6_multilap.png'))
    plt.close(fig)


def fig7():
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, grp, t in ((axs[0], 'heldout', 'held-out drift plans (beta 14 / 21: never trained on)'),
                       (axs[1], 'grip', 'grip plans (beta 0)')):
        rows = []
        for lab, col, zs, hw in METHODS:
            if have(zs):
                rows.append((f'{SHORT[lab]}, zero-shot', col, '//', rate(have(zs), grp)))
            if have(hw):
                rows.append((f'{SHORT[lab]} + 24', col, None, rate(have(hw), grp)))
        bars(ax, rows, 'success (%)')
        ax.set_title(t, fontsize=10)
    fig.suptitle(f'Beyond the training plans{TAG}', fontsize=11)
    fig.savefig(os.path.join(OUT, 'fig7_plans.png'))
    plt.close(fig)


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        try:
            f()
            print('ok', f.__name__)
        except Exception as e:                       # one missing input must not stop the rest
            print('FAILED', f.__name__, repr(e))
    print('->', OUT)
