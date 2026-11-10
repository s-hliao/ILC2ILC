#!/usr/bin/env python3
"""
ladder_figs.py [--out DIR]: the low-friction fix (the reference ladder) and the exact-dynamics oracles, from the 12-run
evaluations (runs/hw/<arm>/eval_big_adaptive.json, else eval_big.json; read only). Style as car_study_figs.py.
Success = on track, RMS e_y <= 10 cm, pace >= 90 % of the plan flown; 95 % Wilson intervals over cars x plans x runs.
  fig16_ladder_per_car   per car (the 5 + friction x0.9 / x0.7 / x0.6), b25 success: ours without the ladder (10 laps),
                         the ladder (10 laps) on sim seeds 0 / 2 / 1, and the exact-dynamics RL oracles at 5,000 laps
  fig17_exact_matching   the car under EXACT dynamics: b25 success and crashes vs real laps per car for RMA (true latent)
                         and PPO+DR trained on each multi-body car, FADA with per-car exact IDMs (+ the fixed lookahead,
                         + adapted without walls where evaluated), and ours (the ladder, 9 laps)
  fig18_ladder_rungs     the rung each car settled on per track and seed, and each rung's speed (mean |V| vs nominal)
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
ap.add_argument('--out', default=os.path.normpath(os.path.join(HERE, '..', '..', 'paper', 'car', 'figures')))
a = ap.parse_args()
OUT = a.out
os.makedirs(OUT, exist_ok=True)
CARS = ['real_nom', 'real_mass', 'real_mu', 'real_act', 'real_lag', 'ax:mu=0.9', 'ax:mu=0.7', 'ax:mu=0.6']
CAR_LAB = {'real_nom': 'nominal', 'real_mass': 'mass +25 %', 'real_mu': 'friction x0.8', 'real_act': 'actuators',
           'real_lag': 'tire / steer lag', 'ax:mu=0.9': 'friction x0.9', 'ax:mu=0.7': 'friction x0.7',
           'ax:mu=0.6': 'friction x0.6'}
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', pink='#CC79A7', sky='#56B4E9', red='#D55E00',
         yellow='#F0E442', grey='#7f7f7f', black='#222222')
SURF = '#fcfcfb'
plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 130, 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'savefig.bbox': 'tight',
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
B25 = ('mocap_square2fast_b25', 'mocap_figfast_b25')


def ev(arm):
    for f in ('eval_big_adaptive.json', 'eval_big.json'):
        p = os.path.join(HERE, 'runs/hw', arm, f)
        if os.path.exists(p):
            return json.load(open(p))
    return None


def runs_n(arm):
    try:
        return int(json.load(open(os.path.join(HERE, 'runs/hw', arm, 'args.json'))).get('eval_runs') or 12)
    except (OSError, ValueError):
        return 12


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    return p, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)


def pooled(e, cars, plans=B25, n_run=12, key='success'):
    """(rate %, CI %) of `key` over cars x plans x runs (12 per plan: reeval.py)."""
    k = n = 0
    for c in cars:
        for p in plans:
            m = (e or {}).get(c, {}).get(p)
            if m is None:
                continue
            k += m[key] * n_run
            n += n_run
    if n == 0:
        return np.nan, np.nan
    p_, ci = wilson(k, n)
    return 100 * p_, 100 * ci


def laps_of(arm):
    s = json.load(open(os.path.join(HERE, 'runs/hw', arm, 'summary.json')))
    v = list((s.get('real_laps') or {}).values())
    return float(np.mean(v)) if v else 0.0


# ---- fig16: per car -------------------------------------------------------------------------------------------------
def fig16():
    nolad = {c: ev('v3drBt_s0_lap10') for c in CARS[:5]}
    nolad.update({'ax:mu=0.9': ev('ax_mu0.9_drBt_lap10'), 'ax:mu=0.7': ev('ax_mu0.7_drBt_lap10'), 'ax:mu=0.6': None})
    series = [('ours, no ladder (v3drBt, 10 laps)', C['grey'], lambda c: nolad.get(c)),
              ('ladder, sim seed 0', C['blue'], lambda c: ev('v3drBtL_s0_ladR_lap10')),
              ('ladder, sim seed 2', C['sky'], lambda c: ev('v3drBtL_s2_ladR_lap10')),
              ('ladder, sim seed 1', C['pink'], lambda c: ev('v3drBtL_s1_ladR_lap10')),
              ('RMA, exact dynamics, 5,000 laps', C['orange'], lambda c: ev('rma_exact_ep5000')),
              ('PPO+DR, exact dynamics, 5,000 laps', C['green'], lambda c: ev('ppo_exact_ep5000'))]
    fig, ax = plt.subplots(figsize=(15, 4.6))
    w = 0.8 / len(series)
    x = np.arange(len(CARS))
    for i, (lab, col, get) in enumerate(series):
        vals, cis = [], []
        for c in CARS:
            e = get(c)
            v, ci = pooled(e, [c]) if e is not None and c in e else (np.nan, np.nan)
            vals.append(v)
            cis.append(ci)
        xs = x - 0.4 + w * (i + 0.5)
        ax.bar(xs, np.nan_to_num(vals), w * 0.92, yerr=np.nan_to_num(cis), color=col, label=lab, capsize=2,
               error_kw=dict(lw=0.8, alpha=0.7))
        for xi, v in zip(xs, vals):
            if np.isnan(v):
                ax.text(xi, 2, 'n/a', ha='center', va='bottom', fontsize=6.5, rotation=90, color=C['grey'])
            elif v < 1:                          # a 0 % bar has no height: say so
                ax.text(xi, 2, '0', ha='center', va='bottom', fontsize=7, color=col, fontweight='bold')
    ax.axvspan(4.5, 7.5, color=C['yellow'], alpha=0.08, lw=0)
    ax.text(6.0, 111, 'friction-axis cars (the RL oracles were not run there)', ha='center', fontsize=8, color=C['grey'])
    ax.set_xticks(x, [CAR_LAB[c] for c in CARS])
    ax.set_ylim(0, 116)
    ax.set_ylabel('b25 success (%)')
    ax.set_title('The reference ladder fixes the low-friction cars (10 real laps); exact-dynamics RL matches at ~5,000 laps')
    ax.legend(ncol=6, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.savefig(os.path.join(OUT, 'fig16_ladder_per_car.png'))
    plt.close(fig)


# ---- fig17: exact-dynamics data matching ---------------------------------------------------------------------------
def fig17():
    cars5 = CARS[:5]
    fams = [('RMA (true latent), exact cars', 'rma_exact', [0, 96, 384, 1536, 3072, 5000, 10000], C['orange'], 'o'),
            ('PPO+DR, exact cars', 'ppo_exact', [0, 96, 384, 1536, 3072, 5000, 10000], C['green'], 's'),
            ('FADA, per-car exact IDMs + oracle planner', 'fada_exact', [0, 24, 96, 384, 1536, 2016], C['red'], 'v'),
            ('FADA, + fixed lookahead', 'fada_exact2', [0, 24, 96, 384, 1536, 2016], C['pink'], '^'),
            ('FADA, + adapted without walls', 'fada_nowall', [0, 24, 96, 384, 1536, 2016], C['grey'], 'D')]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    for lab, pre, eps, col, mk in fams:
        pts = []
        for k in eps:
            e = ev(f'{pre}_ep{k}')
            if e is None:
                continue
            s, ci = pooled(e, cars5)
            f, _ = pooled(e, cars5, key='fail')
            pts.append((max(laps_of(f'{pre}_ep{k}'), 1.0), s, ci, f))
        if not pts:
            continue
        P = np.array(pts)
        n_done = len(pts)
        lab_ = lab + ('' if n_done == len(eps) else f' (partial: {n_done}/{len(eps)} budgets)')
        axs[0].errorbar(P[:, 0], P[:, 1], yerr=P[:, 2], color=col, marker=mk, ms=4, lw=1.4, capsize=2, label=lab_)
        axs[1].plot(P[:, 0], P[:, 3], color=col, marker=mk, ms=4, lw=1.4, label=lab_)
    e = ev('v3drBtL_s0_lad1_lap10')
    s, ci = pooled(e, cars5)
    f, _ = pooled(e, cars5, key='fail')
    L = laps_of('v3drBtL_s0_lad1_lap10')
    axs[0].errorbar([L], [s], yerr=[ci], color=C['blue'], marker='*', ms=13, ls='none', label=f'ours (ladder, {L:.0f} laps)')
    axs[1].plot([L], [f], color=C['blue'], marker='*', ms=13, ls='none', label='ours')
    for axx, yl in ((axs[0], 'b25 success (%)'), (axs[1], 'b25 crashes (%)')):
        axx.set_xscale('log')
        axx.set_xlabel('real laps per car (1 = zero-shot)')
        axx.set_ylabel(yl)
        axx.set_ylim(-3, 105)
    axs[0].axhline(s, color=C['blue'], lw=0.8, ls=':')
    axs[0].set_title('Under exact dynamics RL reaches ours at ~5,000 laps/car (~550x); FADA does not')
    axs[1].set_title('Crashes spent on the way')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.savefig(os.path.join(OUT, 'fig17_exact_matching.png'))
    plt.close(fig)


# ---- fig18: rungs and their speed ----------------------------------------------------------------------------------
def plan_speed(name):
    d = np.load(os.path.join(os.environ.get('F1T_PLANS') or os.path.join(HERE, 'plans_mu02_tight'), name + '.npz'),
                allow_pickle=True)
    return float(np.mean(np.hypot(d['z'][:, 2], d['z'][:, 3])))


def fig18():
    arms = [('seed 0', 'v3drBtL_s0_ladR_lap10'), ('seed 2', 'v3drBtL_s2_ladR_lap10'), ('seed 1', 'v3drBtL_s1_ladR_lap10')]
    tracks = [('mocap_square2fast', 'square'), ('mocap_figfast', 'figure-8')]
    lev = {'': 0, '_mu80': 1, '_mu60': 2}
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw=dict(width_ratios=[3, 1.2]))
    M = np.full((len(arms) * len(tracks), len(CARS)), np.nan)
    rows = []
    for i, (sl, arm) in enumerate(arms):
        rungs = json.load(open(os.path.join(HERE, 'runs/hw', arm, 'summary.json'))).get('rungs', {})
        e = ev(arm)
        for j, (tk, tl) in enumerate(tracks):
            r = i * len(tracks) + j
            rows.append(f'{sl}, {tl}')
            for k, c in enumerate(CARS):
                rg = rungs.get(c, {}).get(tk, '')
                M[r, k] = lev[rg]
                m = e.get(c, {}).get(f'{tk}_b25') if e else None
                txt = {'': 'nominal', '_mu80': '0.8x', '_mu60': '0.6x'}[rg] + (f"\n{100 * m['success']:.0f} %" if m else '')
                axs[0].text(k, r, txt, ha='center', va='center', fontsize=7.5, color='white' if lev[rg] == 2 else C['black'])
    cmap = matplotlib.colors.ListedColormap(['#e8eef4', C['sky'], C['blue']])
    axs[0].imshow(M, cmap=cmap, vmin=-0.5, vmax=2.5, aspect='auto')
    axs[0].set_xticks(range(len(CARS)), [CAR_LAB[c] for c in CARS], rotation=20)
    axs[0].set_yticks(range(len(rows)), rows)
    axs[0].grid(False)
    axs[0].set_title('The rung each car settled on (re-planned for 0.8x / 0.6x friction) and its b25 success')
    sp = {}
    for rg in ('', '_mu80', '_mu60'):
        sp[rg] = np.mean([plan_speed(f'{tk}_b{b}{rg}') / plan_speed(f'{tk}_b{b}') for tk, _ in tracks for b in (0, 25)])
    bars = axs[1].bar(['nominal', '0.8x\nfriction', '0.6x\nfriction'], [100 * sp[r] for r in ('', '_mu80', '_mu60')],
                      color=['#e8eef4', C['sky'], C['blue']], edgecolor=C['grey'], lw=0.6)
    for b_ in bars:
        axs[1].text(b_.get_x() + b_.get_width() / 2, b_.get_height() + 1.5, f'{b_.get_height():.0f} %', ha='center',
                    fontsize=9)
    axs[1].set_ylim(0, 112)
    axs[1].set_ylabel('plan speed (% of nominal, mean |V|)')
    axs[1].set_title("The rungs' price in speed")
    fig.savefig(os.path.join(OUT, 'fig18_ladder_rungs.png'))
    plt.close(fig)


if __name__ == '__main__':
    fig16()
    fig17()
    fig18()
    print('wrote', ', '.join(os.path.join(OUT, f) for f in ('fig16_ladder_per_car.png', 'fig17_exact_matching.png',
                                                            'fig18_ladder_rungs.png')))
