#!/usr/bin/env python3
"""figures.py: the summary figures from the re-evaluated arms (runs/hw/*/eval_big.json; runs/trackilc_big).
  budget.png   full-drift success and RMS lateral error vs real laps per car: ours (nominal sim seed 0:
                    0 / 24 / 48 / 96 laps) and the per-track ILC (24 / 32 / 96 laps), the plans' LQR as reference
  methods.png  full-drift success zero-shot vs after 24 real laps per car, per method (mean over sim seeds)"""
import json
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.normpath(os.path.join(HERE, '..', '..', 'paper', 'car', 'figures', 'old_24lap'))   # superseded: car_figures.py
DRIFT = ['mocap_square2fast_b25', 'mocap_figfast_b25']


def ev(arm):
    a = os.path.join(HERE, 'runs/hw', arm, 'eval_big_adaptive.json')
    if os.path.exists(a):
        return json.load(open(a))
    f = os.path.join(HERE, 'runs/hw', arm, 'eval_big.json')
    return json.load(open(f)) if os.path.exists(f) else None


def drift(e):
    s = [e[c][n]['success'] for c in e for n in DRIFT if n in e[c]]
    y = [e[c][n]['rms_ey'] for c in e for n in DRIFT if n in e[c] and e[c][n]['rms_ey'] is not None]
    return 100 * np.mean(s), 100 * np.mean(y)


def main():
    os.makedirs(OLD, exist_ok=True)
    # budget curve
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ours = [(0, 'v3nom_s0_zs'), (24, 'v3nom_s0_hw24r'), (48, 'v3nom_s0_hw48'), (96, 'v3nom_s0_hw96')]
    pts = [(b, drift(ev(a))) for b, a in ours if ev(a)]
    if pts:
        ax[0].plot([b for b, _ in pts], [d[0] for _, d in pts], 'o-', color='#2b6cb0', label='ILC2Real (ours)')
        ax[1].plot([b for b, _ in pts], [d[1] for _, d in pts], 'o-', color='#2b6cb0', label='ILC2Real (ours)')
    pp = [(0, ['ppo_dr_zs']), (24, ['ppo_dr_hw24', 'ppo_dr_hw24_s2']), (48, ['ppo_dr_hw48', 'ppo_dr_hw48_s2'])]
    pts = [(b, np.mean([drift(ev(x)) for x in arms if ev(x)], axis=0)) for b, arms in pp if any(ev(x) for x in arms)]
    if pts:
        for k in (0, 1):
            ax[k].plot([b for b, _ in pts], [d[k] for _, d in pts], 'D-', color='#2f855a',
                       label='PPO+DR, then our hardware stage')
    f = os.path.join(HERE, 'runs/trackilc_big/summary.json')
    if os.path.exists(f):
        tb = json.load(open(f))['budgets']
        tp = sorted((int(k) * 8, drift(v)) for k, v in tb.items())
        ax[0].plot([b for b, _ in tp], [d[0] for _, d in tp], 's--', color='#c05621', label='per-track ILC (LQR + ff)')
        ax[1].plot([b for b, _ in tp], [d[1] for _, d in tp], 's--', color='#c05621', label='per-track ILC (LQR + ff)')
    lq = ev('lqr_zs')
    if lq:
        for k in (0, 1):
            ax[k].axhline(drift(lq)[k], color='0.5', ls=':', label='plan LQR, zero-shot')
    ax[0].set_ylabel('full-drift success (%)'); ax[1].set_ylabel('RMS lateral error (cm)')
    for k in (0, 1):
        ax[k].set_xlabel('real laps per car'); ax[k].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle('Few-shot adaptation on the multi-body cars (5 cars, 2 tracks, beta* = 25 deg, 5-lap chains)')
    fig.tight_layout()
    fig.savefig(os.path.join(OLD, 'budget.png'), dpi=120)
    # methods
    groups = [('ours (nominal sim)', ['v3nom_s0_zs', 'v3nom_s1_zs', 'v3nom_s2_zs'], ['v3nom_s0_hw24r', 'v3nom_s1_hw24r', 'v3nom_s2_hw24']),
              ('ours + goal fallback\n(14-plan bank)', ['v6nom_s0_zs', 'v6nom_s1_zs'],
               ['v6nom_s0_hw24fb', 'v6nom_s1_hw24fb', 'v6nom_s0_hw24fb3', 'v6nom_s1_hw24fb3']),
              ('ours, DR-A', ['v3drA_s0_zs', 'v3drA_s1_zs'], ['v3drA_s0_hw24', 'v3drA_s1_hw24']),
              ('ours, DR-B', ['v3drB_s0_zs', 'v3drB_s1_zs'], ['v3drB_s0_hw24', 'v3drB_s1_hw24']),
              ('PPO+DR', ['ppo_dr_zs'], ['ppo_dr_hw24', 'ppo_dr_hw24_s2']),
              ('RMA', ['rma_zs'], []),
              ('FADA', ['fada_hw24_zs'], ['fada_hw24']),
              ('plan LQR', ['lqr_zs'], [])]
    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    x = np.arange(len(groups))
    for k, (name, zs, hw) in enumerate(groups):
        for off, arms, col, lab in ((-0.18, zs, '#a0aec0', 'zero-shot'), (0.18, hw, '#2b6cb0', '+24 real laps (our hardware stage / FADA adaptation)')):
            v = [drift(ev(a))[0] for a in arms if ev(a)]
            if v:
                ax.bar(k + off, np.mean(v), 0.34, color=col, label=lab if k == 0 else None,
                       yerr=np.std(v) if len(v) > 1 else None, capsize=3)
                ax.text(k + off, np.mean(v) + (np.std(v) if len(v) > 1 else 0) + 1.2, f'{np.mean(v):.0f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_ylabel('full-drift success (%)'); ax.grid(axis='y', alpha=0.3); ax.legend(fontsize=8)
    ax.set_title('Zero-shot transfer and few-shot adaptation (5 multi-body cars x 2 tracks x 12 five-lap runs)')
    fig.tight_layout()
    fig.savefig(os.path.join(OLD, 'methods.png'), dpi=120)
    print('saved', OLD)


if __name__ == '__main__':
    main()
