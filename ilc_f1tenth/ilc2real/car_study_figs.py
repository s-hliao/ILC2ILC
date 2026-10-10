#!/usr/bin/env python3
"""
car_study_figs.py [--out DIR]: the car's study figures beyond car_figures.py's set (tight tracks, 0.3 m envelope).
Success = on track, RMS e_y <= 10 cm, pace >= 90 %; 95 % Wilson intervals over cars x plans x runs.

  fig8_data_matching   the baselines trained ON THE CARS with privileged data (ppo_real_car.py / fada_real_car.py:
                       PPO+DR fine-tuned with a privileged critic, the RMA teacher with each car's TRUE latent, FADA
                       with its planner gain and with the oracle planner) against real episodes (laps) per car, beside
                       ours + DR after the hardware stage (24 laps); and the crashes each spent
  fig9_chained         the hardware stage's trials: 1 lap from a reset vs 2 chained laps, same 24-lap budget
  fig10_axes           single-axis perturbations of the nominal car (mass, friction, tire stiffness, motor constant,
                       steering lag, delays, steering offset): success per value, no adaptation to the axis
  fig11_conditions     sensing (bad mocap, latency, low rate, EKF lag, noise), starts (slow rolling start, launch) and
                       the held-out track (mocap_squareH, never trained or adapted on)
  fig12_matrix         adapted under one perturbation (rows: the hardware stage run on that car) x tested on another
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
ap.add_argument('--out', default=os.path.join(HERE, 'figs'))
ap.add_argument('--only', nargs='*')
a = ap.parse_args()
OUT = a.out
os.makedirs(OUT, exist_ok=True)
CARS = ['real_nom', 'real_mass', 'real_mu', 'real_act', 'real_lag']
CAR_LAB = {'real_nom': 'nominal', 'real_mass': 'mass +25 %', 'real_mu': 'friction x0.8', 'real_act': 'actuators',
           'real_lag': 'tire / steering lag'}
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', pink='#CC79A7', sky='#56B4E9', red='#D55E00',
         yellow='#F0E442', grey='#7f7f7f', black='#222222')
SURF = '#fcfcfb'
plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 130, 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'savefig.bbox': 'tight',
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
EPS = [0, 24, 48, 96, 192, 384, 768, 1536, 2016]


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    return p, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)


def pooled(e, cars=None, plans=None):
    """e: {car: {plan: metrics}} -> (success %, CI %, crash %)."""
    k = n = f = 0
    for c in cars or list(e):
        for p, m in e.get(c, {}).items():
            if plans and not p.endswith(plans):
                continue
            nn = m.get('n', 6)
            k += m['success'] * nn
            f += m['fail'] * nn
            n += nn
    p, h = wilson(round(k), n)
    return 100 * p, 100 * h, 100 * f / max(n, 1)


def arm_eval(arm):
    f = os.path.join(HERE, 'runs/hw', arm, 'eval_big.json')
    return json.load(open(f)) if os.path.exists(f) else None


def arm_crashes(arm):
    s = json.load(open(os.path.join(HERE, 'runs/hw', arm, 'summary.json')))
    c = s.get('crashes', {})
    return sum(len(v) if isinstance(v, list) else v for v in c.values()) / max(len(c), 1)


def fig8():
    series = [('PPO+DR, fine-tuned on the cars (privileged critic)', 'ppo_real', C['pink'], 's'),
              ('RMA teacher with each car\'s TRUE latent, fine-tuned', 'rma_real', C['red'], 'v'),
              ('FADA, planner gain 0.95', 'fada_real_l095', C['orange'], 'o'),
              ('FADA, oracle planner (refit on our real laps)', 'fada_real_oracle', C['yellow'], 'D')]
    ours = [('ours + DR (B), 24 laps', 'v3drB_s0_hw24', C['green'], '^'),
            ('ours + DR (B, task objective), 24 laps', 'v3drBt_s0_hw24', C['blue'], '*')]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    for lab, arm, col, mk in series:
        pts = [(ep, arm_eval(f'{arm}_ep{ep}')) for ep in EPS]
        pts = [(ep, e) for ep, e in pts if e]
        if not pts:
            continue
        x = [max(ep, 0) for ep, _ in pts]
        r = [pooled(e, plans='_b25') for _, e in pts]
        axs[0].errorbar(x, [q[0] for q in r], yerr=[q[1] for q in r], color=col, marker=mk, ms=5, lw=1.6, capsize=2,
                        label=lab, mec='#333' if col == C['yellow'] else col, mew=0.5)
        axs[1].plot(x, [100 * np.mean([e['real_mu'][p]['success'] for p in e['real_mu'] if p.endswith('_b25')])
                        for _, e in pts], color=col, marker=mk, ms=5, lw=1.6, mec='#333' if col == C['yellow'] else col,
                    mew=0.5)
        axs[2].plot(x[1:], [arm_crashes(f'{arm}_ep{ep}') for ep, _ in pts[1:]], color=col, marker=mk, ms=5, lw=1.6,
                    mec='#333' if col == C['yellow'] else col, mew=0.5)
    for lab, arm, col, mk in ours:
        e = arm_eval(arm)
        if not e:
            continue
        p, h, _ = pooled(e, plans='_b25')
        axs[0].axhspan(p - h, p + h, color=col, alpha=0.10, lw=0)
        axs[0].axhline(p, color=col, lw=1.2, ls='--')
        axs[0].errorbar([24], [p], yerr=[h], color=col, marker=mk, ms=10 if mk == '*' else 7, capsize=3, ls='none',
                        label=f'{lab}: {p:.0f} %', mec='#333', mew=0.5, zorder=5)
        mu = 100 * np.mean([e['real_mu'][q]['success'] for q in e['real_mu'] if q.endswith('_b25')])
        axs[1].plot([24], [mu], color=col, marker=mk, ms=10 if mk == '*' else 7, ls='none', mec='#333', mew=0.5)
        axs[2].plot([24], [arm_crashes(arm)], color=col, marker=mk, ms=10 if mk == '*' else 7, ls='none', mec='#333',
                    mew=0.5, label=lab)
    for ax in axs:
        ax.set_xscale('symlog', linthresh=24, linscale=0.5)
        ax.set_xticks([0, 24, 96, 384, 2016])
        ax.set_xticklabels(['0', '24', '96', '384', '2016'])
        ax.set_xlabel('real episodes (laps) per car')
    axs[0].set_ylabel('full-drift success, 5 cars (%)')
    axs[0].set_ylim(-3, 105)
    axs[0].set_title('success (beta 25 plans)', fontsize=10)
    axs[1].set_ylabel('success on the friction x0.8 car (%)')
    axs[1].set_ylim(-3, 105)
    axs[1].set_title('the low-friction car alone', fontsize=10)
    axs[2].set_ylabel('crashes spent per car (0.3 m envelope)')
    axs[2].set_yscale('symlog', linthresh=1)
    axs[2].set_ylim(0, 4000)
    axs[2].text(30, 0.15, 'ours: 0 crashes', fontsize=8, color='#333')
    axs[2].set_title('real-world cost', fontsize=10)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.1))
    fig.suptitle('Data matching: baselines trained on the cars with privileged data need ~1,500 laps per car to pass '
                 'ours + DR after 24 laps', fontsize=10.5)
    fig.savefig(os.path.join(OUT, 'fig8_data_matching.png'))
    plt.close(fig)


def fig9():
    arms = [('ours', ['v3nom_s0_hw24', 'v3nom_s1_hw24', 'v3nom_s2_hw24'], C['blue']),
            ('ours+DR (A)', ['v3drA_s0_hw24', 'v3drA_s1_hw24'], C['sky']),
            ('ours+DR (B)', ['v3drB_s0_hw24', 'v3drB_s1_hw24'], C['green']),
            ('ours, no exploration', ['v3noexp_s0_hw24'], C['grey']),
            ('BC init + stage', ['bcB_hw24b'], C['black']),
            ('PPO+DR + stage', ['ppo_dr_hw24', 'ppo_dr_hw24_s2'], C['pink'])]
    roots = [('1 lap per trial (reset)', os.path.join(HERE, 'runs_1lap/hw'), '//'),
             ('2 chained laps per trial', os.path.join(HERE, 'runs/hw'), None)]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    w = 0.38
    for j, (rl, root, hatch) in enumerate(roots):
        for i, (lab, aa, col) in enumerate(arms):
            E = [json.load(open(os.path.join(root, x, 'eval_big.json'))) for x in aa
                 if os.path.exists(os.path.join(root, x, 'eval_big.json'))]
            if not E:
                continue
            k = n = f = 0
            for e in E:
                p, _, cr = pooled(e, plans='_b25')
                nn = sum(m.get('n', 6) for c in e for q, m in e[c].items() if q.endswith('_b25'))
                k += p / 100 * nn
                f += cr / 100 * nn
                n += nn
            p, h = wilson(round(k), n)
            x = i + (j - 0.5) * w
            for ax, v, err in ((axs[0], 100 * p, 100 * h), (axs[1], 100 * f / n, None)):
                ax.bar(x, v, w * 0.92, color=col if not hatch else SURF, edgecolor=col, hatch=hatch, lw=1.2,
                       label=rl if i == 0 else None)
                if err is not None:
                    ax.errorbar(x, v, yerr=err, color='#333', capsize=2, lw=0.9)
                ax.text(x, v + (err or 0) + 1, f'{v:.0f}', ha='center', fontsize=7, color='#333')
    for ax in axs:
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([q[0] for q in arms], rotation=20, ha='right', fontsize=8.5)
        ax.grid(axis='x', alpha=0)
    axs[0].set_ylabel('full-drift success after 24 real laps (%)')
    axs[0].set_ylim(0, 105)
    axs[1].set_ylabel('evaluation runs that crash (%)')
    axs[0].legend(fontsize=8, frameon=False, loc='upper right')
    fig.suptitle('Hardware-stage trials: one lap from a reset vs two chained laps (same 24-lap budget; hatched = 1 lap)',
                 fontsize=10.5)
    fig.savefig(os.path.join(OUT, 'fig9_chained.png'))
    plt.close(fig)


NETS = [('ours_drB', 'ours + DR (B), adapted on the nominal car', C['green'], '-', 'o'),
        ('ours_drB_zs', 'ours + DR (B), zero-shot', C['green'], ':', 'o'),
        ('ours_nom', 'ours, adapted on the nominal car', C['blue'], '-', 's'),
        ('ppo_dr_hw', 'PPO+DR + our stage', C['pink'], '-', 'D'),
        ('ppo_dr_zs', 'PPO+DR, zero-shot', C['pink'], ':', 'D'),
        ('rma_zs', 'RMA', C['red'], ':', 'v'),
        ('lqr', 'plan LQR', C['grey'], ':', 'x')]


def fig10():
    S = json.load(open(os.path.join(HERE, 'runs/axes/sweep.json')))['eval']
    nom = json.load(open(os.path.join(HERE, 'runs/axes/sense.json')))['eval']    # no plain real_nom there
    axes = {}
    for car in next(iter(S.values())):
        k, v = car[3:].split('=')
        axes.setdefault(k, []).append(float(v))
    lab = dict(mass='mass (x nominal)', mu='friction (x nominal)', ky='tire cornering stiffness (x)',
               kt='motor torque constant (x)', tau='steering time constant (s)', d_delay='steering delay (periods)',
               i_delay='current delay (periods)', d_off='steering offset (rad)')
    NOMV = dict(mass=1.0, mu=1.0, ky=1.0, kt=1.0, tau=None, d_delay=0, i_delay=0, d_off=0.0)
    ks = list(axes)
    nc = 4
    fig, axs = plt.subplots(int(np.ceil(len(ks) / nc)), nc, figsize=(4.0 * nc, 3.1 * int(np.ceil(len(ks) / nc))),
                            squeeze=False)
    adapted = {}                      # the hardware stage run ON the axis car (ax_<k><v>_drB_hw24)
    for k in ks:
        for v in axes[k]:
            tag = f'ax_{k}{v:g}_drB_hw24'
            e = arm_eval(tag)
            if e:
                adapted[(k, v)] = pooled(e, plans='_b25')[0]
    for i, k in enumerate(ks):
        ax = axs.flat[i]
        vs = sorted(axes[k])
        for net, nl, col, ls, mk in NETS:
            if net not in S:
                continue
            y = [pooled({f'ax:{k}={v:g}': S[net][f'ax:{k}={v:g}']}, plans='_b25')[0] for v in vs]
            ax.plot(vs, y, color=col, ls=ls, marker=mk, ms=4, lw=1.5, label=nl)
        av = [(v, adapted[(k, v)]) for v in vs if (k, v) in adapted]
        if av:
            ax.plot([q[0] for q in av], [q[1] for q in av], color=C['green'], marker='*', ms=11, ls='none', mec='#333',
                    mew=0.5, label='ours + DR (B), adapted ON that car (24 laps)')
        if NOMV.get(k) is not None:
            ax.axvline(NOMV[k], color=C['grey'], lw=0.8, ls='--')
        ax.set_xlabel(lab.get(k, k), fontsize=9)
        ax.set_ylim(-3, 105)
        if i % nc == 0:
            ax.set_ylabel('full-drift success (%)')
    for ax in list(axs.flat)[len(ks):]:
        ax.axis('off')
    h, l = axs.flat[0].get_legend_handles_labels()
    for ax in axs.flat:
        hh, ll = ax.get_legend_handles_labels()
        for x, y in zip(hh, ll):
            if y not in l:
                h.append(x)
                l.append(y)
    fig.legend(h, l, fontsize=8, frameon=False, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.06))
    fig.suptitle('Single-axis perturbations of the nominal car (beta 25 plans; dashed = nominal value; the networks are '
                 'not adapted to the axis unless starred)', fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig10_axes.png'))
    plt.close(fig)


def fig11():
    T = {t: json.load(open(os.path.join(HERE, 'runs/axes', f'{t}.json')))['eval']
         for t in ('sense', 'start_direct', 'start_launch', 'heldout')}
    groups = [('bad mocap', 'sense', '+mocapbad', None), ('latency x2', 'sense', '+latency2', None),
              ('low rate', 'sense', '+lowrate', None), ('EKF lag', 'sense', '+ekflag', None),
              ('noise x3', 'sense', '+noisy3', None),
              ('rolling start\n0.6 m/s', 'start_direct', None, None), ('launch to\n1 m/s', 'start_launch', None, None),
              ('held-out track\nbeta 25', 'heldout', None, '_b25'), ('held-out track\nbeta 21 / 14', 'heldout', None, ('_b21', '_b14')),
              ('held-out track\ngrip', 'heldout', None, '_b0')]
    nets = [q for q in NETS if q[0] != 'ours_nom'] + [NETS[2]]
    fig, ax = plt.subplots(figsize=(16, 4.6))
    w = 0.85 / len(nets)
    for gi, (gl, t, suf, plans) in enumerate(groups):
        for j, (net, nl, col, ls, mk) in enumerate(nets):
            e = T[t].get(net)
            if not e:
                continue
            ee = {c: v for c, v in e.items() if suf is None or c.endswith(suf)}
            p, h, _ = pooled(ee, plans=plans or '_b25')
            x = gi + (j - (len(nets) - 1) / 2) * w
            ax.bar(x, p, w * 0.9, color=col if ls == '-' else SURF, edgecolor=col, hatch=None if ls == '-' else '//',
                   lw=1.0, label=nl if gi == 0 else None)
            ax.errorbar(x, p, yerr=h, color='#333', capsize=1.5, lw=0.7)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.5)
    ax.set_ylabel('success, 5 cars (%)')
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis='x', alpha=0)
    for xx in (4.5, 6.5):
        ax.axvline(xx, color='#999', lw=0.8)
    ax.text(2, 107, 'sensing (on all five cars)', ha='center', fontsize=9)
    ax.text(5.5, 107, 'starting conditions', ha='center', fontsize=9)
    ax.text(8, 107, 'held-out track (never trained / adapted on)', ha='center', fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.14))
    fig.suptitle('Sensing, starting conditions and a held-out track (hatched = zero-shot; adapted networks were adapted '
                 'on each car without the perturbation; 80 % = all cars but friction x0.8)', fontsize=10.5)
    fig.savefig(os.path.join(OUT, 'fig11_conditions.png'))
    plt.close(fig)


def fig12():
    d = json.load(open(os.path.join(HERE, 'runs/axes/tp_matrix.json')))
    E, cars = d['eval'], d['args']['cars']
    rows = [n.split('=')[0] for n in d['args']['net']]
    M = np.array([[pooled({c: E[r][c]}, plans='_b25')[0] for c in cars] for r in rows])
    lab = lambda s: (s.replace('nominal_adapted', 'nominal car').replace('ax:', '').replace('ax_', '')
                     .replace('real_nom+', '').replace('sense_', '').replace('real_nom', 'nominal car'))
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    im = ax.imshow(M, cmap='Greens', vmin=0, vmax=100)
    for i in range(len(rows)):
        for j in range(len(cars)):
            ax.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center', fontsize=8,
                    color='white' if M[i, j] > 60 else '#222', fontweight='bold' if lab(rows[i]).replace('=', '') ==
                    lab(cars[j]).replace('=', '') or (i == 0 and j == 0) else None)
    ax.set_xticks(range(len(cars)))
    ax.set_xticklabels([lab(c) for c in cars], rotation=30, ha='right', fontsize=8.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab(r) for r in rows], fontsize=8.5)
    ax.set_xlabel('tested on')
    ax.set_ylabel('hardware stage run on (24 laps)')
    ax.grid(False)
    fig.colorbar(im, ax=ax, label='full-drift success (%)', shrink=0.8)
    ax.set_title('Adapted under one perturbation, tested under another (ours + DR (B); bold = adapted = tested)',
                 fontsize=10)
    fig.savefig(os.path.join(OUT, 'fig12_matrix.png'))
    plt.close(fig)


if __name__ == '__main__':
    for f in (fig8, fig9, fig10, fig11, fig12):
        if not a.only or f.__name__ in a.only:
            f()
            print('->', os.path.join(OUT, f.__name__), flush=True)
