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
  fig15_pertrack       the per-track ablation: one network for both tracks vs one per track
  fig14_lowmu_completion  the low-friction car: full 5-lap completion against real laps, ours vs PPO+DR
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
ap.add_argument('--out', default=os.path.normpath(os.path.join(HERE, '..', '..', 'paper', 'car', 'figures')))   # src/paper
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
            ('ours + DR (B, task objective), 24 laps', 'v3drBt_s0_hw24', C['blue'], '*')]   # + their 0-10-lap arms
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
    for lab, arm, col, mk in ours:                  # our own real-data budget: zero-shot, 2-10 laps, 24 laps
        net = arm.replace('_hw24', '')
        pts = [(L, arm_eval(f'{net}_{"zs" if L == 0 else f"lap{L}"}')) for L in (0, 2, 4, 6, 8, 10)]
        pts = [(L, e) for L, e in pts if e]
        if not pts:
            continue
        r = [pooled(e, plans='_b25') for _, e in pts]
        x = [L for L, _ in pts]
        p, h, _ = r[-1]
        if mk == '*':                                # the band: the headline network after 10 laps
            axs[0].axhspan(p - h, p + h, color=col, alpha=0.08, lw=0)
            axs[0].axhline(p, color=col, lw=1.0, ls='--')
        axs[0].errorbar(x, [q[0] for q in r], yerr=[q[1] for q in r], color=col, marker=mk, ms=9 if mk == '*' else 6,
                        lw=2.2, capsize=2, label=f'{lab.replace(", 24 laps", "")}: our hardware stage ({p:.0f} % at 10)',
                        mec='#333', mew=0.5, zorder=5)
        axs[1].plot(x, [100 * np.mean([e['real_mu'][q]['success'] for q in e['real_mu'] if q.endswith('_b25')])
                        for _, e in pts], color=col, marker=mk, ms=9 if mk == '*' else 6, lw=2.2, mec='#333', mew=0.5,
                    zorder=5)
        axs[2].plot(x[1:], [arm_crashes(f'{net}_lap{L}') for L in x[1:]],
                    color=col, marker=mk, ms=9 if mk == '*' else 6, lw=2.2, mec='#333', mew=0.5, zorder=5)
    for ax in axs:
        ax.set_xscale('symlog', linthresh=10, linscale=1.0)
        ax.set_xticks([0, 2, 4, 10, 24, 96, 384, 2016])
        ax.set_xticklabels(['0', '2', '4', '10', '24', '96', '384', '2016'])
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
    axs[2].text(2.2, 0.15, 'ours: 0 crashes at every budget', fontsize=8, color='#333')
    axs[2].set_title('real-world cost', fontsize=10)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.1))
    fig.suptitle('Real-data budget: ours + DR reaches 80 % in 4 laps per car (0 crashes); baselines trained on the '
                 'cars with privileged data need ~1,500 laps and hundreds of crashes', fontsize=10.5)
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


L10 = os.path.exists(os.path.join(HERE, 'runs/axes/sweep_l10.json'))     # the 10-lap suite (scheduler_lap10.py)
SFX, BUD = ('_l10', '10 real laps') if L10 else ('', '24 real laps')
NETS = [('ours_drBt', 'ours + DR (task objective), adapted on each car (no perturbation)', C['blue'], '-', '*'),
        ('ours_drBt_zs', 'ours + DR (task objective), zero-shot', C['blue'], ':', '*'),
        ('ours_drB', 'ours + DR (B), adapted on the nominal car', C['green'], '-', 'o'),
        ('ours_drB_zs', 'ours + DR (B), zero-shot', C['green'], ':', 'o'),
        ('ours_nom', 'ours, adapted on the nominal car', C['blue'], '-', 's'),
        ('ppo_dr_hw', 'PPO+DR + our stage', C['pink'], '-', 'D'),
        ('ppo_dr_zs', 'PPO+DR, zero-shot', C['pink'], ':', 'D'),
        ('rma_zs', 'RMA', C['red'], ':', 'v'),
        ('lqr', 'plan LQR', C['grey'], ':', 'x')]


def fig10():
    S = json.load(open(os.path.join(HERE, f'runs/axes/sweep{SFX}.json')))['eval']
    nom = json.load(open(os.path.join(HERE, f'runs/axes/sense{SFX}.json')))['eval']    # no plain real_nom there
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
            tag = f'ax_{k}{v:g}_drBt_lap10' if L10 else f'ax_{k}{v:g}_drB_hw24'
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
                    mew=0.5, label=f'ours + DR, adapted ON that car ({BUD})')
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
    fig.suptitle(f'Single-axis perturbations of the nominal car (beta 25 plans; adapted networks: {BUD}; dashed = '
                 f'nominal value; not adapted to the axis unless starred)', fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig10_axes.png'))
    plt.close(fig)


def fig11():
    T = {t: json.load(open(os.path.join(HERE, 'runs/axes', f'{t}{SFX}.json')))['eval']
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
    fig.suptitle(f'Sensing, starting conditions and a held-out track ({BUD}; hatched = zero-shot; adapted networks were adapted '
                 'on each car without the perturbation; 80 % = all cars but friction x0.8)', fontsize=10.5)
    fig.savefig(os.path.join(OUT, 'fig11_conditions.png'))
    plt.close(fig)


def fig12():
    d = json.load(open(os.path.join(HERE, f'runs/axes/tp_matrix{SFX}.json')))
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
    ax.set_ylabel(f'hardware stage run on ({BUD})')
    ax.grid(False)
    fig.colorbar(im, ax=ax, label='full-drift success (%)', shrink=0.8)
    ax.set_title(f'Adapted under one perturbation ({BUD}), tested under another (bold = adapted = tested)',
                 fontsize=10)
    fig.savefig(os.path.join(OUT, 'fig12_matrix.png'))
    plt.close(fig)


def fig14():
    """Full completion of the 5-lap chained test (no wall crash, no spin-out) on the low-friction car (friction x0.8:
    mu 0.16) against the real laps spent: ours vs PPO+DR, both through our hardware stage at 2-10 laps."""
    series = [('ours + DR (B, task objective)', 'v3drBt_s0', C['blue'], '*'),
              ('ours + DR (B)', 'v3drB_s0', C['green'], '^'),
              ('PPO+DR (+ our hardware stage)', 'ppo_dr', C['pink'], 's'),
              ('PPO+DR, zero-shot (its on-car fine-tuning start)', 'ppo_real', C['pink'], 'D'),
              ('RMA teacher with the TRUE latent, zero-shot', 'rma_real', C['red'], 'v')]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.3), sharey=True)
    for ax, plans, title in ((axs[0], None, 'all 8 plans (drift + grip, both tracks)'),
                             (axs[1], '_b25', 'full-drift plans (beta 25)')):
        for lab, net, col, mk in series:
            if net.endswith('_real'):
                pts = [(L, arm_eval(f'{net}_ep{L}')) for L in (0,)]
            else:
                pts = [(L, arm_eval(f'{net}_{"zs" if L == 0 else f"lap{L}" if L <= 10 else "hw24"}'))
                       for L in (0, 2, 4, 6, 8, 10)]
            pts = [(L, e) for L, e in pts if e]
            x, y, h = [], [], []
            for L, e in pts:
                k = n = 0
                for p, m in e['real_mu'].items():
                    if plans and not p.endswith(plans):
                        continue
                    nn = m.get('n', 6)
                    k += (1 - m['fail']) * nn
                    n += nn
                pp, hh = wilson(round(k), n)
                x.append(L)
                y.append(100 * pp)
                h.append(100 * hh)
            ax.errorbar(x, y, yerr=h, color=col, marker=mk, ms=9 if mk == '*' else 6, lw=2.0 if 'ours' in lab else 1.4,
                        ls='-' if 'ours' in lab or net == 'ppo_dr' else '--', capsize=2, label=lab,
                        mec='#333', mew=0.5, zorder=5 if 'ours' in lab else 3)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_xlim(-0.8, 10.8)
        ax.set_xlabel('real laps on the low-friction car')
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 105)
    axs[0].set_ylabel('runs completing all 5 chained laps (%)')
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, fontsize=8, frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.1))
    fig.suptitle('Low-friction car (mu 0.16): full completion of the 5-lap chained test (0.3 m envelope; 95 % '
                 'intervals)', fontsize=10.5)
    fig.savefig(os.path.join(OUT, 'fig14_lowmu_completion.png'))
    plt.close(fig)


def fig15():
    """The per-track ablation (scheduler_pertrack.py): one network PER TRACK (its own BC / sim stage / hardware stage
    on that track's plans) against the method's ONE network for both tracks, at equal real laps per car (the shared
    network: the two b25 plans alternated; per track: that track's b25 plan, half the laps each). Seeds 0-1 pooled."""
    def sc(arms, key, track=None):
        k = n = 0
        for arm in arms:
            e = arm_eval(arm)
            if not e:
                continue
            for c in CARS:
                for p, m in e[c].items():
                    if p.endswith('_b25') and (track is None or f'_{track}_' in p):
                        nn = m.get('n', 6)
                        k += (m['success'] if key == 'success' else 1 - m['fail']) * nn
                        n += nn
        p, h = wilson(round(k), n)
        return 100 * p, 100 * h
    kinds = [('ours + DR (B)', 'v3drB', C['green']), ('ours, nominal sim', 'v3nom', C['black'])]
    Ls = [('zs', 'zero-shot'), ('lap4', '+ 4 laps'), ('lap8', '+ 8 laps')]
    cols = [(None, 'both tracks'), ('square2fast', 'square track'), ('figfast', 'figure-eight track')]
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.4), sharey=True)
    for ii, (key, yl) in enumerate((('success', 'success (%)'), ('complete', 'completes all 5 laps (%)'))):
        for jj, (track, tl) in enumerate(cols):
            ax = axs[ii, jj]
            w = 0.2
            ticks = []
            for gi, (klab, kind, col) in enumerate(kinds):
                for li, (L, llab) in enumerate(Ls):
                    x0 = gi * 3.6 + li * 1.1
                    ticks.append((x0, llab))
                    for bi, (shared, hatch) in enumerate(((True, None), (False, '//'))):
                        arms = [f'{kind}_s{sd}_{L}' if shared else f'pt_{kind}_s{sd}_{L}' for sd in (0, 1)]
                        p, h = sc(arms, key, track)
                        x = x0 + (bi - 0.5) * 2 * w
                        ax.bar(x, p, 2 * w * 0.92, color=col if shared else SURF, edgecolor=col, hatch=hatch, lw=1.1,
                               label=('one network for both tracks (the method)' if shared else 'one network per track')
                               if (gi, li) == (0, 0) else None)
                        ax.errorbar(x, p, yerr=h, color='#333', capsize=1.5, lw=0.7)
                        ax.text(x, min(p + h + 1.5, 103), f'{p:.0f}', ha='center', fontsize=6.5, color='#333')
            ax.set_xticks([t[0] for t in ticks])
            ax.set_xticklabels([t[1] for t in ticks], fontsize=7, rotation=30, ha='right')
            if ii == 1:
                for gi, (klab, kind, col) in enumerate(kinds):
                    ax.text(gi * 3.6 + 1.1, -0.3, klab, transform=ax.get_xaxis_transform(), ha='center', fontsize=8.5,
                            color=col if col != C['black'] else '#222', fontweight='bold')
            ax.set_ylim(0, 110)
            ax.grid(axis='x', alpha=0)
            if ii == 0:
                ax.set_title(tl, fontsize=10)
            if jj == 0:
                ax.set_ylabel(yl)
    axs[0, 0].legend(fontsize=8, frameon=False, loc='upper right')
    fig.suptitle('Per-track ablation: one goal-conditioned network for both tracks vs a network per track (beta 25 '
                 'plans, 5 cars, seeds 0-1 pooled; equal real laps per car)', fontsize=10.5)
    fig.savefig(os.path.join(OUT, 'fig15_pertrack.png'))
    plt.close(fig)


if __name__ == '__main__':
    for f in (fig8, fig10, fig11, fig12, fig14, fig15):        # fig9 (1-lap vs chained) was 24 laps: figs/old_24lap/
        if f in (fig10, fig11, fig12) and not L10:
            print(f'{f.__name__}: waiting for the 10-lap suite (runs/axes/*_l10.json); the 24-lap version is in figs/old_24lap/')
            continue
        if not a.only or f.__name__ in a.only:
            f()
            print('->', os.path.join(OUT, f.__name__), flush=True)
