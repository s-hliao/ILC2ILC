#!/usr/bin/env python3
"""
stop_analysis.py [--out DIR]: the car's stopping-rule study (the quadruped's snapsel.py analysis). When should the sim
stage stop, using sim data only? Inputs (scheduler_stop.py):

  runs/stopping/sim_val.json   every snapshot (every 5 iterations, 0..90) of the two 90-iteration sim stages
                               (v3drB_s0_long: our learner + DR (B); v3nom_s0_long: nominal sim), scored in the nominal
                               sim on the training and HELD-OUT plans under conditions the stage never trained under
                               (sim_validate.py: nominal / perturbed starts / delay / disturbance / DR / wide DR)
  runs/axes/stop_real.json     the same snapshots' zero-shot transfer to the five multi-body cars (beta 25 + 0 plans,
                               4 runs x 3 laps per car and plan: 80 runs per snapshot, SE ~5 points)

Per sim metric: Spearman rank correlation with real transfer across snapshots (per run), and the snapshot each
stopping rule picks with its regret (real transfer of the best snapshot minus the picked one's; the best is taken on a
3-point running mean of the real curve, so a single lucky snapshot does not set the bar). Rules: last snapshot, the
pipeline's fixed 30 iterations, argmax of the metric, and early stopping on the metric (patience 3 evaluations = 15
iterations, improvement > 0.5 point; and patience 6 = 30 iterations). -> DIR/stop_analysis.json, DIR/stop_analysis.md, src/paper/car/figures/fig13_stopping.png
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
ap.add_argument('--out', default=os.path.join(HERE, 'runs/stopping'))
ap.add_argument('--figs', default=os.path.normpath(os.path.join(HERE, '..', '..', 'paper', 'car', 'figures')))   # src/paper
a = ap.parse_args()
S = json.load(open(os.path.join(HERE, 'runs/stopping/sim_val.json')))
R = json.load(open(os.path.join(HERE, 'runs/axes/stop_real.json')))['eval']
COND = ['nominal', 'pstart', 'delay', 'dist', 'dr', 'drwide']
LAB = dict(v3drB_s0_long='our learner + DR (B)', v3nom_s0_long='ours (nominal sim)')
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', pink='#CC79A7', sky='#56B4E9', red='#D55E00', grey='#7f7f7f')
SURF = '#fcfcfb'


def rank(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind='mergesort')
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    for v in np.unique(x):                          # average ranks over ties
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def real_of(run, it, plans=None):
    r = R.get(f'{run}_it{it:03d}')
    if r is None:
        return None
    return 100 * float(np.mean([v['success'] for c in r.values() for p, v in c.items() if plans is None or p.endswith(plans)]))


def metrics(s):
    """Sim metrics of one snapshot: {name: value}, higher = better."""
    m = {}
    for c in COND:
        for g in ('train', 'heldout'):
            m[f'{c}/{g}/success'] = 100 * s[c][g]['success']
            m[f'{c}/{g}/-fail'] = -100 * s[c][g]['fail']
            if s[c][g]['rms_ey'] is not None:
                m[f'{c}/{g}/-rms_ey'] = -100 * s[c][g]['rms_ey']
    m['mean non-nominal/heldout/success'] = float(np.mean([m[f'{c}/heldout/success'] for c in COND[1:]]))
    m['dr+drwide/heldout/success'] = float(np.mean([m[f'{c}/heldout/success'] for c in ('dr', 'drwide')]))
    return m


def early_stop(vals, patience=3, tol=0.5):
    best, bi, wait = -np.inf, 0, 0
    for i, v in enumerate(vals):
        if v > best + tol:
            best, bi, wait = v, i, 0
        else:
            wait += 1
            if wait >= patience:
                return bi, i                         # (picked, stopped at)
    return bi, len(vals) - 1


def main():
    out, md = {}, ['# Stopping rule for the sim stage (car)', '',
                   'Zero-shot transfer to the five multi-body cars (beta 25 + 0 plans, 80 runs per snapshot, SE ~5 points) '
                   'against SIM-ONLY validation metrics, per snapshot (every 5 iterations). Regret = best real transfer '
                   '(3-point running mean) minus the picked snapshot\'s, in points.', '']
    for run in S:
        its = sorted(int(i) for i in S[run] if real_of(run, int(i)) is not None)
        real = np.array([real_of(run, i) for i in its])
        real25 = np.array([real_of(run, i, '_b25') for i in its])
        smooth = np.convolve(np.pad(real, 1, mode='edge'), np.ones(3) / 3, 'valid')
        best = float(smooth.max())
        M = [metrics(S[run][str(i)]) for i in its]
        names = sorted(set.intersection(*[set(m) for m in M]))
        rho = {n: spearman([m[n] for m in M], real) for n in names}
        picks = {'last snapshot (90 its)': len(its) - 1, 'fixed 30 iterations (the pipeline)': its.index(30)}
        for n in ('nominal/heldout/success', 'nominal/heldout/-rms_ey', 'pstart/heldout/success', 'dr/heldout/success',
                  'drwide/heldout/success', 'dr+drwide/heldout/success', 'mean non-nominal/heldout/success',
                  'drwide/heldout/-rms_ey'):
            v = np.array([m[n] for m in M])
            picks[f'argmax {n}'] = int(np.flatnonzero(v >= v.max() - 1e-9)[0])
            picks[f'early stop {n}'] = early_stop(v)[0]
            picks[f'early stop (patience 30 its) {n}'] = early_stop(v, patience=6)[0]
        rows = {k: dict(it=its[i], real=float(real[i]), real_b25=float(real25[i]), regret=best - float(smooth[i]))
                for k, i in picks.items()}
        out[run] = dict(its=its, real=real.tolist(), real_b25=real25.tolist(), real_smooth=smooth.tolist(),
                        best_smooth=best, spearman=rho, picks=rows,
                        sim={n: [m[n] for m in M] for n in names})
        md += [f'## {LAB.get(run, run)} (`{run}`)', '',
               f'Real transfer {real.min():.0f}-{real.max():.0f} % (beta 25: {real25.min():.0f}-{real25.max():.0f} %), '
               f'best {its[int(np.argmax(smooth))]} iterations (running mean {best:.1f} %).', '',
               '| sim metric (held-out plans) | Spearman with real |', '|---|---|']
        for n in sorted(names, key=lambda n: -np.nan_to_num(rho[n], nan=-9)):
            if '/heldout/' in n and not np.isnan(rho[n]):
                md.append(f'| {n.replace("/heldout", "")} | {rho[n]:+.2f} |')
        md += ['', '| rule | picks (its) | real % | beta 25 % | regret (points) |', '|---|---|---|---|---|']
        for k, r in rows.items():
            md.append(f'| {k} | {r["it"]} | {r["real"]:.1f} | {r["real_b25"]:.1f} | {r["regret"]:.1f} |')
        md.append('')
        print(run, 'best', best, {k: (r['it'], round(r['regret'], 1)) for k, r in rows.items()})
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, 'stop_analysis.json'), 'w'), indent=1)
    open(os.path.join(a.out, 'stop_analysis.md'), 'w').write('\n'.join(md) + '\n')
    figure(out)


def figure(out):
    plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 130, 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'savefig.bbox': 'tight',
                         'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
    runs = list(out)
    fig, axs = plt.subplots(2, len(runs), figsize=(6.2 * len(runs), 7.4), sharex=True, squeeze=False)
    sims = [('nominal/heldout/success', 'nominal sim, held-out plans', C['grey'], '-'),
            ('pstart/heldout/success', 'perturbed starts', C['orange'], '-'),
            ('dr/heldout/success', 'DR dynamics', C['sky'], '-'),
            ('drwide/heldout/success', 'wide DR (x1.5)', C['blue'], '-'),
            ('mean non-nominal/heldout/success', 'mean of the 5 non-nominal conditions', C['pink'], '--')]
    for j, run in enumerate(runs):
        d = out[run]
        its = d['its']
        ax = axs[0, j]
        ax.plot(its, d['real'], color=C['green'], marker='o', ms=4, lw=1.8, label='real transfer (5 cars, beta 25 + 0)')
        ax.fill_between(its, np.array(d['real']) - 4.8, np.array(d['real']) + 4.8, color=C['green'], alpha=0.12, lw=0)
        ax.plot(its, d['real_b25'], color=C['green'], marker='s', ms=3, lw=1.1, ls='--', label='real, beta 25 only')
        for k, mk, col in (('fixed 30 iterations (the pipeline)', 'v', C['grey']),
                           ('argmax nominal/heldout/success', 'X', C['orange']),
                           ('argmax mean non-nominal/heldout/success', '*', C['blue']),
                           ('early stop (patience 30 its) mean non-nominal/heldout/success', 'D', C['pink'])):
            p = d['picks'][k]
            ax.plot(p['it'], p['real'], marker=mk, ms=11 if mk == '*' else 8, color=col, ls='none', mec='#333', mew=0.6,
                    label=f'{k.replace("/heldout/success", "").replace(" (patience 30 its)", ", patience 30")}: {p["it"]} its, regret {p["regret"]:.1f}', zorder=5)
        ax.set_ylabel('zero-shot real success (%)')
        ax.set_title(LAB.get(run, run), fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc='lower right')
        ax.set_ylim(5, 92)
        ax = axs[1, j]
        for n, lab, col, ls in sims:
            r = d['spearman'][n]
            ax.plot(its, d['sim'][n], color=col, lw=1.6, ls=ls,
                    label=f'{lab} ' + ('(flat: no ranking signal)' if np.isnan(r) else f'(Spearman {r:+.2f})'))
        ax.set_ylabel('SIM-ONLY validation success (%)')
        ax.set_xlabel('sim-stage iteration (snapshot)')
        ax.legend(fontsize=7.5, frameon=False, loc='lower right')
        ax.set_ylim(15, 102)
    fig.suptitle('When to stop the sim stage: the nominal-sim score is flat; DR-family held-out scores track real '
                 'transfer', fontsize=10.5)
    os.makedirs(a.figs, exist_ok=True)
    fig.savefig(os.path.join(a.figs, 'fig13_stopping.png'))
    plt.close(fig)


if __name__ == '__main__':
    main()
