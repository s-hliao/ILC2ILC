#!/usr/bin/env python3
"""quad_figures.py: the quadruped (Go1 goal-plane jumping) results as figures, recomputed from the per-jump records with
master_table.py's definitions (success = no fall, |ex| <= 5 cm, |ez| <= 3 cm; reserved test goals planefinal; 95 %
Wilson intervals). Reads ~/ilc_ws/log/dilc/plane (outside the repo) -> src/paper/quadruped/figures_summary/*.png
  fig1_factorial    sim stage {nominal, DR-A, DR-B} x {zero-shot, +24 nominal starts, +24 varied starts}, under
                    nominal / start-perturbed / sensing-perturbed evaluation (robots r1 s1 r4m r5)
  fig2_methods      ours vs every baseline, nominal evaluation (robots r1 s1 r5, shared by all runs)
  fig3_budget       success vs real jumps per robot: our hardware stage vs per-goal ILC (JumpILC) on the test goals
  fig4_hw_arms      every hardware-stage arm (master table section 2), grouped by what it changes
  fig5_sim_variants the sim-stage variants' CPU zero-shot on the validation goals (it20 / it30, per seed)
  fig6_efficiency   sim samples and wall time to train (abstract notes)
  fig7_goal_maps    success per goal on the (x, h) plane, zero-shot vs after the 24-jump stage (5 robots pooled)"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'log', 'dilc', 'plane'))    # the per-jump records (not in git)
OUT = os.path.normpath(os.path.join(HERE, '..', '..', 'paper', 'quadruped', 'figures_summary'))     # src/paper: the paper's material
os.makedirs(OUT, exist_ok=True)
ok = lambda r: (not r['fell']) and abs(r['ex']) <= 0.05 and abs(r['ez']) <= 0.03
START = ['blk15', 'blk2', 'crouch', 'tall', 'noseup', 'nosedn']
SENSE = ['mocapbad', 'delay10']
R4 = ['real_r1', 'real_s1', 'real_r4m', 'real_r5']
R3 = ['real_r1', 'real_s1', 'real_r5']
# Okabe-Ito (colour-blind safe), fixed order
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', pink='#CC79A7', sky='#56B4E9', red='#D55E00',
         yellow='#F0E442', grey='#7f7f7f', black='#222222')
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': 0.25, 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'savefig.dpi': 160,
                     'savefig.bbox': 'tight'})


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, h


def jumps(src, tag, robots, rob=False):
    out = []
    for rb in robots:
        d = src.get(rb) if isinstance(src, dict) else src
        if d is None:
            continue
        f = os.path.join(P, d, rb, f"eval_planefinal_{'rob_' if rob else ''}{tag}.json")
        if os.path.exists(f):
            out += json.load(open(f))
    return out


def rate(rs):
    p, h = wilson(sum(ok(r) for r in rs), len(rs))
    return 100 * p, 100 * h, len(rs)


def three(src, tag, robots):
    nom = jumps(src, tag, robots)
    rob = jumps(src, tag, robots, rob=True)
    st = [r for r in rob if r.get('cond', '').split('+')[-1] in START]
    se = [r for r in rob if r.get('cond', '').split('+')[-1] in SENSE]
    return rate(nom), rate(st), rate(se)


mix = lambda base, r4m: {'real_r1': base, 'real_s1': base, 'real_r5': base, 'real_r4m': r4m}


# ---- fig 1 --------------------------------------------------------------------------------------------------------
def fig1():
    rows = {'nominal\nsim': [(mix('hw6g16_v10s2it20', 'base6_r4m'), 'start'), (mix('gate_loose', 'base6_r4m'), 'final'), ('vs24', 'final')],
            'DR fine-\ntune (A)': [(mix('drA_hw24', 'drA_hw24_r4m'), 'start'), (mix('drA_hw24', 'drA_hw24_r4m'), 'final'), ('drA_vs24', 'final')],
            'DR from\nscratch (B)': [(mix('drB_hw24', 'drB_hw24_r4m'), 'start'), (mix('drB_hw24', 'drB_hw24_r4m'), 'final'), ('drB_vs24', 'final')]}
    conds = ['zero-shot', '+24 jumps, nominal starts', '+24 jumps, varied starts']
    cols = [C['grey'], C['blue'], C['sky']]
    evals = ['nominal evaluation', 'start perturbations (6)', 'sensing perturbations (2)']
    data = {g: [three(s, t, R4) for s, t in v] for g, v in rows.items()}
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.1), sharey=True)
    x = np.arange(len(rows))
    for e, ax in enumerate(axs):
        for c in range(3):
            v = [data[g][c][e] for g in rows]
            ax.bar(x + (c - 1) * 0.27, [q[0] for q in v], 0.25, yerr=[q[1] for q in v], color=cols[c], capsize=2,
                   error_kw=dict(lw=0.8), label=conds[c] if e == 0 else None, edgecolor='white', linewidth=1.5)
            for xi, q in zip(x, v):
                ax.text(xi + (c - 1) * 0.27, q[0] + q[1] + 1.2, f'{q[0]:.0f}', ha='center', va='bottom', fontsize=7.5,
                        color='#333')
        ax.set_xticks(x)
        ax.set_xticklabels(list(rows), fontsize=9)
        ax.set_title(evals[e], fontsize=10)
        ax.set_ylim(0, 62)
    axs[0].set_ylabel('success on the reserved test goals (%)')
    axs[0].legend(fontsize=8, loc='upper left', frameon=False)
    fig.suptitle('Sim stage x hardware stage (robots r1 s1 r4m r5, n = 128 per bar, 95 % Wilson)', fontsize=11)
    fig.savefig(os.path.join(OUT, 'fig1_factorial.png'))
    plt.close(fig)


# ---- per-goal ILC (JumpILC) ------------------------------------------------------------------------------------------
okf = lambda r: (not r['fell']) and abs(r['ex']) <= 0.05 and abs(r['ez']) <= 0.03


def _land(z, base):
    gx, gh = float(base.split('_g')[1].split('_')[0]), float(base.split('_g')[1].split('_')[1])
    xr, lx = np.asarray(z['x_ref'], float), np.asarray(z['log_X'], float)
    target = xr[-1].copy()
    target[:2] = xr[0, :2] + np.array([gx, gh])
    target[2] = 0.0
    e = lx[-1] - target
    return dict(robot='real_' + base.split('_real_')[1].split('_s')[0], goal=[gx, gh], ex=float(e[0]), ez=float(e[1]),
                fell=bool(z['log_fell']))


def jilc_at(k, robots):
    rs = []
    for d in sorted(glob.glob(os.path.join(P, 'jilc_v3', 'scratch_g*'))):
        tr = sorted(glob.glob(os.path.join(d, 'trial_*.npz')))
        if not tr:
            continue
        r = _land(np.load(tr[min(k + 1, len(tr)) - 1]), os.path.basename(d))
        if r['robot'] in robots:
            rs.append(r)
    return rs


def jilc_eval(k, robots):
    rs = []
    for d in sorted(glob.glob(os.path.join(P, 'jilc_eval', f'b{k}', 'retarget_g*'))):
        tr = sorted(glob.glob(os.path.join(d, 'trial_*.npz')))
        if not tr:
            continue
        r = _land(np.load(tr[0]), os.path.basename(d))
        if r['robot'] in robots:
            rs.append(r)
    return rs


def rate_f(rs):
    p, h = wilson(sum(okf(r) for r in rs), len(rs))
    return 100 * p, 100 * h, len(rs)


# ---- fig 2 --------------------------------------------------------------------------------------------------------
def fig2():
    rows = [('ours, zero-shot', rate(jumps('hw6g16_v10s2it20', 'start', R3)), C['grey']),
            ('ours + 24 real jumps', rate(jumps('gate_loose', 'final', R3)), C['blue']),
            ('ours + 96 real jumps', rate(jumps('hw6g16_v10s2it20', 'final', R3)), C['blue']),
            ('our learner + DR (A: DR fine-tune), zero-shot', rate(jumps('drA_hw24', 'start', R3)), C['grey']),
            ('our learner + DR (A) + 24 real jumps', rate(jumps('drA_hw24', 'final', R3)), C['sky']),
            ('our learner + DR (B: DR from scratch), zero-shot', rate(jumps('drB_hw24', 'start', R3)), C['grey']),
            ('our learner + DR (B) + 24 real jumps', rate(jumps('drB_hw24', 'final', R3)), C['sky']),
            ('per-goal ILC, 3 trials / test goal (24)', rate_f(jilc_eval(3, R3)), C['orange']),
            ('per-goal ILC, 12 trials / test goal (96)', rate_f(jilc_eval(12, R3)), C['orange']),
            ('PPO + DR, zero-shot', rate(jumps('base/eval_ppo_plain', 'start', R3)), C['pink']),
            ('RMA, zero-shot', rate(jumps('base/eval_rma', 'start', R3)), C['pink']),
            ('RMA cross-trial (6 calibration jumps)', rate(jumps('base/eval_rma_ctx', 'final', R3)), C['pink']),
            ('FADA (48 LoRA jumps)', rate(jumps('base/eval_fada', 'final', R3)), C['pink'])]
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, (p, h, n), col) in zip(y, rows):
        ax.barh(yi, p, 0.7, xerr=h, color=col, capsize=2, error_kw=dict(lw=0.8), edgecolor='white', linewidth=1.5)
        ax.text(p + h + 0.8, yi, f'{p:.0f}%  (n {n})', va='center', fontsize=8, color='#333')
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel('success on the reserved test goals, nominal evaluation (%)')
    ax.set_xlim(0, 65)
    ax.grid(axis='y', alpha=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C['grey'], C['blue'], C['sky'], C['orange'], C['pink'])]
    ax.legend(handles, ['zero-shot (no real jumps)', 'ours + hardware stage', 'our learner + DR + hardware stage',
                        'per-goal ILC (JumpILC)', 'RL / adaptation baselines'], fontsize=8, frameon=False,
              loc='upper center', bbox_to_anchor=(0.35, -0.1), ncol=3)
    ax.set_title('ILC2Real vs baselines (robots r1 s1 r5, 95 % Wilson)', fontsize=11)
    fig.savefig(os.path.join(OUT, 'fig2_methods.png'))
    plt.close(fig)


# ---- fig 3 --------------------------------------------------------------------------------------------------------
def fig3():
    ours = [(0, jumps('hw6g16_v10s2it20', 'start', R3)), (12, jumps('igfix_b12', 'final', R3)),
            (24, jumps('gate_loose', 'final', R3)), (48, jumps('hw6g_v10s2it20', 'final', R3)),
            (96, jumps('hw6g16_v10s2it20', 'final', R3))]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    o = [(b, *rate(rs)) for b, rs in ours if rs]
    ax.errorbar([q[0] for q in o], [q[1] for q in o], yerr=[q[2] for q in o], color=C['blue'], marker='o', ms=6, lw=2,
                capsize=3, label='ILC2Real: one network, 6 training goals (never the test goals)')
    for lab, run, col, mk in (('our learner + DR (A: DR fine-tune) + our hardware stage', 'drA_hw24', C['sky'], 'v'),
                              ('our learner + DR (B: DR from scratch) + our hardware stage', 'drB_hw24', C['green'], '^')):
        d = [(0, *rate(jumps(run, 'start', R3))), (24, *rate(jumps(run, 'final', R3)))]
        ax.errorbar([q[0] + (1 if mk == 'v' else -1) for q in d], [q[1] for q in d], yerr=[q[2] for q in d], color=col,
                    marker=mk, ms=6, lw=1.3, capsize=3, label=lab)
    ks = list(range(0, 13))
    jt = [(8 * k, *rate_f(jilc_at(k, R3))) for k in ks]
    ax.plot([q[0] for q in jt], [q[1] for q in jt], color=C['orange'], lw=1.5, ls='--',
            label='per-goal ILC on the 8 test goals, trial k+1')
    je = [(8 * k, *rate_f(jilc_eval(k, R3))) for k in (3, 12)]
    ax.errorbar([q[0] for q in je], [q[1] for q in je], yerr=[q[2] for q in je], color=C['orange'], marker='s', ms=6,
                ls='none', capsize=3, label='per-goal ILC, re-flown with our protocol')
    ax.set_xlabel('real jumps per robot')
    ax.set_ylabel('success on the reserved test goals (%)')
    ax.set_xticks([0, 12, 24, 48, 72, 96])
    ax.set_ylim(0, 60)
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    ax.set_title('Few-shot adaptation budget (robots r1 s1 r5)', fontsize=11)
    fig.savefig(os.path.join(OUT, 'fig3_budget.png'))
    plt.close(fig)


# ---- fig 4 --------------------------------------------------------------------------------------------------------
def fig4():
    groups = [
        ('budget', C['blue'], [('zero-shot', 'hw6g16_v10s2it20', 'start'), ('+12 fixed goals', 'igfix_b12', 'final'),
                               ('+24 (deployment recipe)', 'gate_loose', 'final'), ('+24 run 1', 'igfix_b24', 'final'),
                               ('+24 run 2', 'sens_none', 'final'), ('+48', 'hw6g_v10s2it20', 'final'),
                               ('+96', 'hw6g16_v10s2it20', 'final')]),
        ('goal selection', C['sky'], [('info-gain 12', 'ig_b12', 'final'), ('info-gain 24', 'ig_b24', 'final'),
                                      ('info-gain v2 12', 'ig2_b12', 'final'), ('info-gain v2 24', 'ig2_b24', 'final'),
                                      ('sequential info-gain 12', 'ig1_b12', 'final'), ('random 12', 'igrnd_b12', 'final'),
                                      ('random 24', 'igrnd_b24', 'final'), ('8 goals (6 boxes) x 3', 'box8', 'final')]),
        ('sensitivity / direction', C['red'], [('secant per goal', 'sens_goal', 'final'), ('secant shared', 'sens_shared', 'final'),
                                                 ('value residual, ridge 1', 'sens_vres1', 'final'),
                                                 ('value residual, ridge 10', 'sens_vres10', 'final'),
                                                 ('Abbeel-style direction', 'hw_abbeel', 'final'),
                                                 ('Jacobians along the real trajectory', 'hw_jacreal', 'final'),
                                                 ("model's landing (no measurement)", 'hw_model_res', 'final')]),
        ('update details', C['orange'], [('default gate', 'gate_default', 'final'), ('anchor 0.3', 'anc03', 'final'),
                                         ('anchor 0.1', 'anc01', 'final'), ('2 reps x 2 its', 'reps2', 'final'),
                                         ('varied starts 24', 'vs24', 'final'), ('varied starts 48', 'vs48', 'final'),
                                         ('varied starts, sim from measured start', 'vs24_meas', 'final')]),
        ('trained under a constant perturbation', C['pink'], [(p, f'tp_{p}', 'final') for p in START + SENSE]),
        ('our learner + DR + our hardware stage', C['green'], [('DR (A) zero-shot', 'drA_hw24', 'start'),
                                                              ('DR (A) + 24', 'drA_hw24', 'final'),
                                                              ('DR (A) + 24, varied starts', 'drA_vs24', 'final'),
                                                              ('DR (B) zero-shot', 'drB_hw24', 'start'),
                                                              ('DR (B) + 24', 'drB_hw24', 'final'),
                                                              ('DR (B) + 24, varied starts', 'drB_vs24', 'final')]),
    ]
    rows = [(g, col, lab, rate(jumps(src, tag, R3))) for g, col, items in groups for lab, src, tag in items]
    rows = [r for r in rows if r[3][2] > 0]
    fig, ax = plt.subplots(figsize=(8.6, 0.27 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    for yi, (g, col, lab, (p, h, n)) in zip(y, rows):
        ax.barh(yi, p, 0.72, xerr=h, color=col, capsize=1.5, error_kw=dict(lw=0.7), edgecolor='white', linewidth=1)
        ax.text(p + h + 0.6, yi, f'{p:.0f}', va='center', fontsize=7.5, color='#333')
    zs = rate(jumps('hw6g16_v10s2it20', 'start', R3))[0]
    ax.axvline(zs, color=C['grey'], lw=1, ls=':', label=f'zero-shot ({zs:.0f} %)')
    ax.set_yticks(y)
    ax.set_yticklabels([r[2] for r in rows], fontsize=8)
    ax.set_xlabel('success on the reserved test goals, nominal evaluation (%)')
    ax.set_xlim(0, 58)
    ax.grid(axis='y', alpha=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for _, col, _ in groups]
    ax.legend(handles + [plt.Line2D([], [], color=C['grey'], ls=':')], [g for g, _, _ in groups] + ['zero-shot'],
              fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.4, -0.05), ncol=3)
    ax.set_title('Every hardware-stage arm (24 real jumps per robot unless noted; robots r1 s1 r5, n = 96)', fontsize=10)
    fig.savefig(os.path.join(OUT, 'fig4_hw_arms.png'))
    plt.close(fig)


# ---- fig 5 --------------------------------------------------------------------------------------------------------
def fig5():
    groups = [('nominal (control)', ['st_ctl_s0', 'st_ctl_s2']),
              ('Nguyen-style stages', ['st_sched_s0', 'st_sched_s2', 'st_floor_s0', 'st_floor_s2']),
              ('value-gradient critic', ['vg1_s0', 'vg1_s2', 'vg2_s0', 'vg2_s2']),
              ('residual critic', ['vgr1_s0', 'vgr1_s2', 'vgr2_s0', 'vgr2_s2']),
              ('iLQR feedback matching', ['km001_s0', 'km01_s0', 'km1_s0', 'km01_s2', 'km1_s2', 'km3_s0']),
              ('GPS-style distillation', ['gps2_km1_s0', 'gps2_km1_s2', 'gps2_s0']),
              ('DR from scratch (B)', ['drB_s0', 'drB_s2'])]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for gi, (g, runs) in enumerate(groups):
        for it, mk, col in ((20, 'o', C['sky']), (30, 's', C['blue'])):
            v = []
            for r in runs:
                f = os.path.join(P, r, f'cpu_val_g0_plane_it{it}.json')
                if os.path.exists(f):
                    v.append(100 * np.mean([ok(x) for x in json.load(open(f))]))
            if v:
                xs = gi + (0.12 if it == 30 else -0.12) + np.linspace(-0.06, 0.06, len(v))
                ax.scatter(xs, v, marker=mk, s=26, color=col, zorder=3, label=f'iteration {it}' if gi == 0 else None,
                           edgecolor='white', linewidth=0.6)
                ax.hlines(np.mean(v), gi + (0.12 if it == 30 else -0.12) - 0.1, gi + (0.12 if it == 30 else -0.12) + 0.1,
                          color=col, lw=2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g for g, _ in groups], fontsize=8.5, rotation=15, ha='right')
    ax.set_ylabel('CPU zero-shot success (%)')
    ax.set_ylim(0, 50)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title('Sim-stage variants: zero-shot on the validation goals (4 robots x 8 goals x 2; one point per seed / '
                 'setting, bar = mean)', fontsize=9.5)
    fig.savefig(os.path.join(OUT, 'fig5_sim_variants.png'))
    plt.close(fig)


# ---- fig 6 --------------------------------------------------------------------------------------------------------
def fig6():
    rows = [('ours (nominal sim)', 1920, 32 / 60, C['blue']), ('ours, DR fine-tune (A)', 2560, 1.0, C['sky']),
            ('ours, DR from scratch (B)', 2880, 2.0, C['sky']), ('PPO + DR', 76800, 5.0, C['pink']),
            ('RMA', 76800, 5.0, C['pink']), ('FADA', 102400, 16.0, C['pink'])]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, k, lab, fmt in ((axs[0], 1, 'simulated jumps to train (log)', '{:,.0f}'),
                            (axs[1], 2, 'wall time to train, hours (log)', '{:.1f} h')):
        for yi, r in zip(y, rows):
            ax.barh(yi, r[k], 0.7, color=r[3], edgecolor='white', linewidth=1.5)
            ax.text(r[k] * 1.08, yi, fmt.format(r[k]), va='center', fontsize=8, color='#333')
        ax.set_xscale('log')
        ax.set_xlabel(lab)
        ax.grid(axis='y', alpha=0)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([r[0] for r in rows], fontsize=9)
    axs[0].set_xlim(1e3, 4e5)
    axs[1].set_xlim(0.3, 50)
    fig.suptitle('Sim cost (from the run logs; RL / FADA times approximate, shared GPUs)', fontsize=11)
    fig.savefig(os.path.join(OUT, 'fig6_efficiency.png'))
    plt.close(fig)


# ---- fig 7 --------------------------------------------------------------------------------------------------------
def fig7():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    for ax, tag, title in ((axs[0], 'zeroshot', 'zero-shot'), (axs[1], 'after24', 'after the 24-jump hardware stage')):
        rs = []
        for f in glob.glob(os.path.join(P, 'figdata', f'grid_{tag}_real_*.json')):
            rs += json.load(open(f))
        by = {}
        for r in rs:
            by.setdefault((round(r['goal'][0], 3), round(r['goal'][1], 3)), []).append(ok(r))
        g = np.array([[k[0], k[1], 100 * np.mean(v), len(v)] for k, v in by.items()])
        sc = ax.scatter(g[:, 0], g[:, 1], c=g[:, 2], cmap='viridis', vmin=0, vmax=100, s=70, marker='s',
                        edgecolor='white', linewidth=0.5)
        ax.set_title(f'{title}: {np.mean([ok(r) for r in rs]) * 100:.0f} % of {len(rs)} jumps', fontsize=10)
        ax.set_xlabel('goal distance x (m)')
        ax.grid(alpha=0.15)
    axs[0].set_ylabel('goal height h (m, box)')
    cb = fig.colorbar(sc, ax=axs, fraction=0.025, pad=0.02)
    cb.set_label('success (%), 5 robots pooled')
    fig.suptitle('Success over the goal plane (flat jumps at h = 0, box jumps above)', fontsize=11)
    fig.savefig(os.path.join(OUT, 'fig7_goal_maps.png'))
    plt.close(fig)


# ---- fig 8 --------------------------------------------------------------------------------------------------------
M = os.path.join(P, 'match')


def match_curve(name):
    """A data-matching run's checkpoints {real jumps per robot: (success %, CI half-width, n)} on the test goals,
    pooled over r1 s1 r5 (quad_real_ppo.py summary.json / quad_eval_ckpts.py ckpt_summary.json)."""
    for f in ('summary.json', 'ckpt_summary.json'):
        fp = os.path.join(M, name, f)
        d = json.load(open(fp)) if os.path.exists(fp) else None
        if isinstance(d, dict) and 'hist' not in d:      # fada_adapt.py's summary.json is a per-robot list
            out = []
            for k, rob in sorted(d.items(), key=lambda kv: int(kv[0])):
                v = [x for x in rob.values() if x.get('success') is not None and x.get('n')]
                if len(v) < 3:
                    continue
                n = sum(x['n'] for x in v)
                p, h = wilson(round(sum(x['success'] * x['n'] for x in v)), n)
                out.append((int(k), 100 * p, 100 * h, n))
            return out
    return []


def fig8():
    """The data-matching study: the baselines trained ON THE ROBOTS with privileged data, from their sim policies --
    how many real jumps until they match our best model after the hardware stage?"""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    tgt = rate(jumps('drB_hw24', 'final', R3))
    ax.axhspan(tgt[0] - tgt[1], tgt[0] + tgt[1], color=C['green'], alpha=0.12, lw=0)
    ax.axhline(tgt[0], color=C['green'], lw=1.2, ls='--')
    ax.errorbar([24], [tgt[0]], yerr=[tgt[1]], color=C['green'], marker='^', ms=8, capsize=3, lw=0,
                label=f'ours: our learner + DR (B) + 24 real jumps ({tgt[0]:.1f} %)', zorder=5)
    ours = [(0, jumps('hw6g16_v10s2it20', 'start', R3)), (24, jumps('gate_loose', 'final', R3)),
            (48, jumps('hw6g_v10s2it20', 'final', R3)), (96, jumps('hw6g16_v10s2it20', 'final', R3))]
    o = [(b, *rate(rs)) for b, rs in ours if rs]
    snap = lambda run, its: [(6 * k, rs) for k in its for rs in [[r for rb in R3 for f in [os.path.join(
        P, 'fixbox', run, rb, f'it{k}', 'eval_planefinal.json')] if os.path.exists(f) for r in json.load(open(f))]]
        if len(rs) >= 32 * len(R3)]                  # our runs every 6 real jumps (quad_eval_ckpts.py snapshots)
    for run, its, col, mk, lab in (('hw6g16_v10s2it20', range(17), C['blue'], 'o',
                                    'ours, nominal-sim learner: our hardware stage, every 6 jumps'),
                                   ('drB_hw24', range(5), C['green'], '^',
                                    'ours: our learner + DR (B): our hardware stage, every 6 jumps')):
        s_ = [(b, *rate(rs)) for b, rs in snap(run, its)]
        if len(s_) > 2:
            ax.errorbar([q[0] for q in s_], [q[1] for q in s_], yerr=[q[2] for q in s_], color=col, marker=mk, ms=4,
                        lw=1.8, capsize=1.5, label=lab, zorder=4)
            if run.startswith('hw6g16'):
                o = []                                   # the snapshot curve replaces the separate-run points
    if o:
        ax.plot([q[0] for q in o], [q[1] for q in o], color=C['blue'], marker='o', ms=5, lw=1.5,
                label='ours, nominal-sim learner + our hardware stage')
    runs = [('ppo_real', 'PPO+DR, then PPO on the robot (privileged critic)', C['pink'], 's', '-'),
            ('rma_real', 'RMA teacher with the TRUE dynamics, then PPO on the robot', C['red'], 'D', '-'),
            ('fada_real', "FADA: LoRA on the robot, its own (sim) planner", C['orange'], 'v', '-'),
            ('fada_real_oracle', 'FADA with an ORACLE planner (fit to our real jumps)', C['orange'], 'v', '--')]
    for name, lab, col, mk, ls in runs:
        c = match_curve(name)
        if not c:
            continue
        x = [q[0] for q in c]
        ax.errorbar(x, [q[1] for q in c], yerr=[q[2] for q in c], color=col, marker=mk, ms=5, lw=1.5, ls=ls,
                    capsize=2, mfc='white' if ls == '--' else col, label=lab)
        hit = [q[0] for q in c if q[1] >= tgt[0]]
        if hit:
            ax.annotate(f'matches at {hit[0]}', (hit[0], tgt[0]), textcoords='offset points', xytext=(4, 8),
                        fontsize=8, color=col)
    ax.set_xscale('symlog', linthresh=12, linscale=1.0)
    ticks = [0, 6, 12, 24, 48, 96, 192, 384, 768, 2016]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=8)
    ax.set_xlim(-2, 2300)
    ax.set_ylim(0, 70)
    ax.set_xlabel('real jumps per robot (the baselines: on the robot itself, with privileged information)')
    ax.set_ylabel('success on the reserved test goals (%)')
    ax.legend(fontsize=7.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax.set_title('Real data needed to match ILC2Real (robots r1 s1 r5; 95 % intervals)', fontsize=11)
    fig.savefig(os.path.join(OUT, 'fig8_real_data_match.png'))
    fig.savefig(os.path.join(HERE, '..', '..', 'paper', 'quadruped', 'figures', '13_real_data_matching.png'))   # the browsed set
    plt.close(fig)


if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        try:
            f()
            print('ok', f.__name__)
        except Exception as e:                       # one missing input must not stop the rest
            print('FAILED', f.__name__, repr(e))
    print('->', OUT)
