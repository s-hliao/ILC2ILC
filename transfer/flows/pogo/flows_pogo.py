#!/usr/bin/env python3
"""
flows_pogo.py: the closed-loop flow of the pogo stick (apex-to-apex return map) under the start policies and along the
DistILC hardware stage -- how the real sticks corrupt it at iteration 0 and how the stage re-corrects it (SPEC.md).

The map. State = the flight apex (h, vx) (vz = 0, x irrelevant; the MLP / FADA policies are memoryless, so the map
is exact; RMA keeps a 50-period history, longer than a hop, so its lanes first fly one warm-up hop: see fly_hop). F^s_pi(x, g): start in flight at the apex on a control-period boundary, fly
the policy on system s until liftoff; the next apex is ballistic: h' = z + vz^2 / 2g, vx' = vx. V = F(x) - x.
  s = sim:   the nominal model (common.substep, nominal parameters, 0.2 ms steps as pogo_jax.SUB = 50)
  s = robot: real_pogo's sticks (robot_params, 0.1 ms steps as real_pogo.SUB_REAL = 100)
Reference flow: V^sim_{pi_0}. Errors are normalized by the task tolerances (ERR_SCALE: 2 cm apex, 0.1 m/s speed).

States (per start, seed, robot, goal): the tube = the apex states visited by sim pi_0 and by the robot's pi_i (every
iteration) from the goal's 4 validation drops (seed 301) over 20 hops, a random subsample of N_TUBE of them plus N_TUBE
copies perturbed uniformly by +-1 tolerance (+-2 cm, +-0.1 m/s); 'far' = N_TUBE copies perturbed by +-3 tolerances.
Goals: IN = common.TRAIN_GOALS (the stage's 6); the validation goals are goals_from_seed(16, VAL_SEED = 301) (the
first 8 are the val evaluation goals), HULL = inside the IN hull or within HULL_MARGIN of it in the normalized goal box,
OUT = the rest. Never the spent test seeds.
Task link: the fixed point x_bar = mean of the apexes of hops 10..20 from each drop; error |(x_bar - g) / tol|.

    python flows_pogo.py compute [--jobs 8]      -> cache/<start>_s<seed>.npz
    python flows_pogo.py portraits                -> fig_portraits_<start>.png
    python flows_pogo.py aggregate                -> summary.json, summary_extra.json, fig_field_err.png, fig_task_err.png
"""
import argparse
import json
import os
import sys

POGO = '/home/henry/ilc_ws/src/ilc_pogo'
HERE = os.path.dirname(os.path.abspath(__file__))
for k in ('POGO_EVAL_SET', 'POGO_TEST_SEED', 'POGO_EVAL_TAG'):
    os.environ.pop(k, None)
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, POGO)

import numpy as np                                    # noqa: E402

import common as C                                    # noqa: E402
import policies                                       # noqa: E402
import real_pogo as rp                                # noqa: E402

assert C.EVAL_SET == 'val' and C.VAL_SEED == 301
TOL = C.ERR_SCALE                                     # (apex h 0.02 m, vx 0.10 m/s)
SUB_SIM, SUB_REAL = 50, rp.SUB_REAL
ROBOTS = rp.DEFAULT_ROBOTS
SEEDS = (0, 1, 2)
N_DROPS, N_HOPS, FP_FROM = 4, 20, 10
N_TUBE = 100
HULL_MARGIN = 0.10
RUNS = os.path.join(HERE, 'runs')
V3 = os.path.join(POGO, 'runs', 'v3')
CACHE = os.path.join(HERE, 'cache')
REDO_V1 = os.environ.get('FLOWS_REDO_V1', '0') == '1'   # recompute caches written before goal_err / the raw maps existed

STARTS = {   # name -> (iterations, policy path for (seed, robot, it))
    'ours': (list(range(7)), 'nom'),
    'ours+DR': (list(range(7)), 'drNomInit'),
    'PPO+DR': (list(range(7)), 'ppo_dr'),
    'RMA': ([0], None),
    'FADA': ([0, 4], None),
}
BASE = {'nom': 'runs/nom_s{s}/policy_pick.npz', 'drNomInit': 'runs/drNomInit_s{s}/policy_pick.npz',
        'ppo_dr': 'runs/ppo_dr_s{s}/policy_final.npz'}


def policy_path(start, s, robot, it):
    if start == 'RMA':
        return os.path.join(POGO, 'runs/rmaR1_s0/policy_final.npz' if s == 0 else f'runs/rmaR1full_s{s}/policy_final.npz')
    if start == 'FADA':        # FADA's own adaptation (fada.py --iters 4, 6 trials each): it0 = zs, it4 = hw24
        return os.path.join(V3, f'fadaF2_s{s}_' + ('zs' if it == 0 else 'hw24'), f'{robot}_policy.npz')
    arm = STARTS[start][1]
    if it == 0:
        return os.path.join(POGO, BASE[arm].format(s=s))
    d = os.path.join(RUNS, f'{arm}_s{s}_PF')
    return os.path.join(d, f'{robot}_policy.npz') if it == 6 else os.path.join(d, f'it{it}', f'{robot}_policy.npz')


def goal_sets():
    from scipy.spatial import ConvexHull
    nz = lambda g: (np.asarray(g) - C.GOAL_LO) / (C.GOAL_HI - C.GOAL_LO)
    hull = ConvexHull(nz(C.TRAIN_GOALS))
    val = C.goals_from_seed(16, C.VAL_SEED)
    dist = (nz(val) @ hull.equations[:, :2].T + hull.equations[:, 2]).max(1)
    return {'IN': C.TRAIN_GOALS.copy(), 'HULL': val[dist <= HULL_MARGIN], 'OUT': val[dist > HULL_MARGIN]}


def system_params(name):
    p = dict(C.NOMINAL) if name == 'sim' else rp.robot_params(name)
    return {k: p[k] for k in C.PKEYS}


def fly_hop(pol, H, VX, G, system, tmax=150):
    """One hop of the return map, vectorized over lanes: start in flight at the apex (z = H, vx = VX, vz = 0) on a
    control-period boundary, fly `pol` on `system` ('sim' = nominal model, else a real_pogo robot) until liftoff.
    Policies with memory (RMA: a 50-period = 0.5 s history, longer than a hop) first fly one warm-up hop from the same
    apex and back up to its next apex, then the lane is reset to the apex (H, VX) with the history kept: the measured
    hop then sees a history of genuine hopping on that system near x (memoryless policies: no warm-up, exact map).
    -> (next apex (n, 2) [h, vx], fell (n,)); NaN where it fell or did not lift off within tmax periods."""
    H, VX = np.asarray(H, float), np.asarray(VX, float)
    G = np.asarray(G, float)
    n = len(H)
    x0 = np.zeros((n, C.NX))
    x0[:, 1], x0[:, 2] = H, VX
    x = x0.copy()
    p = {k: np.full(n, v) for k, v in system_params(system).items()}
    sub = SUB_SIM if system == 'sim' else SUB_REAL
    h = C.DT / sub
    pol = policies.as_policy(pol)
    memory = type(pol).__name__ == 'RMAPolicy'
    pol.reset(n)
    out = np.full((n, 2), np.nan)
    done = ~np.isfinite(H) | ~np.isfinite(VX)
    fell = done.copy()
    phase = np.full(n, 0 if memory else 1)
    lifted = np.zeros(n, bool)
    for t in range(tmax * (2 if memory else 1)):
        if done.all():
            break
        a = pol(C.features(x, G, np), x, G)
        for j in range(sub):
            st0 = x[:, 5]
            xn = C.substep(x, a, p, h, np)
            lo = (st0 > 0.5) & (xn[:, 5] < 0.5) & ~done
            rec = lo & (phase == 1)
            if rec.any():
                out[rec, 0] = xn[rec, 1] + 0.5 * np.maximum(xn[rec, 3], 0.0) ** 2 / C.GRAV
                out[rec, 1] = xn[rec, 2]
                done |= rec
            lifted |= lo & (phase == 0)
            f = C.failed(xn, np) & ~done
            fell |= f
            done |= f
            x = np.where(done[:, None], x, xn)
        if memory:                                   # warm-up lanes back at their apex: reset the state to (H, VX)
            tp = (phase == 0) & lifted & (x[:, 3] <= 0) & ~done
            x[tp], phase[tp] = x0[tp], 1
    fell |= ~done
    return out, fell


def orbit(pol, X0, G, system, hops=N_HOPS):
    """Iterates the return map from apex states X0 (n, 2) -> (hops + 1, n, 2), NaN after a fall."""
    xs = [np.asarray(X0, float)]
    for _ in range(hops):
        xs.append(fly_hop(pol, xs[-1][:, 0], xs[-1][:, 1], G, system)[0])
    return np.stack(xs)


def drop_apexes(G):
    """The validation drops (real_pogo.start_states, pert 1, seed 301; N_DROPS per goal) as their ballistic apex."""
    Gr = np.repeat(G, N_DROPS, 0)
    x0 = rp.start_states(Gr, 1.0, np.random.default_rng(C.VAL_SEED))
    return np.stack([x0[:, 1] + 0.5 * x0[:, 3] ** 2 / C.GRAV, x0[:, 2]], -1), Gr


def fixed_point_err(orb, Gr):
    """orb (hops + 1, n, 2): per lane the mean apex of hops FP_FROM.. and its normalized error to the goal, plus the
    normalized spread (a period-2 gait / no convergence) -> (err (n,), spread (n,), fell (n,))."""
    late = orb[FP_FROM:]
    fell = ~np.isfinite(late).all((0, 2))
    xb = late.mean(0)
    err = np.linalg.norm((xb - Gr[:, ::-1]) / TOL, axis=1)       # goal g = (v*, h*) -> (h*, v*)
    spread = np.linalg.norm(late.std(0) / TOL, axis=1)
    err[fell], spread[fell] = np.nan, np.nan
    return err, spread, fell


def compute_job(job):
    start, s = job
    out_path = os.path.join(CACHE, f'{start}_s{s}.npz')
    if os.path.exists(out_path) and REDO_V1 and 'goal_err' not in np.load(out_path).files:
        os.makedirs(os.path.join(HERE, 'cache_old'), exist_ok=True)     # a v1 cache (no raw maps / goal-flow error)
        os.replace(out_path, os.path.join(HERE, 'cache_old', f'{start}_s{s}_v1.npz'))
    if os.path.exists(out_path):
        return out_path
    iters = STARTS[start][0]
    if STARTS[start][1] and not os.path.exists(os.path.join(RUNS, f'{STARTS[start][1]}_s{s}_PF', 'summary.json')):
        print(f'skip {start} s{s}: the stage run {STARTS[start][1]}_s{s}_PF is missing / unfinished', flush=True)
        return None
    GS = goal_sets()
    names = [k for k in ('IN', 'HULL', 'OUT') for _ in GS[k]]
    G = np.concatenate([GS[k] for k in ('IN', 'HULL', 'OUT')])
    ng = len(G)
    X0, Gr = drop_apexes(G)
    rs = np.random.default_rng(1000 + s)
    pol0 = policy_path(start, s, ROBOTS[0], 0)        # it0 is the same network on every robot
    orb_sim = orbit(pol0, X0, Gr, 'sim')
    fp_sim = fixed_point_err(orb_sim, Gr)
    res = dict(goals=G, goal_set=np.array(names), iters=np.array(iters), robots=np.array(ROBOTS),
               fp_sim=fp_sim[0].reshape(ng, N_DROPS), spread_sim=fp_sim[1].reshape(ng, N_DROPS))
    F_err = np.full((len(ROBOTS), len(iters), ng), np.nan)
    F_far = np.full_like(F_err, np.nan)
    F_abs = np.full_like(F_err, np.nan)              # |V^real_{pi_i}| on the tube (scale reference)
    F_invalid = np.full_like(F_err, np.nan)
    FP = np.full((len(ROBOTS), len(iters), ng, N_DROPS), np.nan)
    SP = np.full_like(FP, np.nan)
    REF_abs = np.full((len(ROBOTS), ng), np.nan)
    GOAL_err = np.full_like(F_err, np.nan)          # |F^real_{pi_i}(x) - g| on the tube: the stage's own target flow
    GOAL_sim = np.full((len(ROBOTS), ng), np.nan)   # (V* = g - x, every apex at the goal) and the same for sim pi_0
    MAPS = dict(states=[], sim=[], real=[])         # the raw maps, so any other reference can be computed post hoc
    for ri, r in enumerate(ROBOTS):
        pols = [policy_path(start, s, r, it) for it in iters]
        orbs = [orbit(p, X0, Gr, r) for p in pols]
        for k, o in enumerate(orbs):
            e, sp, _ = fixed_point_err(o, Gr)
            FP[ri, k], SP[ri, k] = e.reshape(ng, N_DROPS), sp.reshape(ng, N_DROPS)
        # the tube per goal: visited apexes of sim pi_0 and every real pi_i
        tube, far, gt = [], [], []
        for gi in range(ng):
            lanes = slice(gi * N_DROPS, (gi + 1) * N_DROPS)
            vis = np.concatenate([o[:, lanes].reshape(-1, 2) for o in [orb_sim] + orbs])
            vis = vis[np.isfinite(vis).all(1)]
            pick = vis[rs.integers(0, len(vis), N_TUBE)]
            pert = vis[rs.integers(0, len(vis), N_TUBE)] + rs.uniform(-1, 1, (N_TUBE, 2)) * TOL   # (h, vx)
            farp = vis[rs.integers(0, len(vis), N_TUBE)] + rs.uniform(-3, 3, (N_TUBE, 2)) * TOL
            tube.append(np.concatenate([pick, pert]))
            far.append(farp)
        tube, far = np.stack(tube), np.stack(far)          # (ng, 2 N, 2), (ng, N, 2)
        tube[..., 0], far[..., 0] = np.maximum(tube[..., 0], 0.52), np.maximum(far[..., 0], 0.52)
        allx = np.concatenate([tube, far], 1)              # (ng, 3 N, 2)
        Gl = np.repeat(G, allx.shape[1], 0)
        flat = allx.reshape(-1, 2)
        Fs = fly_hop(pol0, flat[:, 0], flat[:, 1], Gl, 'sim')[0].reshape(allx.shape)
        Vs = (Fs - allx) / TOL
        REF_abs[ri] = np.nanmean(np.linalg.norm(Vs[:, :2 * N_TUBE], axis=2), 1)
        ghv = G[:, None, ::-1]                             # goal as (h*, v*)
        GOAL_sim[ri] = np.nanmean(np.linalg.norm((Fs - ghv)[:, :2 * N_TUBE] / TOL, axis=2), 1)
        MAPS['states'].append(allx.astype(np.float32))
        MAPS['sim'].append(Fs.astype(np.float32))
        MAPS['real'].append([])
        for k, p in enumerate(pols):
            Fr = fly_hop(p, flat[:, 0], flat[:, 1], Gl, r)[0].reshape(allx.shape)
            Vr = (Fr - allx) / TOL
            d = np.linalg.norm(Vr - Vs, axis=2)            # (ng, 3 N) normalized |dV_i|
            F_err[ri, k] = np.nanmean(d[:, :2 * N_TUBE], 1)
            F_far[ri, k] = np.nanmean(d[:, 2 * N_TUBE:], 1)
            F_abs[ri, k] = np.nanmean(np.linalg.norm(Vr[:, :2 * N_TUBE], axis=2), 1)
            F_invalid[ri, k] = np.mean(~np.isfinite(d[:, :2 * N_TUBE]), 1)
            GOAL_err[ri, k] = np.nanmean(np.linalg.norm((Fr - ghv)[:, :2 * N_TUBE] / TOL, axis=2), 1)
            MAPS['real'][-1].append(Fr.astype(np.float32))
        print(json.dumps(dict(start=start, seed=s, robot=r, field_IN=[round(float(np.nanmean(F_err[ri, k, :6])), 3)
                                                                      for k in range(len(iters))])), flush=True)
    res.update(field_err=F_err, field_far=F_far, field_abs=F_abs, field_invalid=F_invalid, fp=FP, spread=SP, ref_abs=REF_abs,
               goal_err=GOAL_err, goal_sim=GOAL_sim, map_states=np.array(MAPS['states']), map_sim=np.array(MAPS['sim']),
               map_real=np.array(MAPS['real']))
    os.makedirs(CACHE, exist_ok=True)
    np.savez(out_path, **res)
    return out_path


# ---------------------------------------------------------------------------------------------------- aggregation
COL = {'ours': '#2a78d6', 'ours+DR': '#eb6834', 'PPO+DR': '#1baf7a', 'RMA': '#6250d6', 'FADA': '#e87ba4'}
MARK = {'ours': 'o', 'ours+DR': 's', 'PPO+DR': '^', 'RMA': 'D', 'FADA': 'v'}
LABEL = {'ours': 'ours (nominal) + stage', 'ours+DR': 'ours + DR + stage', 'PPO+DR': 'PPO+DR + our stage',
         'RMA': 'RMA (zero-shot)', 'FADA': 'FADA (own adaptation)'}
SETS = ('IN', 'HULL', 'OUT')
OUT_DR = ('real_act', 'real_offset', 'real_prog')


def boot_ci(v, B=10000, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return [np.nan] * 3
    idx = np.random.default_rng(seed).integers(0, len(v), (B, len(v)))
    m = v[idx].mean(1)
    return [float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def load_all():
    D = {}
    for st in STARTS:
        L = [np.load(os.path.join(CACHE, f'{st}_s{s}.npz')) for s in SEEDS if os.path.exists(os.path.join(CACHE, f'{st}_s{s}.npz'))]
        if L:
            D[st] = L
    return D


def per_unit(L, key, gset, robots=None):
    """-> (units, iters): per (robot, seed) the mean over the set's goals (and drops)."""
    rows, units = [], []
    for s, d in zip(SEEDS, L):
        m = d['goal_set'] == gset
        for ri, r in enumerate(d['robots']):
            if robots is not None and r not in robots:
                continue
            a = d[key][ri][:, m]
            rows.append(np.nanmean(a.reshape(a.shape[0], -1), 1))
            units.append(f'{r}:s{s}')
    return np.array(rows), units


def aggregate():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    D = load_all()
    GS = goal_sets()
    summary = dict(domain='pogo', map='apex-to-apex return map', proj=['apex h', 'vx'], starts={}, notes='')
    extra = dict(starts={}, goal_sets={k: v.tolist() for k, v in GS.items()})
    for st, L in D.items():
        iters = L[0]['iters'].tolist()
        ent = dict(iters=iters, field_err={}, task_err={})
        ex = dict(field_err_ci={}, task_err_ci={}, field_far_ci={}, field_invalid={}, field_rel_ci={},
                  field_err_ci_inDR={}, field_err_ci_outDR={}, task_err_ci_inDR={}, task_err_ci_outDR={},
                  task_err_sim={}, fall_frac={}, spread={}, success_frac={})
        for g in SETS:
            fe, units = per_unit(L, 'field_err', g)
            te, _ = per_unit(L, 'fp', g)
            ff, _ = per_unit(L, 'field_far', g)
            ra = np.array([np.nanmean(d['ref_abs'][ri][d['goal_set'] == g]) for d in L for ri in range(len(d['robots']))])
            ent['robots'] = units
            ent['field_err'][g] = fe.T.tolist()            # [iter][unit]
            ent['task_err'][g] = te.T.tolist()
            ex['field_err_ci'][g] = [boot_ci(fe[:, k]) for k in range(len(iters))]
            ex['task_err_ci'][g] = [boot_ci(te[:, k]) for k in range(len(iters))]
            ex['field_far_ci'][g] = [boot_ci(ff[:, k]) for k in range(len(iters))]
            ex['field_rel_ci'][g] = [boot_ci(fe[:, k] / ra) for k in range(len(iters))]
            for tag, rob in (('inDR', [r for r in ROBOTS if r not in OUT_DR]), ('outDR', list(OUT_DR))):
                a, _ = per_unit(L, 'field_err', g, rob)
                b, _ = per_unit(L, 'fp', g, rob)
                ex[f'field_err_ci_{tag}'][g] = [boot_ci(a[:, k]) for k in range(len(iters))]
                ex[f'task_err_ci_{tag}'][g] = [boot_ci(b[:, k]) for k in range(len(iters))]
            inv, _ = per_unit(L, 'field_invalid', g)
            ex['field_invalid'][g] = np.nanmean(inv, 0).tolist()
            fallf = []
            succ = []
            for k in range(len(iters)):
                vals = np.concatenate([d['fp'][:, k][:, d['goal_set'] == g].ravel() for d in L])
                fallf.append(float(np.mean(~np.isfinite(vals))))
                # |h err| <= 2 cm and |v err| <= 0.1 on the fixed point is implied by err <= 1 (a sufficient proxy)
                succ.append(float(np.mean(np.nan_to_num(vals, nan=np.inf) <= 1.0)))
            ex['fall_frac'][g], ex['success_frac'][g] = fallf, succ
            ex['spread'][g] = [float(np.nanmean(np.concatenate([d['spread'][:, k][:, d['goal_set'] == g].ravel() for d in L])))
                               for k in range(len(iters))]
            ex['task_err_sim'][g] = boot_ci([np.nanmean(d['fp_sim'][d['goal_set'] == g]) for d in L])
            if all('goal_err' in d.files for d in L):      # field error against the stage's target flow V* = g - x
                ge, _ = per_unit(L, 'goal_err', g)
                ex.setdefault('goal_err_ci', {})[g] = [boot_ci(ge[:, k]) for k in range(len(iters))]
                ex.setdefault('goal_err_sim', {})[g] = boot_ci([np.nanmean(d['goal_sim'][:, d['goal_set'] == g]) for d in L])
        summary['starts'][st] = ent
        extra['starts'][st] = ex
    summary['notes'] = (
        'Pogo (toy 2-D SLIP). Map: apex (h, vx) -> next apex under pi(.|g), started in flight on a control-period boundary; '
        'sim = nominal model (0.2 ms), real = real_pogo sticks (0.1 ms, damping 15 N s/m + each perturbation). '
        'field_err = mean over a tube of |V^real_{pi_i} - V^sim_{pi_0}| in tolerance units (2 cm apex, 0.1 m/s); '
        'tube = apexes visited by sim pi_0 and real pi_0..pi_N from the 4 validation drops per goal over 20 hops, '
        '100 sampled + 100 perturbed by +-1 tolerance. task_err = |fixed point - goal| in tolerance units, fixed point = '
        'mean apex of hops 10-20 per drop (NaN = fall). Lists are [iter][unit], units = robot:seed (6 robots x 3 seeds). '
        'IN = 6 TRAIN_GOALS; HULL / OUT = 16 goals of the validation seed 301 inside / outside the IN hull (+0.10 margin, '
        'normalized goal box). ours / ours+DR / PPO+DR: the frozen PF hardware stage (6 iterations x 6 trials) re-run with '
        'per-iteration checkpoints; RMA zero-shot only (the stage adapts MLPs only); FADA: its own LoRA adaptation, '
        'it4 = 24 trials. No plan / reference trajectory in this domain: the reference flow is V^sim_{pi_0}.')
    json.dump(summary, open(os.path.join(HERE, 'summary.json'), 'w'), indent=1)
    json.dump(extra, open(os.path.join(HERE, 'summary_extra.json'), 'w'), indent=1)
    plot_curves(plt, summary, extra, 'field_err_ci', 'normalized mean |dV_i|  (tolerance units)', 'fig_field_err.png',
                ref=None, logy=True)
    plot_curves(plt, summary, extra, 'task_err_ci', 'fixed-point error |x* - g|  (tolerance units)', 'fig_task_err.png',
                ref=1.0, logy=True)
    if all('goal_err_ci' in e for e in extra['starts'].values()):
        plot_curves(plt, summary, extra, 'goal_err_ci', 'mean |F(x) - g| on the tube  (tolerance units)',
                    'fig_goalflow_err.png', ref=None, logy=True)
    return summary, extra


def plot_curves(plt, summary, extra, key, ylabel, fname, ref=None, logy=False):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, g in zip(axs, SETS):
        for st, ent in summary['starts'].items():
            it = np.array(ent['iters'])
            ci = np.array(extra['starts'][st][key][g])
            if len(it) > 2:
                ax.fill_between(it, ci[:, 1], ci[:, 2], color=COL[st], alpha=0.15, lw=0)
                ax.plot(it, ci[:, 0], '-', color=COL[st], lw=2, marker=MARK[st], ms=6, label=LABEL[st])
            else:
                off = {'RMA': -0.12, 'FADA': 0.12}[st]
                ax.errorbar(it + off, ci[:, 0], yerr=[ci[:, 0] - ci[:, 1], ci[:, 2] - ci[:, 0]], color=COL[st],
                            marker=MARK[st], ms=7, ls='--' if len(it) > 1 else 'none', lw=1.5, capsize=3, label=LABEL[st])
        if ref is not None:
            ax.axhline(ref, color='#888888', lw=1, ls=':')
            ax.text(6.1, ref, 'tolerance', color='#52514e', fontsize=8, va='bottom', ha='right')
        ax.set_title({'IN': 'IN (6 training goals)', 'HULL': f'HULL ({len(extra["goal_sets"]["HULL"])} val goals)',
                      'OUT': f'OUT ({len(extra["goal_sets"]["OUT"])} val goals)'}[g], fontsize=10)
        ax.set_xlabel('hardware-stage iteration (6 real trials each)')
        ax.set_xticks(range(7))
        if logy:
            ax.set_yscale('log')
        ax.grid(alpha=0.25, lw=0.5)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel(ylabel)
    axs[-1].legend(fontsize=8, frameon=False, loc='best')
    fig.suptitle('pogo, mean over 6 robots x 3 seeds, 95 % bootstrap CI', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, fname), dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------- phase portraits
def portraits(start='ours+DR', seed=0, robots=('real_mass', 'real_offset'), goal=(0.8, 0.62), its=(1, 2, 6), n=17,
              fname=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    g = np.array(goal)
    hs = np.linspace(g[1] - 0.08, g[1] + 0.08, n)
    vs = np.linspace(g[0] - 0.5, g[0] + 0.5, n)
    VV, HH = np.meshgrid(vs, hs)
    flat = np.stack([HH.ravel(), VV.ravel()], -1)
    Gl = np.repeat(g[None], len(flat), 0)
    X0, Gr = drop_apexes(g[None])

    def field(p, system):
        F = fly_hop(p, flat[:, 0], flat[:, 1], Gl, system)[0]
        o = orbit(p, X0, Gr, system, hops=N_HOPS)
        return F - flat, o

    pol0 = policy_path(start, seed, robots[0], 0)
    Vs, os_ = field(pol0, 'sim')
    cols = [('sim', 0)] + [('real', 0)] + [('real', k) for k in its]
    fig, axs = plt.subplots(len(robots), len(cols), figsize=(3.1 * len(cols), 3.0 * len(robots)), sharex=True, sharey=True)
    axs = np.atleast_2d(axs)
    vmax = 3.0
    for ri, r in enumerate(robots):
        for ci, (kind, k) in enumerate(cols):
            ax = axs[ri, ci]
            if kind == 'sim':
                V, o, title = Vs, os_, 'sim, pi_0 (reference)'
            else:
                V, o = field(policy_path(start, seed, r, k), r)
                title = f'{r[5:]}, pi_{k}' + (' (corrupted)' if k == 0 else '')
            dV = np.linalg.norm((V - Vs) / TOL, axis=1).reshape(HH.shape)
            im = ax.pcolormesh(VV, HH, dV, cmap='Oranges', vmin=0, vmax=vmax, shading='auto', alpha=0.85)
            U, W = V[:, 1].reshape(HH.shape), V[:, 0].reshape(HH.shape)
            ax.quiver(VV, HH, U, W, angles='xy', scale_units='xy', scale=1.0, width=0.004, color='#2b2b2b', alpha=0.8)
            for j in range(o.shape[1]):
                ax.plot(o[:, j, 1], o[:, j, 0], '-', color='#2a78d6', lw=0.8, alpha=0.7)
            late = o[FP_FROM:]
            if np.isfinite(late).all():
                xb = late.reshape(-1, 2).mean(0)
                ax.plot(xb[1], xb[0], 'o', color='#2a78d6', ms=7, mec='white', mew=1.2, zorder=5)
            ax.plot(g[0], g[1], '*', color='#e34948', ms=13, mec='white', mew=1, zorder=6)
            ax.add_patch(plt.Rectangle((g[0] - TOL[1], g[1] - TOL[0]), 2 * TOL[1], 2 * TOL[0], fill=False, ec='#e34948',
                                       lw=0.8, ls='--'))
            ax.set_xlim(vs[0], vs[-1])
            ax.set_ylim(hs[0], hs[-1])
            ax.set_title(title, fontsize=9)
            if ci == 0:
                ax.set_ylabel(f'{r}\napex h (m)', fontsize=9)
            if ri == len(robots) - 1:
                ax.set_xlabel('apex vx (m/s)', fontsize=9)
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.01)
    cb.set_label('|V - V_sim,pi0| (tolerance units)', fontsize=8)
    fig.suptitle(f'pogo apex return map, {start} (seed {seed}), goal v*={g[0]}, h*={g[1]}: arrows = F(x) - x (to the next '
                 f'apex); blue = orbits from the 4 val drops, dot = their fixed point; star / dashed box = goal / tolerance',
                 fontsize=9)
    fname = fname or os.path.join(HERE, f'fig_portraits_{start.replace("+", "")}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('compute', 'portraits', 'aggregate', 'all'))
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--starts', nargs='+', default=list(STARTS))
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    ap.add_argument('--portrait-start', default='ours+DR')
    ap.add_argument('--portrait-seed', type=int, default=0)
    ap.add_argument('--portrait-goal', type=float, nargs=2, default=[0.8, 0.62])
    ap.add_argument('--portrait-robots', nargs='+', default=['real_mass', 'real_offset'])
    a = ap.parse_args()
    if a.cmd in ('compute', 'all'):
        jobs = [(st, s) for st in a.starts for s in a.seeds]
        if a.jobs > 1:
            from multiprocessing import Pool
            with Pool(min(a.jobs, 8)) as P:
                for p in P.imap_unordered(compute_job, jobs):
                    print('done', p, flush=True)
        else:
            for j in jobs:
                print('done', compute_job(j), flush=True)
    if a.cmd in ('aggregate', 'all'):
        aggregate()
    if a.cmd in ('portraits', 'all'):
        print(portraits(a.portrait_start, a.portrait_seed, tuple(a.portrait_robots), tuple(a.portrait_goal)))


if __name__ == '__main__':
    main()
