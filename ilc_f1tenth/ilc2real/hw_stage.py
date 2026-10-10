#!/usr/bin/env python3
"""
hw_stage.py --policy NPZ --out DIR: the hardware stage of ILC2Real on the F1TENTH (deploy.py's analog) -- few-shot
ILC of the sim network on each "real" car (mb_car's multi-body cars), the structure from the nominal model, every
lap used, then the multi-lap evaluation. Per car, from the same starting network:

  each iteration (--iters; one trial of --trial-laps (default 2) chained laps at each of --goals: 2 x 6 x 2 = 24 laps)
    1. the car's current network flies each goal's plan for one lap from the plan's start (--train-starts S: from a
       random arc length with a start perturbation of relative size S instead), measured with sensing noise
    2. per lap the closed-loop GN ILC step from the NOMINAL model's Jacobians at the MEASURED states (ilc_core),
       the network's own feedback at the measured observations, the measured tracking error; rms capped (--cap)
    3. targets mu_t + step_t at the lap's observations; the network regresses onto every lap so far (weights
       halving with age), anchored to its previous outputs on nominal-sim states of the goals (trust region,
       weight --anchor-w x N0 / (N0 + laps))
  then the evaluation on the car (--eval-plans): --eval-runs runs per plan, each --eval-laps chained laps (no reset)
  from a start spread over the lap with a perturbation (reserved seeds from --eval-seed); success = no spin-out /
  departure, RMS lateral error <= 0.10 m and RMS sideslip error <= 8 deg over all laps.
--iters 0: zero-shot evaluation only. --policy lqr: the plans' LQR (no network; evaluation only).
-> DIR/summary.json (per car: trials, per-plan eval metrics), DIR/<car>_policy.npz.
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--policy', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--cars', nargs='+', default=['real_nom', 'real_mass', 'real_mu', 'real_act', 'real_lag'])
ap.add_argument('--goals', nargs='+', default=[f'mocap_{t}_b{b}' for t in ('square2fast', 'figfast') for b in (10, 18, 25)])
ap.add_argument('--eval-plans', nargs='+', default=None)
ap.add_argument('--anchor-plans', nargs='+', default=[f'mocap_{t}_b{b}' for t in ('square2fast', 'figfast') for b in (0, 10, 18, 25)],
                help='the trust region holds the network on nominal-sim states of these plans (the sim stage\'s bank)')
ap.add_argument('--rollback', type=float, default=0.7, help='a car whose laps this iteration progress less than this '
                'fraction of the last kept iteration\'s (a new spin-out) is rolled back to its previous network and its '
                'step cap halved (0: off)')
ap.add_argument('--crash-rollback', type=int, default=1, help='a car that crashes (wall / spin) on a goal it flew '
                'cleanly in the last kept iteration is rolled back to that network and its step cap halved (1: on)')
ap.add_argument('--max-crashes', type=int, default=0, help='crash budget per car: after this many crashed trials the '
                'car stops adapting and keeps its last kept network (0: no budget)')
ap.add_argument('--record', default='', help='save every trial lap (true state, plan index, actions) to this npz '
                '(record_car.py: the hardware-stage videos)')
ap.add_argument('--iters', type=int, default=4)
ap.add_argument('--beta', type=float, default=0.5)
ap.add_argument('--delta', type=float, default=0.05)
ap.add_argument('--cap', type=float, default=0.05)
ap.add_argument('--jac', default='closed', choices=('closed', 'open', 'flip'))
ap.add_argument('--objective', default='task', choices=('task', 'plan'),
                help='task: the ILC asks only for the path (e_y) and the pace (|V| against the plan\'s) -- the car may '
                     'grip or drift, whichever it does better; plan: track the whole plan (sideslip, yaw rate, ...)')
ap.add_argument('--err-scale', type=float, nargs=6, default=None, metavar=('EY', 'EPSI', 'VX', 'VY', 'R', 'V'),
                help='explicit ILC error-row scales (overrides --objective)')
ap.add_argument('--train-starts', type=float, default=0.0)
ap.add_argument('--goal-cycle', type=int, default=0, help='1: each iteration flies ONE trial, cycling through --goals '
                '(budget = iters x trial laps; an update after every trial: the small-budget lap ablation, 2..10 laps)')
ap.add_argument('--pace-local', type=float, default=0.0, help='a LOCAL pace target: 1 - slack x (the plan\'s lateral '
                'acceleration V^2 |kappa| / its maximum): slower only where the plan corners hardest (the friction-limited '
                'sections), full pace on the straights; slack 0.1 = 90 %% at the peak')
ap.add_argument('--pace-target', type=float, default=1.0, help='the task objective asks for this fraction of the plan\'s '
                'speed (|V| - p |V_plan|; 1: the plan\'s pace). < 1 (e.g. 0.95, inside the >= 90 %% success criterion) lets the '
                'stage trade speed for the path at a friction limit the nominal model does not know')
ap.add_argument('--trial-laps', type=int, default=2, help='laps per trial, chained without a reset (2: the second lap '
                'starts where the first ended, so the ILC sees the lap-to-lap behaviour the multi-lap evaluation tests); '
                'each trial costs that many real laps')
ap.add_argument('--anchor-w', type=float, default=1.0)
ap.add_argument('--anchor-n0', type=float, default=6.0)
ap.add_argument('--age-decay', type=float, default=0.5)
ap.add_argument('--steps', type=int, default=800)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--fallback-suffix', default='', help='goal fallback (the quadruped\'s stall guard for unreachable goals): '
                'a goal whose laps spin out --fallback-after times in a row switches, for that car, to its variant with '
                'this suffix (e.g. _mu80: the same drift planned for 0.8x friction) -- triggered by outcomes only, no '
                'parameter estimation; the evaluation then also uses the variants on that car (summary eval_adaptive)')
ap.add_argument('--fallback-after', type=int, default=2)
ap.add_argument('--eval-runs', type=int, default=4)
ap.add_argument('--eval-laps', type=float, default=5.0)
ap.add_argument('--eval-seed', type=int, default=701)
ap.add_argument('--seed', type=int, default=9001)
ap.add_argument('--gpu', default=None)
a = ap.parse_args()
if a.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax                                       # noqa: E402
import jax.numpy as jnp                          # noqa: E402
import numpy as np                               # noqa: E402
import optax                                     # noqa: E402

import bank                                      # noqa: E402
import ilc_core                                  # noqa: E402
import real_car as rc                            # noqa: E402
import sim_jax as sj                             # noqa: E402

DEFAULT_EVAL = [f'mocap_{t}_b{b}' for t in ('square2fast', 'figfast') for b in (25, 21, 14, 0)]


def load_policy(path):
    d = np.load(path)
    n = len([k for k in d.files if k.startswith('W')])
    pol = [(np.asarray(d[f'W{i}']), np.asarray(d[f'b{i}'])) for i in range(n)]
    ad = os.path.join(os.path.dirname(path), 'rma_adapt.npz')
    if 'mode' in d.files and str(d['mode']) == 'rma' and os.path.exists(ad):
        e = np.load(ad)
        m = len([k for k in e.files if k.startswith('W')])
        return dict(actor=pol, adapt=[(np.asarray(e[f'W{i}']), np.asarray(e[f'b{i}'])) for i in range(m)],
                    hist=int(e['hist']))
    return pol


def evaluate(policies, plans_by_name, eval_names, seed, laps, runs):
    """policies: {car: params (numpy) or None (LQR)} -> {car: {plan: metrics}}."""
    jobs, keys = [], []
    for car, pol in policies.items():
        for name in eval_names:
            pl = plans_by_name[name]
            rs = np.random.default_rng(seed + sum(map(ord, name)))
            for k in range(runs):
                s0 = (k / runs) * float(pl['length'])
                pert = rs.uniform(-0.5, 0.5, 5)
                jobs.append(dict(car_name=car, plan=pl, policy=pol, laps=laps, s0=s0, pert=pert,
                                 seed=seed * 7 + k))
                keys.append((car, name))
    res = rc.run_many(jobs)
    out = {}
    for (car, name), r, j in zip(keys, res, jobs):
        m = rc.lap_metrics(r, j['plan'])
        out.setdefault(car, {}).setdefault(name, []).append(m)
    summ = {}
    for car, d in out.items():
        summ[car] = {}
        for name, ms in d.items():
            ok = [m for m in ms if not m['failed']]
            summ[car][name] = dict(
                success=float(np.mean([m['success'] for m in ms])), fail=float(np.mean([m['failed'] for m in ms])),
                wall=float(np.mean([m.get('crash') == 'wall' for m in ms])),          # stopped at the room's envelope
                peak_ey=float(np.max([m.get('peak_ey', 0.0) for m in ms])),
                laps=float(np.mean([m['laps'] for m in ms])),
                rms_ey=float(np.mean([m['rms_ey'] for m in ok])) if ok else None,
                rms_dbeta=float(np.mean([m['rms_dbeta'] for m in ok])) if ok else None,
                pace=float(np.mean([m['pace'] for m in ok])) if ok else None,
                mean_abs_beta=float(np.mean([m['mean_abs_beta'] for m in ok])) if ok else None,
                last_lap_ey=float(np.mean([m['per_lap_ey'][-1] for m in ok if m['per_lap_ey']])) if ok else None,
                first_lap_ey=float(np.mean([m['per_lap_ey'][0] for m in ok if m['per_lap_ey']])) if ok else None)
    return summ


ERRS = np.array(a.err_scale) if a.err_scale else (sj.TASK_SCALE if a.objective == 'task' else sj.ERR_SCALE)


def main():
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)
    eval_names = a.eval_plans or DEFAULT_EVAL
    fb = lambda n: n + a.fallback_suffix
    have = lambda n: os.path.exists(os.path.join(bank.PLAN_DIR, n + '.npz'))
    extra = [fb(n) for n in a.goals + eval_names + a.anchor_plans] if a.fallback_suffix else []
    extra = [n for n in extra if have(n)]
    names = list(dict.fromkeys(a.goals + eval_names + a.anchor_plans + extra))
    plans = bank.load_bank([n for n in names if os.path.exists(os.path.join(bank.PLAN_DIR, n + '.npz'))])
    by_name = {p['name']: p for p in plans}
    eval_names = [n for n in eval_names if n in by_name]
    eval_all = list(dict.fromkeys(eval_names + [fb(n) for n in eval_names if a.fallback_suffix and fb(n) in by_name]))
    t0 = time.time()
    summary = dict(args=vars(a), cars={})
    if a.policy == 'lqr':
        summary['eval'] = evaluate({c: None for c in a.cars}, by_name, eval_names, a.eval_seed, a.eval_laps, a.eval_runs)
        json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
        return
    base = load_policy(a.policy)
    if isinstance(base, dict):                       # RMA: zero-shot evaluation only
        assert a.iters == 0
        summary['eval'] = evaluate({c: base for c in a.cars}, by_name, eval_names, a.eval_seed, a.eval_laps, a.eval_runs)
        json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
        return
    gn_names = list(dict.fromkeys(a.goals + [fb(g) for g in a.goals if a.fallback_suffix and fb(g) in by_name]))
    goal_plans = [by_name[g] for g in gn_names]
    P = sj.Plans(goal_plans, gn_names)
    T1 = int(np.ceil(1.05 * max(P.lap_time) / sj.DT))                 # one lap (the trust-region pool)
    T = int(np.ceil((a.trial_laps + 0.05) * max(P.lap_time) / sj.DT))  # one trial
    policy_fn = lambda ps, o: sj.mlp(ps, o)
    gns = {}

    def gn_for(cap):
        if cap not in gns:
            gns[cap] = ilc_core.make_gn(P, T, policy_fn, a.beta, a.delta, cap, a.jac, err_scale=ERRS)
        return gns[cap]

    # trust-region pool: nominal-sim states of the bank's plans flown by the starting network
    anc = [by_name[n] for n in list(dict.fromkeys(a.anchor_plans + [fb(n) for n in a.anchor_plans if a.fallback_suffix]))
           if n in by_name]
    PA = sj.Plans(anc, [p['name'] for p in anc])
    n_pool = 8 * len(anc)
    pid = np.repeat(np.arange(len(anc)), 8)
    xs, ids = [], []
    rs = np.random.default_rng(a.seed)
    for k, q in enumerate(pid):
        x0, i0 = sj.start_states(PA, jax.random.PRNGKey(k), [q], [rs.uniform() * float(PA.L[q])], pert=0.5)
        xs.append(x0[0]); ids.append(i0[0])
    jb = [(jnp.array(W), jnp.array(b)) for W, b in base]
    pool = sj.rollout(PA, lambda o: policy_fn(jb, o), T1, jnp.stack(xs), jnp.stack(ids), jnp.array(pid),
                      sj.nominal_params(n_pool))
    ok = ~np.asarray(pool['failed'])
    O_anchor = np.asarray(pool['obs'])[ok]

    opt = optax.adam(a.lr)

    @jax.jit
    def fit_step(ps, st, o, y, w, oa, ya, wa):
        def loss(ps):
            l = jnp.sum(w[:, None] * (policy_fn(ps, o) - y) ** 2) / (jnp.sum(w) + 1e-9)
            return l + wa * jnp.mean((policy_fn(ps, oa) - ya) ** 2)
        l, g = jax.value_and_grad(loss)(ps)
        u, st = opt.update(g, st, ps)
        return optax.apply_updates(ps, u), st, l

    params = {c: [(jnp.array(W), jnp.array(b)) for W, b in base] for c in a.cars}
    prev_params = dict(params)
    cap = {c: a.cap for c in a.cars}
    last_prog = {c: None for c in a.cars}
    rollbacks = {c: [] for c in a.cars}
    goal_now = {c: {g: g for g in a.goals} for c in a.cars}
    fails_row = {c: {g: 0 for g in a.goals} for c in a.cars}
    switched = {c: {} for c in a.cars}
    data = {c: [] for c in a.cars}
    real_laps = {c: 0.0 for c in a.cars}
    trials = {c: [] for c in a.cars}
    recs = []                                         # --record: every trial lap
    crashes = {c: [] for c in a.cars}                 # every crashed training trial: it, goal, reason, peak |e_y|
    last_crashed = {c: set() for c in a.cars}         # goals that crashed in the last kept iteration
    frozen = {c: False for c in a.cars}               # crash budget used up: no more trials, last kept network
    for it in range(a.iters):
        goals_it = [a.goals[it % len(a.goals)]] if a.goal_cycle else a.goals
        jobs = []
        for c in [c for c in a.cars if not frozen[c]]:
            pol = [(np.asarray(W), np.asarray(b)) for W, b in params[c]]
            for gi, g in enumerate(goals_it):
                pl = by_name[goal_now[c][g]]
                if a.train_starts > 0:
                    s0 = rs.uniform() * float(pl['length'])
                    pert = rs.uniform(-1, 1, 5) * a.train_starts
                else:
                    s0, pert = 0.0, rs.uniform(-0.1, 0.1, 5)
                jobs.append(dict(car_name=c, plan=pl, policy=pol, laps=a.trial_laps + 0.05, s0=s0, pert=pert,
                                 seed=a.seed + 100 * it + gi))
        res = rc.run_many(jobs)
        for c in [c for c in a.cars if not frozen[c]]:
            rr = [r for r, j in zip(res, jobs) if j['car_name'] == c]
            crashed = {g for g, r in zip(goals_it, rr) if r['failed']}
            if a.record:
                for g, r in zip(goals_it, rr):
                    recs.append((dict(car=c, it=it, plan=goal_now[c][g], failed=bool(r['failed']), crash=r.get('crash', ''),
                                      laps=float(r['laps'])), r))
            for g, r in zip(goals_it, rr):
                if r['failed']:
                    crashes[c].append(dict(it=it, goal=goal_now[c][g], reason=r.get('crash', ''),
                                           peak_ey=r.get('peak_ey'), laps=r['laps']))
            pid_c = np.array([gn_names.index(j['plan']['name']) for j in jobs if j['car_name'] == c])
            for gi, (g, r) in enumerate(zip(goals_it, rr)):          # goal fallback on repeated spin-outs
                fails_row[c][g] = fails_row[c][g] + 1 if r['failed'] else 0
                if (a.fallback_suffix and goal_now[c][g] == g and fails_row[c][g] >= a.fallback_after
                        and fb(g) in by_name):
                    goal_now[c][g] = fb(g)
                    switched[c][g] = dict(it=it, to=fb(g))
                    print(json.dumps(dict(it=it, car=c, fallback=g, to=fb(g))), flush=True)
            prog = float(np.mean([max(r['laps'], 0.0) for r in rr]))
            for r in rr:
                real_laps[c] += max(r['laps'], 0.0)
            new_crash = sorted(crashed - last_crashed[c]) if a.crash_rollback and last_prog[c] is not None else []
            if a.max_crashes and len(crashes[c]) >= a.max_crashes:
                frozen[c] = True                           # crash budget used up: keep the last kept network
                if new_crash or (last_prog[c] is not None and prog < a.rollback * last_prog[c]):
                    params[c] = prev_params[c]
                print(json.dumps(dict(it=it, car=c, frozen=True, crashes=len(crashes[c]))), flush=True)
                continue
            if (a.rollback > 0 and last_prog[c] is not None and prog < a.rollback * last_prog[c]) or new_crash:
                params[c] = prev_params[c]                 # less progress or a new crash: back to the last kept network
                cap[c] *= 0.5
                rollbacks[c].append(dict(it=it, prog=prog, last=last_prog[c], cap=cap[c], new_crash=new_crash))
                print(json.dumps(dict(it=it, car=c, rollback=True, prog=prog, last=last_prog[c], cap=cap[c],
                                      new_crash=new_crash)), flush=True)
                continue
            last_prog[c] = prog
            last_crashed[c] = crashed
            prev_params[c] = params[c]
            xs, idx, acts, err, valid = ilc_core.pad_runs(rr, T)
            if a.pace_target != 1.0 or a.pace_local > 0:   # the pace row against the target fraction of the plan's speed
                for k in range(len(rr)):
                    pl_k = by_name[gn_names[pid_c[k]]]
                    zz = np.asarray(pl_k['z'])
                    vp = np.hypot(zz[:, 2], zz[:, 3])
                    frac = np.full(len(zz), a.pace_target)
                    if a.pace_local > 0:
                        alat = vp ** 2 * np.abs(np.asarray(pl_k['kappa_s']))
                        frac = frac - a.pace_local * alat / max(alat.max(), 1e-6)
                    j = idx[k] % len(zz)
                    err[k, :, 5] += (1.0 - frac[j]) * vp[j]
            step, J, Jp = gn_for(cap[c])(params[c], jnp.array(xs), jnp.array(idx), jnp.array(acts), jnp.array(pid_c),
                             jnp.array(valid), jnp.array(err))
            step, J, Jp = np.asarray(step), np.asarray(J), np.asarray(Jp)
            bad = ~(np.isfinite(step).all(axis=(1, 2)) & np.isfinite(J))
            valid[bad] = False                          # a non-finite step (an odd measured state) gives no targets
            step = np.nan_to_num(step)
            J, Jp = np.nan_to_num(J), np.nan_to_num(Jp)
            for k, r in enumerate(rr):
                m = min(len(r['obs']), T)
                v = valid[k, :m]
                data[c].append(dict(obs=r['obs'][:m][v], y=(r['mu'][:m] + step[k, :m])[v], it=it))
                mm = rc.lap_metrics(r, by_name[gn_names[pid_c[k]]])
                trials[c].append(dict(it=it, goal=gn_names[pid_c[k]], failed=mm.get('failed'), rms_ey=mm.get('rms_ey'),
                                      rms_dbeta=mm.get('rms_dbeta'), J=float(J[k]), J_pred=float(Jp[k]),
                                      crash=r.get('crash', ''), peak_ey=r.get('peak_ey')))
            # fit
            O = np.concatenate([d['obs'] for d in data[c]])
            Y = np.concatenate([d['y'] for d in data[c]])
            W = np.concatenate([np.full(len(d['obs']), a.age_decay ** (it - d['it'])) for d in data[c]])
            Ya = np.asarray(policy_fn(params[c], jnp.array(O_anchor)))
            wa = a.anchor_w * a.anchor_n0 / (a.anchor_n0 + len(data[c]))
            ps, st = params[c], opt.init(params[c])
            O_, Y_, W_, Oa_, Ya_ = map(jnp.array, (O, Y, W, O_anchor, Ya))
            for s in range(a.steps):
                b = jnp.array(rs.integers(0, len(O), min(2048, len(O))))
                ba = jnp.array(rs.integers(0, len(O_anchor), min(2048, len(O_anchor))))
                ps, st, l = fit_step(ps, st, O_[b], Y_[b], W_[b], Oa_[ba], Ya_[ba], wa)
            if np.isfinite(float(l)) and all(np.isfinite(np.asarray(W)).all() for W, b in ps):
                params[c] = ps                           # a non-finite fit is discarded
            last = [t for t in trials[c] if t['it'] == it]
            print(json.dumps(dict(it=it, car=c, loss=float(l), fails=sum(bool(t['failed']) for t in last),
                                  ey=float(np.nanmean([t['rms_ey'] or np.nan for t in last])),
                                  dbeta=float(np.nanmean([t['rms_dbeta'] or np.nan for t in last])),
                                  J=float(np.mean([t['J'] for t in last])), Jp=float(np.mean([t['J_pred'] for t in last])),
                                  t=round(time.time() - t0, 1))), flush=True)
    pols = {c: [(np.asarray(W), np.asarray(b)) for W, b in params[c]] for c in a.cars}
    for c in a.cars:
        np.savez(os.path.join(a.out, f'{c}_policy.npz'), **{f'W{i}': W for i, (W, b) in enumerate(pols[c])},
                 **{f'b{i}': b for i, (W, b) in enumerate(pols[c])})
    summary['eval'] = evaluate(pols, by_name, eval_all, a.eval_seed, a.eval_laps, a.eval_runs)
    summary['switched'] = switched
    if a.fallback_suffix:
        summary['eval_adaptive'] = {c: {n: summary['eval'][c][fb(n) if switched[c] and fb(n) in summary['eval'][c] else n]
                                        for n in eval_names} for c in a.cars}
    summary['trials'] = trials
    summary['real_laps'] = real_laps
    summary['rollbacks'] = rollbacks
    summary['crashes'] = crashes
    if a.record and recs:
        import record_car
        record_car.save(a.record, recs, [by_name[n] for n in dict.fromkeys(m['plan'] for m, _ in recs)])
    summary['frozen'] = frozen
    summary['safety_ey'] = float(os.environ.get('F1T_SAFETY_EY') or 0.6)
    summary['seconds'] = time.time() - t0
    json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    for c in a.cars:
        print(c, json.dumps({k: (round(v['success'], 2), v['rms_ey'] and round(v['rms_ey'], 3),
                                 v['rms_dbeta'] and round(v['rms_dbeta'], 1)) for k, v in summary['eval'][c].items()}))


if __name__ == '__main__':
    main()
