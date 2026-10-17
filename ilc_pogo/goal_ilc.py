#!/usr/bin/env python3
"""
goal_ilc.py: the per-goal classical ILC baseline (the car's track_ilc.py / the quadruped's per-goal JumpILC analog).
On each real pogo stick and each trained goal separately, from scratch: the Raibert controller plus a time-indexed
feedforward ff_t (t = 0 .. T-1, the whole 10 s evaluation run, so the feedforward covers it) learned by the same
closed-loop Gauss-Newton ILC step as ours -- the NOMINAL model's Jacobians at the MEASURED states, the Raibert
controller's own feedback in the closed loop (pogo_jax.lane_jacobians(feedback='expert'), pogo_jax.gn_lane). One
trial per goal per iteration, always from the same clean drop at the goal's apex (pert 0).
After each budget (trials per goal) one arm dir DIR_PREFIX_b{N}: the evaluation flies the TRAINED goals only (a
feedforward exists only for them), 4 perturbed drops each (common.EVAL_SEED, ff time-indexed from t = 0), 10 s.
Also stored: the clean-start evaluation (the drop it was trained from).
    python goal_ilc.py --out runs/hw/goalilc --budgets 1 2 4 8 --gpu 3
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='runs/hw/goalilc', help='arm dirs are <out>_b<N>')
ap.add_argument('--robots', nargs='+', default=None)
ap.add_argument('--goals', type=int, default=None, help='use only the first N training goals (smoke tests)')
ap.add_argument('--budgets', type=int, nargs='+', default=[1, 2, 4, 8])
ap.add_argument('--seconds', type=float, default=10.0, help='trial = evaluation length')
ap.add_argument('--beta', type=float, default=0.5)
ap.add_argument('--delta', type=float, default=0.02)
ap.add_argument('--cap', type=float, default=0.1)
ap.add_argument('--eval-runs', type=int, default=4)
ap.add_argument('--eval-seed', type=int, default=None, help='drop seed (default: common.EVAL_SEED of POGO_EVAL_SET)')
ap.add_argument('--gpu', default=None)
a = ap.parse_args()
if a.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax                                       # noqa: E402
import jax.numpy as jnp                          # noqa: E402
import numpy as np                               # noqa: E402

import common as C                               # noqa: E402
import pogo_jax as pj                            # noqa: E402
import real_pogo as rp                           # noqa: E402

ROBOTS = a.robots or rp.DEFAULT_ROBOTS
GOALS = C.TRAIN_GOALS[:a.goals] if a.goals else C.TRAIN_GOALS
T = int(round(a.seconds / C.DT))


@jax.jit
def gn(xs, acts, g, err, failed):
    """Per lane (robot, goal): the closed-loop GN step from the nominal model at the measured states."""
    p1 = {k: jnp.float32(v) for k, v in C.NOMINAL.items()}

    def lane(X, Ap, gg, E, F):
        one = pj.lane_jacobians(None, None, gg, p1, 'expert')
        A, B, Cm, K = jax.vmap(one)(X[:-1], Ap)
        return pj.gn_lane(A, B, Cm, K, E, ~F, a.beta, a.delta, a.cap)
    return jax.vmap(lane)(xs, acts, g, err, failed)


def robot_param_arrays(keys):
    """Per-run parameter arrays (runs of different robots in one batch)."""
    ps = [rp.robot_params(r) for r in keys]
    return {k: np.array([p[k] for p in ps]) for k in ps[0]}


def evaluate(ff):
    """Every robot x trained goal: eval_runs perturbed drops (common.EVAL_SEED) + the clean drop, the goal's ff from t = 0."""
    out, clean = {}, {}
    rows = [(r, gi) for r in ROBOTS for gi in range(len(GOALS))]
    rs = np.random.default_rng(C.EVAL_SEED if a.eval_seed is None else a.eval_seed)
    G = np.repeat(GOALS, a.eval_runs, 0)
    x0g = rp.start_states(G, 1.0, rs)                       # the same drops for every robot (as real_pogo.evaluate)
    keys, goals, x0, ffs = [], [], [], []
    for r, gi in rows:
        for j in range(a.eval_runs):
            keys.append((r, gi, 'pert'))
            goals.append(GOALS[gi]); x0.append(x0g[gi * a.eval_runs + j]); ffs.append(ff[(r, gi)])
        keys.append((r, gi, 'clean'))
        goals.append(GOALS[gi]); x0.append(rp.start_states(GOALS[gi:gi + 1], 0.0, rs)[0]); ffs.append(ff[(r, gi)])
    goals, x0, ffs = np.array(goals), np.array(x0), np.array(ffs)
    res = rp.simulate(None, goals, x0, robot_param_arrays([k[0] for k in keys]), T, ff=ffs)
    for i, (r, gi, kind) in enumerate(keys):
        m = C.hop_metrics(res['x'][i], goals[i], res['failed_end'][i])
        m['goal'] = goals[i].tolist()
        (out if kind == 'pert' else clean).setdefault(r, []).append(m)
    return out, clean


def main():
    t0 = time.time()
    ff = {(r, gi): np.zeros((T, 2)) for r in ROBOTS for gi in range(len(GOALS))}
    rows = [(r, gi) for r in ROBOTS for gi in range(len(GOALS))]
    goals = np.array([GOALS[gi] for r, gi in rows])
    x0 = rp.start_states(goals, 0.0, np.random.default_rng(0))
    pp = robot_param_arrays([r for r, gi in rows])
    trials = {r: 0 for r in ROBOTS}
    log = []
    done = 0
    for budget in sorted(a.budgets):
        for it in range(done, budget):
            ffs = np.array([ff[k] for k in rows])
            res = rp.simulate(None, goals, x0, pp, T, ff=ffs)
            for r, gi in rows:
                trials[r] += 1
            step, J, Jp = gn(jnp.array(res['x'], jnp.float32), jnp.array(res['a'], jnp.float32), jnp.array(goals, jnp.float32),
                             jnp.array(res['err'], jnp.float32), jnp.array(res['failed']))
            step, J, Jp = np.nan_to_num(np.asarray(step)), np.asarray(J), np.asarray(Jp)
            for n, k in enumerate(rows):
                ff[k] = ff[k] + step[n]
            rec = dict(it=it, J=float(np.mean(J)), J_pred=float(np.mean(Jp)), fails=float(np.mean(res['failed_end'])),
                       J_robot={r: float(np.mean([J[n] for n, k in enumerate(rows) if k[0] == r])) for r in ROBOTS},
                       t=round(time.time() - t0, 1))
            log.append(rec)
            print(json.dumps(rec), flush=True)
        done = budget
        ev, clean = evaluate(ff)
        d = f'{a.out}_b{budget}'
        os.makedirs(d, exist_ok=True)
        for r in ROBOTS:
            np.savez(os.path.join(d, f'{r}_ff.npz'), ff=np.array([ff[(r, gi)] for gi in range(len(GOALS))]), goals=GOALS)
        summary = dict(args=dict(vars(a), budget=budget), method='goal_ilc',
                       eval={r: rp.summarize(ev[r]) for r in ROBOTS}, runs=ev,
                       eval_clean={r: rp.summarize(clean[r]) for r in ROBOTS}, runs_clean=clean,
                       real_trials={r: budget * len(GOALS) for r in ROBOTS}, goals_eval=GOALS.tolist(),
                       eval_goals_note='trained goals (a feedforward exists only for them), not the reserved test goals',
                       trials_per_goal=budget, ilc_log=log, seconds=time.time() - t0)
        json.dump(summary, open(os.path.join(d, 'summary.json'), 'w'), indent=1)
        print('BUDGET', budget, json.dumps({r: (round(summary['eval'][r]['success'], 2),
                                                summary['eval'][r]['rms_apex'] and round(summary['eval'][r]['rms_apex'] * 100, 2),
                                                round(summary['eval_clean'][r]['success'], 2)) for r in ROBOTS}), flush=True)


if __name__ == '__main__':
    main()
