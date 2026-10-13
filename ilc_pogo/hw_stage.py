#!/usr/bin/env python3
"""
hw_stage.py --policy NPZ --out DIR: the "hardware" stage of DistILC on the pogo sticks (the car's hw_stage.py analog)
-- few-shot ILC of the sim network on each "real" pogo stick (real_pogo: the numpy simulator, idealized sensing),
the structure from the NOMINAL model, every trial used, then the multi-hop evaluation. Per robot, from the same
starting network:

  each iteration (--iters): one trial of --T periods (3 s, ~6 hops) at each of --goals from that goal's drop
    1. the robot's current network flies it, measured exactly (perfect mocap, no delay)
    2. per trial the closed-loop GN ILC step from the NOMINAL model's Jacobians at the MEASURED states, the
       network's own feedback at the measured observations, the measured task errors; a full GN step (beta 1, light
       damping delta 0.005), rms capped (--cap); defaults chosen on the validation goals (seed 301), see NOTES.md
    3. targets mu_t + step_t at the trial's observations; the network regresses onto every trial so far (weights
       --age-decay per iteration of age), optionally anchored to its previous outputs on nominal-sim states (trust
       region, weight --anchor-w x N0 / (N0 + trials); off by default)
    a new fall on a goal that did not fall in the last kept iteration, or a mean task cost above 1 / --rollback x the
    last kept one, rolls the robot back to its last kept network and halves its cap
  positivity: for every goal flown in two consecutive kept iterations, cos(G step, e_next - e) between the nominal
    model's predicted and the measured change of the (scaled) error vector
  then the evaluation on the robot: real_pogo.evaluate (8 reserved test goals x --eval-runs drops x 10 s)
--iters 0: zero-shot evaluation only. --policy expert: the Raibert controller (evaluation only).
-> DIR/summary.json ({args, eval, runs, real_trials, trials, positivity, rollbacks, goals_eval, seconds}),
   DIR/<robot>_policy.npz.
"""
import argparse
import json
import os
import sys
import time
from functools import partial

ap = argparse.ArgumentParser()
ap.add_argument('--policy', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--robots', nargs='+', default=None)
ap.add_argument('--goals', type=float, nargs='+', default=None, help='flat v h pairs (default common.TRAIN_GOALS)')
ap.add_argument('--iters', type=int, default=4)
ap.add_argument('--T', type=int, default=300)
ap.add_argument('--beta', type=float, default=1.0)
ap.add_argument('--delta', type=float, default=0.005)
ap.add_argument('--cap', type=float, default=0.1)
ap.add_argument('--jac', default='closed', choices=('closed', 'open', 'flip'))
ap.add_argument('--anchor-w', type=float, default=0.0, help='trust region (0: off, the default '
                'chosen on the validation goals: a full GN step without the anchor converges fastest)')
ap.add_argument('--anchor-n0', type=float, default=6.0)
ap.add_argument('--age-decay', type=float, default=0.5)
ap.add_argument('--steps', type=int, default=800)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--rollback', type=float, default=0.7, help='0: off')
ap.add_argument('--crash-rollback', type=int, default=1)
ap.add_argument('--train-pert', type=float, default=0.0, help='drop perturbation of the training trials (0: the goal\'s '
                'clean drop, the same start every iteration)')
ap.add_argument('--eval-runs', type=int, default=4)
ap.add_argument('--eval-seconds', type=float, default=10.0)
ap.add_argument('--eval-seed', type=int, default=701)
ap.add_argument('--eval-goals', type=int, default=8)
ap.add_argument('--seed', type=int, default=9001)
ap.add_argument('--fit-samples', type=int, default=0, help='GPS-style fitted local dynamics (the baseline of '
                'Proposition "identifiability" in src/paper/theory): per goal and iteration, the noise-free trial plus N-1 '
                'trials with i.i.d. action noise (--fit-noise) from the same drop; per period a time-varying linear model '
                'of the deviations (x, z, vx, vz, xf; actions) is fitted by ridge regression toward the nominal Jacobian '
                '(prior worth --fit-prior samples) and replaces the nominal A_t, B_t in the GN step. Every rollout counts '
                'as a real trial. 0 / 1: off (ours)')
ap.add_argument('--fit-noise', type=float, default=0.05)
ap.add_argument('--fit-prior', type=float, default=1e-3)
ap.add_argument('--record', default='', help='save every training trial (x, goal, robot, it) to this npz')
ap.add_argument('--gpu', default=None)
a = ap.parse_args()
if a.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                               # noqa: E402

import common as C                               # noqa: E402
import policies                                  # noqa: E402
import real_pogo as rp                           # noqa: E402

ROBOTS = a.robots or rp.DEFAULT_ROBOTS
GOALS = np.array(a.goals).reshape(-1, 2) if a.goals else C.TRAIN_GOALS
EVAL_GOALS = C.test_goals(a.eval_goals)


class GroupPolicy:
    """One batch, several policies: rows[k] flown by pols[k]."""
    def __init__(self, pols, rows):
        self.pols, self.rows = [policies.as_policy(p) for p in pols], rows

    def reset(self, n):
        for p, r in zip(self.pols, self.rows):
            p.reset(len(r))

    def __call__(self, o, x, g):
        out = np.zeros((len(o), 2))
        for p, r in zip(self.pols, self.rows):
            out[r] = p(o[r], x[r], g[r])
        return out


def batch_params(names):
    ps = [rp.robot_params(n) for n in names]
    return {k: np.array([p[k] for p in ps]) for k in ps[0]}


def evaluate(pols):
    res = rp.evaluate(pols, EVAL_GOALS, list(pols), runs=a.eval_runs, seconds=a.eval_seconds, seed=a.eval_seed)
    return {r: rp.summarize(ms) for r, ms in res.items()}, res


def write(summary):
    json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)


def main():
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)
    t0 = time.time()
    summary = dict(args=vars(a), goals_eval=EVAL_GOALS.tolist(), goals_train=GOALS.tolist())
    base = policies.load_any(a.policy)
    if a.iters == 0 or not isinstance(base, policies.MLPPolicy):
        assert a.iters == 0, 'only MLP policies can be adapted'
        ev, runs = evaluate({r: base for r in ROBOTS})
        summary.update(eval=ev, runs=runs, real_trials={r: 0 for r in ROBOTS}, seconds=time.time() - t0)
        write(summary)
        print(json.dumps({r: (round(v['success'], 3), v['rms_apex'] and round(v['rms_apex'] * 100, 2)) for r, v in ev.items()}))
        return

    import jax
    import jax.numpy as jnp
    import optax
    import pogo_jax as pj

    fb = 'none' if a.jac == 'open' else 'policy'

    @jax.jit
    def parts(params, xs, acts, g):
        pn = {k: v[0] for k, v in pj.nominal_params(1).items()}

        def lane(X, Ap, gg):
            return jax.vmap(pj.lane_jacobians(pj.mlp, params, gg, pn, fb))(X[:-1], Ap)
        return jax.vmap(lane)(xs, acts, g)

    @partial(jax.jit, static_argnames=('cap',))
    def gn_parts(A, B, Cm, K, E, valid, cap):
        return jax.vmap(lambda A, B, Cm, K, E, V: pj.gn_lane(A, B, Cm, K, E, V, a.beta, a.delta, cap,
                                                              flip=a.jac == 'flip', vectors=True))(A, B, Cm, K, E, valid)

    CI = np.array([0, 1, 2, 3, 4])                  # the continuous coordinates the fitted model covers

    def fitted_AB(A, B, X0, U0, Xs, Us, ok):
        """Ridge fit per period of the deviations about the noise-free trial (X0, U0) from the noisy samples (Xs, Us),
        toward the nominal (A, B); ok (S, T) masks samples after a fall."""
        A, B = np.array(A), np.array(B)
        w = a.fit_prior * a.fit_noise ** 2
        for t in range(A.shape[0]):
            m = ok[:, t]
            if m.sum() == 0:
                continue
            Z = np.concatenate([(Xs[m, t] - X0[t])[:, CI], Us[m, t] - U0[t]], 1)          # (S, 7)
            Y = (Xs[m, t + 1] - X0[t + 1])[:, CI]                                        # (S, 5)
            P = np.concatenate([A[t][np.ix_(CI, CI)], B[t][CI]], 1)                       # (5, 7) prior
            Th = (Y.T @ Z + w * P) @ np.linalg.inv(Z.T @ Z + w * np.eye(Z.shape[1]))
            A[t][np.ix_(CI, CI)], B[t][CI] = Th[:, :len(CI)], Th[:, len(CI):]
        return A, B

    @partial(jax.jit, static_argnames=('cap',))
    def gn(params, xs, acts, g, E, valid, cap):
        pn = {k: v[0] for k, v in pj.nominal_params(1).items()}

        def lane(X, Ap, gg, Ee, V):
            one = pj.lane_jacobians(pj.mlp, params, gg, pn, fb)
            A, B, Cm, K = jax.vmap(one)(X[:-1], Ap)
            return pj.gn_lane(A, B, Cm, K, Ee, V, a.beta, a.delta, cap, flip=a.jac == 'flip', vectors=True)
        return jax.vmap(lane)(xs, acts, g, E, valid)

    # trust-region pool: nominal-sim states of the starting network over random goals
    rs = np.random.default_rng(a.seed)
    jb = [(jnp.array(W), jnp.array(b)) for W, b in base.params]
    Ga = C.GOAL_LO + (C.GOAL_HI - C.GOAL_LO) * rs.uniform(size=(64, 2))
    pool = pj.rollout(lambda o: pj.mlp(jb, o), a.T, pj.start_states(jax.random.PRNGKey(a.seed), Ga, 0.5), jnp.asarray(Ga),
                      pj.nominal_params(64))
    O_anchor = np.asarray(pool['obs'])[~np.asarray(pool['failed'])]

    opt = optax.adam(a.lr)

    @jax.jit
    def fit_step(ps, st, o, y, w, oa, ya, wa):
        def loss(ps):
            l = jnp.sum(w[:, None] * (pj.mlp(ps, o) - y) ** 2) / (jnp.sum(w) + 1e-9)
            return l + wa * jnp.mean((pj.mlp(ps, oa) - ya) ** 2)
        l, g = jax.value_and_grad(loss)(ps)
        u, st = opt.update(g, st, ps)
        return optax.apply_updates(ps, u), st, l

    params = {r: [(jnp.array(W), jnp.array(b)) for W, b in base.params] for r in ROBOTS}
    prev = dict(params)
    cap = {r: a.cap for r in ROBOTS}
    lastJ = {r: None for r in ROBOTS}
    last_fell = {r: set() for r in ROBOTS}
    data = {r: [] for r in ROBOTS}
    trials = {r: [] for r in ROBOTS}
    rollbacks = {r: [] for r in ROBOTS}
    pos = {r: [] for r in ROBOTS}
    pending = {r: {} for r in ROBOTS}          # goal -> (valid mask, e, de) of the last kept trial
    n_trials = {r: 0 for r in ROBOTS}
    recs = []
    names = [r for r in ROBOTS for _ in GOALS]
    Gb = np.tile(GOALS, (len(ROBOTS), 1))
    pb = batch_params(names)
    rows = [np.arange(k * len(GOALS), (k + 1) * len(GOALS)) for k in range(len(ROBOTS))]
    for it in range(a.iters):
        x0 = rp.start_states(Gb, a.train_pert, np.random.default_rng(a.seed + it if a.train_pert > 0 else a.seed))
        pol = GroupPolicy([[(np.asarray(W), np.asarray(b)) for W, b in params[r]] for r in ROBOTS], rows)
        res = rp.simulate(pol, Gb, x0, pb, a.T)
        NS = max(a.fit_samples - 1, 0)
        if NS:                                      # the fitted-dynamics baseline's extra, perturbed rollouts
            nb = len(Gb)
            rows2 = [np.concatenate([np.arange(k * len(GOALS), (k + 1) * len(GOALS)) + j * nb for j in range(NS)])
                     for k in range(len(ROBOTS))]
            pol2 = GroupPolicy([[(np.asarray(W), np.asarray(b)) for W, b in params[r]] for r in ROBOTS], rows2)
            nz = a.fit_noise * np.random.default_rng(a.seed + 1000 + it).standard_normal((NS * nb, a.T, 2))
            res2 = rp.simulate(pol2, np.tile(Gb, (NS, 1)), np.tile(x0, (NS, 1)), {kk: np.tile(v, NS) for kk, v in pb.items()},
                               a.T, noise=nz)
        for k, r in enumerate(ROBOTS):
            rr = rows[k]
            n_trials[r] += len(rr) * (1 + NS)
            F = res['failed'][rr]
            fell = {gi for gi in range(len(GOALS)) if res['failed_end'][rr][gi]}
            w = 1.0 / C.ERR_SCALE
            Jm = 0.5 * np.sum((res['err'][rr] * w) ** 2 * (~F)[..., None], axis=(1, 2)) / a.T
            if a.record:
                recs += [dict(robot=r, it=it, goal=GOALS[gi].tolist(), x=res['x'][rr][gi]) for gi in range(len(GOALS))]
            new_fall = sorted(fell - last_fell[r]) if a.crash_rollback and lastJ[r] is not None else []
            if lastJ[r] is not None and ((a.rollback > 0 and Jm.mean() > lastJ[r] / a.rollback) or new_fall):
                params[r] = prev[r]
                cap[r] *= 0.5
                rollbacks[r].append(dict(it=it, J=float(Jm.mean()), last=lastJ[r], cap=cap[r], new_fall=new_fall))
                pending[r] = {}
                print(json.dumps(dict(it=it, robot=r, rollback=True, J=float(Jm.mean()), last=lastJ[r])), flush=True)
                continue
            lastJ[r], last_fell[r], prev[r] = float(Jm.mean()), fell, params[r]
            valid = ~F
            for gi in fell:                          # the last 20 periods before a fall are not trusted
                kk = np.where(F[gi])[0]
                if len(kk):
                    valid[gi, max(kk[0] - 20, 0):] = False
            if NS:
                A, B, Cm, K = (np.asarray(q) for q in parts(params[r], jnp.asarray(res['x'][rr]), jnp.asarray(res['a'][rr]),
                                                             jnp.asarray(Gb[rr])))
                A, B = A.copy(), B.copy()
                for gi in range(len(GOALS)):
                    sr = rr[gi] + np.arange(NS) * len(Gb)
                    ok = ~res2['failed'][sr]
                    A[gi], B[gi] = fitted_AB(A[gi], B[gi], res['x'][rr][gi], res['a'][rr][gi], res2['x'][sr], res2['a'][sr], ok)
                step, J, Jp, e, de = gn_parts(*(jnp.asarray(q) for q in (A, B, Cm, K)), jnp.asarray(res['err'][rr]),
                                              jnp.asarray(valid), cap=float(cap[r]))
            else:
                step, J, Jp, e, de = gn(params[r], jnp.asarray(res['x'][rr]), jnp.asarray(res['a'][rr]),
                                        jnp.asarray(Gb[rr]), jnp.asarray(res['err'][rr]), jnp.asarray(valid), cap=float(cap[r]))
            step, e, de = np.nan_to_num(np.asarray(step)), np.asarray(e), np.asarray(de)
            vrow = np.repeat(valid, C.NE, axis=1)
            for gi in range(len(GOALS)):              # positivity: last kept prediction vs this measured change
                if gi in pending[r]:
                    v0, e0, de0 = pending[r][gi]
                    m = v0 & vrow[gi]
                    d_meas = (e[gi] - e0)[m]
                    d_pred = de0[m]
                    if m.sum() > 10 and np.linalg.norm(d_pred) > 0 and np.linalg.norm(d_meas) > 0:
                        pos[r].append(dict(it=it, goal=gi, cos=float(d_pred @ d_meas / np.linalg.norm(d_pred) / np.linalg.norm(d_meas)),
                                           ratio=float(np.linalg.norm(d_meas) / np.linalg.norm(d_pred))))
                pending[r][gi] = (vrow[gi], e[gi], de[gi])
            for gi in range(len(GOALS)):
                v = valid[gi]
                data[r].append(dict(obs=res['obs'][rr][gi][v], y=(res['mu'][rr][gi] + step[gi])[v], it=it))
                trials[r].append(dict(it=it, goal=GOALS[gi].tolist(), fell=bool(res['failed_end'][rr][gi]), J=float(Jm[gi]),
                                      J_gn=float(np.asarray(J)[gi]), J_pred=float(np.asarray(Jp)[gi])))
            O = np.concatenate([d['obs'] for d in data[r]])
            Y = np.concatenate([d['y'] for d in data[r]])
            W = np.concatenate([np.full(len(d['obs']), a.age_decay ** (it - d['it'])) for d in data[r]])
            Ya = np.asarray(pj.mlp(params[r], jnp.asarray(O_anchor)))
            wa = a.anchor_w * a.anchor_n0 / (a.anchor_n0 + len(data[r]))
            ps, st = params[r], opt.init(params[r])
            O_, Y_, W_, Oa_, Ya_ = map(jnp.asarray, (O, Y, W, O_anchor, Ya))
            for s in range(a.steps):
                b = jnp.asarray(rs.integers(0, len(O), min(2048, len(O))))
                ba = jnp.asarray(rs.integers(0, len(O_anchor), min(2048, len(O_anchor))))
                ps, st, l = fit_step(ps, st, O_[b], Y_[b], W_[b], Oa_[ba], Ya_[ba], wa)
            if np.isfinite(float(l)) and all(np.isfinite(np.asarray(W)).all() for W, b in ps):
                params[r] = ps
            print(json.dumps(dict(it=it, robot=r, J=round(float(Jm.mean()), 3), falls=len(fell), loss=float(l),
                                  cos=[round(q['cos'], 2) for q in pos[r] if q['it'] == it], t=round(time.time() - t0, 1))),
                  flush=True)
    pols = {r: [(np.asarray(W), np.asarray(b)) for W, b in params[r]] for r in ROBOTS}
    for r in ROBOTS:
        C.save_policy(os.path.join(a.out, f'{r}_policy.npz'), pols[r], kind='mlp')
    if a.record and recs:
        np.savez(a.record, recs=np.array(recs, dtype=object))
    ev, runs = evaluate({r: pols[r] for r in ROBOTS})
    summary.update(eval=ev, runs=runs, real_trials=n_trials, trials=trials, positivity=pos, rollbacks=rollbacks,
                   seconds=time.time() - t0)
    write(summary)
    for r in ROBOTS:
        cs = [q['cos'] for q in pos[r]]
        print(r, json.dumps(dict(success=round(ev[r]['success'], 3), rms_apex=ev[r]['rms_apex'], v_err=ev[r]['v_err'],
                                 pos_frac=round(float(np.mean(np.array(cs) > 0)), 2) if cs else None,
                                 cos_med=round(float(np.median(cs)), 2) if cs else None)))


if __name__ == '__main__':
    main()
