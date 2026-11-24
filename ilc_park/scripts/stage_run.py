"""The parking stage against a "real" car in sim (a mismatched model), and the step-0 positivity check.

The "real" car: the nominal Fiala model with --real-mu friction (hardware evidence: the real car held its 13-18 deg drift
at 2.4-2.6 m/s where the nominal model's equilibrium is 2.9 m/s, i.e. an effective mu ~0.44 against the nominal 0.6),
released from ITS OWN steady drift (what the real car holds while circling). Jacobians: the nominal model only.

  --positivity N   step 0: for N training parks, the GN step du from the measured park, then the park re-flown with the
                   offsets added to the policy's actions; reports the cosine and gain between the predicted error change
                   G du and the measured one (drone soccer: cosine 0.85 median, 86 % positive, gain 2.3).
  (default)        the stage: --parks real parks on training slots (one refit per --parks-per-iter), evaluated on the
                   held-out slots after every --eval-every parks.

    F1T_MU=0.6 python scripts/stage_run.py --policy runs/ppo_nominal_s0/policy_it0600.npz --slots results/feasibility_R1.5_b15.json \
        --out runs/stage_mu044_s0 --positivity 32
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import jax                                           # noqa: E402
import jax.numpy as jnp                              # noqa: E402
import optax                                         # noqa: E402

from ilc_park import car_model as cm                 # noqa: E402
from ilc_park import sim, stage                      # noqa: E402
from ilc_park.ppo import load_slots, mlp             # noqa: E402

actor_fn = lambda params, o: jnp.tanh(mlp(params, o))          # noqa: E731


def load_actor(path):
    d = np.load(path)
    n = len([k for k in d.files if k.endswith('_w')])
    return [(jnp.asarray(d[f'actor_{i}_w']), jnp.asarray(d[f'actor_{i}_b'])) for i in range(n)]


def save_actor(actor, path):
    np.savez(path, **{f'actor_{i}_{n}': np.asarray(t) for i, l in enumerate(actor) for n, t in zip('wb', l)})


def split_slots(slots, every=4):
    """Held-out slots: every `every`-th feasible slot (a fixed, spread-out subset); the rest are trained on."""
    idx = np.arange(len(slots))
    held = idx % every == every // 2
    return slots[~held], slots[held]


def fly(actor, x0, slot, p, offsets=None):
    """One park in the given car: states (T+1, 9), actions (T, 2) (offsets (T, 2) added to the policy's actions)."""
    T = int(round(sim.T_MAX / sim.DT))
    noise = jnp.zeros((T, 1, 2)) if offsets is None else jnp.asarray(offsets)[:, None]
    xs, os_, as_ = sim.rollout(actor, jnp.asarray(x0)[None], jnp.asarray(slot)[None], actor_fn, p, noise)
    return np.asarray(xs[:, 0]), np.asarray(os_[:, 0]), np.asarray(as_[:, 0])


def park_metrics(xs, slot):
    t = stage.stop_tick(xs)
    pos, dpsi = sim.slot_frame(jnp.asarray(xs[t]), jnp.asarray(slot))
    pe, he = float(np.linalg.norm(pos)), float(np.degrees(abs(dpsi)))
    return dict(pos_err=pe, psi_err_deg=he, parked=bool(pe < 0.15 and he < 10.0), stop_s=t * sim.DT,
                beta_peak_deg=float(np.degrees(np.abs(np.asarray(sim.sideslip(xs))[np.asarray(sim.speed(xs)) > 1.2]).max()))
                if (np.asarray(sim.speed(xs)) > 1.2).any() else 0.0)


def ilc_step(actor, xs, as_, slot, p_nom, a, horizon):
    """Rows, lifted G and the GN step for one measured park; returns (du (T, 2), rows e, G, t_stop, row ticks)."""
    t_stop = stage.stop_tick(xs)
    A, B, K = (np.asarray(m) for m in stage.linearize(actor, jnp.asarray(xs), jnp.asarray(as_), jnp.asarray(slot),
                                                          actor_fn, p_nom))
    e, C, ticks = stage.rows_and_outputs(actor, jnp.asarray(xs), jnp.asarray(slot), actor_fn, p_nom, t_stop, horizon,
                                         beta_row=False, pred=a.pred)
    if a.beta_row:                                   # the sideslip rows, appended (positivity compares the pose rows)
        eb, Cb, tb = stage.rows_and_outputs(actor, jnp.asarray(xs), jnp.asarray(slot), actor_fn, p_nom, t_stop, horizon,
                                            beta_row=True, ticks_p=np.zeros(0, int))
        e, C, ticks = np.r_[e, eb[3:]], np.r_[C, Cb[3:]], np.r_[ticks, tb[3:]]
    mean = np.asarray(actor_fn(actor, jnp.asarray(sim.observe_batch(xs[:-1], slot))))
    masks = (np.abs(mean) < 0.98).astype(float)
    G = stage.lifted_rows_jacobian(A[:t_stop], B[:t_stop], K[:t_stop], masks[:t_stop], C, ticks)
    du = np.zeros_like(as_)
    du[:t_stop] = stage.gn_step(G, e, a.beta, a.delta, a.cap_rms)
    return du, e, G, t_stop, ticks


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--policy', required=True)
    ap.add_argument('--slots', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--real-mu', type=float, default=0.44)
    ap.add_argument('--R0', type=float, nargs='+', default=[1.5])
    ap.add_argument('--beta0', type=float, default=-15.0)
    ap.add_argument('--positivity', type=int, default=0)
    ap.add_argument('--parks', type=int, default=120)
    ap.add_argument('--parks-per-iter', type=int, default=4)
    ap.add_argument('--eval-every', type=int, default=24)
    ap.add_argument('--horizon', type=int, default=80, help='predicted-stop rollout length (periods, 2 s)')
    ap.add_argument('--beta', type=float, default=0.5)
    ap.add_argument('--delta', type=float, default=0.5)
    ap.add_argument('--cap-rms', type=float, default=0.3)
    ap.add_argument('--beta-row', action='store_true')
    ap.add_argument('--pred', default='none', choices=('rollout', 'kinematic', 'none'),
                    help='per-tick stop-pose rows. none (default, 2026-11-16): the measured final pose only (3 rows; classic terminal ILC) --\n'
                    'on 4 fixed parks x 4 iterations it converged monotonically (0.58/1.93/0.47 m -> 0.13/0.12/0.07 m) where rollout\n'
                    '(nominal closed-loop stop: huge ill-conditioned Jacobians, steps ~0.01) stalled and kinematic (velocity line at\n'
                    'A_STOP) overshot')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--anchor-weight', type=float, default=4.0)
    ap.add_argument('--anchor-n0', type=float, default=6.0)
    ap.add_argument('--age-decay', type=float, default=0.5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)
    rng = np.random.default_rng(a.seed)

    p_nom = {k: float(v) for k, v in cm.NOMINAL.items()}
    p_nom = jax.tree.map(jnp.asarray, p_nom)
    p_real = dict(p_nom, muf=jnp.asarray(a.real_mu), mur=jnp.asarray(a.real_mu))
    p_real_b = sim.nominal_params(1, muf=a.real_mu, mur=a.real_mu)
    real_cm = dict(cm.NOMINAL, muf=a.real_mu, mur=a.real_mu)
    releases, _ = sim.drift_table(a.R0, [a.beta0], real_cm)          # the REAL car's own steady drift
    slots = load_slots(a.slots)
    train_slots, held_slots = split_slots(slots)
    actor = load_actor(a.policy)
    log = open(os.path.join(a.out, 'log.jsonl'), 'a')

    def draw():
        x0 = releases[rng.integers(len(releases))].copy()
        sl = train_slots[rng.integers(len(train_slots))].copy()
        if rng.random() < 0.5:
            x0 *= sim.MIRROR_STATE
            sl *= np.array([1, -1, -1])
        return x0, sl

    if a.positivity:
        out = []
        for k in range(a.positivity):
            x0, sl = draw()
            xs, _, as_ = fly(actor, x0, sl, p_real_b)
            du, e, G, t_stop, ticks = ilc_step(actor, xs, as_, sl, p_nom, a, a.horizon)
            xs2, _, _ = fly(actor, x0, sl, p_real_b, offsets=du)
            pred = np.einsum('rtc,tc->r', G, du[:G.shape[1]])
            # measured change of the same rows: recompute the rows on the re-flown park at the same ticks
            tp = np.unique(ticks[ticks < t_stop])
            e1, _, _ = stage.rows_and_outputs(actor, jnp.asarray(xs), jnp.asarray(sl), actor_fn, p_nom, t_stop, a.horizon,
                                              ticks_p=tp, pred=a.pred)
            e2, _, _ = stage.rows_and_outputs(actor, jnp.asarray(xs2), jnp.asarray(sl), actor_fn, p_nom, t_stop, a.horizon,
                                              ticks_p=tp, pred=a.pred)
            n = len(e1)                              # the pose rows, same ticks on both flights
            meas = e2 - e1
            cos = float(pred[:n] @ meas / (np.linalg.norm(pred[:n]) * np.linalg.norm(meas) + 1e-12))
            gain = float(meas @ pred[:n] / (pred[:n] @ pred[:n] + 1e-12))
            m1, m2 = park_metrics(xs, sl), park_metrics(xs2, sl)
            row = dict(k=k, cos=cos, gain=gain, pos_err_before=m1['pos_err'], pos_err_after=m2['pos_err'],
                       psi_before=m1['psi_err_deg'], psi_after=m2['psi_err_deg'], rows=int(len(e)))
            out.append(row)
            print(json.dumps(row), flush=True)
            log.write(json.dumps(dict(positivity=row)) + '\n'); log.flush()
        c = np.array([r['cos'] for r in out])
        g = np.array([r['gain'] for r in out])
        imp = np.array([r['pos_err_after'] < r['pos_err_before'] for r in out])
        summ = dict(cos_median=float(np.median(c)), positive=float(np.mean(c > 0)), gain_median=float(np.median(g)),
                    pos_err_improved=float(imp.mean()),
                    pos_err_before=float(np.mean([r['pos_err_before'] for r in out])),
                    pos_err_after=float(np.mean([r['pos_err_after'] for r in out])))
        print('POSITIVITY', json.dumps(summ), flush=True)
        json.dump(dict(summary=summ, rows=out), open(os.path.join(a.out, 'positivity.json'), 'w'), indent=1)
        return

    # ---- the stage ----
    anchor_obs = []
    for _ in range(64):                                # anchor pool: the initial policy's NOMINAL parks
        x0n, sln = draw()
        x0n = sim.drift_table(a.R0, [a.beta0])[0][0] * (sim.MIRROR_STATE if sln[1] < 0 else 1)
        _, on, _ = fly(actor, x0n, sln, sim.nominal_params(1))
        anchor_obs.append(on)
    anchor_obs = jnp.asarray(np.concatenate(anchor_obs))
    opt = optax.adam(a.lr)

    def evaluate(act, parks_done):
        res = []
        for sl in held_slots:
            for mirror in (False, True):
                x0 = releases[0] * (sim.MIRROR_STATE if mirror else 1)
                s = sl * (np.array([1, -1, -1]) if mirror else 1)
                xs, _, _ = fly(act, x0, s, p_real_b)
                res.append(park_metrics(xs, s))
        line = dict(parks=parks_done, held_pos_err=float(np.mean([r['pos_err'] for r in res])),
                    held_psi_err=float(np.mean([r['psi_err_deg'] for r in res])),
                    held_parked=float(np.mean([r['parked'] for r in res])),
                    held_beta_peak=float(np.median([r['beta_peak_deg'] for r in res])), held_n=len(res))
        print(json.dumps(line), flush=True)
        log.write(json.dumps(line) + '\n'); log.flush()

    buffer = []
    evaluate(actor, 0)
    save_actor(actor, os.path.join(a.out, 'policy_p0000.npz'))
    for parks in range(a.parks_per_iter, a.parks + 1, a.parks_per_iter):
        recs = []
        for _ in range(a.parks_per_iter):
            x0, sl = draw()
            xs, os_, as_ = fly(actor, x0, sl, p_real_b)
            du, e, _, t_stop, _ = ilc_step(actor, xs, as_, sl, p_nom, a, a.horizon)
            mean = np.asarray(actor_fn(actor, jnp.asarray(os_)))
            buffer.append((os_[:t_stop], np.clip(mean[:t_stop] + du[:t_stop], -0.999, 0.999)))
            recs.append(park_metrics(xs, sl))
        prev = jax.tree.map(lambda z: z, actor)
        w_anchor = a.anchor_weight * a.anchor_n0 / (a.anchor_n0 + parks)
        O = jnp.asarray(np.concatenate([b[0] for b in buffer]))
        Y = jnp.asarray(np.concatenate([b[1] for b in buffer]))
        ages = np.concatenate([np.full(len(b[0]), (len(buffer) - 1 - i) // a.parks_per_iter) for i, b in enumerate(buffer)])
        prob = a.age_decay ** ages
        prob /= prob.sum()
        y_anchor = actor_fn(prev, anchor_obs)
        opt_state = opt.init(actor)

        @jax.jit
        def fit_step(act, st, o, y, oa, ya):
            def loss(pp):
                return (jnp.mean(jnp.sum((actor_fn(pp, o) - y) ** 2, -1))
                        + w_anchor * jnp.mean(jnp.sum((actor_fn(pp, oa) - ya) ** 2, -1)))
            l, g = jax.value_and_grad(loss)(act)
            upd, st = opt.update(g, st, act)
            return optax.apply_updates(act, upd), st, l
        for _ in range(a.steps):
            idx = rng.choice(len(O), a.batch, p=prob)
            ai = rng.integers(0, len(anchor_obs), a.batch)
            actor, opt_state, l = fit_step(actor, opt_state, O[idx], Y[idx], anchor_obs[ai], y_anchor[ai])
        line = dict(parks=parks, train_pos_err=float(np.mean([r['pos_err'] for r in recs])),
                    train_parked=float(np.mean([r['parked'] for r in recs])), fit_loss=float(l), w_anchor=w_anchor)
        print(json.dumps(line), flush=True)
        log.write(json.dumps(line) + '\n'); log.flush()
        if parks % a.eval_every == 0:
            evaluate(actor, parks)
            save_actor(actor, os.path.join(a.out, f'policy_p{parks:04d}.npz'))


if __name__ == '__main__':
    main()
