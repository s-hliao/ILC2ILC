"""The Go1 ball stage against the perturbed robot (model.PERTURBED: the sim stand-in for the real Go1), and the step-0
positivity check. Jacobians from the NOMINAL model only.

  --positivity N  N single-goal segments: GN step from the measured segment, the segment re-flown with the offsets;
                  cosine / gain between the predicted row change G du and the measured one, and the metric change
  (default)       the stage: flights of --goals-per-flight goals (switch on reach), one refit per flight; validation
                  (seeds 200-215, 8 goals x 8 lanes each) in the perturbed robot at the --eval-at goal counts

    python scripts/stage_run.py --policy runs/ppo_nominal_s0/policy_it0800.npz --out runs/stage_nom_s0 --seed 0
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import jax                                           # noqa: E402
import jax.numpy as jnp                              # noqa: E402
import optax                                         # noqa: E402

from ilc_dogball import model as m                   # noqa: E402
from ilc_dogball import sim, stage                   # noqa: E402
from ilc_dogball.ppo import actor_fn, eval_sets, score   # noqa: E402

SEG_T = int(sim.TIMEOUT / m.DT)


def load_actor(path):
    d = np.load(path)
    n = len([k for k in d.files if k.endswith('_w')])
    return [(jnp.asarray(d[f'actor_{i}_w']), jnp.asarray(d[f'actor_{i}_b'])) for i in range(n)]


def save_actor(actor, path):
    np.savez(path, **{f'actor_{i}_{n}': np.asarray(t) for i, l in enumerate(actor) for n, t in zip('wb', l)})


STOP_MODE = False


def fly(actor, x0, goals, base, T, offsets=None, reset_ball=True):
    noise = jnp.zeros((T, 1, m.NA)) if offsets is None else jnp.asarray(offsets)[:, None]
    out = sim.rollout(actor, jnp.asarray(x0)[None], jnp.asarray(goals)[None], sim.params(base, 1), actor_fn, noise,
                      reset_ball=reset_ball, stop_mode=STOP_MODE)
    return {k: np.asarray(v)[:, 0] for k, v in out.items()}


def all_rows(xs, g):
    eb, _, em, _ = (np.asarray(z) for z in stage._row_jacobians(jnp.asarray(xs), jnp.asarray(g)))
    return eb, em


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--policy', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--positivity', type=int, default=0)
    ap.add_argument('--goals', type=int, default=216)
    ap.add_argument('--goals-per-flight', type=int, default=6)
    ap.add_argument('--eval-at', type=int, nargs='+', default=[0, 6, 12, 24, 48, 96, 144, 216])
    ap.add_argument('--no-ball-rows', action='store_true')
    ap.add_argument('--no-approach-rows', action='store_true')
    ap.add_argument('--miss-scale', type=float, default=stage.MISS_SCALE, help='approach-row scale (m)')
    ap.add_argument('--task', default='reach', choices=('reach', 'stop'), help='stop: goals end when the dog comes to rest (sim stop_mode); '
                    'rows = the measured stop error + ball rows')
    ap.add_argument('--nominal', default='NOMINAL', help='model.<NAME> for the Jacobians and anchors (e.g. FILLED)')
    ap.add_argument('--real', default='PERTURBED', help='model.<NAME> flown as the real robot (e.g. HOLLOW)')
    ap.add_argument('--speed-row', action='store_true', help='drone soccer\'s speed row: |v| at the reach tick vs V*, the initial '
                    'policy\'s median reach speed in the NOMINAL sim (measured at start)')
    ap.add_argument('--no-contact-mask', action='store_true', help='keep ball rows while the ball touches a wall (the nominal '
                    'wall is smooth: its Jacobian exists)')
    ap.add_argument('--beta', type=float, default=0.5)
    ap.add_argument('--delta', type=float, default=0.5)
    ap.add_argument('--cap-rms', type=float, default=0.3)
    ap.add_argument('--steps', type=int, default=800)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--anchor-weight', type=float, default=4.0)
    ap.add_argument('--anchor-n0', type=float, default=6.0)
    ap.add_argument('--age-decay', type=float, default=0.5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)
    stage.MISS_SCALE = a.miss_scale                          # module constants read at trace time
    stage.MASK_CONTACT = not a.no_contact_mask
    rng = np.random.default_rng(1000 + a.seed)                 # training draws: never the validation / test seeds
    NOM, REAL, STOP = getattr(m, a.nominal), getattr(m, a.real), a.task == 'stop'
    global STOP_MODE
    STOP_MODE = STOP
    p_nom = sim.params(NOM)
    actor = load_actor(a.policy)
    kw = dict(beta=a.beta, delta=a.delta, cap_rms=a.cap_rms)
    use = dict(ball=not a.no_ball_rows, approach=not a.no_approach_rows)
    log = open(os.path.join(a.out, 'log.jsonl'), 'a')

    def write(line):
        print(json.dumps(line), flush=True)
        log.write(json.dumps(line) + '\n')
        log.flush()

    if a.positivity:
        res = []
        for k in range(a.positivity):
            x0, g = sim.sample_goals(rng, 1, 1)
            f1 = fly(actor, x0[0], g[0], REAL, SEG_T)
            done = np.where(f1['reached'] | f1['timeout'])[0]
            T = int(done[0]) + 1 if len(done) else SEG_T
            xs = np.concatenate([x0[:1], f1['x'][:T]])
            du, e = stage.segment_step(actor, actor_fn, xs, f1['act'][:T], x0[0, 13:16], g[0, 0], 0.0, p_nom, **use, **kw)
            off = np.zeros((SEG_T, m.NA))
            off[:T] = du
            f2 = fly(actor, x0[0], g[0], REAL, SEG_T, offsets=off)
            xs2 = np.concatenate([x0[:1], f2['x'][:T]])
            xm1 = stage.measured(xs, np.concatenate([x0[:1, 13:16], f1['act'][:T]]))
            xm2 = stage.measured(xs2, np.concatenate([x0[:1, 13:16], f2['act'][:T]]))
            e1, C, tk, kd = stage.rows(xm1, g[0, 0], **use, with_kinds=True)
            if len(e1) == 0:
                continue
            A, B, K = (np.asarray(z) for z in stage.linearize(actor, jnp.asarray(xm1), jnp.asarray(f1['act'][:T]),
                                                               jnp.asarray(g[0, 0]), 0.0, actor_fn, p_nom))
            mean = np.asarray(actor_fn(actor, jax.vmap(sim.observe, (0, None))(jnp.asarray(xm1[:-1]), jnp.asarray(g[0, 0]))))
            G = stage.lifted(A, B, K, (np.abs(mean) < 0.98).astype(float), C, tk)
            pred = np.einsum('rtc,tc->r', G, du)
            # the same rows (same ticks, same kinds) on the re-flown segment
            eb1, em1 = all_rows(xm1, g[0, 0])
            eb2, em2 = all_rows(xm2, g[0, 0])
            meas = [eb2[t, k] - eb1[t, k] if k < 2 else em2[t] - em1[t] for t, k in zip(tk, kd)]
            meas = np.asarray(meas)
            cos = float(pred @ meas / (np.linalg.norm(pred) * np.linalg.norm(meas) + 1e-12))
            gain = float(meas @ pred / (pred @ pred + 1e-12))
            b1 = 100 * np.sqrt(np.mean(np.sum(xs[1:, 6:8] ** 2, -1)))
            b2 = 100 * np.sqrt(np.mean(np.sum(xs2[1:, 6:8] ** 2, -1)))
            d2 = f2['reached'] | f2['timeout']
            T2 = (int(np.where(d2)[0][0]) + 1) * m.DT if d2.any() else sim.TIMEOUT
            row = dict(k=k, cos=cos, gain=gain, rows=int(len(e1)), ball_rms_cm_before=b1, ball_rms_cm_after=b2,
                       t_reach_before=T * m.DT, t_reach_after=T2, e_rms_before=float(np.sqrt(np.mean(e1 ** 2))))
            res.append(row)
            write(dict(positivity=row))
        c = np.array([r['cos'] for r in res])
        summ = dict(n=len(res), cos_median=float(np.median(c)), positive=float(np.mean(c > 0)),
                    gain_median=float(np.median([r['gain'] for r in res])),
                    ball_rms_before=float(np.mean([r['ball_rms_cm_before'] for r in res])),
                    ball_rms_after=float(np.mean([r['ball_rms_cm_after'] for r in res])),
                    t_reach_before=float(np.mean([r['t_reach_before'] for r in res])),
                    t_reach_after=float(np.mean([r['t_reach_after'] for r in res])))
        write(dict(POSITIVITY=summ))
        json.dump(dict(summary=summ, rows=res), open(os.path.join(a.out, 'positivity.json'), 'w'), indent=1)
        return

    ex0, egoals = eval_sets()
    eT = 8 * SEG_T // 2

    def evaluate(act, goals_done):
        out = sim.rollout(act, jnp.asarray(ex0), jnp.asarray(egoals), sim.params(REAL, len(ex0)), actor_fn,
                          jnp.zeros((eT, len(ex0), m.NA)), reset_ball=True, stop_mode=STOP)
        write(dict(goals=goals_done, **{f'val_{k}': v for k, v in score(out, ex0, 8).items()}))
        save_actor(act, os.path.join(a.out, f'policy_g{goals_done:04d}.npz'))

    # anchor pool: the initial policy's observations in nominal flights
    ax0, ag = sim.sample_goals(rng, 32, a.goals_per_flight)
    aout = sim.rollout(actor, jnp.asarray(ax0), jnp.asarray(ag), sim.params(NOM, 32), actor_fn,
                       jnp.zeros((a.goals_per_flight * SEG_T // 2, 32, m.NA)), reset_ball=True, stop_mode=STOP)
    anchor_obs = jnp.asarray(np.asarray(aout['obs']).reshape(-1, sim.NOBS))
    if a.speed_row:
        ax = np.asarray(aout['x'])
        reach_speed = np.hypot(ax[..., 3], ax[..., 4])[np.asarray(aout['reached'])]
        stage.V_STAR = float(np.median(reach_speed))
        write(dict(v_star=stage.V_STAR, n=int(reach_speed.size)))
    opt = optax.adam(a.lr)
    if 0 in a.eval_at:
        evaluate(actor, 0)
    buffer = []                                              # (flight index, obs, targets)
    T_flight = a.goals_per_flight * SEG_T
    goals_done, flight = 0, 0
    while goals_done < a.goals:
        x0, g = sim.sample_goals(rng, 1, a.goals_per_flight)
        f = fly(actor, x0[0], g[0], REAL, T_flight)
        done = f['reached'] | f['timeout']
        ends = np.where(done)[0][:a.goals_per_flight]
        xs_all = np.concatenate([x0[:1], f['x']])
        starts = np.r_[0, ends[:-1] + 1]
        n_rows, e_rms = 0, []
        for s, e_ in zip(starts, ends):
            T = e_ - s + 1
            esc = np.where(f['escaped'][s:e_ + 1])[0]
            if len(esc):                                     # rows up to the escape only
                T = int(esc[0]) + 1
            xs = xs_all[s:s + T + 1]
            a_prev = x0[0, 13:16] if s == 0 else f['act'][s - 1]
            gi = int(f['goal_idx'][s])
            du, e = stage.segment_step(actor, actor_fn, xs, f['act'][s:s + T], a_prev, g[0, gi], s * m.DT, p_nom,
                                       reached=bool(f['reached'][s + T - 1]) and not len(esc), stop=STOP, **use, **kw)
            if len(e) == 0:
                continue
            mean = np.asarray(actor_fn(actor, jnp.asarray(f['obs'][s:s + T])))
            buffer.append((flight, f['obs'][s:s + T], np.clip(mean + du, -0.999, 0.999)))
            n_rows += len(e)
            e_rms.append(float(np.sqrt(np.mean(e ** 2))))
        goals_done += len(ends)
        prev = actor
        w_anchor = a.anchor_weight * a.anchor_n0 / (a.anchor_n0 + goals_done)
        O = jnp.asarray(np.concatenate([b[1] for b in buffer]))
        Y = jnp.asarray(np.concatenate([b[2] for b in buffer]))
        prob = np.concatenate([np.full(len(b[1]), a.age_decay ** (flight - b[0])) for b in buffer])
        prob /= prob.sum()
        y_anchor = actor_fn(prev, anchor_obs)
        opt_state = opt.init(actor)

        @jax.jit
        def fit_step(act, st, o, y, oa, ya):
            def loss(pp):
                return (jnp.mean(jnp.sum((actor_fn(pp, o) - y) ** 2, -1))
                        + w_anchor * jnp.mean(jnp.sum((actor_fn(pp, oa) - ya) ** 2, -1)))
            l, gr = jax.value_and_grad(loss)(act)
            upd, st = opt.update(gr, st, act)
            return optax.apply_updates(act, upd), st, l
        for _ in range(a.steps):
            idx = rng.choice(len(O), a.batch, p=prob)
            ai = rng.integers(0, len(anchor_obs), a.batch)
            actor, opt_state, l = fit_step(actor, opt_state, O[idx], Y[idx], anchor_obs[ai], y_anchor[ai])
        tr = done[:ends[-1] + 1] if len(ends) else done
        write(dict(goals=goals_done, flight=flight, rows=n_rows, e_rms=float(np.mean(e_rms)) if e_rms else None,
                   flight_s=float((ends[-1] + 1) * m.DT) if len(ends) else None,
                   esc=int(f['escaped'][:ends[-1] + 1].sum()) if len(ends) else None, fit_loss=float(l),
                   w_anchor=w_anchor))
        flight += 1
        for ge in a.eval_at:
            if goals_done - len(ends) < ge <= goals_done:
                evaluate(actor, goals_done)


if __name__ == '__main__':
    main()
