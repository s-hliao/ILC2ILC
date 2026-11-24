"""PPO for the Go1 ball-carrying policy in the NOMINAL reduced model (or a randomized family: --dr, the PPO+DR baseline).

Reward per 50 ms period (lanes end when the ball escapes):
  progress  W.prog * (d_before - d_after) / 0.05 m toward the goal in force; + W.reach on reaching it
  ball      - W.ball * sum relu(|b| / travel - MARGIN_FRAC)^2   (keep the ball in its safe zone)
  wall      - W.wall * sum relu(|b| - travel) / 0.01 m   (padded-wall penetration ~ impact speed)
  escape    - W.esc once
  smooth    - W.du |a_t - a_{t-1}|^2
Evaluation (every --eval-every): validation goal sequences (seeds 200-215; test seeds 100-103 are never used here),
in the nominal and the perturbed robot, ball put back after an escape.

    python -m ilc_dogball.ppo --out runs/ppo_nominal_s0 --seed 0
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ilc_dogball import model as m
from ilc_dogball import sim

W = dict(prog=0.1, reach=1.0, ball=2.0, wall=2.0, contact=1.0, esc=5.0, du=0.02, stop=3.0, stop_scale=0.08)   # 2026-11-16 option 1: contact
# penalty (per period touching a wall) + 4x penetration weight: the nominal policy keeps the ball off the walls, so the
# robot mismatch's wall hits are a clean, learnable gap (v2 policies traded wall hits for speed: .22 hits/goal nominal)
MARGIN_FRAC = 0.6                 # the ball's safe zone: within 60 % of its travel (the stage's ball rows use the same)
VAL_SEEDS = tuple(range(200, 216))


def init_mlp(key, sizes, scale_last=0.01):
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for i, (k, (n_in, n_out)) in enumerate(zip(keys, zip(sizes[:-1], sizes[1:]))):
        s = scale_last if i == len(sizes) - 2 else np.sqrt(2.0 / n_in)
        params.append((jax.random.normal(k, (n_in, n_out)) * s, jnp.zeros(n_out)))
    return params


def mlp(params, x):
    for w, b in params[:-1]:
        x = jnp.tanh(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def actor_fn(params, o):
    return jnp.tanh(mlp(params, o))


def init_agent(key, hidden=128, log_std=-1.0):
    ka, kc = jax.random.split(key)
    return dict(actor=init_mlp(ka, [sim.NOBS, hidden, hidden, m.NA]),
                critic=init_mlp(kc, [sim.NOBS, hidden, hidden, 1], scale_last=1.0),
                log_std=jnp.full((m.NA,), log_std))


def rewards(out, x0, stop_mode=False):
    xs = jnp.concatenate([x0[None], out['x']])
    d0 = jnp.linalg.norm(xs[:-1, :, :2] - out['goal'], axis=-1)
    d1 = jnp.linalg.norm(xs[1:, :, :2] - out['goal'], axis=-1)
    active = out['goal_idx'] < out['goal_idx'].max() + 1          # all periods (held last goal included)
    b = xs[1:, :, 6:8]
    r = W['prog'] * (d0 - d1) / 0.05 * active + W['reach'] * out['reached']
    if stop_mode:          # precise stop: the bonus decays with the stop error (8 cm scale)
        r += W['stop'] * out['reached'] * jnp.exp(-(out['dist'] / W['stop_scale']) ** 2)
    r -= W['ball'] * jnp.sum(jax.nn.relu(jnp.abs(b) / m.TRAVEL - MARGIN_FRAC) ** 2, -1)
    r -= W['wall'] * jnp.sum(jax.nn.relu(jnp.abs(b) - m.TRAVEL), -1) / 0.01
    r -= W['contact'] * jnp.any(jnp.abs(b) > m.TRAVEL - 0.005, -1)
    r -= W['esc'] * out['escaped']
    da = out['act'] - jnp.concatenate([x0[None, :, 13:16], out['act'][:-1]])
    r -= W['du'] * jnp.sum(da ** 2, -1)
    return r * out['alive']


def gaussian_logp(mean, log_std, a):
    return jnp.sum(-0.5 * ((a - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * np.log(2 * np.pi), -1)


def gae(values, r, alive, gamma, lam):
    """values (T+1, n), r (T, n), alive (T, n) (alive at the period's start; an escape ends the lane after it)."""
    nxt = jnp.concatenate([alive[1:], alive[-1:]]) * (1.0 - 0.0)

    def back(carry, inp):
        v, v1, rr, al, al1 = inp
        d = rr + gamma * v1 * al1 - v
        carry = (d + gamma * lam * carry * al1) * al
        return carry, carry
    _, adv = jax.lax.scan(back, jnp.zeros_like(r[0]), (values[:-1], values[1:], r, alive, nxt), reverse=True)
    return adv, adv + values[:-1]


def make_update(opt, clip, ent):
    @jax.jit
    def update(agent, opt_state, o, a, logp_old, adv, ret, w):
        def loss(ag):
            mean = actor_fn(ag['actor'], o)
            logp = gaussian_logp(mean, ag['log_std'], a)
            ratio = jnp.exp(logp - logp_old)
            pg = -jnp.sum(w * jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip, 1 + clip) * adv)) / jnp.sum(w)
            v = mlp(ag['critic'], o)[:, 0]
            vl = 0.5 * jnp.sum(w * (v - ret) ** 2) / jnp.sum(w)
            return pg + vl - ent * jnp.sum(ag['log_std'])
        l, g = jax.value_and_grad(loss)(agent)
        upd, opt_state = opt.update(g, opt_state, agent)
        return optax.apply_updates(agent, upd), opt_state, l
    return update


def dr_params(rng, n, scale=1.0):
    """The PPO+DR family: at scale 1 a box around the nominal robot that covers model.PERTURBED in every parameter."""
    u = lambda lo, hi: rng.uniform(lo, hi, n)                 # noqa: E731
    lerp = lambda k, lo, hi: (m.NOMINAL[k] + scale * (lo - m.NOMINAL[k]), m.NOMINAL[k] + scale * (hi - m.NOMINAL[k]))  # noqa: E731
    box = dict(tau_c=(0.02, 0.10), tau_v=(0.10, 0.30), tau_w=(0.08, 0.22), a_max=(2.0, 3.6), al_max=(6.0, 11.0),
               gain_v=(0.9, 1.2), gain_w=(0.9, 1.2), c_rr=(0.2, 0.6), tilt0_x=(-np.radians(0.5), np.radians(0.5)),
               tilt0_y=(-np.radians(0.5), np.radians(0.5)), sway_a=(0.0, 1.0), pitch_a=(0.0, np.radians(2.0)))
    d = {k: np.full(n, float(v)) for k, v in m.NOMINAL.items()}
    for k, (lo, hi) in box.items():
        assert lo <= m.PERTURBED[k] <= hi, k
        d[k] = u(*lerp(k, lo, hi))
    return {k: jnp.asarray(v) for k, v in d.items()}


def eval_sets(n_goals=8, lanes_per_seed=8, seeds=VAL_SEEDS):
    xs, gs = [], []
    for s in seeds:
        x0, g = sim.sample_goals(np.random.default_rng(s), lanes_per_seed, n_goals)
        xs.append(x0)
        gs.append(g)
    return np.concatenate(xs), np.concatenate(gs)


def score(out, x0, n_goals):
    """Per-lane metrics of a flown evaluation: time to reach per goal (timeouts count TIMEOUT), ball offset rms (cm),
    escapes per goal, wall-contact fraction."""
    gi = np.asarray(out['goal_idx'])
    reached = np.asarray(out['reached'])
    timeout = np.asarray(out['timeout'])
    tg = np.asarray(out['t_goal'])
    done = reached | timeout
    ttr = np.where(timeout, sim.TIMEOUT, tg + m.DT)
    n = gi.shape[1]
    rows = []
    xs = np.asarray(out['x'])
    dist = np.asarray(out['dist']) if 'dist' in out else None
    for i in range(n):
        ends = np.where(done[:, i])[0]
        t_goals = ttr[ends, i]
        if len(t_goals) == 0:
            continue
        upto = ends[-1] + 1
        b = xs[:upto, i, 6:8]
        stop_err = dist[ends, i][reached[ends, i]] if dist is not None else np.zeros(0)
        rows.append(dict(time_to_reach_s=float(t_goals.mean()), goals=int(len(t_goals)),
                         stop_err_cm=float(100 * stop_err.mean()) if len(stop_err) else float('nan'),
                         reached_frac=float(reached[ends, i].mean()),
                         ball_rms_cm=float(100 * np.sqrt(np.mean(np.sum(b ** 2, -1)))),
                         esc_per_goal=float(np.asarray(out['escaped'])[:upto, i].sum() / len(t_goals)),
                         wall_frac=float(np.mean(np.any(np.abs(b) > m.TRAVEL - 0.005, -1))),
                         wall_hits_per_goal=float(np.sum(np.diff(np.any(np.abs(b) > m.TRAVEL, -1).astype(int)) == 1)
                                                  / len(t_goals)),
                         outside_margin_frac=float(np.mean(np.any(np.abs(b) > MARGIN_FRAC * m.TRAVEL, -1)))))
    agg = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0] if k != 'goals'}
    agg['goals'] = int(sum(r['goals'] for r in rows))
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dr', type=float, default=0.0, help='domain randomization scale (0: nominal)')
    ap.add_argument('--iters', type=int, default=400)
    ap.add_argument('--lanes', type=int, default=2048)
    ap.add_argument('--goals', type=int, default=6)
    ap.add_argument('--T', type=int, default=400, help='periods per lane (20 s)')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--mb', type=int, default=32768)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--ent', type=float, default=1e-3)
    ap.add_argument('--eval-every', type=int, default=50)
    ap.add_argument('--task', default='reach', choices=('reach', 'stop'))
    ap.add_argument('--nominal', default='NOMINAL', help='model dict trained in (model.<NAME>, e.g. FILLED)')
    ap.add_argument('--real', default='PERTURBED', help='model dict evaluated as the real robot (e.g. HOLLOW)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(vars(a), weights=W), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)
    rng = np.random.default_rng(a.seed)
    key = jax.random.PRNGKey(a.seed)
    key, k0 = jax.random.split(key)
    agent = init_agent(k0)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr))
    opt_state = opt.init(agent)
    update = make_update(opt, a.clip, a.ent)
    critic_v = jax.jit(lambda c, o: mlp(c, o)[..., 0])
    ex0, egoals = eval_sets()
    eT = 8 * int(sim.TIMEOUT / m.DT) // 2
    NOM, REAL, STOP = getattr(m, a.nominal), getattr(m, a.real), a.task == 'stop'
    p_nom_train = sim.params(NOM, a.lanes)
    log = open(os.path.join(a.out, 'log.jsonl'), 'a')

    def write(line):
        print(json.dumps(line), flush=True)
        log.write(json.dumps(line) + '\n')
        log.flush()

    for it in range(a.iters + 1):
        if it % a.eval_every == 0:
            line = dict(it=it)
            for name, base in (('nom', NOM), ('pert', REAL)):
                out = sim.rollout(agent['actor'], jnp.asarray(ex0), jnp.asarray(egoals), sim.params(base, len(ex0)),
                                  actor_fn, jnp.zeros((eT, len(ex0), m.NA)), reset_ball=True, stop_mode=STOP)
                line.update({f'{name}_{k}': v for k, v in score(out, ex0, 8).items()})
            write(line)
            np.savez(os.path.join(a.out, f'policy_it{it:04d}.npz'),
                     **{f'actor_{i}_{n}': np.asarray(t) for i, l in enumerate(agent['actor']) for n, t in zip('wb', l)})
        if it == a.iters:
            break
        t0 = time.time()
        x0, goals = sim.sample_goals(rng, a.lanes, a.goals)
        p = dr_params(rng, a.lanes, a.dr) if a.dr > 0 else p_nom_train
        key, kn = jax.random.split(key)
        noise = jax.random.normal(kn, (a.T, a.lanes, m.NA)) * jnp.exp(agent['log_std'])
        x0j = jnp.asarray(x0)
        out = sim.rollout(agent['actor'], x0j, jnp.asarray(goals), p, actor_fn, noise, stop_mode=STOP)
        r = rewards(out, x0j, STOP)
        v = critic_v(agent['critic'], out['obs'])
        v = jnp.concatenate([v, jnp.zeros_like(v[:1])])
        adv, ret = gae(v, r, out['alive'], a.gamma, a.lam)
        logp = gaussian_logp(actor_fn(agent['actor'], out['obs']), agent['log_std'], out['act'])
        flat = lambda z: z.reshape((-1,) + z.shape[2:])            # noqa: E731
        O, A, LP, ADV, RET, WT = map(flat, (out['obs'], out['act'], logp, adv, ret, out['alive']))
        mu = jnp.sum(WT * ADV) / jnp.sum(WT)
        sd = jnp.sqrt(jnp.sum(WT * (ADV - mu) ** 2) / jnp.sum(WT))
        ADV = (ADV - mu) / (sd + 1e-8)
        N = O.shape[0]
        for _ in range(a.epochs):
            key, kp = jax.random.split(key)
            perm = jax.random.permutation(kp, N)
            for s in range(0, N - a.mb + 1, a.mb):
                idx = perm[s:s + a.mb]
                agent, opt_state, l = update(agent, opt_state, O[idx], A[idx], LP[idx], ADV[idx], RET[idx], WT[idx])
        if it % 10 == 0:
            write(dict(it=it, reward=float(r.sum(0).mean()), goals=float(np.asarray(out['reached']).sum(0).mean()),
                       esc=float(np.asarray(out['escaped'] * out['alive']).sum(0).mean()), loss=float(l),
                       log_std=np.asarray(agent['log_std']).round(2).tolist(), seconds=round(time.time() - t0, 2)))


if __name__ == '__main__':
    main()
