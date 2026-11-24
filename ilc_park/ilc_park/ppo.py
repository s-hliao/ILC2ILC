"""PPO for the drift-parking policy in the nominal car (or a domain-randomized family: the PPO+DR baseline).

Every iteration: --lanes episodes of T_MAX seconds, each released from a random steady drift (the drift table; half the
lanes mirrored to the right-turning circle) toward a random feasible slot (--slots, from scripts/feasibility.py, with
jitter). Reward per 25 ms period:
  potential shaping  Phi(s') - Phi(s), Phi = -(|p| / 0.5 m + |dpsi| / 0.5 rad)      (p, dpsi: the car in the slot frame)
  drift              + w_drift * min(|beta| / beta_min, 1) while V > v_drift            (the entry must be a drift)
  smoothness         - w_du |a_t - a_{t-1}|^2
  park               + w_park * exp(-(|p| / 0.15)^2 - (dpsi / 0.2)^2) * [V < 0.1]       (every period it sits parked)
Deterministic evaluation on the held-out slots (--eval-slots) after every --eval-every iterations.

    F1T_MU=0.6 python -m ilc_park.ppo --slots results/feasibility_R1.5_b15.json --out runs/ppo_nominal_s0 --seed 0
"""
import argparse
import json
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ilc_park import sim

W = dict(shape=1.0, drift=0.05, du=0.02, park=0.2, beta_min_deg=10.0, v_drift=1.2)


# ---------------------------------------------------------------------------------------------------------------
# networks
# ---------------------------------------------------------------------------------------------------------------

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


def policy_mean(params, o):
    """The deployed policy: the actor's mean, squashed to the action box."""
    return jnp.tanh(mlp(params['actor'], o))


def init_agent(key, hidden=128, log_std=-1.0):
    ka, kc = jax.random.split(key)
    return dict(actor=init_mlp(ka, [sim.NOBS, hidden, hidden, sim.NACT]),
                critic=init_mlp(kc, [sim.NOBS, hidden, hidden, 1], scale_last=1.0),
                log_std=jnp.full((sim.NACT,), log_std))


# ---------------------------------------------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------------------------------------------

def load_slots(path):
    rows = [r for r in json.load(open(path)) if r['success']]
    return np.array([[r['dx'], r['dy'], np.radians(r['dpsi'])] for r in rows])


def sample_episodes(rng, n, drift_states, slots, jitter=(0.15, 0.15, np.radians(10))):
    """Release states (n, 9) and slots (n, 3): a random drift, a random feasible slot (+ jitter), half mirrored."""
    x0 = drift_states[rng.integers(len(drift_states), size=n)].copy()
    sl = slots[rng.integers(len(slots), size=n)] + rng.uniform(-1, 1, (n, 3)) * np.asarray(jitter)
    mirror = rng.random(n) < 0.5
    x0[mirror] *= sim.MIRROR_STATE
    sl[mirror] *= np.array([1, -1, -1])
    return x0, sl


MU_GRID = (0.36, 0.42, 0.48, 0.54, 0.60, 0.68, 0.76)


def dr_batch(rng, n, mu_tables, slots):
    """PPO+DR lanes: per-lane parameters and release states from that lane's own friction's drift table."""
    from ilc_park import car_model as cm
    mi = rng.integers(len(MU_GRID), size=n)
    mu = np.asarray(MU_GRID)[mi]
    x0 = np.stack([mu_tables[k][rng.integers(len(mu_tables[k]))] for k in mi])
    sl = slots[rng.integers(len(slots), size=n)] + rng.uniform(-1, 1, (n, 3)) * np.array([0.15, 0.15, np.radians(10)])
    mirror = rng.random(n) < 0.5
    x0[mirror] *= sim.MIRROR_STATE
    sl[mirror] *= np.array([1, -1, -1])
    ms = rng.uniform(0.75, 1.25, n)
    ks = rng.uniform(0.8, 1.2, n)
    p = sim.nominal_params(n, muf=mu, mur=mu, m=cm.NOMINAL['m'] * ms, Iz=cm.NOMINAL['Iz'] * ms,
                           Cf=cm.NOMINAL['Cf'] * ks, Cr=cm.NOMINAL['Cr'] * ks,
                           kt=cm.NOMINAL['kt'] * rng.uniform(0.8, 1.2, n), tau_s=rng.uniform(0.03, 0.08, n))
    return p, x0, sl


def rewards(xs, as_, slots):
    """Per-period rewards (T, n) of a flown batch (states (T+1, n, 9), actions (T, n, 2))."""
    def phi(x):
        pos, dpsi = jax.vmap(jax.vmap(sim.slot_frame, (0, 0)), (0, None))(x, slots)
        return -(jnp.linalg.norm(pos, axis=-1) / 0.5 + jnp.abs(dpsi) / 0.5), pos, dpsi
    ph, pos, dpsi = phi(xs)
    r = W['shape'] * (ph[1:] - ph[:-1])
    V = sim.speed(xs[1:])
    beta = jnp.abs(sim.sideslip(xs[1:]))
    r += W['drift'] * jnp.minimum(beta / np.radians(W['beta_min_deg']), 1.0) * (V > W['v_drift'])
    da = jnp.diff(as_, axis=0, prepend=as_[:1])
    r -= W['du'] * jnp.sum(da ** 2, -1)
    parked = jnp.exp(-(jnp.linalg.norm(pos[1:], axis=-1) / 0.15) ** 2 - (dpsi[1:] / 0.2) ** 2)
    r += W['park'] * parked * (V < 0.1)
    return r


def episode_metrics(xs, slots):
    """Final pose error (m, deg) and final speed (m/s) per lane, and the peak sideslip while fast."""
    pos, dpsi = jax.vmap(sim.slot_frame)(xs[-1], slots)
    V = sim.speed(xs)
    beta = jnp.where(V > W['v_drift'], jnp.abs(sim.sideslip(xs)), 0.0)
    return dict(pos_err=jnp.linalg.norm(pos, axis=-1), psi_err_deg=jnp.degrees(jnp.abs(dpsi)), v_end=V[-1],
                beta_peak_deg=jnp.degrees(beta.max(0)))


# ---------------------------------------------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------------------------------------------

def gaussian_logp(mean, log_std, a):
    return jnp.sum(-0.5 * ((a - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * np.log(2 * np.pi), -1)


def gae(values, r, gamma, lam):
    """values (T+1, n) (last = 0 bootstrap at the episode end), r (T, n) -> advantages, returns."""
    def back(carry, inp):
        v, v1, rr = inp
        d = rr + gamma * v1 - v
        carry = d + gamma * lam * carry
        return carry, carry
    _, adv = jax.lax.scan(back, jnp.zeros_like(r[0]), (values[:-1], values[1:], r), reverse=True)
    return adv, adv + values[:-1]


def make_update(opt, clip, ent):
    @jax.jit
    def update(agent, opt_state, o, a, logp_old, adv, ret):
        def loss(ag):
            mean = jnp.tanh(mlp(ag['actor'], o))
            logp = gaussian_logp(mean, ag['log_std'], a)
            ratio = jnp.exp(logp - logp_old)
            pg = -jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip, 1 + clip) * adv))
            v = mlp(ag['critic'], o)[:, 0]
            vl = 0.5 * jnp.mean((v - ret) ** 2)
            return pg + vl - ent * jnp.sum(ag['log_std'])
        l, g = jax.value_and_grad(loss)(agent)
        upd, opt_state = opt.update(g, opt_state, agent)
        return optax.apply_updates(agent, upd), opt_state, l
    return update


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--slots', required=True, help='feasibility json (training slots: the feasible ones)')
    ap.add_argument('--eval-slots', default=None, help='held-out slots json (default: a fixed draw of --slots)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--iters', type=int, default=600)
    ap.add_argument('--lanes', type=int, default=4096)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--mb', type=int, default=32768)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--ent', type=float, default=1e-3)
    ap.add_argument('--R0', type=float, nargs='+', default=[1.3, 1.5, 1.7])
    ap.add_argument('--beta0', type=float, nargs='+', default=[-12.0, -15.0, -18.0])
    ap.add_argument('--eval-every', type=int, default=25)
    ap.add_argument('--dr', action='store_true', help='PPO+DR: friction from MU_GRID (each with its own drift table: the\n'
                    'lanes are released from their own car\'s steady drift), mass +-25 %%, tire stiffness / motor constant +-20 %%,\n'
                    'servo time constant 0.03-0.08 s')
    ap.add_argument('--real-mu', type=float, default=0.44, help='the stand-in real car (evaluation only)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(vars(a), weights=W), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)

    rng = np.random.default_rng(a.seed)
    key = jax.random.PRNGKey(a.seed)
    drift_states, drift_meta = sim.drift_table(a.R0, a.beta0)
    from ilc_park import car_model as cm
    mu_tables = [sim.drift_table(a.R0, a.beta0, dict(cm.NOMINAL, muf=mu, mur=mu))[0] for mu in MU_GRID] if a.dr else None
    real_states = sim.drift_table(a.R0, a.beta0, dict(cm.NOMINAL, muf=a.real_mu, mur=a.real_mu))[0]
    slots = load_slots(a.slots)
    eval_slots = load_slots(a.eval_slots) if a.eval_slots else slots
    T = int(round(sim.T_MAX / sim.DT))
    p = sim.nominal_params(a.lanes)

    key, k0 = jax.random.split(key)
    agent = init_agent(k0)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr))
    opt_state = opt.init(agent)
    update = make_update(opt, a.clip, a.ent)
    actor_fn = lambda params, o: jnp.tanh(mlp(params, o))          # noqa: E731

    @jax.jit
    def values_of(critic, o):
        return mlp(critic, o)[..., 0]

    # the evaluation set: every held-out slot from every drift, both directions, deterministic
    ex0 = np.concatenate([np.repeat(drift_states, len(eval_slots), 0)] * 2)
    esl = np.concatenate([np.tile(eval_slots, (len(drift_states), 1))] * 2)
    half = len(ex0) // 2
    ex0[half:] *= sim.MIRROR_STATE
    esl[half:] *= np.array([1, -1, -1])
    ep = sim.nominal_params(len(ex0))
    # the same held-out set in the stand-in real car, released from ITS steady drifts
    rx0 = np.concatenate([np.repeat(real_states, len(eval_slots), 0)] * 2)
    rsl = np.concatenate([np.tile(eval_slots, (len(real_states), 1))] * 2)
    rx0[len(rx0) // 2:] *= sim.MIRROR_STATE
    rsl[len(rsl) // 2:] *= np.array([1, -1, -1])
    rp = sim.nominal_params(len(rx0), muf=a.real_mu, mur=a.real_mu)

    log = open(os.path.join(a.out, 'log.jsonl'), 'a')
    for it in range(a.iters + 1):
        if it % a.eval_every == 0:
            xs, _, _ = sim.rollout(agent['actor'], jnp.asarray(ex0), jnp.asarray(esl), actor_fn, ep,
                                   jnp.zeros((T, len(ex0), 2)))
            m = {k: np.asarray(v) for k, v in episode_metrics(xs, jnp.asarray(esl)).items()}
            parked = (m['pos_err'] < 0.15) & (m['psi_err_deg'] < 10) & (m['v_end'] < 0.1)
            line = dict(it=it, eval_pos_err=float(np.mean(m['pos_err'])), eval_psi_err=float(np.mean(m['psi_err_deg'])),
                        eval_parked=float(parked.mean()), eval_beta_peak=float(np.median(m['beta_peak_deg'])),
                        eval_v_end=float(np.mean(m['v_end'])))
            xr, _, _ = sim.rollout(agent['actor'], jnp.asarray(rx0), jnp.asarray(rsl), actor_fn, rp,
                                   jnp.zeros((T, len(rx0), 2)))
            mr = {k: np.asarray(v) for k, v in episode_metrics(xr, jnp.asarray(rsl)).items()}
            line.update(real_pos_err=float(np.mean(mr['pos_err'])), real_psi_err=float(np.mean(mr['psi_err_deg'])),
                        real_parked=float(((mr['pos_err'] < 0.15) & (mr['psi_err_deg'] < 10) & (mr['v_end'] < 0.1)).mean()))
            print(json.dumps(line), flush=True)
            log.write(json.dumps(line) + '\n'); log.flush()
            np.savez(os.path.join(a.out, f'policy_it{it:04d}.npz'),
                     **{f'actor_{i}_{n}': np.asarray(t) for i, l in enumerate(agent['actor']) for n, t in zip('wb', l)})
        if it == a.iters:
            break
        t0 = time.time()
        if a.dr:
            p, x0, sl = dr_batch(rng, a.lanes, mu_tables, slots)
        else:
            x0, sl = sample_episodes(rng, a.lanes, drift_states, slots)
        key, kn = jax.random.split(key)
        noise = jax.random.normal(kn, (T, a.lanes, 2)) * jnp.exp(agent['log_std'])
        xs, os_, as_ = sim.rollout(agent['actor'], jnp.asarray(x0), jnp.asarray(sl), actor_fn, p, noise)
        r = rewards(xs, as_, jnp.asarray(sl))
        v = values_of(agent['critic'], os_)
        v = jnp.concatenate([v, jnp.zeros_like(v[:1])])
        adv, ret = gae(v, r, a.gamma, a.lam)
        mean = jnp.tanh(mlp(agent['actor'], os_))
        logp = gaussian_logp(mean, agent['log_std'], as_)
        flat = lambda z: z.reshape((-1,) + z.shape[2:])            # noqa: E731
        O, A, LP, ADV, RET = map(flat, (os_, as_, logp, adv, ret))
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        N = O.shape[0]
        for _ in range(a.epochs):
            key, kp = jax.random.split(key)
            perm = jax.random.permutation(kp, N)
            for s in range(0, N - a.mb + 1, a.mb):
                idx = perm[s:s + a.mb]
                agent, opt_state, l = update(agent, opt_state, O[idx], A[idx], LP[idx], ADV[idx], RET[idx])
        if it % 5 == 0:
            line = dict(it=it, reward=float(r.sum(0).mean()), loss=float(l), log_std=np.asarray(agent['log_std']).tolist(),
                        seconds=round(time.time() - t0, 2))
            print(json.dumps(line), flush=True)
            log.write(json.dumps(line) + '\n'); log.flush()


if __name__ == '__main__':
    main()
