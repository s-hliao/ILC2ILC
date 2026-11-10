#!/usr/bin/env python3
"""
dagger_car.py --oracle runs/fig1/rma_s<s> --out runs/fig1/dagger_s<s> (2026-11-07 ~12:30, DIAGNOSTIC; user: "yes, do it"; rule in
runs/screen/FADA_AUDIT_1107.md written before results): is FADA's zero-shot edge over RMA from DAgger distillation of the privileged
teacher itself, or from FADA's planner-IDM / history? The student here is ONE memoryless MLP pi(obs) -> a, the RMA actor's size
(55 -> 256 -> 256 -> 2, tanh), no z, no history, no planner / IDM. Everything else is fada_car_tf.py's source recipe: oracle = the RMA teacher
rma_s<s>/policy_final.npz reading the exact z, sim_jax DR scale 1, rma_s<s>'s 6 plans, 8 iterations (it 0 oracle rollouts, then the
student's), 512 lanes x 160 periods, 2 suboptimal-checkpoint rollouts per iteration, 2000 Adam steps of batch 512 at lr 3e-4 per
iteration on the aggregate; labels = the oracle's action at EVERY visited state (DAgger). Saves OUT/it<k>_policy.npz (plain MLP:
policy_runtime / fig1_eval read it) + train.log.
"""
import argparse
import glob
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--oracle', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--iters', type=int, default=8)
ap.add_argument('--lanes', type=int, default=512)
ap.add_argument('--T', type=int, default=160)
ap.add_argument('--steps', type=int, default=2000)
ap.add_argument('--batch', type=int, default=512)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--data', default='fada', choices=('fada', 'rma'), help='(2026-11-07 ~13:55) rma: EXACTLY fadaR / fadaR1 data (fada_car_tf.py --data rma = '
                'ppo_dr.py phase 2): on-policy only (it 0 the teacher), --iters 150, buffer of the last 8 iterations, every 4th period, --steps 200')
ap.add_argument('--prev-a', type=int, default=0, help='1: the student reads [obs, a_{t-1}] (fadaR1\'s inputs: o_t + the previous-action token)')
a = ap.parse_args()
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax                                       # noqa: E402
import jax.numpy as jnp                          # noqa: E402
import numpy as np                               # noqa: E402
import optax                                     # noqa: E402

import bank                                      # noqa: E402
import car_model as cm                           # noqa: E402
import sim_jax as sj                             # noqa: E402
from consts import FAIL_EY, FAIL_EPSI            # noqa: E402

ZKEYS = ['m', 'muf', 'Cf', 'kt', 'tau_s', 'd_off', 'd_delay', 'i_delay']                         # ppo_dr.py / fada_car_tf.py
ZNOM = jnp.array([cm.NOMINAL['m'], cm.NOMINAL['muf'], 250.0, 0.0258, 0.055, 0.0, 0.5, 0.5])
ZSC = jnp.array([0.6, 0.12, 50.0, 0.005, 0.025, 0.03, 0.5, 0.5])


def zvec(p):
    return (jnp.stack([p[k].astype(jnp.float32) for k in ZKEYS], -1) - ZNOM) / ZSC


def load_actor(f):
    d = np.load(f)
    return [(jnp.array(d[f'W{i}']), jnp.array(d[f'b{i}'])) for i in range(3)]


def main():
    os.makedirs(a.out, exist_ok=True)
    src = json.load(open(os.path.join(a.oracle, 'args.json')))
    plans = bank.load_bank(src['plans'])
    P = sj.Plans(plans, src['plans'])
    oracle = load_actor(os.path.join(a.oracle, 'policy_final.npz'))
    subs = [load_actor(f) for f in sorted(glob.glob(os.path.join(a.oracle, 'policy_it*.npz')))
            if not f.endswith(f"policy_it{int(src['iters']):03d}.npz")]
    rng = np.random.default_rng(a.seed)
    key = jax.random.PRNGKey(a.seed)
    k0 = jax.random.split(jax.random.PRNGKey(a.seed + 2), 3)
    sizes = [sj.NOBS + 2 * a.prev_a, 256, 256, 2]
    stu = [(jax.random.normal(k, (i, o)) * np.sqrt(1.0 / i), jnp.zeros(o)) for k, (i, o) in zip(k0, zip(sizes[:-1], sizes[1:]))]
    stu[-1] = (stu[-1][0] * 0.01, stu[-1][1])
    log = open(os.path.join(a.out, 'train.log'), 'a')
    T0 = time.time()

    def say(**kw):
        kw['t'] = round(time.time() - T0, 1)
        print(json.dumps(kw), flush=True); log.write(json.dumps(kw) + '\n'); log.flush()

    def make_fly(kind):
        def lane(x0, i0, pid, p, actor):
            z = zvec(p)

            def body(c, _):
                x, i, failed, buf, prev = c
                i_new = sj.project(P, pid, i, x[:2])
                obs, err = sj.features(P, pid, i_new, x)
                si = jnp.concatenate([obs, prev]) if a.prev_a else obs
                astar = sj.mlp(oracle, jnp.concatenate([obs, z]))
                mu = sj.mlp(actor, si) if kind == 'student' else sj.mlp(actor, jnp.concatenate([obs, z]))
                a_phys = mu * sj.A_SCALE + sj.A_OFF
                buf_new = jnp.stack([a_phys, buf[0]])
                d_cmd = jnp.where(p['d_delay'] > 0, buf[0][0], a_phys[0]) + p['d_off']
                i_cmd = jnp.where(p['i_delay'] > 0, buf[0][1], a_phys[1])
                x_new = sj.step(x, jnp.stack([d_cmd, i_cmd]), p)
                bad = (jnp.abs(err[0]) > FAIL_EY) | (jnp.abs(err[1]) > FAIL_EPSI) | (x[3] < 0.3) | ~jnp.all(jnp.isfinite(x_new))
                fn = failed | bad
                return (jnp.where(fn, x, x_new), i_new, fn, buf_new, mu), (si, astar, fn)
            _, out = jax.lax.scan(body, (x0, i0, False, jnp.zeros((2, 2)), jnp.zeros(2)), None, length=a.T)
            return out
        return jax.jit(jax.vmap(lane, in_axes=(0, 0, 0, 0, None)))
    fly_student, fly_actor = make_fly('student'), make_fly('actor')

    def collect(kind, actor):
        nonlocal key
        pid = rng.integers(0, len(src['plans']), a.lanes)
        s0 = rng.uniform(0, 1, a.lanes) * np.asarray(P.L)[pid]
        xs, ids = [], []
        for k in range(a.lanes):
            x0, i0 = sj.start_states(P, jax.random.PRNGKey(int(rng.integers(1 << 30))), [pid[k]], [s0[k]], pert=1.0)
            xs.append(x0[0]); ids.append(i0[0])
        key, k1 = jax.random.split(key)
        p = sj.dr_params(k1, a.lanes, 1.0)
        fly = fly_student if kind == 'student' else fly_actor
        obs, astar, fn = fly(jnp.stack(xs), jnp.stack(ids), jnp.array(pid), p, actor)
        fn = np.asarray(fn)
        live = np.concatenate([np.ones((a.lanes, 1), bool), ~fn[:, :-1]], 1)            # states reached before the failure
        laps = float(np.sum(live.sum(1) * sj.DT / np.asarray(P.lap_time)[pid]))
        if a.data == 'rma':                           # ppo_dr.py phase 2: every 4th period
            live = live & (np.arange(a.T)[None] % 4 == 0)
        return dict(o=np.asarray(obs)[live], a=np.asarray(astar)[live], alive=float((~fn[:, -1]).mean())), laps

    opt = optax.adam(a.lr)
    st = opt.init(stu)

    @jax.jit
    def step(q, s, o, t):
        l, g = jax.value_and_grad(lambda w: jnp.mean((sj.mlp(w, o) - t) ** 2))(q)
        u, s = opt.update(g, s, q)
        return optax.apply_updates(q, u), s, l

    O, A, sim_laps = [], [], 0.0
    for it in range(a.iters):
        kind = 'oracle' if it == 0 else 'student'
        blk, lp = collect(kind, oracle if kind == 'oracle' else stu); sim_laps += lp
        O.append(blk['o']); A.append(blk['a'])
        if a.data == 'rma':                           # on-policy only, buffer of the last 8 iterations
            O, A = O[-8:], A[-8:]
        for j in range(2 if a.data == 'fada' else 0):                            # suboptimal: 2x, from random intermediate checkpoints (fada_car_tf.py's)
            b2, lp2 = collect('actor', subs[rng.integers(len(subs))]); sim_laps += lp2
            O.append(b2['o']); A.append(b2['a'])
        Oc, Ac = jnp.asarray(np.concatenate(O), jnp.float32), jnp.asarray(np.concatenate(A), jnp.float32)
        ls = []
        for s_ in range(a.steps):
            b = jnp.asarray(rng.integers(0, len(Ac), a.batch))
            stu, st, l = step(stu, st, Oc[b], Ac[b]); ls.append(float(l))
        if a.data == 'rma' and (it + 1) not in (40, 75, a.iters):
            continue
        np.savez(os.path.join(a.out, f'it{it + 1}_policy.npz'), **{f'W{i}': np.asarray(W) for i, (W, b) in enumerate(stu)},
                 **{f'b{i}': np.asarray(b) for i, (W, b) in enumerate(stu)}, **(dict(mode='mlp_prev') if a.prev_a else {}))
        say(it=it + 1, rollout=kind, alive_end=round(blk['alive'], 3), loss=[round(np.mean(ls[:50]), 5), round(np.mean(ls[-50:]), 5)],
            samples=int(len(Ac)), sim_laps=round(sim_laps, 1))


if __name__ == '__main__':
    main()
