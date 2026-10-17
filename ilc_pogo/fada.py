#!/usr/bin/env python3
"""
fada.py: the FADA-style baseline on the pogo stick (the car's fada_car.py / the quadruped's fada_train.py +
fada_adapt.py). Policy a_t = IDM(o_t, P(o_t)):
  planner  P(o_t) -> d_{t+H}: the state deviation H control periods ahead that the SOURCE policy reaches, d = the
           observation's continuous state entries --dims (default 0 1 2 4 5: z - h*, vx - v*, vz, the foot offset
           x - xf and the leg compression L - l0 in stance; the binary stance flag is left out); trained on the
           source's rollouts over the DR family without action noise (FADA's recipe: the DR'd source; frozen after)
  IDM      I(o_t, d_{t+H}) -> a_t: the action that reaches a given deviation; rollouts of the source plus smooth
           action offsets (rms --a-off) over the DR family for coverage, the action as flown; initialized from the
           source's weights (the extra d inputs at zero weight)
  adapt    on each real pogo stick, iters x the 6 training goals (3 s trials from mildly perturbed drops): every real
           transition relabelled in hindsight (o_t, d_{t+H} as reached) -> a_t as flown -- reward-free, the task
           error is never used -- and a LoRA of rank --rank on each IDM layer (base frozen) fitted to all of the
           stick's transitions so far
Writes <out>_zs (the IDM before adaptation) and <out>_hw{6 iters} arms in the common summary format.
    python fada.py --out runs/hw/fada --init runs/nom_s0/policy_final.npz --iters 4 --gpu 3
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='runs/hw/fada', help='arm dirs <out>_zs and <out>_hw<6 iters>')
ap.add_argument('--init', default='runs/ppo_dr_s0/policy_final.npz', help='the source policy (planner targets, IDM init); '
                'FADA\'s recipe: a DR-trained source (PPO+DR)')
ap.add_argument('--target', default='hstep', choices=('hstep', 'apex'), help='the planner / IDM target: the obs deviation '
                'H periods ahead (hstep), or the [apex error, speed error] at the first flight apex after the next stance '
                '(apex: the keyframe every action until then can move -- the pogo\'s actions act only at touchdown / in '
                'stance, so most a_t do not affect d_{t+H})')
ap.add_argument('--real-noise', type=float, default=0.0, help='OU action noise (rms) on the real training trials: '
                'excitation for the hindsight relabelling (same trials, same seconds)')
ap.add_argument('--idm-clean', type=int, default=0, help='also train the IDM on the noise-free source rollouts (the '
                'states and targets it is queried at; the noisy ones give coverage)')
ap.add_argument('--sim-check', type=int, default=1, help='score IDM(o, P(o)) against its source in DR sim (128 lanes, 10 s)')
ap.add_argument('--robots', nargs='+', default=None)
ap.add_argument('--dims', type=int, nargs='+', default=[0, 1, 2, 4, 5])
ap.add_argument('--H', type=int, default=4, help='the deviation H control periods ahead (40 ms)')
ap.add_argument('--dr', type=float, default=1.0)
ap.add_argument('--data-iters', type=int, default=30)
ap.add_argument('--lanes', type=int, default=256)
ap.add_argument('--T', type=int, default=300)
ap.add_argument('--a-off', type=float, default=0.15)
ap.add_argument('--train-steps', type=int, default=20000)
ap.add_argument('--iters', type=int, default=4)
ap.add_argument('--trial-seconds', type=float, default=3.0)
ap.add_argument('--rank', type=int, default=4)
ap.add_argument('--adapt-steps', type=int, default=500)
ap.add_argument('--eval-runs', type=int, default=4)
ap.add_argument('--eval-seconds', type=float, default=10.0)
ap.add_argument('--eval-goals', type=int, default=8)
ap.add_argument('--seed', type=int, default=0)
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

import common as C                               # noqa: E402
import fada_np                                   # noqa: E402
import pogo_jax as pj                            # noqa: E402
import real_pogo as rp                           # noqa: E402

ROBOTS = a.robots or rp.DEFAULT_ROBOTS
DIMS = np.array(a.dims) if a.target == 'hstep' else np.array([-1, -1])     # apex: [apex error, speed error]
ND, H = len(DIMS), a.H


def ou_noise(rng, n, T, rms, rho=0.9):
    e = rng.standard_normal((T, n, 2)) * rms * np.sqrt(1 - rho ** 2)
    out = np.zeros_like(e)
    out[0] = rng.standard_normal((n, 2)) * rms
    for t in range(1, T):
        out[t] = rho * out[t - 1] + e[t]
    return out.transpose(1, 0, 2)


def pairs(obs, act, alive):
    """(o_t, d, a_t) with both ends before a fall; d = the obs deviation H periods ahead (hstep) or the apex keyframe."""
    act = np.clip(act, -1.0, 1.0)                    # the action as applied (the model clips)
    if a.target == 'hstep':
        ok = alive[:, :-H] & alive[:, H:]
        return obs[:, :-H][ok], obs[:, H:][..., DIMS][ok], act[:, :-H][ok]
    O, D, A = [], [], []
    n, T = obs.shape[:2]
    for i in range(n):
        st, vz = obs[i, :, 3] > 0.5, obs[i, :, 2]
        apx = np.where(~st[1:] & (vz[:-1] > 0) & (vz[1:] <= 0))[0] + 1          # flight apexes
        sts = np.where(st)[0]
        if len(apx) == 0 or len(sts) == 0:
            continue
        t = np.arange(T)
        s_next = sts[np.minimum(np.searchsorted(sts, t), len(sts) - 1)]         # the next stance period (>= t)
        has_s = np.searchsorted(sts, t) < len(sts)
        k = np.searchsorted(apx, s_next, side='right')                          # the first apex after it
        ok = has_s & (k < len(apx))
        ta = apx[np.minimum(k, len(apx) - 1)]
        ok &= alive[i] & alive[i, ta]
        d = np.stack([obs[i, ta, 0] + 0.5 * obs[i, ta, 2] ** 2 / C.GRAV, obs[i, ta, 1]], -1)
        O.append(obs[i][ok]); D.append(d[ok]); A.append(act[i][ok])
    if not O:
        return np.zeros((0, obs.shape[-1])), np.zeros((0, 2)), np.zeros((0, 2))
    return np.concatenate(O), np.concatenate(D), np.concatenate(A)


def fit(params, X, Y, steps, lr, rng, mb=8192):
    opt = optax.adam(lr)
    st = opt.init(params)

    @jax.jit
    def step(ps, st, x, y):
        l, g = jax.value_and_grad(lambda q: jnp.mean((pj.mlp(q, x) - y) ** 2))(ps)
        u, st = opt.update(g, st, ps)
        return optax.apply_updates(ps, u), st, l
    Xj, Yj = jnp.asarray(X, jnp.float32), jnp.asarray(Y, jnp.float32)
    for s in range(steps):
        b = jnp.asarray(rng.integers(0, len(X), min(mb, len(X))))
        params, st, l = step(params, st, Xj[b], Yj[b])
    return params, float(l)


def sim_check(src, idm, planner, rng, n=128, T=1000):
    """IDM(o, P(o)) vs its source in DR sim: success / apex / speed (common.hop_metrics), the same lanes."""
    G = C.GOAL_LO + (C.GOAL_HI - C.GOAL_LO) * rng.uniform(size=(n, 2))
    k1, k2 = jax.random.split(jax.random.PRNGKey(123))
    x0 = pj.start_states(k1, G, rng.uniform(0, 1, n))
    p = pj.dr_params(k2, n, a.dr)
    out = {}
    for nm, fn in (('source', lambda o: pj.mlp(src, o)),
                   ('fada', lambda o: pj.mlp(idm, jnp.concatenate([o, pj.mlp(planner, o)], -1)))):
        r = jax.jit(lambda x0, g, p: pj.rollout(fn, T, x0, g, p))(x0, jnp.asarray(G), p)
        ms = [C.hop_metrics(np.asarray(r['x'][i]), G[i], bool(r['failed_end'][i])) for i in range(n)]
        out[nm] = dict(success=float(np.mean([m['success'] for m in ms])), fail=float(np.mean([m['failed'] for m in ms])),
                       apex=float(np.nanmean([m['rms_apex'] if m['rms_apex'] is not None else np.nan for m in ms])),
                       v_err=float(np.nanmean([abs(m['v_err']) if m['v_err'] is not None else np.nan for m in ms])))
    return out


def robot_param_arrays(keys):
    ps = [rp.robot_params(r) for r in keys]
    return {k: np.array([p[k] for p in ps]) for k in ps[0]}


def write_arm(d, pols, trials, extra, t0):
    os.makedirs(d, exist_ok=True)
    goals = C.eval_goals(a.eval_goals)
    for r in ROBOTS:
        fada_np.save(os.path.join(d, f'{r}_policy.npz'), pols[r].idm, pols[r].planner, DIMS, H, target=a.target)
    runs = rp.evaluate({r: pols[r] for r in ROBOTS}, goals, ROBOTS, runs=a.eval_runs, seconds=a.eval_seconds)
    summ = dict(args=vars(a), method="fada", eval_set=C.EVAL_SET, eval={r: rp.summarize(runs[r]) for r in ROBOTS}, runs=runs,
                real_trials={r: len(trials.get(r, [])) for r in ROBOTS}, goals_eval=goals.tolist(), trials=trials,
                seconds=time.time() - t0, **extra)
    json.dump(summ, open(os.path.join(d, 'summary.json'), 'w'), indent=1)
    print('ARM', d, json.dumps({r: (round(summ['eval'][r]['success'], 2), summ['eval'][r]['rms_apex'] and
                                   round(summ['eval'][r]['rms_apex'] * 100, 2), round(summ['eval'][r]['fail'], 2))
                                for r in ROBOTS}), flush=True)


def main():
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    key = jax.random.PRNGKey(a.seed)
    while not os.path.exists(a.init):                    # the source policy may still be training
        time.sleep(30)
    time.sleep(2)
    src = [(jnp.array(W), jnp.array(b)) for W, b in C.load_policy(a.init)]
    sim_seconds = 0.0

    @jax.jit
    def fly(x0, g, p, nz):
        return pj.rollout(lambda o: pj.mlp(src, o), a.T, x0, g, p, nz)

    PX, PY, IX, IY = [], [], [], []
    for it in range(a.data_iters):
        for noisy in (False, True):
            G = C.GOAL_LO + (C.GOAL_HI - C.GOAL_LO) * rng.uniform(size=(a.lanes, 2))
            key, k1, k2 = jax.random.split(key, 3)
            x0 = pj.start_states(k1, G, rng.uniform(0, 1, a.lanes))
            p = pj.dr_params(k2, a.lanes, a.dr) if a.dr > 0 else pj.nominal_params(a.lanes)
            nz = ou_noise(rng, a.lanes, a.T, a.a_off) if noisy else np.zeros((a.lanes, a.T, 2))
            out = fly(x0, jnp.asarray(G), p, jnp.asarray(nz, jnp.float32))
            sim_seconds += a.lanes * a.T * C.DT
            o, d, act = pairs(np.asarray(out['obs']), np.asarray(out['a']), ~np.asarray(out['failed']))
            if noisy or a.idm_clean:
                IX.append(np.concatenate([o, d], -1)); IY.append(act)
            if not noisy:
                PX.append(o); PY.append(d)
    PX, PY, IX, IY = map(np.concatenate, (PX, PY, IX, IY))
    print(json.dumps(dict(event='data', planner=len(PX), idm=len(IX), sim_hours=sim_seconds / 3600, t=time.time() - t0)),
          flush=True)
    hidden = src[0][0].shape[1]
    key, k0 = jax.random.split(key)
    planner = pj.init_mlp(k0, [C.NOBS, hidden, hidden, ND])
    planner, lp = fit(planner, PX, PY, a.train_steps, 3e-4, rng)
    W0 = jnp.concatenate([src[0][0], jnp.zeros((ND, hidden))], 0)            # the IDM starts as the source policy
    idm = [(W0, src[0][1])] + list(src[1:])
    idm, li = fit(idm, IX, IY, a.train_steps, 3e-4, rng)
    print(json.dumps(dict(event='fit', planner_loss=lp, planner_var=float(np.mean(np.var(PY, 0))), idm_loss=li,
                          idm_var=float(np.mean(np.var(IY, 0))), t=time.time() - t0)), flush=True)
    base = [(np.asarray(W), np.asarray(b)) for W, b in idm]
    pl_np = [(np.asarray(W), np.asarray(b)) for W, b in planner]
    extra = dict(sim_hours=sim_seconds / 3600, planner_loss=lp, idm_loss=li, target=a.target)
    if a.sim_check:
        extra['sim_check'] = sim_check(src, idm, planner, rng)
        print(json.dumps(dict(event='sim_check', **extra['sim_check'])), flush=True)
    zs_pols = {r: fada_np.FADAPolicy(base, pl_np, DIMS, H) for r in ROBOTS}
    write_arm(f'{a.out}_zs', zs_pols, {}, extra, t0)

    # ---- adaptation: LoRA on every IDM layer, hindsight relabelling of the real transitions
    def lora_init(k):
        out = []
        for W, b in base:
            k, kk = jax.random.split(k)
            out.append((jax.random.normal(kk, (W.shape[0], a.rank)) * 0.01, jnp.zeros((a.rank, W.shape[1]))))
        return out
    merged = lambda lo: [(W + np.asarray(A) @ np.asarray(B), b) for (W, b), (A, B) in zip(base, lo)]
    opt2 = optax.adam(1e-3)
    base_j = [(jnp.array(W), jnp.array(b)) for W, b in base]

    @jax.jit
    def astep(lo, st, x, y):
        def loss(lo):
            ps = [(W + A @ B, b) for (W, b), (A, B) in zip(base_j, lo)]
            return jnp.mean((pj.mlp(ps, x) - y) ** 2)
        l, g = jax.value_and_grad(loss)(lo)
        u, st = opt2.update(g, st, lo)
        return optax.apply_updates(lo, u), st, l

    lora = {r: lora_init(jax.random.PRNGKey(7)) for r in ROBOTS}
    st2 = {r: opt2.init(lora[r]) for r in ROBOTS}
    data = {r: ([], []) for r in ROBOTS}
    trials = {r: [] for r in ROBOTS}
    T = int(round(a.trial_seconds / C.DT))
    G6 = C.TRAIN_GOALS
    for it in range(a.iters):
        keys = [(r, gi) for r in ROBOTS for gi in range(len(G6))]
        goals = np.array([G6[gi] for r, gi in keys])
        x0 = rp.start_states(goals, 0.2, rng)
        pols = {r: fada_np.FADAPolicy(merged(lora[r]), pl_np, DIMS, H) for r in ROBOTS}

        class Mixed:                                     # each run flies its own robot's adapted policy
            def reset(self, n):
                pass

            def __call__(self, o, x, g):
                out = np.zeros((len(o), 2))
                for r in ROBOTS:
                    m = np.array([k[0] == r for k in keys])
                    out[m] = pols[r](o[m], x[m], g[m])
                return out
        nz = ou_noise(rng, len(keys), T, a.real_noise) if a.real_noise > 0 else None
        res = rp.simulate(Mixed(), goals, x0, robot_param_arrays([k[0] for k in keys]), T, noise=nz)
        alive = ~res['failed']
        for n, (r, gi) in enumerate(keys):
            m = C.hop_metrics(res['x'][n], goals[n], res['failed_end'][n])
            trials[r].append(dict(it=it, goal=goals[n].tolist(), failed=m['failed'], rms_apex=m['rms_apex'],
                                  v_err=m['v_err']))
            o, d, act = pairs(res['obs'][n:n + 1], res['a'][n:n + 1], alive[n:n + 1])
            data[r][0].append(np.concatenate([o, d], -1)); data[r][1].append(act)
        for r in ROBOTS:
            Xa = jnp.asarray(np.concatenate(data[r][0]), jnp.float32)
            Ya = jnp.asarray(np.concatenate(data[r][1]), jnp.float32)
            for s in range(a.adapt_steps):
                b = jnp.asarray(rng.integers(0, Xa.shape[0], min(2048, Xa.shape[0])))
                lora[r], st2[r], l = astep(lora[r], st2[r], Xa[b], Ya[b])
            last = [t for t in trials[r] if t['it'] == it]
            print(json.dumps(dict(robot=r, it=it, loss=float(l), n=int(Xa.shape[0]),
                                  fails=int(sum(t['failed'] for t in last)),
                                  apex=float(np.nanmean([t['rms_apex'] if t['rms_apex'] is not None else np.nan for t in last])),
                                  t=round(time.time() - t0, 1))), flush=True)
    pols = {r: fada_np.FADAPolicy(merged(lora[r]), pl_np, DIMS, H) for r in ROBOTS}
    write_arm(f'{a.out}_hw{a.iters * len(G6)}', pols, trials, extra, t0)
    print(json.dumps(dict(event='done', t=time.time() - t0)), flush=True)


if __name__ == '__main__':
    main()
