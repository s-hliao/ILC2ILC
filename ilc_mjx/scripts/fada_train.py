#!/usr/bin/env python3
"""
fada_train.py --policy POLICY.npz --out DIR [--dr] [--priv]: a FADA-style planner-IDM (Xie et al., arXiv 2606.28476,
re-built for the jump; a baseline) distilled on the GPU sim from a source policy's rollouts:

  planner  P(o_k) -> e_{k+1}: the next sample's normalized SRB error the source policy reaches (rollouts with the
           stance exploration only, no action noise)
  IDM      I(o_k, e_{k+1}) -> a_k: the action that reaches a given next state (rollouts with smooth action offsets
           too, for coverage; the action as flown)
  policy   a_k = I(o_k, P(o_k)); o_k the deployable observation (no privileged z)

FADA's recipe: --dr --priv with the DR'd privileged teacher as the source (it reads z; the planner and IDM do not).
--policy our nominal sim policy without --dr isolates the adaptation signal from DR. The IDM starts from the source
policy's weights (the extra inputs at zero weight). Writes OUT/policy (deploy format; fada_adapt.py adapts its IDM).
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True)
ap.add_argument("--bank", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--dr", action="store_true")
ap.add_argument("--priv", action="store_true", help="the source reads the dynamics z (the DR teacher)")
ap.add_argument("--batches", type=int, default=100, help="x 256 jumps for each of the planner and the IDM")
ap.add_argument("--explore", type=float, nargs=2, default=[0.08, 0.15])
ap.add_argument("--goals", type=float, nargs=2, default=[0.40, 0.60])
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--horizon", type=int, default=1, help="the planner's target and the IDM's input: the error H samples "
                "ahead (H > 1: a less delay-sensitive inverse model; stored as P_h)")
ap.add_argument("--load-data", default="", help="skip the collection: OUT/data.npz of an earlier run")
ap.add_argument("--gpu", default="0")
ap.add_argument("--plane", action="store_true", help="2D goal plane: goals uniform over the bank's valid region (flat and box jumps), evals on its grid")
ap.add_argument("--est-window", type=float, default=0.0, help="s: the policy observes the robots' estimator (jump.Config)")
a = ap.parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ilc_mjx.jump import Config, JumpEnv  # noqa: E402

os.makedirs(a.out, exist_ok=True)
pdir0 = os.path.dirname(os.path.abspath(a.policy))
bank = json.load(open(a.bank or os.path.join(pdir0, "bank.json")))
env = JumpEnv(bank, Config(est_window=a.est_window))
Nc, Ndc, N = env.Nc, env.Ndc, env.N
NZ = len(JumpEnv.DYN_KEYS)
mask = np.ones((Nc, 4))
mask[Ndc:, :2] = 0.0
logf = open(os.path.join(a.out, "train.log"), "a")


def say(msg):
    print(msg, flush=True)
    logf.write(msg + "\n")
    logf.flush()


ws = {k: jnp.asarray(v, jnp.float32) for k, v in np.load(a.policy).items() if k != "n_hidden"}
OD = 31
rng = np.random.default_rng(a.seed)
nominal = lambda B: np.tile([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0], (B, 1))
t0 = time.time()
Po, Pt, Io, It, Ia, Im = [], [], [], [], [], []
env.priv_dim = NZ if a.priv else 0
for b in range(0 if a.load_data else a.batches):
    B = 256
    for kind in ("planner", "idm"):
        goals = env.sample_plane_goals(B, rng) if a.plane else np.stack([rng.uniform(*a.goals, B), np.zeros(B)], 1)
        ref = env.references(goals)
        dyn = env.dyn_arrays(env.sample_dyn(B, rng) if a.dr else nominal(B))
        q_off, a_off = env.explore_noise(B, rng, a.explore[0], 0.0 if kind == "planner" else a.explore[1])
        out = env.rollout(ws, ref, jax.random.PRNGKey(int(rng.integers(2 ** 31))), stochastic=False,
                          a_offset=a_off, q_offset=q_off, dyn=dyn)
        # a fallen or non-finite jump never reaches the data (the 2026-10-07 run's NaN states made every fit NaN)
        ok = ~np.asarray(out["fell"], bool) & np.isfinite(out["obs"][:, :Nc + 1, :OD]).all((1, 2)) \
            & np.isfinite(out["act"]).all((1, 2))
        o = out["obs"][ok][:, :, :OD]                                  # the deployable observation
        o_k, e_n = o[:, :Nc], o[:, np.minimum(np.arange(Nc) + a.horizon, Nc), :6]
        if kind == "planner":
            Po.append(o_k.reshape(-1, OD))
            Pt.append(e_n.reshape(-1, 6))
        else:
            Io.append(o_k.reshape(-1, OD))
            It.append(e_n.reshape(-1, 6))
            Ia.append(out["act"][ok].reshape(-1, 4))
            Im.append(np.tile(mask, (int(ok.sum()), 1)))
    if b % 20 == 0:
        say(f"collect {b}/{a.batches}  [{time.time() - t0:.0f} s]")
f32 = lambda x: jnp.asarray(np.concatenate(x) if isinstance(x, list) else x, jnp.float32)
if a.load_data:
    _d = np.load(a.load_data)
    Po, Pt, Io, It, Ia, Im = (_d[k] for k in ("Po", "Pt", "Io", "It", "Ia", "Im"))
else:
    np.savez(os.path.join(a.out, "data.npz"), **{k: np.concatenate(v) for k, v in
                                                  dict(Po=Po, Pt=Pt, Io=Io, It=It, Ia=Ia, Im=Im).items()})
Po, Pt, Io, It, Ia, Im = map(f32, (Po, Pt, Io, It, Ia, Im))
assert all(bool(jnp.isfinite(x).all()) for x in (Po, Pt, Io, It, Ia, Im)), "non-finite FADA data"
t_col = time.time() - t0
say(f"data: planner {Po.shape[0]}, IDM {Io.shape[0]} samples ({'DR' if a.dr else 'nominal'}, source "
    f"{'privileged teacher' if a.priv else 'policy'}), collected in {t_col:.0f} s")

key = jax.random.PRNGKey(a.seed)
k1, k2, k3 = jax.random.split(key, 3)
glorot = lambda k, o, i: jax.random.normal(k, (o, i)) * jnp.sqrt(2.0 / i)
P = dict(P_W0=glorot(k1, 256, OD), P_b0=jnp.zeros(256), P_W1=glorot(k2, 256, 256), P_b1=jnp.zeros(256),
         P_Wo=glorot(k3, 6, 256) * 0.1, P_bo=jnp.zeros(6))
I = dict(ws)
I["W0"] = jnp.concatenate([ws["W0"][:, :OD], jnp.zeros((ws["W0"].shape[0], 6))], 1)


def planner_out(p, o):
    return jax.vmap(lambda o_: JumpEnv.planner(p, o_))(o)


def idm_out(w, o, e):
    return jax.vmap(lambda x: JumpEnv.actor(w, x))(jnp.concatenate([o, e], 1))


def adam(params, loss, data, steps, lr, key):
    vg = jax.value_and_grad(loss)
    n = data[0].shape[0]

    @jax.jit
    def run(params, key):
        mo = jax.tree_util.tree_map(jnp.zeros_like, params)
        vo = jax.tree_util.tree_map(jnp.zeros_like, params)

        def one(c, s):
            p, mo, vo, key = c
            key, sub = jax.random.split(key)
            idx = jax.random.randint(sub, (4096,), 0, n)
            l, g = vg(p, *[d[idx] for d in data])
            t = s + 1
            mo = jax.tree_util.tree_map(lambda m_, g_: 0.9 * m_ + 0.1 * g_, mo, g)
            vo = jax.tree_util.tree_map(lambda v_, g_: 0.999 * v_ + 0.001 * g_ ** 2, vo, g)
            lr_t = lr * (0.1 ** (s / steps))
            p = jax.tree_util.tree_map(lambda q, m_, v_: q - lr_t * (m_ / (1 - 0.9 ** t))
                                       / (jnp.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8), p, mo, vo)
            return (p, mo, vo, key), l
        (p, _, _, _), ls = jax.lax.scan(one, (params, mo, vo, key), jnp.arange(steps))
        return p, ls
    return run(params, key)


t1 = time.time()
P, lp = adam(P, lambda p, o, t: ((planner_out(p, o) - t) ** 2).mean(), (Po, Pt), a.steps, a.lr, k1)
say(f"planner: mse {float(lp[:100].mean()):.3f} -> {float(lp[-100:].mean()):.4f} (target var {float(Pt.var()):.3f})")
I, li = adam(I, lambda w, o, e, t, m: (((idm_out(w, o, e) - t) * m) ** 2).sum(-1).mean(), (Io, It, Ia, Im),
             a.steps, a.lr, k2)
say(f"IDM: mse {float(li[:100].mean()):.4f} -> {float(li[-100:].mean()):.5f} (action var {float(Ia.var()):.4f})")
t_fit = time.time() - t1

W = dict(I, **P, P_h=jnp.asarray(a.horizon))
d = os.path.join(a.out, "policy")
os.makedirs(d, exist_ok=True)
np.savez(os.path.join(d, "policy.npz"), **{k: np.asarray(v) for k, v in W.items()}, n_hidden=np.array(2))
for f in ("bank.json", "ilc_U.npz", "ilc_best.json"):
    if os.path.exists(os.path.join(pdir0, f)):
        shutil.copy(os.path.join(pdir0, f), d)
json.dump(dict(best="policy", eval_goals=None, members=[dict(name="policy", group=0, variant="policy", tag="policy",
                                                             episodes=0, updates=0, score=None)]),
          open(os.path.join(d, "manifest.json"), "w"))

# zero-shot in the GPU sim: the planner-IDM against its source, nominal and over a fixed DR set
env.priv_dim = 0
r_scale, qe = float(bank["r_scale"]), np.asarray(bank["qe"], float)
for name, B, Pd in (("nominal", 16, nominal(16)), ("DR", 128, env.sample_dyn(128, np.random.default_rng(12345)))):
    goals = env.sample_plane_goals(B, rng) if a.plane else np.stack([np.linspace(*a.goals, B), np.zeros(B)], 1)
    ref = env.references(goals)
    xr = np.asarray(ref["x_ref"])
    tg = xr[:, N].copy()
    tg[:, :2] = xr[:, 0, :2] + goals
    tg[:, 2] = 0.0
    dyn = env.dyn_arrays(Pd)
    res = []
    for tag, w_, pv in (("planner-IDM", W, 0), ("source", ws, NZ if a.priv else 0)):
        env.priv_dim = pv
        o = env.rollout(w_, ref, jax.random.PRNGKey(0), stochastic=False, dyn=dyn)
        e = o["X"][:, N] - tg
        res.append(f"{tag} {(0.01 * r_scale * (e * qe * e).sum(1) + 20 * o['fell']).mean():.2f}")
    say(f"zero-shot {name}: " + ", ".join(res))
say(f"done: {2 * a.batches * 256} sim jumps, collect {t_col:.0f} s + fit {t_fit:.0f} s")
