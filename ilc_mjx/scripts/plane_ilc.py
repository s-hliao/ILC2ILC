#!/usr/bin/env python3
"""
plane_ilc.py --bank BANK --out FILE: per-goal ILC in the nominal GPU sim, open loop (no network): for each goal a
sequence of normalized contact actions a (Nc, 4) on top of the interpolated reference, flown deterministically; each
iteration the Gauss-Newton step on the landing error e = (x, z, pitch) / sx through the open-loop landing sensitivity
S = d x_N / d a (the sim's own one-step FD Jacobians, chained, and the ballistic flight), a <- a - beta S'(SS' +
delta tr/3 I)^-1 e, rms capped. Can the ILC reach each goal of the plane from its reference? Writes per goal the
error per iteration and the final actions (the per-goal ILC solutions).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--bank", required=True)
ap.add_argument("--goals", nargs="*", default=None, help="x,h ...; default: a grid over the plans' hull")
ap.add_argument("--grid", type=float, default=0.025)
ap.add_argument("--iters", type=int, default=15)
ap.add_argument("--beta", type=float, default=0.7)
ap.add_argument("--cap", type=float, default=0.15)
ap.add_argument("--delta", type=float, default=0.05)
ap.add_argument("--init", default=None, help="a previous --out: start from its actions (warm start)")
ap.add_argument("--out", required=True)
ap.add_argument("--gpu", default=None)
a = ap.parse_args()
if a.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ilc_mjx.jump import JumpEnv  # noqa: E402

bank = json.load(open(a.bank))
env = JumpEnv(bank)
N, Nc = env.N, env.Nc
Gp = env.goals


def in_hull(g):
    w = env.weights([g])[0]
    return w.min() > -1e-9


if a.goals:
    G = np.array([[float(v) for v in s.split(",")] for s in a.goals])
else:
    xs = np.arange(Gp[:, 0].min(), Gp[:, 0].max() + 1e-9, a.grid)
    hs = np.arange(0.0, Gp[:, 1].max() + 1e-9, a.grid)
    G = np.array([(x, h) for h in hs for x in xs])
    G = G[[in_hull(g) for g in G]]
B = len(G)
ref = env.references(G)
xr = np.asarray(ref["x_ref"])
tg = xr[:, N, :3].copy()
tg[:, :2] = xr[:, 0, :2] + G
tg[:, 2] = 0.0
sx = np.asarray(bank["sx"], float)
# a zero network (its action 0): the actions are the offsets alone
tmpl = next(f for f in sorted(os.listdir(os.path.dirname(a.bank))) if f.endswith(".npz") and f.startswith("g0_")) \
    if any(f.startswith("g0_") for f in os.listdir(os.path.dirname(a.bank))) else None
od = 9 + 6 * env.H + (4 if env.PA else 0)
hid = 256
w = {"W0": jnp.zeros((hid, od)), "b0": jnp.zeros(hid), "W1": jnp.zeros((hid, hid)), "b1": jnp.zeros(hid),
     "W_mu": jnp.zeros((4, hid)), "b_mu": jnp.zeros(4), "W_ls": jnp.zeros((4, hid)), "b_ls": jnp.zeros(4)}
swing = np.asarray(env.swing)
U = np.zeros((B, Nc, 4))
if a.init:
    prev = json.load(open(a.init))
    pg = {tuple(np.round(r["goal"], 4)): np.array(r["a"]) for r in prev["rows"]}
    for i, g in enumerate(G):
        if tuple(np.round(g, 4)) in pg:
            U[i] = pg[tuple(np.round(g, 4))]
Tf = env.Nfl * env.dt
Phi = np.eye(6)
Phi[:3, 3:] = Tf * np.eye(3)
hist = [[] for _ in range(B)]
best = [(np.inf, U[i].copy()) for i in range(B)]
for it in range(a.iters + 1):
    o = env.rollout(w, ref, jax.random.PRNGKey(0), stochastic=False, fd=True, a_offset=U)
    e = o["X"][:, N, :3] - tg
    fell = np.asarray(o["fell"], bool)
    en = e / sx[:3]
    for i in range(B):
        sc = float(np.sqrt((en[i] ** 2).sum())) + (1e3 if fell[i] else 0.0)
        hist[i].append(dict(ex=float(e[i, 0]), ez=float(e[i, 1]), eth=float(e[i, 2]), fell=bool(fell[i])))
        if sc < best[i][0]:
            best[i] = (sc, U[i].copy())
    ok = (~fell) & (np.abs(e[:, 0]) <= 0.05) & (np.abs(e[:, 1]) <= 0.03)
    print(f"it {it:2d}: ok {ok.mean():4.0%} fell {fell.mean():4.0%} |ex| {np.abs(e[:, 0]).mean() * 100:5.1f} cm "
          f"|ez| {np.abs(e[:, 1]).mean() * 100:4.1f} cm |eth| {np.degrees(np.abs(e[:, 2])).mean():4.1f} deg", flush=True)
    if it == a.iters:
        break
    A_, B_ = np.nan_to_num(o["A"]), np.nan_to_num(o["B"])                # (B, Nc, 6, 6), (B, Nc, 6, 4)
    for i in range(B):
        q = (Phi[:3] / sx[:3, None])                                     # rows: d e_n / d x_Nc
        S = np.zeros((3, Nc, 4))
        for k in range(Nc - 1, -1, -1):
            if k < Nc - 1:
                q = q @ A_[i, k + 1]
            S[:, k] = q @ B_[i, k]
        S[:, swing] = 0.0
        Sf = S.reshape(3, -1)
        M = Sf @ Sf.T
        M = M + a.delta * np.trace(M) / 3 * np.eye(3) + 1e-9 * np.eye(3)
        stp = (Sf.T @ np.linalg.solve(M, np.nan_to_num(en[i]))).reshape(Nc, 4) * a.beta
        rms = np.sqrt((stp ** 2).sum() / (~swing).sum())
        stp *= min(1.0, a.cap / max(rms, 1e-12))
        if fell[i] or not np.isfinite(e[i]).all():          # a fall: back to the best so far, half the step
            U[i] = best[i][1] - 0.5 * stp if np.isfinite(best[i][0]) else U[i] * 0.5
        else:
            U[i] = U[i] - stp
        U[i] = np.clip(U[i], -0.999, 0.999) * ~swing
rows = [dict(goal=list(map(float, g)), hist=h, a=best[i][1].tolist(), best=float(best[i][0])) for i, (g, h) in
        enumerate(zip(G, hist))]
json.dump(dict(bank=a.bank, rows=rows, args=vars(a)), open(a.out, "w"))
okb = np.mean([r["best"] < np.sqrt((5 / 1) ** 2 + 0) and not r["hist"][-1]["fell"] for r in rows])
print("per goal, the last iteration: ex cm (F fell)")
for r in rows:
    h = r["hist"][-1]
    print(f"  ({r['goal'][0]:.3f}, {r['goal'][1]:.3f}): " + ("F" if h["fell"] else f"ex {h['ex'] * 100:5.1f} ez {h['ez'] * 100:5.1f}")
          + "   path " + " ".join("F" if x["fell"] else f"{x['ex'] * 100:.0f}" for x in r["hist"]))
