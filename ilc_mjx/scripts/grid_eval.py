#!/usr/bin/env python3
"""
grid_eval.py --policy DIR [--member NAME|ref] --out FILE: the sim network over the 2D goal plane, in the nominal GPU sim
(deterministic, no exploration): a grid of goals (x, h) -- h = 0 a flat jump, h > 0 onto a box -- each flown once, and
per goal the landing error (ex, ez, pitch), a fall, and success (no fall, |ex| <= 5 cm, |ez| <= 3 cm). --member ref
flies the interpolated reference alone (the action 0: the plans' forces), to tell the reference's misses from the
network's. Prints the success map and writes the rows to --out (json).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True, help="a policy folder (bank.json, manifest.json, member npz)")
ap.add_argument("--member", default="best", help="a member name, 'best', or 'ref' (the reference alone)")
ap.add_argument("--xs", type=float, nargs=3, default=[0.40, 0.65, 0.025], metavar=("LO", "HI", "STEP"))
ap.add_argument("--hs", type=float, nargs=3, default=[0.0, 0.20, 0.025], metavar=("LO", "HI", "STEP"))
ap.add_argument("--hull", action="store_true", help="only goals inside the bank's goal regions")
ap.add_argument("--q-off", type=float, default=0.0, help="stance offsets ~ N(0, this) (perturbed starts)")
ap.add_argument("--out", default=None)
ap.add_argument("--gpu", default=None)
ap.add_argument("--est-window", type=float, default=0.0, help="s: the policy observes the robots' estimator")
a = ap.parse_args()
if a.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ilc_mjx.jump import Config, JumpEnv  # noqa: E402

bank = json.load(open(os.path.join(a.policy, "bank.json")))
env = JumpEnv(bank, Config(est_window=a.est_window))
N = env.N
man = json.load(open(os.path.join(a.policy, "manifest.json"))) if os.path.exists(
    os.path.join(a.policy, "manifest.json")) else {}
member = man.get("best") if a.member == "best" else a.member
if member == "ref" or member is None:
    any_npz = sorted(f for f in os.listdir(a.policy) if f.startswith("g0_") and f.endswith(".npz"))[0]
    w = {k: jnp.asarray(v, jnp.float32) for k, v in np.load(os.path.join(a.policy, any_npz)).items() if k != "n_hidden"}
    w["W_mu"], w["b_mu"] = jnp.zeros_like(w["W_mu"]), jnp.zeros_like(w["b_mu"])
    member = "ref"
else:
    w = {k: jnp.asarray(v, jnp.float32) for k, v in np.load(os.path.join(a.policy, member + ".npz")).items()
         if k != "n_hidden"}

xs = np.arange(a.xs[0], a.xs[1] + 1e-9, a.xs[2])
hs = np.arange(a.hs[0], a.hs[1] + 1e-9, a.hs[2])
G = np.array([(x, h) for h in hs for x in xs])
if a.hull:
    from dilc_execute import GoalBank
    gb = GoalBank(bank)
    inside = np.array([gb.inside(g) if hasattr(gb, "inside") else True for g in G])
    G = G[inside]
ref = env.references(G)
q_off = env.explore_noise(len(G), np.random.default_rng(5), a.q_off, 0.0)[0] if a.q_off > 0 else None
o = env.rollout(w, ref, jax.random.PRNGKey(0), stochastic=False, q_offset=q_off)
xr = np.asarray(ref["x_ref"])
tg = xr[:, N].copy()
tg[:, :2] = xr[:, 0, :2] + G
e = o["X"][:, N] - tg
fell = np.asarray(o["fell"], bool)
ok = (~fell) & (np.abs(e[:, 0]) <= 0.05) & (np.abs(e[:, 1]) <= 0.03)
rows = [dict(goal=list(map(float, g)), ex=float(e_[0]), ez=float(e_[1]), eth=float(e_[2]), fell=bool(f), ok=bool(k))
        for g, e_, f, k in zip(G, e, fell, ok)]
print(f"{a.policy} member {member}: {ok.mean():.0%} ok, {fell.mean():.0%} fell, |ex| {np.abs(e[:, 0]).mean() * 100:.1f} cm, "
      f"|ez| {np.abs(e[:, 1]).mean() * 100:.1f} cm over {len(G)} goals")
print("rows h (cm), cols x (cm); cell: ex cm (F fell, * ok)")
print("      " + " ".join(f"{x * 100:5.1f}" for x in xs))
for h in hs[::-1]:
    line = f"{h * 100:5.1f} "
    for x in xs:
        i = np.where((np.abs(G[:, 0] - x) < 1e-9) & (np.abs(G[:, 1] - h) < 1e-9))[0]
        if not len(i):
            line += "    ."
            continue
        r = rows[i[0]]
        line += " " + ("    F" if r["fell"] else f"{r['ex'] * 100:4.0f}" + ("*" if r["ok"] else " "))
    print(line)
if a.out:
    json.dump(dict(policy=a.policy, member=member, rows=rows), open(a.out, "w"), indent=0)
