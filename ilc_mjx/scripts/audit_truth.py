#!/usr/bin/env python3
"""
audit_truth.py RUN_DIR MEMBER OUT.npz: the ground truth the critic should learn, on the GPU sim (deterministic,
so a perturbed jump differs from its base by the perturbation alone):

  base      J jumps of the policy (deterministic, goals across the bank): states X, forces, actions, obs, falls,
            the FD Jacobians A, B along them, and the policy's own Jacobian d a / d o at every sample
  truth     for every contact sample k and action channel c (swing channels skipped), the jump flown again with
            a_k[c] -/+ eps, the policy's feedback acting from k+1 on: the landing cost and fall of each, at two
            eps (linearity check). From these: dJ/da_k, the closed-loop derivative of the landing score
            J = 0.01 x Stage III cost (the critic's reward, before the fall penalty) -- exactly what
            -dQ/da_k / gamma^(Nc-1-k) of a perfect critic is.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("member")
ap.add_argument("out")
ap.add_argument("--goals", type=float, nargs="*", default=[0.41, 0.44, 0.47, 0.5, 0.53, 0.56, 0.59, 0.455,
                                                            0.485, 0.515, 0.545, 0.425, 0.575, 0.505, 0.465, 0.535])
ap.add_argument("--eps", type=float, nargs=2, default=[0.05, 0.01])
ap.add_argument("--batch", type=int, default=480)
ap.add_argument("--gpu", default="1")
a = ap.parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ilc_mjx.jump import JumpEnv  # noqa: E402

P = os.path.join(a.run, "policy")
bank = json.load(open(os.path.join(P, "bank.json")))
env = JumpEnv(bank)
w = {k: jnp.asarray(v, jnp.float32) for k, v in np.load(os.path.join(P, a.member + ".npz")).items() if k != "n_hidden"}
goals = np.stack([np.asarray(a.goals), np.zeros(len(a.goals))], 1)
J = len(goals)
key = jax.random.PRNGKey(0)
qe, rs = np.asarray(bank["qe"], float), float(bank["r_scale"])
Nc, N = env.Nc, env.N


gamma = float(bank.get("gamma", 0.99))
Ndc = env.Ndc


def score(X, ref_x, g):
    """0.01 x the Stage III cost (EpisodeMaker.make's rc[-1, 2]): the landing against the goal, level."""
    tgt = ref_x[:, N].copy()
    tgt[:, :2] = ref_x[:, 0, :2] + g
    tgt[:, 2] = 0.0
    e = X[:, N] - tgt
    return 0.01 * rs * np.einsum("bi,i,bi->b", e, qe, e)


def stages(X, ref_x, g):
    """(B, 3): 0.01 x each stage's cost as the critic's reward sums it, discounted from sample 0 (transition j
    weighs gamma^j; the flight and landing rows belong to the last transition, as in EpisodeMaker.make)."""
    e = X - ref_x
    c = 0.01 * rs * np.einsum("bmi,i,bmi->bm", e, qe, e)               # (B, N+1) per state
    wj = gamma ** np.arange(Nc)                                         # transition j -> state j+1
    s1 = (c[:, 1:Nc + 1] * wj).sum(1)
    s2 = (c[:, 1:Nc + 1] * wj * (np.arange(Nc) + 1 >= Ndc)).sum(1) + gamma ** (Nc - 1) * c[:, Nc + 1:N + 1].sum(1)
    s3 = gamma ** (Nc - 1) * score(X, ref_x, g)
    return np.stack([s1, s2, s3], 1)


t0 = time.time()
ref = env.references(goals)
base = env.rollout(w, ref, key, stochastic=False, fd=True)
ref_x = np.asarray(ref["x_ref"])
J0 = score(base["X"], ref_x, goals[:, :2])
S0 = stages(base["X"], ref_x, goals[:, :2])
# the policy's Jacobian d a / d o along the base jumps
dpi = jax.vmap(jax.vmap(jax.jacfwd(lambda o: JumpEnv.actor(w, o))))(jnp.asarray(base["obs"][:, :Nc]))
print(f"base: {J} jumps, J0 mean {J0.mean():.3f}, fell {int(base['fell'].sum())}; {time.time() - t0:.0f} s", flush=True)

chans = [(k, c) for k in range(Nc) for c in range(4) if not (k >= env.Ndc and c < 2)]
jobs = [(j, k, c, s * e) for j in range(J) for (k, c) in chans for e in a.eps for s in (-1.0, 1.0)]
Jpert, Fpert = np.zeros(len(jobs)), np.zeros(len(jobs), bool)
Spert = np.zeros((len(jobs), 3))
for i0 in range(0, len(jobs), a.batch):
    blk = jobs[i0:i0 + a.batch]
    gi = np.array([j for j, _, _, _ in blk])
    off = np.zeros((len(blk), Nc, 4))
    for b, (_, k, c, d) in enumerate(blk):
        off[b, k, c] = d
    r_b = jax.tree_util.tree_map(lambda v: v[gi], ref)
    out = env.rollout(w, r_b, key, stochastic=False, a_offset=off)
    Jpert[i0:i0 + len(blk)] = score(out["X"], ref_x[gi], goals[gi, :2])
    Spert[i0:i0 + len(blk)] = stages(out["X"], ref_x[gi], goals[gi, :2])
    Fpert[i0:i0 + len(blk)] = out["fell"]
    print(f"  {i0 + len(blk)}/{len(jobs)} perturbed jumps, {time.time() - t0:.0f} s", flush=True)

G = {e: np.full((J, Nc, 4), np.nan) for e in a.eps}
GS = {e: np.full((J, Nc, 4, 3), np.nan) for e in a.eps}
FL = {e: np.zeros((J, Nc, 4), bool) for e in a.eps}
idx = {jb: i for i, jb in enumerate(jobs)}
for j in range(J):
    for (k, c) in chans:
        for e in a.eps:
            im, ip = idx[(j, k, c, -e)], idx[(j, k, c, e)]
            G[e][j, k, c] = (Jpert[ip] - Jpert[im]) / (2 * e)
            GS[e][j, k, c] = (Spert[ip] - Spert[im]) / (2 * e) / gamma ** k   # d(discounted from k)/d a_k
            FL[e][j, k, c] = Fpert[ip] or Fpert[im]
np.savez(a.out, goals=goals, X=base["X"], U=base["U"], act=base["act"], obs=base["obs"], fell=base["fell"],
         A=base["A"], B=base["B"], dpi=np.asarray(dpi), J0=J0, x_ref=ref_x, eps=np.asarray(a.eps),
         S0=S0, **{f"dJda_{i}": G[e] for i, e in enumerate(a.eps)}, **{f"dSda_{i}": GS[e] for i, e in enumerate(a.eps)}, **{f"fell_{i}": FL[e] for i, e in enumerate(a.eps)})
lin = np.isfinite(G[a.eps[0]]) & np.isfinite(G[a.eps[1]])
print(f"-> {a.out}; linearity: corr(dJ/da at eps {a.eps[0]}, at {a.eps[1]}) "
      f"{np.corrcoef(G[a.eps[0]][lin], G[a.eps[1]][lin])[0, 1]:.3f}; {time.time() - t0:.0f} s")
