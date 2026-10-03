#!/usr/bin/env python3
"""fd_outliers.py: the GPU FD Jacobians' tails -- over a batch of stochastic policy jumps, how often A, B (normalized)
are far from typical, and what that does to the closed-loop co-states chained over the jump."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ilc_mjx.jump import JumpEnv  # noqa: E402

P = sys.argv[1] if len(sys.argv) > 1 else "/home/henry/ilc_ws/log/dilc/pre_vgofdg2s2/policy"
MEM = sys.argv[2] if len(sys.argv) > 2 else "g0_vgofdpre.s0_e1200"
bank = json.load(open(os.path.join(P, "bank.json")))
env = JumpEnv(bank)
w = {k: jnp.asarray(v, jnp.float32) for k, v in np.load(os.path.join(P, MEM + ".npz")).items() if k != "n_hidden"}
Bn = 128
goals = np.stack([np.random.default_rng(0).uniform(0.4, 0.6, Bn), np.zeros(Bn)], 1)
out = env.rollout(w, env.references(goals), jax.random.PRNGKey(1), stochastic=True, fd=True)
D = np.asarray(env.sx)
An = out["A"] * D[None, None, None, :] / D[None, None, :, None]
Bn_ = out["B"] / D[None, None, :, None]
na = np.abs(An - np.eye(6)).max((2, 3))
nb = np.abs(Bn_).max((2, 3))
q = lambda x: " ".join(f"p{p} {np.percentile(x, p):.3g}" for p in (50, 90, 99, 99.9)) + f" max {x.max():.3g}"
print(f"|A_n - I|max per sample: {q(na)}")
print(f"|B_n|max per sample:     {q(nb)}")
print(f"fell {int(out['fell'].sum())}/{Bn}")
# closed-loop products: how far a unit co-state grows when chained back over the jump
dpi = jax.vmap(jax.vmap(jax.jacfwd(lambda o: JumpEnv.actor(w, o))))(jnp.asarray(out["obs"][:, :env.Nc]))
K = np.asarray(dpi)[..., :6]
grow = np.zeros(Bn)
for b in range(Bn):
    v = np.ones(6) / np.sqrt(6)
    for k in range(env.Nc - 1, 0, -1):
        v = 0.99 * (An[b, k] + Bn_[b, k] @ K[b, k]).T @ v
    grow[b] = np.linalg.norm(v)
print(f"|co-state| after chaining back over the jump (start 1): {q(grow)}")
worst = np.argsort(-na.max(1))[:3]
for b in worst:
    k = int(np.argmax(na[b]))
    print(f"  jump {b} sample {k}: |A-I| {na[b, k]:.3g}, |B| {nb[b, k]:.3g}, fell {bool(out['fell'][b])}")
