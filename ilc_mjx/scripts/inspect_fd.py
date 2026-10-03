#!/usr/bin/env python3
"""inspect_fd.py: the GPU FD Jacobians up close -- A at a few samples (normalized, as the trainer sees it),
the measured state perturbations dS0 (how close to eps_s sx along each coordinate), against the SRB model."""
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

np.set_printoptions(precision=3, suppress=True, linewidth=150)
P = "/home/henry/ilc_ws/log/dilc/pre_vgofd2s1/policy"
bank = json.load(open(os.path.join(P, "bank.json")))
env = JumpEnv(bank)
w = {k: jnp.asarray(v, jnp.float32) for k, v in np.load(os.path.join(P, "g0_vgofdpre.s0_e600.npz")).items()
     if k != "n_hidden"}
goals = np.array([[0.5, 0.0]] * 4)
ref = env.references(goals)
# the perturbations themselves, at sample 10 of the settled stance
d = env.stand()
s0 = env.srb(d, jnp.zeros(2))
print("stance state", np.asarray(s0))
for i in range(6):
    dp = env._perturb(d, i, jnp.int32(10))
    print(f"perturb {i}: dS0 / (eps sx) = {np.asarray((env.srb(dp, jnp.zeros(2)) - s0) / (env.cfg.eps_s * env.sx))}")
out = env.rollout(w, ref, jax.random.PRNGKey(0), stochastic=False, fd=True)
D = np.asarray(env.sx)
for k in (5, 25, 45):
    A = out["A"][0, k]
    print(f"\nk={k}: A (normalized: D^-1 A D)\n{A * D[None, :] / D[:, None]}")
