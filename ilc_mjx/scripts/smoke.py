#!/usr/bin/env python3
"""smoke.py: a batch of policy jumps on the GPU (deterministic, then with finite differences): the landings,
the time, and the FD Jacobians against the SRB model's along the same jumps."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--policy", default="/home/henry/ilc_ws/log/dilc/pre_vgofd2s1/policy")
ap.add_argument("--member", default="g0_vgofdpre.s0_e600")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--gpu", default="1")
a = ap.parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ilc_mjx.jump import JumpEnv  # noqa: E402

bank = json.load(open(os.path.join(a.policy, "bank.json")))
env = JumpEnv(bank)
w = {k: jnp.asarray(v, jnp.float32) for k, v in np.load(os.path.join(a.policy, a.member + ".npz")).items()
     if k != "n_hidden"}
goals = np.stack([np.linspace(0.40, 0.60, a.batch), np.zeros(a.batch)], 1)
ref = env.references(goals)
for fd in (False, True):
    t0 = time.time()
    out = env.rollout(w, ref, jax.random.PRNGKey(0), stochastic=False, fd=fd)
    t1 = time.time()
    out = env.rollout(w, ref, jax.random.PRNGKey(0), stochastic=False, fd=fd)
    t2 = time.time()
    X = out["X"]
    land = X[:, env.N, :3] - np.asarray(ref["x_ref"][:, env.N, :3])
    print(f"fd={fd}: compile+run {t1 - t0:.0f} s, run {t2 - t1:.2f} s = {a.batch / (t2 - t1):.1f} jumps/s; "
          f"fell {int(out['fell'].sum())}/{a.batch}; landing error vs plan rms x {np.sqrt((land[:, 0] ** 2).mean()) * 100:.1f} cm "
          f"z {np.sqrt((land[:, 1] ** 2).mean()) * 100:.1f} cm pitch {np.degrees(np.sqrt((land[:, 2] ** 2).mean())):.1f} deg")
    if fd:
        A, B = out["A"], out["B"]
        print(f"   FD: finite A {np.isfinite(A).all((2, 3)).mean():.0%}, |B| rms (SRB units per action) "
              f"{np.sqrt((B ** 2).mean()):.4f}, |A - I| rms {np.sqrt(((A - np.eye(6)) ** 2).mean()):.4f}")
        # against the SRB model's along the same jumps (EpisodeMaker's, normalized the same way)
        from dilc_execute import EpisodeMaker, GoalBank
        from ilc_mjx import host_path
        bk = dict(bank, plans=[dict(p, path=host_path(p["path"])) for p in bank["plans"]])
        gb = GoalBank(bk)
        mk = EpisodeMaker(gb, "/home/henry/mujoco_menagerie")
        D = gb.sx
        cs = []
        for b in range(0, a.batch, max(1, a.batch // 8)):
            rf = gb.reference(goals[b])[0]
            e_srb = mk.make(goals[b], rf, X[b], out["U"][b], X[b], bool(out["fell"][b]))
            e_fd = mk.make(goals[b], rf, X[b], out["U"][b], X[b], bool(out["fell"][b]), jac=(A[b], B[b]))
            cs.append(np.corrcoef(e_srb["B"].ravel(), e_fd["B"].ravel())[0, 1])
            if b == 0:
                ob = np.abs(e_fd["obs"] - out["obs"][b]).max()
        print(f"   corr(B_FD, B_SRB) per jump: {' '.join(f'{c:.2f}' for c in cs)}; "
              f"obs rebuilt on the host vs flown: max |diff| {ob:.1e}")
