#!/usr/bin/env python3
"""
descent_test.py --policy DIR --member M --robot real_r1 --out OUT.npz: is the nominal GPU sim's gradient a descent
direction on the robot? (the premise of deep ILC's hardware stage)

  base     the policy on the robot at --goals x --seeds (value_bench.py base, container): J0, its spread over seeds
           (the trial-to-trial noise), the real states and landing errors
  model    per base jump, the landing cost's gradient in the actions: the GPU sim's FD Jacobians along the same
           goal, the policy's feedback at the REAL states, the REAL landing error (closed-loop co-states) -- and the
           SRB model's (value_bench's g_srb) and two random directions
  measure  the robot flown again at the same seeds, the actions moved by -/+ eps along each (value_bench.py perturb):
           does J fall along -G, and does G . d predict the measured change?
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True)
ap.add_argument("--member", required=True)
ap.add_argument("--robot", default="real_r1")
ap.add_argument("--out", required=True)
ap.add_argument("--goals", type=float, nargs="+", default=[0.425, 0.475, 0.525, 0.575])
ap.add_argument("--seeds", type=int, nargs="+", default=[811, 812, 813, 814, 815, 816])
ap.add_argument("--eps", type=float, default=0.1)
ap.add_argument("--gpu", default="1")
a = ap.parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ilc_mjx.jump import JumpEnv  # noqa: E402

WS = "/home/henry/ilc_ws"
cp = lambda p: "/ilc_ws" + os.path.abspath(p)[len(WS):]
VB = "python3 src/ilc_quad/scripts/value_bench.py"


def box(cmd):
    full = f"cd /ilc_ws && source install/setup.bash && export ROS_LOCALHOST_ONLY=1 OMP_NUM_THREADS=1 && {cmd}"
    subprocess.run(["sg", "docker", "-c", f"docker exec ilc_quad bash -lc '{full}'"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


base_f, dirs_f, pert_f = (a.out.replace(".npz", s) for s in ("_base.npz", "_dirs.npz", "_pert.npz"))
P = f"--policy {cp(a.policy)} --member {a.member} --cond {a.robot} --jobs 12"
if not os.path.exists(base_f):
    box(f"{VB} base {P} --goals {' '.join(map(str, a.goals))} --seeds {' '.join(map(str, a.seeds))} --out {cp(base_f)}")
b = np.load(base_f)
bank = json.load(open(os.path.join(a.policy, "bank.json")))
env = JumpEnv(bank)
Nc, Ndc, N = env.Nc, env.Ndc, env.N
gamma = float(bank.get("gamma", 0.99))
D = np.asarray(bank["sx"], float)
qe, rs = np.asarray(bank["qe"], float), float(bank["r_scale"])
mask = np.ones((Nc, 4))
mask[Ndc:, :2] = 0.0
w = {k: jnp.asarray(v, jnp.float32) for k, v in np.load(os.path.join(a.policy, a.member + ".npz")).items() if k != "n_hidden"}
dpi = jax.jit(jax.vmap(jax.jacfwd(lambda o: JumpEnv.actor(w, o))))
# the nominal sim's Jacobians along each goal (deterministic: one per goal)
ug = sorted(set(float(g) for g in b["goal"]))
sim = env.rollout(w, env.references(np.array([[g, 0.0] for g in ug])), jax.random.PRNGKey(0), stochastic=False, fd=True)
An = sim["A"] * D[None, None, None, :] / D[None, None, :, None]
Bn = sim["B"] / D[None, None, :, None]
# the ballistic flight map from takeoff to touchdown (SRB, exact for the CoM), normalized at takeoff
from dilc_execute import EpisodeMaker, GoalBank  # noqa: E402
from ilc_mjx import host_path  # noqa: E402
gb = GoalBank(dict(bank, plans=[dict(p, path=host_path(p["path"])) for p in bank["plans"]]))
mk = EpisodeMaker(gb, "/home/henry/mujoco_menagerie")
Af = mk.srb.f_lin(np.zeros(6), np.zeros(4), np.zeros(2), np.zeros(2), env.dt)[0].full()
Phi = np.linalg.matrix_power(Af, N - Nc)
Pn = len(b["goal"])
G = np.zeros((Pn, Nc, 4))
for p in range(Pn):
    gi = ug.index(float(b["goal"][p]))
    K = np.asarray(dpi(jnp.asarray(b["obs"][p, :Nc], jnp.float32)))[..., :6]
    m = 0.01 * (Phi.T @ (2 * rs * qe * b["e0"][p])) * D            # d J / d e_Nc (normalized), measured error
    for k in range(Nc - 1, -1, -1):
        if k < Nc - 1:
            m = gamma * (An[gi, k + 1] + Bn[gi, k + 1] @ K[k + 1]).T @ m
        G[p, k] = (Bn[gi, k].T @ m) * mask[k]
G_srb = 0.01 * b["g_srb"] * mask[None]
rng = np.random.default_rng(0)
nel = mask.sum()
unit = lambda d: d * 0.5 * np.sqrt(nel) / max(np.linalg.norm(d), 1e-12)   # rms 0.5 per element
labels = ["-G_fd", "-G_srb", "rand0", "rand1"]
Dd = np.stack([np.stack([unit(-G[p]), unit(-G_srb[p]), unit(rng.standard_normal((Nc, 4)) * mask),
                         unit(rng.standard_normal((Nc, 4)) * mask)]) for p in range(Pn)])
np.savez(dirs_f, D=Dd, labels=np.array(labels), G=G, G_srb=G_srb)
if not os.path.exists(pert_f):
    box(f"{VB} perturb {P} --base {cp(base_f)} --dirs {cp(dirs_f)} --eps {a.eps} --out {cp(pert_f)}")
pr = np.load(pert_f)
J0 = b["J0"]
print(f"{a.robot}: {Pn} base jumps; J0 mean {np.nanmean(J0):.3f}; trial-to-trial spread at a goal (std over seeds): "
      + " ".join(f"g{g:.3f} {np.nanstd(J0[b['goal'] == g]):.3f} (mean {np.nanmean(J0[b['goal'] == g]):.2f})" for g in ug))
meas = (pr["Jp"] - pr["Jm"]) / (2 * a.eps)
for j, l in enumerate(labels):
    ok = np.isfinite(meas[:, j])
    print(f"  {l:7s}: J {np.nanmean(J0):.3f} -> {np.nanmean(pr['Jp'][:, j]):.3f} along +d (-> {np.nanmean(pr['Jm'][:, j]):.3f} along -d); "
          f"measured dJ/deps {np.nanmean(meas[:, j]):+.3f} (<0 for {np.mean(meas[ok, j] < 0):.0%})")
for name, Gm in (("G_fd", G), ("G_srb", G_srb)):
    pred = np.einsum("pka,pdka->pd", Gm, Dd)
    rj = [2, 3]
    ok = np.isfinite(meas[:, rj])
    x, y = pred[:, rj][ok], meas[:, rj][ok]
    print(f"  {name}: predicted vs measured dJ/deps along random directions r {np.corrcoef(x, y)[0, 1]:+.2f}, slope {(x @ y) / (x @ x):.2f}")
