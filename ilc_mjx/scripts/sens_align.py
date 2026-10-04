#!/usr/bin/env python3
"""
sens_align.py --policy DIR --member M --robot real_r1 --out OUT: how wrong may the ILC's model be? The hardware stage's
Gauss-Newton step is da = -beta L e with L = S'(S S' + delta tr/3 I)^-1, S the MODEL's closed-loop landing sensitivity
(3 x Nc*4). On the robot the landing error then changes by -beta S_true L e, so the ILC converges iff the eigenvalues
of M = S_true L lie in the disc |1 - beta lambda| < 1 (for small beta: Re lambda > 0 -- the model's sensitivity
need only point the right way). This measures M on the robot, for several models:

  base     the policy at --goals x --seeds (value_bench.py base, container)
  models   per base jump, each model's S (its GPU sim's FD Jacobians along the goal, the policy's feedback at the REAL
           states) and L; the 3 columns of L, scaled to rms 0.5, are the directions
  measure  the robot flown at the same seeds with the actions moved by +/- eps along each (value_bench.py perturb):
           S_true L, column by column, from the landing errors' central differences (common random numbers)
-> per model: M's eigenvalues (median over base jumps), whether |1 - 0.3 lambda| < 1, and the angle between the
predicted and measured landing-error changes. A negated nominal model (the 180-degree control) follows without flights.
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
ap.add_argument("--out", required=True, help="output prefix (.npz files and the report)")
ap.add_argument("--goals", type=float, nargs="+", default=[0.425, 0.5, 0.575])
ap.add_argument("--seeds", type=int, nargs="+", default=[811, 812])
ap.add_argument("--eps", type=float, default=0.1)
ap.add_argument("--delta", type=float, default=0.1)
ap.add_argument("--beta", type=float, default=0.3)
ap.add_argument("--gpu", default="1")
a = ap.parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ilc_mjx.jump import JumpEnv  # noqa: E402

# the models: thrust-to-weight wrong both ways (mass x m, motors / m), the nominal one, and the GPU sim's friction low
MODELS = {"m0.70": (0.70, 1 / 0.70), "m0.85": (0.85, 1 / 0.85), "nominal": (1.0, 1.0), "m1.15": (1.15, 1 / 1.15),
          "m1.30": (1.30, 1 / 1.30)}
WS = "/home/henry/ilc_ws"
cp = lambda p: "/ilc_ws" + os.path.abspath(p)[len(WS):]
VB = "python3 src/ilc_quad/scripts/value_bench.py"


def box(cmd):
    full = f"cd /ilc_ws && source install/setup.bash && export ROS_LOCALHOST_ONLY=1 OMP_NUM_THREADS=1 && {cmd}"
    subprocess.run(["sg", "docker", "-c", f"docker exec ilc_quad bash -lc '{full}'"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


base_f, dirs_f, pert_f = a.out + "_base.npz", a.out + "_dirs.npz", a.out + "_pert.npz"
P = f"--policy {cp(a.policy)} --member {a.member} --cond {a.robot} --jobs 16"
if not os.path.exists(base_f):
    box(f"{VB} base {P} --goals {' '.join(map(str, a.goals))} --seeds {' '.join(map(str, a.seeds))} --out {cp(base_f)}")
b = np.load(base_f)
bank = json.load(open(os.path.join(a.policy, "bank.json")))
env = JumpEnv(bank)
Nc, Ndc, N = env.Nc, env.Ndc, env.N
D = np.asarray(bank["sx"], float)
mask = np.ones((Nc, 4))
mask[Ndc:, :2] = 0.0
w = {k: jnp.asarray(v, jnp.float32) for k, v in np.load(os.path.join(a.policy, a.member + ".npz")).items() if k != "n_hidden"}
dpi = jax.jit(jax.vmap(jax.jacfwd(lambda o: JumpEnv.actor(w, o))))
T = (N - Nc) * env.dt
PHI = np.zeros((3, 6))
PHI[:, :3] = np.eye(3)
PHI[:, 3:] = T * np.eye(3)
ug = sorted(set(float(g) for g in b["goal"]))
Pn = len(b["goal"])


def sens(An, Bn, K):
    S = np.zeros((3, Nc, 4))
    for i in range(3):
        m = PHI[i] * D
        for k in range(Nc - 1, -1, -1):
            if k < Nc - 1:
                m = (An[k + 1] + Bn[k + 1] @ K[k + 1]).T @ m
            S[i, k] = (Bn[k].T @ m) * mask[k]
    return S.reshape(3, -1)


names = list(MODELS)
Ls, Ss, scale = {}, {}, {}
Dd = np.zeros((Pn, 3 * len(names), Nc, 4))
nel = mask.sum()
for mi, nm in enumerate(names):
    ms, mot = MODELS[nm]
    Pm = np.array([ms, 0.0, mot, 0.0, 1.0, 1.0, 0.0, 0.0])
    sim = env.rollout(w, env.references(np.array([[g, 0.0] for g in ug])), jax.random.PRNGKey(0), stochastic=False,
                      fd=True, dyn=env.dyn_arrays(np.tile(Pm, (len(ug), 1))))
    An = sim["A"] * D[None, None, None, :] / D[None, None, :, None]
    Bn = sim["B"] / D[None, None, :, None]
    Ls[nm], Ss[nm], scale[nm] = [], [], []
    for p in range(Pn):
        gi = ug.index(float(b["goal"][p]))
        K = np.asarray(dpi(jnp.asarray(b["obs"][p, :Nc], jnp.float32)))[..., :6]
        S = sens(An[gi], Bn[gi], K)
        Mm = S @ S.T
        L = S.T @ np.linalg.inv(Mm + a.delta * np.trace(Mm) / 3 * np.eye(3))          # (Nc*4, 3)
        Ls[nm].append(L)
        Ss[nm].append(S)
        c = []
        for j in range(3):
            col = L[:, j].reshape(Nc, 4)
            cj = 0.5 * np.sqrt(nel) / max(np.linalg.norm(col), 1e-12)                    # rms 0.5 per element
            Dd[p, 3 * mi + j] = col * cj
            c.append(cj)
        scale[nm].append(c)
np.savez(dirs_f, D=Dd, names=np.array(names))
if not os.path.exists(pert_f):
    box(f"{VB} perturb {P} --base {cp(base_f)} --dirs {cp(dirs_f)} --eps {a.eps} --out {cp(pert_f)}")
pr = np.load(pert_f)
dE = (pr["Ep"][..., :3] - pr["Em"][..., :3]) / (2 * a.eps)          # (Pn, ndirs, 3): S_true (c_j L_j)
lines = [f"{a.robot}: {Pn} base jumps (goals {ug}, seeds {list(a.seeds)}); M = S_true L per model, ideal I; "
         f"converges iff |1 - {a.beta} lambda| < 1 for all eigenvalues"]
summary = {}
for mi, nm in enumerate(names):
    eigs, convs, cosines = [], [], []
    for p in range(Pn):
        cols = dE[p, 3 * mi:3 * mi + 3]                                # (3 dirs, 3)
        if not np.isfinite(cols).all():
            continue
        Mt = np.stack([cols[j] / scale[nm][p][j] for j in range(3)], 1)  # columns: S_true L_j
        lam = np.linalg.eigvals(Mt)
        eigs.append(np.sort_complex(lam))
        convs.append(bool(np.all(np.abs(1 - a.beta * lam) < 1)))
        pred = Ss[nm][p] @ Ls[nm][p]                                   # the model's own S L (~ I)
        for j in range(3):
            u, v = Mt[:, j], pred[:, j]
            cosines.append(float(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12)))
    if not eigs:
        continue
    E = np.array(eigs)
    summary[nm] = dict(eig_real_median=np.median(E.real, 0).tolist(), converges=float(np.mean(convs)),
                       cos_median=float(np.median(cosines)), min_real=float(np.median(E.real.min(1))))
    lines.append(f"  {nm:8s} eigenvalues (median real parts) {' '.join(f'{x:+.2f}' for x in np.median(E.real, 0))}; "
                 f"converges for {np.mean(convs):.0%} of base jumps; predicted vs measured direction cos {np.median(cosines):+.2f}")
if "nominal" in summary:
    lines.append(f"  flipped  (the nominal model negated: eigenvalues negate) converges for 0%: real parts "
                 f"{' '.join(f'{-x:+.2f}' for x in summary['nominal']['eig_real_median'])}")
open(a.out + "_report.txt", "w").write("\n".join(lines) + "\n")
json.dump(summary, open(a.out + "_summary.json", "w"), indent=1)
print("\n".join(lines))
