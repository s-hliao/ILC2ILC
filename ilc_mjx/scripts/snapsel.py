#!/usr/bin/env python3
"""
snapsel.py --runs pre_coilcex1 ... --out DIR: choosing the sim policy without the robot. Every snapshot of the given
sim stages (OUT/policy/*.npz), scored two ways:
  sim metrics (GPU sim only, what one has before hardware): nominal score over the goal range; the score from
              perturbed stances (planar joint offsets ~ N(0, 0.08)); the score under the GPU DR family; the policy's
              mean feedback gain |d pi / d e|
  transfer    zero-shot on the val robots (real_r1, real_s1, real_r4, real_r5; goals 0.45/0.4625/0.5375/0.55, seed 301,
              2 jumps each: 32 per snapshot), in the container
then the rank correlation of each sim metric with transfer, over all snapshots and within each run (does it pick the
right moment to stop?). -> DIR/snapshots.json, DIR/report.txt
"""
import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="sim-stage folders under log/dilc")
ap.add_argument("--out", required=True)
ap.add_argument("--gpu", default="0")
a = ap.parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ilc_mjx.jump import JumpEnv  # noqa: E402

D = "/home/henry/ilc_ws/log/dilc"
WS, CWS = "/home/henry/ilc_ws", "/ilc_ws"
cpath = lambda p: CWS + os.path.abspath(p)[len(WS):]
os.makedirs(a.out, exist_ok=True)
J = os.path.join(a.out, "snapshots.json")
rows = json.load(open(J)) if os.path.exists(J) else {}

snaps = []
for r in a.runs:
    for f in sorted(glob.glob(os.path.join(D, r, "policy", "g0_*.npz"))):
        ep = int(os.path.basename(f).rsplit("_e", 1)[1][:-4])
        snaps.append((r, os.path.basename(f)[:-4], ep, f))
bank = json.load(open(os.path.join(D, a.runs[0], "policy", "bank.json")))
env = JumpEnv(bank)
N = env.N
qe, rs = np.asarray(bank["qe"], float), float(bank["r_scale"])


def targets(goals, ref):
    xr = np.asarray(ref["x_ref"])
    tg = xr[:, N].copy()
    tg[:, :2] = xr[:, 0, :2] + goals
    tg[:, 2] = 0.0
    return tg


def score(o, tg):
    e = o["X"][:, N] - tg
    return 0.01 * rs * (e * qe * e).sum(1) + 20 * o["fell"]


G = np.stack([np.linspace(0.40, 0.60, 32), np.zeros(32)], 1)
REF = env.references(G)
TG = targets(G, REF)
Q_OFF, _ = env.explore_noise(32, np.random.default_rng(5), 0.08, 0.0)
DYN = env.dyn_arrays(env.sample_dyn(32, np.random.default_rng(12345)))
dpi = jax.jit(jax.vmap(jax.jacfwd(lambda o, w_: JumpEnv.actor(w_, o)), (0, None)))
for r, name, ep, f in snaps:
    key = f"{r}/{name}"
    if key in rows and "sim_nominal" in rows[key]:
        continue
    w = {k: jnp.asarray(v, jnp.float32) for k, v in np.load(f).items() if k != "n_hidden"}
    o = env.rollout(w, REF, jax.random.PRNGKey(0), stochastic=False)
    nom = score(o, TG)
    K = np.asarray(dpi(jnp.asarray(o["obs"][:, :env.Nc].reshape(-1, o["obs"].shape[-1]), jnp.float32), w))[..., :6]
    op = env.rollout(w, REF, jax.random.PRNGKey(0), stochastic=False, q_offset=Q_OFF)
    od = env.rollout(w, REF, jax.random.PRNGKey(0), stochastic=False, dyn=DYN)
    rows[key] = dict(run=r, member=name, episodes=ep, sim_nominal=float(nom.mean()),
                     sim_perturbed=float(score(op, TG).mean()), sim_dr=float(score(od, TG).mean()),
                     gain=float(np.abs(K).mean()))
    json.dump(rows, open(J, "w"), indent=1)
    print(f"{key}: sim nominal {rows[key]['sim_nominal']:.2f}, perturbed {rows[key]['sim_perturbed']:.2f}, "
          f"DR {rows[key]['sim_dr']:.2f}, gain {rows[key]['gain']:.3f}", flush=True)

# transfer: zero-shot on the val robots
X = "install/ilc_quad/lib/ilc_quad/dilc_execute.py"
JUMPS = "--jump 0.45 0 --jump 0.4625 0 --jump 0.5375 0 --jump 0.55 0"
for r, name, ep, f in snaps:
    key = f"{r}/{name}"
    if "transfer" in rows[key]:
        continue
    out = os.path.join(a.out, f"{r}_{name}.json")
    cmd = (f"cd /ilc_ws && source install/setup.bash && export ROS_LOCALHOST_ONLY=1 OMP_NUM_THREADS=1 && python3 {X} run "
           f"--policy {cpath(os.path.join(D, r, 'policy'))} --member {name} --conds real_r1,real_s1,real_r4,real_r5 "
           f"{JUMPS} --episodes 2 --seed 301 --jobs 16 --json {cpath(out)}")
    try:
        subprocess.run(["sg", "docker", "-c", f"docker exec ilc_quad bash -lc '{cmd}'"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=1800)
    except subprocess.TimeoutExpired:                  # a hung flight: kill it, leave this snapshot unscored
        subprocess.run(["sg", "docker", "-c", f"docker exec ilc_quad pkill -9 -f {cpath(out)}"], check=False)
        print(f"{key}: transfer flight timed out, skipped", flush=True)
    if os.path.exists(out):
        res = json.load(open(out))
        per = {}
        for x in res:
            per.setdefault(x["cond"], []).append(0.01 * x["landing_cost"] + 20 * x["fell"])
        rows[key]["transfer"] = float(np.mean([np.mean(v) for v in per.values()]))
        rows[key]["transfer_per_robot"] = {k: float(np.mean(v)) for k, v in per.items()}
        json.dump(rows, open(J, "w"), indent=1)
        print(f"{key}: transfer (val, zero-shot) {rows[key]['transfer']:.2f}", flush=True)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


ok = [v for v in rows.values() if "transfer" in v]
lines = [f"{len(ok)} snapshots with transfer; rank correlation of each sim metric with zero-shot val transfer "
         f"(lower is better for all)"]
for m in ("sim_nominal", "sim_perturbed", "sim_dr", "gain", "episodes"):
    allr = spearman([v[m] for v in ok], [v["transfer"] for v in ok])
    within = []
    picks = []
    for r in a.runs:
        vs = [v for v in ok if v["run"] == r]
        if len(vs) >= 3:
            within.append(spearman([v[m] for v in vs], [v["transfer"] for v in vs]))
            best_by_metric = min(vs, key=lambda v: v[m])
            picks.append(best_by_metric["transfer"] - min(v["transfer"] for v in vs))
    lines.append(f"  {m:14s} all {allr:+.2f}   within runs (mean) {np.mean(within):+.2f}   "
                 f"regret of picking each run's snapshot by it {np.mean(picks):.2f}")
lines.append("  (regret: the val transfer of the snapshot the metric picks, minus the run's best; final snapshot's regret: "
             f"{np.mean([[v for v in ok if v['run'] == r and v['episodes'] == max(x['episodes'] for x in ok if x['run'] == r)][0]['transfer'] - min(v['transfer'] for v in ok if v['run'] == r) for r in a.runs if any(v['run'] == r for v in ok)]):.2f})")
open(os.path.join(a.out, "report.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
