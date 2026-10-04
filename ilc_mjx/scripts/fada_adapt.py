#!/usr/bin/env python3
"""
fada_adapt.py --policy DIR --robots ... --out OUT: FADA's adaptation (a baseline) on the robot, in deploy.py's
budget and protocol -- per robot 10 iterations x one jump at each of 3 training goals (30 jumps), the same trial
seeds, then the same evaluation (--eval final: the reserved test, --perturbed its 8 perturbations).

  each iteration  the planner-IDM (fada_train.py) flies the goals; every real transition is relabelled in
                  hindsight, (o_k, e_{k+1} as reached) -> a_k as flown -- reward-free: the landing error is never
                  used; a LoRA of rank --rank on each IDM layer (the base and the planner frozen) is fitted to all
                  the robot's transitions so far, and merged for the next flights.
--rank 0 --zero-shot: no adaptation (the evaluation of the policy as distilled).
-> OUT/<robot>/summary.json as deploy.py's (tab.py reads it).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True, help="policy dir (fada_train.py's OUT/policy)")
ap.add_argument("--robots", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--goals", type=float, nargs="+", default=[0.425, 0.5, 0.575])
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--rank", type=int, default=4)
ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=9001)
ap.add_argument("--eval", default="final", choices=("val", "final", "holdout", "none"))
ap.add_argument("--perturbed", action="store_true")
ap.add_argument("--eval-episodes", type=int, default=4)
ap.add_argument("--gpu", default="0")
a = ap.parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ilc_mjx.jump import JumpEnv  # noqa: E402

WS, CWS = "/home/henry/ilc_ws", "/ilc_ws"
cpath = lambda p: CWS + os.path.abspath(p)[len(WS):]
EVAL = dict(val=([0.45, 0.4625, 0.5375, 0.55], 301), final=([0.4375, 0.4875, 0.5125, 0.5625], 701),
            holdout=([0.4375, 0.4625, 0.4875, 0.5125, 0.5375, 0.5625], 1301))   # holdout: the frozen final test only
bank = json.load(open(os.path.join(a.policy, "bank.json")))
Ndc, Nsc, _ = bank["phases"]
Nc = Ndc + Nsc
OD = 31
mask = np.ones((Nc, 4))
mask[Ndc:, :2] = 0.0
_w = dict(np.load(os.path.join(a.policy, "policy.npz")))
H = int(_w.pop("P_h", 1))                       # the IDM's target horizon (fada_train --horizon)
W0 = {k: jnp.asarray(v, jnp.float32) for k, v in _w.items() if k != "n_hidden"}
LAYERS = ("W0", "W1", "W_mu")


def export(w, d):
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "policy.npz"), **{k: np.asarray(v) for k, v in w.items()}, n_hidden=np.array(2),
             P_h=np.array(H))
    for f in ("bank.json", "ilc_U.npz", "ilc_best.json"):
        if os.path.exists(os.path.join(a.policy, f)):
            shutil.copy(os.path.join(a.policy, f), d)
    json.dump(dict(best="policy", eval_goals=None, members=[dict(name="policy", group=0, variant="policy", tag="policy",
                                                                 episodes=0, updates=0, score=None)]),
              open(os.path.join(d, "manifest.json"), "w"))


def fly(pdir, conds, jumps, seed, json_out, save_dir="", episodes=1):
    j = " ".join(f"--jump {g} 0" for g in jumps)
    sd = f"--save-dir {cpath(save_dir)}" if save_dir else ""
    cmd = (f"cd /ilc_ws && source install/setup.bash && export ROS_LOCALHOST_ONLY=1 OMP_NUM_THREADS=1 && "
           f"python3 install/ilc_quad/lib/ilc_quad/dilc_execute.py run --policy {cpath(pdir)} --member policy "
           f"--conds {conds} {j} --episodes {episodes} --seed {seed} --jobs 8 --json {cpath(json_out)} {sd}")
    for attempt in range(2):
        try:
            subprocess.run(["sg", "docker", "-c", f"docker exec ilc_quad bash -lc '{cmd}'"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False, timeout=900)
            break
        except subprocess.TimeoutExpired:
            tag = os.path.abspath(json_out)[len(WS):]
            subprocess.run(["sg", "docker", "-c", f"docker exec ilc_quad pkill -f {CWS + tag}"], check=False)
            print(f"flight timed out ({json_out}), attempt {attempt + 1}", flush=True)
    return json.load(open(json_out)) if os.path.exists(json_out) else []


def score(r):
    return 0.01 * r["landing_cost"] + 20.0 * r["fell"]


def merged(base, lora):
    w = dict(base)
    for k in LAYERS:
        w[k] = base[k] + lora[k + "_B"] @ lora[k + "_A"]
    return w


def lora_fit(base, O, E, A_, M, key):
    """LoRA on the IDM's layers (rank a.rank; B at zero, so it starts at the base), fitted to the transitions."""
    ks = jax.random.split(key, len(LAYERS))
    lora = {}
    for k, kk in zip(LAYERS, ks):
        o_, i_ = base[k].shape
        lora[k + "_A"] = jax.random.normal(kk, (a.rank, i_)) / np.sqrt(i_)
        lora[k + "_B"] = jnp.zeros((o_, a.rank))
    X = jnp.concatenate([O, E], 1)

    def loss(lo):
        w = merged(base, lo)
        pred = jax.vmap(lambda x: JumpEnv.actor({k: v for k, v in w.items() if not k.startswith("P_")}, x))(X)
        return (((pred - A_) * M) ** 2).sum(-1).mean()
    vg = jax.jit(jax.value_and_grad(loss))
    mo = jax.tree_util.tree_map(jnp.zeros_like, lora)
    vo = jax.tree_util.tree_map(jnp.zeros_like, lora)
    l0 = None
    for s in range(1, a.steps + 1):
        l, g = vg(lora)
        l0 = l if l0 is None else l0
        mo = jax.tree_util.tree_map(lambda m_, g_: 0.9 * m_ + 0.1 * g_, mo, g)
        vo = jax.tree_util.tree_map(lambda v_, g_: 0.999 * v_ + 0.001 * g_ ** 2, vo, g)
        lora = jax.tree_util.tree_map(lambda p, m_, v_: p - a.lr * (m_ / (1 - 0.9 ** s))
                                      / (jnp.sqrt(v_ / (1 - 0.999 ** s)) + 1e-8), lora, mo, vo)
    return merged(base, lora), float(l0), float(l)


def run_robot(robot):
    out = os.path.join(a.out, robot)
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "deploy.log"), "w")

    def say(msg):
        print(f"[{robot}] {msg}", flush=True)
        logf.write(msg + "\n")
        logf.flush()
    w = dict(W0)
    data, hist = [], []
    iters = a.iters if a.rank > 0 else 0
    for it in range(iters + 1):
        pdir = os.path.join(out, f"it{it}", "policy")
        export(w, pdir)
        if it == iters:
            break
        t0 = time.time()
        rdir = os.path.join(out, f"it{it}", "real")
        shutil.rmtree(rdir, ignore_errors=True)
        fly(pdir, robot, a.goals, a.seed + 101 * it, os.path.join(out, f"it{it}", "real.json"), rdir)
        line = []
        for f in sorted(os.listdir(rdir)) if os.path.isdir(rdir) else []:
            z = np.load(os.path.join(rdir, f))
            g = float(z["goal"][0])
            obs, act, fell = np.asarray(z["obs"], float), np.asarray(z["act"], float), bool(z["fell"])
            J = float(0.01 * np.asarray(z["rc"])[:, 2].sum())
            hist.append(dict(it=it, goal=g, J=J, fell=fell))
            line.append(f"g{g:.3f} J {J:.2f}{' FELL' if fell else ''}")
            if fell:                       # a fall's transitions are not the robot's dynamics near the plan
                continue
            data.append((obs[:Nc, :OD], obs[np.minimum(np.arange(Nc) + H, Nc), :6], act[:Nc] * mask, mask))
        if not data:
            continue
        O, E, A_, M = (jnp.asarray(np.concatenate([d[i] for d in data]), jnp.float32) for i in range(4))
        w, l0, l1 = lora_fit(W0, O, E, A_, M, jax.random.PRNGKey(it))
        say(f"it {it}: " + " | ".join(line) + f"; LoRA on {len(data)} trials: IDM mse {l0:.2e} -> {l1:.2e}"
            f"  [{time.time() - t0:.0f} s]")
    res = dict(robot=robot, hist=hist, args=vars(a))
    if a.eval != "none":
        goals, seed = EVAL[a.eval]
        for tag, pd_ in (("start", os.path.join(out, "it0", "policy")), ("final", os.path.join(out, f"it{iters}", "policy"))):
            if tag == "final" and iters == 0:
                for k in ("", "_falls"):
                    res[f"{a.eval}_final{k}"] = res[f"{a.eval}_start{k}"]
                    if a.perturbed:
                        res[f"rob_final{k}"] = res[f"rob_start{k}"]
                continue
            rr = fly(pd_, robot, goals, seed, os.path.join(out, f"eval_{a.eval}_{tag}.json"), episodes=a.eval_episodes)
            res[f"{a.eval}_{tag}"] = float(np.mean([score(r) for r in rr])) if rr else None
            res[f"{a.eval}_{tag}_falls"] = int(sum(r["fell"] for r in rr))
            if a.perturbed:
                rp = fly(pd_, f"/ilc_ws/log/dilc/final701/rob_{robot}.txt", goals, seed,
                         os.path.join(out, f"eval_{a.eval}_rob_{tag}.json"), episodes=a.eval_episodes)
                res[f"rob_{tag}"] = float(np.mean([score(r) for r in rp])) if rp else None
                res[f"rob_{tag}_falls"] = int(sum(r["fell"] for r in rp))
        fm = lambda v: "n/a" if v is None else f"{v:.2f}"
        say(f"{a.eval}: start {fm(res[f'{a.eval}_start'])} -> final {fm(res[f'{a.eval}_final'])}"
            + (f"; perturbed {fm(res.get('rob_start'))} -> {fm(res.get('rob_final'))}" if a.perturbed else ""))
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)
    return res


allres = [run_robot(r) for r in a.robots]
json.dump(allres, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
