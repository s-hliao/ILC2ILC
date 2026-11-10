#!/usr/bin/env python3
"""
fada_adapt_tf.py --policy fig4/fadaT_s<s>/it8/policy --robots ... --out OUT (2026-11-07, FADA_AUDIT_1107.md FULL FIDELITY, the quad):
FADA's few-shot target adaptation as the paper (Algorithm 1 T1-T3, Eq. 4.4-4.6, B.3), in fada_adapt.py's robot budget and evaluation:
  T1  ONE round of target jumps by FADA-zs only: --rounds x one jump at each of --goals (fada_adapt.py's goals, its trial seeds
      seed + 101 r), --train-cond as fada_adapt.py (every practice jump under that perturbation);
  T2  per budget (the first --budgets jumps, goal-cycled) ONE offline LoRA fit from the base IDM: windows (O^H_k, A^H_k, Y_exec =
      o_{k+1..k+K}, capped at the takeoff sample) -> a_k as executed, from the robot's own recorded observations (fallen jumps dropped,
      as fada_adapt.py), r 8 / alpha 16 / dropout 0.05 on every IDM Linear, planner + base frozen, 800 Adam steps, lr 3e-4, batch 4096
      with replacement, first-action MSE with the swing legs' unused entries masked;
  T3  the merged IDM -> OUT/<robot>/ep<K>/policy; ep0 = FADA-zs. Evaluation of every budget = fada_adapt.py's (--eval, 4 episodes,
      --perturbed: final701/rob_<robot>.txt). -> OUT/<robot>/summary.json {budget: scores}.
"""
import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/henry/ilc_ws/src/ilc_f1tenth/ilc2real")
ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True)
ap.add_argument("--robots", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--goals", nargs="+", default=["0.45", "0.60", "0.50,0.15", "0.60,0.10", "0.50,0.05", "0.575,0.15"])
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--budgets", type=int, nargs="+", default=[6, 12, 24, 36])
ap.add_argument("--steps", type=int, default=800)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--seed", type=int, default=9001)
ap.add_argument("--eval", default="planeval", choices=("planeval", "planefinal", "none"))
ap.add_argument("--perturbed", action="store_true")
ap.add_argument("--train-cond", default="")
ap.add_argument("--eval-episodes", type=int, default=4)
ap.add_argument("--gpu", default="0")
a = ap.parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import fada_tf as F  # noqa: E402
from fada_quad_tf_core import Adam, apply_updates  # noqa: E402

WS, CWS = "/home/henry/ilc_ws", "/ilc_ws"
cpath = lambda p: CWS + os.path.abspath(p)[len(WS):]
EVAL = dict(planeval=([0.4375, 0.5625, (0.53, 0.11), (0.51, 0.13), (0.56, 0.12), (0.52, 0.16), (0.4625, 0.06), (0.6125, 0.07)], 301),
            planefinal=([0.4875, 0.6125, (0.52, 0.12), (0.54, 0.14), (0.57, 0.11), (0.51, 0.17), (0.4875, 0.035), (0.6375, 0.09)], 701))
gvec = lambda g: (tuple(float(v) for v in g) if isinstance(g, (tuple, list)) else
                  tuple(float(v) for v in g.split(",")) if isinstance(g, str) and "," in g else (float(g), 0.0))
goals = [gvec(g) for g in a.goals]
bank = json.load(open(os.path.join(a.policy, "bank.json")))
Ndc, Nsc, _ = bank["phases"]
Nc = Ndc + Nsc
H, K, DA = F.H_HIST, F.K_FUT, 4
MASK = np.ones((Nc, DA), np.float32)
MASK[Ndc:, :2] = 0.0
_W = dict(np.load(os.path.join(a.policy, "policy.npz")))
P0, I0 = F.unflatten(_W, "FP"), F.unflatten(_W, "FI")
H = F.h_hist(I0)                              # (2026-11-07) the model's own history length
OVERLAY = f"export PYTHONPATH={os.environ['ILC_QUAD_OVERLAY']}:$PYTHONPATH && " if os.environ.get("ILC_QUAD_OVERLAY") else ""
# (2026-11-07 FIX) the robots' settings passed into the container exactly as deploy.py does (they were missing here: every flight ran the
# robots at the overlay's default joint damping and without the free-landing / tail-landing task config -- all results before invalid)
if os.environ.get("ILC_ROBOT_DAMPING"):
    OVERLAY += f"export ILC_SIM_DAMPING={os.environ['ILC_ROBOT_DAMPING']} && "
for _v in ("ILC_FREE_LANDING", "ILC_LAND_ZMIN", "ILC_FTH_MAX", "ILC_LAND_FEET"):
    if os.environ.get(_v):
        OVERLAY += f"export {_v}={os.environ[_v]} && "
if os.environ.get("DILC_NODE_PARAMS"):
    OVERLAY += f"export DILC_NODE_PARAMS='{os.environ['DILC_NODE_PARAMS']}' && "


def flown(*args, **kw):
    from fly_tokens import flight_token
    with flight_token():
        return subprocess.run(*args, **kw)


def fly(pdir, conds, jumps, seed, json_out, save_dir="", episodes=1):            # = fada_adapt.py's
    j = " ".join(f"--jump {gvec(g)[0]} {gvec(g)[1]}" for g in jumps)
    sd = f"--save-dir {cpath(save_dir)}" if save_dir else ""
    cmd = (f"cd /ilc_ws && source install/setup.bash && export ROS_LOCALHOST_ONLY=1 OMP_NUM_THREADS=1 && {OVERLAY}"
           f"python3 install/ilc_quad/lib/ilc_quad/dilc_execute.py run --policy {cpath(pdir)} --member policy "
           f"--conds {conds} {j} --episodes {episodes} --seed {seed} --jobs {os.environ.get('DILC_FLY_JOBS', '8')} --json {cpath(json_out)} {sd}")
    for attempt in range(2):
        try:
            flown(["sg", "docker", "-c", f"docker exec ilc_quad bash -lc '{cmd}'"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                  check=False, timeout=900)
            break
        except subprocess.TimeoutExpired:
            subprocess.run(["sg", "docker", "-c", f"docker exec ilc_quad pkill -f {CWS + os.path.abspath(json_out)[len(WS):]}"], check=False)
            print(f"flight timed out ({json_out}), attempt {attempt + 1}", flush=True)
    return json.load(open(json_out)) if os.path.exists(json_out) else []


score = lambda r: 0.01 * r["landing_cost"] + 20.0 * r["fell"]


def export(I, d, meta):
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "policy.npz"), fada_tf=np.array(1), n_hidden=np.array(2), **F.flatten(F.to_np(P0), "FP"), **F.flatten(F.to_np(I), "FI"))
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), default=float)
    for f in ("bank.json", "ilc_U.npz", "ilc_best.json"):
        if os.path.exists(os.path.join(a.policy, f)):
            shutil.copy(os.path.join(a.policy, f), d)
    json.dump(dict(best="policy", eval_goals=None, members=[dict(name="policy", group=0, variant="policy", tag="policy", episodes=0,
                                                                 updates=0, score=None)]), open(os.path.join(d, "manifest.json"), "w"))


def windows(eps, d_o=7):
    """Every (O^H, A^H, mask, Y_exec, a_k, swing mask) of the recorded jumps: o_k = [obs[k, :7], obs[k, -4:]] (the executed previous
    action, bank prev_action), A tokens the action executed before each history sample."""
    Os, As, Ms, Ys, Ts, Ks = [], [], [], [], [], []
    for obs, act in eps:
        o = (np.concatenate([obs[:, :7], obs[:, -4:]], 1) if d_o == 11 else obs[:, :7]).astype(np.float32)   # 2026-11-07 leak fix: d_o 7
        ap_ = np.concatenate([np.zeros((1, DA)), act], 0)                             # ap_[j]: executed before sample j
        for k in range(Nc):
            lo = max(0, k - H + 1); n = k - lo + 1
            Oh = np.zeros((H, d_o), np.float32); Ah = np.zeros((H, DA), np.float32); m = np.zeros(H, bool)
            Oh[H - n:], Ah[H - n:], m[H - n:] = o[lo:k + 1], ap_[lo:k + 1], True
            Os.append(Oh); As.append(Ah); Ms.append(m); Ys.append(o[np.minimum(k + 1 + np.arange(K), Nc)]); Ts.append(act[k]); Ks.append(MASK[k])
    return [np.stack(x).astype(np.float32 if i != 2 else bool) for i, x in enumerate((Os, As, Ms, Ys, Ts, Ks))]


Ij = jax.tree_util.tree_map(jnp.asarray, I0)
opt = Adam(a.lr)


@jax.jit
def step(L, s, key, Dv):
    kb, kd = jax.random.split(key)
    b = jax.random.randint(kb, (a.batch,), 0, Dv[4].shape[0])
    Oh, Ah, m, Y, tgt, mk = (x[b] for x in Dv)

    def loss(q):
        return jnp.mean(jnp.sum(((F.idm_fwd(Ij, Oh, Ah, Y, m, xp=jnp, lora=q, key=kd, jax=jax)[:, 0] - tgt) * mk) ** 2, -1))
    l, g = jax.value_and_grad(loss)(L)
    u, s = opt.update(g, s, L)
    return apply_updates(L, u), s, l


def run_robot(robot):
    out = os.path.join(a.out, robot)
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "deploy.log"), "a")

    def say(msg):
        print(f"[{robot}] {msg}", flush=True)
        logf.write(msg + "\n"); logf.flush()
    conds = robot
    if a.train_cond:
        lines = [l for l in open(f"/home/henry/ilc_ws/log/dilc/final701/rob_{robot}.txt").read().splitlines() if l.startswith(f"{robot}+{a.train_cond}|")]
        if len(lines) != 1:
            raise SystemExit(f"{robot}: no perturbation {a.train_cond}")
        tcf = os.path.join(out, f"train_cond_{a.train_cond}.txt"); open(tcf, "w").write(lines[0] + "\n"); conds = cpath(tcf)
    p0 = os.path.join(out, "ep0", "policy")
    export(I0, p0, dict(budget=0))
    eps, jumps, falls, t0 = [], [], 0, time.time()                   # T1: one round, FADA-zs only, goal-cycled
    for r in range(a.rounds):
        rdir = os.path.join(out, "collect", f"r{r}")
        shutil.rmtree(rdir, ignore_errors=True)
        fly(p0, conds, goals, a.seed + 101 * r, os.path.join(out, "collect", f"r{r}.json"), rdir)
        for g in goals:
            fs = [f for f in sorted(os.listdir(rdir)) if f.startswith(f"policy_g{g[0]:.3f}_{g[1]:.3f}_")] if os.path.isdir(rdir) else []
            for f in fs[:1]:
                z = np.load(os.path.join(rdir, f))
                jumps.append(bool(z["fell"]))
                eps.append(None if bool(z["fell"]) else (np.asarray(z["obs"], float), np.asarray(z["act"], float)))
    say(f"collected {len(eps)} FADA-zs jumps ({sum(jumps)} fell) [{time.time() - t0:.0f} s]")
    res = dict(robot=robot, args=vars(a), collected=len(eps), collect_falls=int(sum(jumps)), budgets={})
    pdirs = {0: p0}
    for kb in a.budgets:                                              # T2: one offline fit per budget, from the base
        use = [e for e in eps[:kb] if e is not None]
        if not use:
            say(f"budget {kb}: no unfallen jumps"); continue
        Dv = tuple(jnp.asarray(x) for x in windows(use, F.d_obs(I0, DA)))
        L = jax.tree_util.tree_map(jnp.asarray, F.init_lora(a.seed + kb, I0))
        s, key, ls = opt.init(L), jax.random.PRNGKey(a.seed + kb), []
        for st in range(a.steps):
            key, ks = jax.random.split(key)
            L, s, l = step(L, s, ks, Dv)
            ls.append(l)
        ls = [float(x) for x in ls]
        pd_ = os.path.join(out, f"ep{kb}", "policy")
        export(F.merge_lora(I0, F.to_np(L)), pd_, dict(budget=kb, jumps_used=len(use), loss=[np.mean(ls[:20]), np.mean(ls[-20:])]))
        pdirs[kb] = pd_
        res["budgets"][str(kb)] = dict(jumps=kb, used=len(use), loss0=float(np.mean(ls[:20])), loss1=float(np.mean(ls[-20:])))
        say(f"budget {kb}: {len(use)} jumps, IDM mse {np.mean(ls[:20]):.2e} -> {np.mean(ls[-20:]):.2e}")
    if a.eval != "none":                                              # T3 evaluation, fada_adapt.py's
        eg, seed = EVAL[a.eval]
        for kb, pd_ in pdirs.items():
            e = res["budgets"].setdefault(str(kb), {})
            rr = fly(pd_, robot, eg, seed, os.path.join(out, f"ep{kb}", f"eval_{a.eval}.json"), episodes=a.eval_episodes)
            e[a.eval] = float(np.mean([score(x) for x in rr])) if rr else None
            e[f"{a.eval}_falls"] = int(sum(x["fell"] for x in rr))
            if a.perturbed:
                rp = fly(pd_, f"/ilc_ws/log/dilc/final701/rob_{robot}.txt", eg, seed, os.path.join(out, f"ep{kb}", f"eval_{a.eval}_rob.json"),
                         episodes=a.eval_episodes)
                e["rob"] = float(np.mean([score(x) for x in rp])) if rp else None
                e["rob_falls"] = int(sum(x["fell"] for x in rp))
            say(f"budget {kb}: {a.eval} {e.get(a.eval)} perturbed {e.get('rob')}")
            json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)
    return res


allres = [run_robot(r) for r in a.robots]
json.dump(allres, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
