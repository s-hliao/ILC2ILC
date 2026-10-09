#!/usr/bin/env python3
"""
rma_ctx_eval.py --run DIR/it<k> --robots ... --out OUT: RMA with cross-trial context (ppo_train.py --mode student_ctx)
on the robot. Per robot: the teacher at the nominal robot's latent flies --calib goals (one jump each: the calibration,
counted in the robot budget); psi reads each jump -> a latent, their mean (the fallen jumps' only if all fell) is
folded into the teacher; then the evaluation (as fada_adapt.py / deploy.py: --eval planefinal, --perturbed), "start"
the nominal latent (no context), "final" the calibrated one.
-> OUT/<robot>/: it0 (nominal latent) and it1 (calibrated) policies, calib/ episodes, eval_*.json, summary.json.
Run from src/ilc_mjx with the ilcmjx env; the container must be up.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="the student_ctx run's iteration dir (OUT/it<k>: policy/, ctx_params.npz, ctx.json)")
ap.add_argument("--robots", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--calib", nargs="+", default=["0.45", "0.60", "0.50,0.15", "0.60,0.10", "0.50,0.05", "0.575,0.15"],
                help="calibration goals (default: our hardware stage's training goals)")
ap.add_argument("--eval", default="planefinal", choices=("planeval", "planefinal"))
ap.add_argument("--perturbed", action="store_true")
ap.add_argument("--eval-episodes", type=int, default=4)
ap.add_argument("--seed", type=int, default=9001)
ap.add_argument("--gpu", default="0")
a = ap.parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ilc_mjx import rma_ctx  # noqa: E402

WS, CWS = "/home/henry/ilc_ws", "/ilc_ws"
cpath = lambda p: CWS + os.path.abspath(p)[len(WS):]
EVAL = dict(planeval=([0.4375, 0.5625, (0.53, 0.11), (0.51, 0.13), (0.56, 0.12), (0.52, 0.16), (0.4625, 0.06),
                       (0.6125, 0.07)], 301),
            planefinal=([0.4875, 0.6125, (0.52, 0.12), (0.54, 0.14), (0.57, 0.11), (0.51, 0.17), (0.4875, 0.035),
                         (0.6375, 0.09)], 701))
gvec = lambda g: (tuple(float(v) for v in g) if isinstance(g, (tuple, list)) else
                  tuple(float(v) for v in g.split(",")) if isinstance(g, str) and "," in g else (float(g), 0.0))
calib = [gvec(g) for g in a.calib]
pdir_run = os.path.join(a.run, "policy")
bank = json.load(open(os.path.join(pdir_run, "bank.json")))
Ndc, Nsc, _ = bank["phases"]
Nc = Ndc + Nsc
sx = np.asarray(bank["sx"], float)
ctx = json.load(open(os.path.join(a.run, "ctx.json")))
T = dict(np.load(ctx["teacher"]))
L0 = np.asarray(ctx["latent0"], float)
PSI = {k: jnp.asarray(v, jnp.float32) for k, v in np.load(os.path.join(a.run, "ctx_params.npz")).items()}


def export(w, d):
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "policy.npz"), **{k: np.asarray(v) for k, v in w.items()}, n_hidden=np.array(2))
    for f in ("bank.json", "ilc_U.npz", "ilc_best.json"):
        if os.path.exists(os.path.join(pdir_run, f)):
            shutil.copy(os.path.join(pdir_run, f), d)
    json.dump(dict(best="policy", eval_goals=None, members=[dict(name="policy", group=0, variant="policy", tag="policy",
                                                                 episodes=0, updates=0, score=None)]),
              open(os.path.join(d, "manifest.json"), "w"))


# ILC_QUAD_OVERLAY (a container path): an ilc_quad package put first on the robots' PYTHONPATH -- the box-fix reruns
# (log/dilc/plane/fixbox) fly the fixed robots while other runs keep the installed ones
OVERLAY = f"export PYTHONPATH={os.environ['ILC_QUAD_OVERLAY']}:$PYTHONPATH && " if os.environ.get("ILC_QUAD_OVERLAY") else ""


def flown(*args, **kw):
    """subprocess.run of a flight, holding a flight token (fly_tokens.py: the reruns' CPU budget)"""
    from fly_tokens import flight_token
    with flight_token():
        return subprocess.run(*args, **kw)


def fly(pdir, conds, jumps, seed, json_out, save_dir="", episodes=1):
    j = " ".join(f"--jump {gvec(g)[0]} {gvec(g)[1]}" for g in jumps)
    sd = f"--save-dir {cpath(save_dir)}" if save_dir else ""
    cmd = (f"cd /ilc_ws && source install/setup.bash && export ROS_LOCALHOST_ONLY=1 OMP_NUM_THREADS=1 && {OVERLAY}"
           f"python3 install/ilc_quad/lib/ilc_quad/dilc_execute.py run --policy {cpath(pdir)} --member policy "
           f"--conds {conds} {j} --episodes {episodes} --seed {seed} --jobs 8 --json {cpath(json_out)} {sd}")
    for attempt in range(2):
        try:
            flown(["sg", "docker", "-c", f"docker exec ilc_quad bash -lc '{cmd}'"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False, timeout=900)
            break
        except subprocess.TimeoutExpired:
            tag = os.path.abspath(json_out)[len(WS):]
            subprocess.run(["sg", "docker", "-c", f"docker exec ilc_quad pkill -9 -f {CWS + tag}"], check=False)
            print(f"flight timed out ({json_out}), attempt {attempt + 1}", flush=True)
    return json.load(open(json_out)) if os.path.exists(json_out) else []


def score(r):
    return 0.01 * r["landing_cost"] + 20.0 * r["fell"]


def run_robot(robot):
    out = os.path.join(a.out, robot)
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "deploy.log"), "w")

    def say(msg):
        print(f"[{robot}] {msg}", flush=True)
        logf.write(msg + "\n")
        logf.flush()
    p0 = os.path.join(out, "it0", "policy")
    export(rma_ctx.fold(T, L0), p0)
    rdir = os.path.join(out, "calib")
    shutil.rmtree(rdir, ignore_errors=True)
    for attempt in range(3):                         # a hung flight returns no jumps: fly again (same seed)
        fly(p0, robot, calib, a.seed, os.path.join(out, "calib.json"), rdir)
        eps = [np.load(os.path.join(rdir, f)) for f in sorted(os.listdir(rdir)) if f.endswith(".npz") and not f.startswith("ref_")] \
            if os.path.isdir(rdir) else []
        if len(eps) >= len(calib):
            break
    if not eps:
        raise SystemExit(f"{robot}: no calibration jumps")
    F = np.stack([rma_ctx.features(z["obs"], z["act"], np.asarray(z["info"], float)[:3], bool(z["fell"]), sx, Nc) for z in eps])
    fell = np.array([bool(z["fell"]) for z in eps])
    ls = np.asarray(rma_ctx.psi_b(PSI, jnp.asarray(F)))
    l = ls[~fell].mean(0) if (~fell).any() else ls.mean(0)
    say(f"calibration: {len(eps)} jumps ({int(fell.sum())} fell); latent {np.round(l, 2).tolist()} "
        f"(nominal {np.round(L0, 2).tolist()}, spread across jumps {float(ls.std(0).mean()):.3f})")
    p1 = os.path.join(out, "it1", "policy")
    export(rma_ctx.fold(T, l), p1)
    res = dict(robot=robot, latent=l.tolist(), calib_jumps=len(eps), calib_fell=int(fell.sum()), args=vars(a))
    goals, seed = EVAL[a.eval]
    for tag, pd_ in (("start", p0), ("final", p1)):
        rr = fly(pd_, robot, goals, seed, os.path.join(out, f"eval_{a.eval}_{tag}.json"), episodes=a.eval_episodes)
        res[f"{a.eval}_{tag}"] = float(np.mean([score(r) for r in rr])) if rr else None
        res[f"{a.eval}_{tag}_falls"] = int(sum(r["fell"] for r in rr))
        if a.perturbed:
            rp = fly(pd_, f"/ilc_ws/log/dilc/final701/rob_{robot}.txt", goals, seed,
                     os.path.join(out, f"eval_{a.eval}_rob_{tag}.json"), episodes=a.eval_episodes)
            res[f"rob_{tag}"] = float(np.mean([score(r) for r in rp])) if rp else None
            res[f"rob_{tag}_falls"] = int(sum(r["fell"] for r in rp))
    fm = lambda v: "n/a" if v is None else f"{v:.2f}"
    say(f"{a.eval}: nominal latent {fm(res[f'{a.eval}_start'])} -> calibrated {fm(res[f'{a.eval}_final'])}"
        + (f"; perturbed {fm(res.get('rob_start'))} -> {fm(res.get('rob_final'))}" if a.perturbed else ""))
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)
    return res


allres = [run_robot(r) for r in a.robots]
json.dump(allres, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
