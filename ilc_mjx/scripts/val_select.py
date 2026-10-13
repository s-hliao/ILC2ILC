#!/usr/bin/env python3
"""
val_select.py --kind rma|fada --runs RUN [RUN ...] --out OUT.json: per-robot checkpoint selection on a VALIDATION goal set
(never the reserved test goals), the standard early-stopping protocol for the data-matching baselines whose curves oscillate
(the exact-dynamics RMA: per-robot peaks r1 56 / s1 56 / r5 22 % at different checkpoints, 32 test jumps each).

  validation goals  8 goals in the test region's box, disjoint from the 8 test goals (VAL below), 4 episodes each, seed 702
  candidates        --kind rma:  RUN/ck<k>/<robot>/policy (quad_real_ppo.py), budget k jumps (+ --offset RUN=N)
                    --kind fada: RUN/<robot>/it<k>/policy (fada_adapt.py), budget from RUN/ckpt_summary.json's keys
  pick              per robot: the best validation success (ties -> fewer jumps); its TEST success is read from the
                    run's own test evaluation (RUN/ck<k>/<robot>/eval_planefinal.json or RUN/<robot>/it<k>/eval_planefinal.json)
-> OUT.json: every candidate's validation success, the picks and their test success (mean over robots = the selected
baseline's score, comparable to ours: 43.8 %).
"""
import argparse
import glob
import json
import os
import subprocess

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=("rma", "fada"), required=True)
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--offset", nargs="*", default=[], help="RUN=N: the run continued another for N jumps (budget offset)")
ap.add_argument("--robots", nargs="+", default=["real_r1", "real_s1", "real_r5"])
ap.add_argument("--episodes", type=int, default=4)
ap.add_argument("--jobs", type=int, default=6)
ap.add_argument("--overlay", default="/ilc_ws/log/dilc/plane/fixbox/overlay")
ap.add_argument("--out", required=True)
a = ap.parse_args()
WS, CWS = "/home/henry/ilc_ws", "/ilc_ws"
cpath = lambda p: CWS + os.path.realpath(p)[len(WS):]
TEST = [(0.4875, 0.0), (0.6125, 0.0), (0.52, 0.12), (0.54, 0.14), (0.57, 0.11), (0.51, 0.17), (0.4875, 0.035), (0.6375, 0.09)]
VAL = [(0.50, 0.0), (0.575, 0.0), (0.55, 0.06), (0.60, 0.05), (0.53, 0.10), (0.58, 0.16), (0.49, 0.08), (0.62, 0.12)]
assert not set(VAL) & set(TEST)
XC = (f"cd /ilc_ws && source install/setup.bash && export ROS_LOCALHOST_ONLY=1 OMP_NUM_THREADS=1 && "
      f"export PYTHONPATH={a.overlay}:$PYTHONPATH && python3 install/ilc_quad/lib/ilc_quad/dilc_execute.py")
ok = lambda x: (not x["fell"]) and abs(x["ex"]) <= 0.05 and abs(x["ez"]) <= 0.03
offset = {kv.split("=")[0].rstrip("/"): int(kv.split("=")[1]) for kv in a.offset}


def candidates():
    for run in a.runs:
        run = run.rstrip("/")
        name = os.path.basename(run)
        if a.kind == "rma":
            for d in glob.glob(os.path.join(run, "ck*")):
                k = int(os.path.basename(d)[2:])
                for r in a.robots:
                    p = os.path.join(d, r, "policy")
                    if os.path.exists(p):
                        yield dict(run=name, k=k, budget=k + offset.get(run, offset.get(name, 0)), robot=r, policy=p,
                                   test=os.path.join(d, r, "eval_planefinal.json"))
        else:
            for r in a.robots:                      # budget: the adaptation iteration (x the run's jumps per iteration)
                for d in glob.glob(os.path.join(run, r, "it*")):
                    k = int(os.path.basename(d)[2:])
                    p = os.path.join(d, "policy")
                    if os.path.exists(p):
                        yield dict(run=name, k=k, budget=k, robot=r, policy=p, test=os.path.join(d, "eval_planefinal.json"))


def score(js):
    rr = json.load(open(js))
    rr = rr.get("episodes", rr) if isinstance(rr, dict) else rr
    return (100.0 * float(np.mean([ok(x) for x in rr])), len(rr)) if rr else (None, 0)


jumps = " ".join(f"--jump {g[0]} {g[1]}" for g in VAL)
rows = []
for c in candidates():
    if not os.path.exists(c["test"]):              # no test evaluation to report for it: not a candidate
        continue
    vj = os.path.join(os.path.dirname(c["policy"]), f"{c['robot']}_eval_val.json" if a.kind == "rma" else "eval_val.json")
    if not os.path.exists(vj):
        cmd = (f"{XC} run --policy {cpath(c['policy'])} --member policy --conds {c['robot']} {jumps} --episodes {a.episodes} "
               f"--seed 702 --jobs {a.jobs} --json {cpath(vj)}")
        subprocess.run(["sg", "docker", "-c", f"docker exec ilc_quad bash -lc '{cmd}'"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=3600)
    c["val"], c["n_val"] = score(vj) if os.path.exists(vj) else (None, 0)
    c["test_score"] = score(c["test"])[0] if os.path.exists(c["test"]) else None
    rows.append(c)
    print(json.dumps({k: c[k] for k in ("run", "k", "robot", "val", "test_score")}), flush=True)
picks = {}
for r in a.robots:
    cand = [c for c in rows if c["robot"] == r and c["val"] is not None and c["test_score"] is not None]
    if cand:
        best = max(cand, key=lambda c: (c["val"], -c["budget"]))
        picks[r] = {k: best[k] for k in ("run", "k", "budget", "val", "test_score", "policy")}
mean = float(np.mean([p["test_score"] for p in picks.values()])) if picks else None
json.dump(dict(val_goals=VAL, picks=picks, selected_test_mean=mean, max_budget=max((p["budget"] for p in picks.values()), default=0),
               candidates=rows), open(a.out, "w"), indent=1)
print(json.dumps(dict(picks={r: (p["run"], p["k"], round(p["val"], 1), round(p["test_score"], 1)) for r, p in picks.items()},
                      selected_test_mean=mean)), flush=True)
