#!/usr/bin/env python3
"""
quad_eval_ckpts.py --run OUT --per-iter 24 --iters 0 1 2 4 8 16 32 64 84: the reserved-test-goal evaluation
(planefinal: 8 goals x 4 episodes, seed 701, deterministic, as quad_real_ppo.py / fada_adapt.py / deploy.py) of a
fada_adapt.py run's per-iteration policies OUT/<robot>/it<k>/policy, at the given iterations -- the data-matching
study's FADA curve (k x --per-iter real jumps per robot). -> OUT/ckpt_summary.json: {jumps: {robot: success, n, falls}}.
"""
import argparse
import json
import os
import subprocess

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--robots", nargs="+", default=["real_r1", "real_s1", "real_r5"])
ap.add_argument("--iters", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32, 64, 84])
ap.add_argument("--per-iter", type=int, default=24)
ap.add_argument("--episodes", type=int, default=4)
ap.add_argument("--jobs", type=int, default=8)
ap.add_argument("--overlay", default="/ilc_ws/log/dilc/plane/fixbox/overlay")
a = ap.parse_args()
WS, CWS = "/home/henry/ilc_ws", "/ilc_ws"
cpath = lambda p: CWS + os.path.abspath(p)[len(WS):]
TEST = [(0.4875, 0.0), (0.6125, 0.0), (0.52, 0.12), (0.54, 0.14), (0.57, 0.11), (0.51, 0.17), (0.4875, 0.035),
        (0.6375, 0.09)]
XC = (f"cd /ilc_ws && source install/setup.bash && export ROS_LOCALHOST_ONLY=1 OMP_NUM_THREADS=1 && "
      f"export PYTHONPATH={a.overlay}:$PYTHONPATH && python3 install/ilc_quad/lib/ilc_quad/dilc_execute.py")
jumps = " ".join(f"--jump {g[0]} {g[1]}" for g in TEST)
res = {}
for k in a.iters:
    for r in a.robots:
        pdir = os.path.join(a.run, r, f"it{k}", "policy")
        if not os.path.exists(pdir):
            continue
        js = os.path.join(a.run, r, f"it{k}", "eval_planefinal.json")
        if not os.path.exists(js):
            cmd = (f"{XC} run --policy {cpath(pdir)} --member policy --conds {r} {jumps} --episodes {a.episodes} "
                   f"--seed 701 --jobs {a.jobs} --json {cpath(js)}")
            subprocess.run(["sg", "docker", "-c", f"docker exec ilc_quad bash -lc '{cmd}'"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=3600)
        rr = json.load(open(js)) if os.path.exists(js) else []
        ok = [(not x["fell"]) and abs(x["ex"]) <= 0.05 and abs(x["ez"]) <= 0.03 for x in rr]
        res.setdefault(str(k * a.per_iter), {})[r] = dict(n=len(rr), success=float(np.mean(ok)) if rr else None,
                                                          falls=int(sum(x["fell"] for x in rr)))
    tot = [v for v in res.get(str(k * a.per_iter), {}).values() if v["success"] is not None]
    if tot:
        p = sum(v["success"] * v["n"] for v in tot) / sum(v["n"] for v in tot)
        print(f"{k * a.per_iter} jumps/robot: success {100 * p:.1f}% over {sum(v['n'] for v in tot)} test jumps", flush=True)
    json.dump(res, open(os.path.join(a.run, "ckpt_summary.json"), "w"), indent=1)
