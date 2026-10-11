#!/usr/bin/env python3
"""eval_picks.py: the quadruped's picked RMA / FADA baselines (src/paper/baselines.json: per method the best oracle and the best
non-oracle variant, at its best real-data budget) as ordinary runs for the figure scripts -> log/dilc/plane/fixbox/base/pick_<role>/
<robot>/eval_planefinal_{start,rob_start}.json, the protocol of every other run (planefinal: the 8 reserved test goals x 4
episodes, seed 701; perturbed: the robot's 8 rob_<robot>.txt perturbations, same goals / seed). A pick that is a fixed run (kind
'point') is copied; a data-matching checkpoint ('curve') is flown. Re-runnable: a role is redone only when its pick changed.
Run from src/ilc_mjx with the ilcmjx env; the ilc_quad container must be up."""
import argparse
import json
import os
import shutil
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", type=int, default=4)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
PL = "/home/henry/ilc_ws/log/dilc/plane"
WS, CWS = "/home/henry/ilc_ws", "/ilc_ws"
cpath = lambda p: CWS + os.path.abspath(p)[len(WS):]
TEST = [(0.4875, 0.0), (0.6125, 0.0), (0.52, 0.12), (0.54, 0.14), (0.57, 0.11), (0.51, 0.17), (0.4875, 0.035), (0.6375, 0.09)]
JUMPS = " ".join(f"--jump {g[0]} {g[1]}" for g in TEST)
XC = ("cd /ilc_ws && source install/setup.bash && export ROS_LOCALHOST_ONLY=1 OMP_NUM_THREADS=1 && "
      "export PYTHONPATH=/ilc_ws/log/dilc/plane/fixbox/overlay:$PYTHONPATH && python3 install/ilc_quad/lib/ilc_quad/dilc_execute.py")
PER_ITER = {"fada_real": 24, "fada_real_oracle": 24, "fada_real_oracle16": 64, "fada_real_true": 32}
R3 = ["real_r1", "real_s1", "real_r5"]


def fly(pdir, conds, out):
    cmd = (f"{XC} run --policy {cpath(pdir)} --member policy --conds {conds} {JUMPS} --episodes 4 --seed 701 "
           f"--jobs {a.jobs} --json {cpath(out)}")
    subprocess.run(["sg", "docker", "-c", f"docker exec ilc_quad bash -lc '{cmd}'"], stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=7200)


picks = json.load(open(os.path.join(HERE, "..", "..", "paper", "baselines.json")))["quadruped"]
for role, v in picks.items():
    b = v["best"]
    out = os.path.join(PL, "fixbox", "base", f"pick_{role}")
    meta = dict(label=b["label"], kind=b["kind"], source=b["source"], budget=b["best_budget"], oracle=b["oracle"])
    mf = os.path.join(out, "pick.json")
    if os.path.exists(mf) and json.load(open(mf)) == meta and all(
            os.path.exists(os.path.join(out, rb, "eval_planefinal_rob_start.json")) for rb in R3):
        print(f"{role}: up to date ({b['label']})", flush=True)
        continue
    if os.path.exists(out):
        shutil.rmtree(out)                               # a changed pick: the old one's evaluations go
    for rb in R3:
        d = os.path.join(out, rb)
        os.makedirs(d, exist_ok=True)
        if b["kind"] == "point":                         # a fixed run: its own nominal / perturbed evaluations at its tag
            run, tag = b["source"]
            for pre in ("", "rob_"):
                shutil.copy(os.path.join(PL, "fixbox", run, rb, f"eval_planefinal_{pre}{tag}.json"),
                            os.path.join(d, f"eval_planefinal_{pre}start.json"))
            continue
        name, k = b["source"], b["best_budget"]          # a data-matching checkpoint
        src = (os.path.join(PL, "match", name, f"ck{k}", rb) if name not in PER_ITER else
               os.path.join(PL, "match", name, rb, f"it{k // PER_ITER[name]}"))
        shutil.copy(os.path.join(src, "eval_planefinal.json"), os.path.join(d, "eval_planefinal_start.json"))
        fly(os.path.join(src, "policy"), f"/ilc_ws/log/dilc/final701/rob_{rb}.txt", os.path.join(d, "eval_planefinal_rob_start.json"))
        print(f"{role} {rb}: flown", flush=True)
    json.dump(meta, open(mf, "w"), indent=1)
    print(f"{role}: {b['label']} at {b['best_budget']} real jumps", flush=True)
print("EVAL_PICKS_DONE", flush=True)
