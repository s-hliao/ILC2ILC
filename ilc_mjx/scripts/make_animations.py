#!/usr/bin/env python3
"""
make_animations.py [--robots R ...] [--format gif|mp4] [--jobs N] [--only hwstage|compare]: every animation from the
trajectories in ../trajectories, into ../figures/ilc2real/anim/:

  hwstage_<file>_<robot>   for every hardware-stage file (gate_loose, hwstage_*): the stage batch by batch
  compare_transfer_<robot> the sim-to-sim transfer of every method that has an eval_<method> file, side by side

Defaults are small enough to commit (GIF, dpi 100, a 48-colour palette, real time at 17 fps; each well under 10 MB). --format mp4 needs ffmpeg. Runs the renders in
parallel (--jobs). Example: python scripts/make_animations.py --robots real_s1 --format mp4
"""
import argparse, glob, json, os, subprocess, sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TR = os.path.join(HERE, "..", "trajectories")
OUT = os.path.join(HERE, "..", "figures", "ilc2real", "anim")
ap = argparse.ArgumentParser()
ap.add_argument("--robots", nargs="+", default=["real_r1", "real_s1", "real_r4", "real_r5", "real_r4m"],
                help="each file is rendered for the ones of these it has")
ap.add_argument("--format", default="gif", choices=("gif", "mp4"))
ap.add_argument("--jobs", type=int, default=4)
ap.add_argument("--only", choices=("hwstage", "compare"))
ap.add_argument("--dpi", type=int, default=100)
a = ap.parse_args()
os.makedirs(OUT, exist_ok=True)
# the methods for the transfer comparison, in the figures' order, with their labels
METHODS = [("ours_zeroshot", "ours, zero-shot"), ("ours_24jumps", "ours, 24 real jumps"), ("learner_dr", "our learner + DR (A), zero-shot"),
           ("dr_finetune_24jumps", "our learner + DR (A), 24 real jumps"),
           ("dr_scratch_24jumps", "our learner + DR (B), 24 real jumps"), ("ppo_dr", "PPO + DR"), ("rma", "RMA"), ("rma_crosstrial", "RMA, cross-trial"), ("fada", "FADA (adapted)")]
jobs = []
py = sys.executable
robots_in = lambda f: {m["robot"] for m in json.loads(str(np.load(f, allow_pickle=True)["meta"]))}
anim = os.path.join(HERE, "animate_hw_stage.py")
if a.only != "compare":
    for f in sorted(glob.glob(os.path.join(TR, "gate_loose.npz")) + glob.glob(os.path.join(TR, "hwstage_*.npz"))):
        name = os.path.splitext(os.path.basename(f))[0]
        for rb in [r for r in a.robots if r in robots_in(f)]:
            jobs.append([py, anim, "hwstage", f, "--robot", rb, "--format", a.format, "--dpi", str(a.dpi),
                         "--out", os.path.join(OUT, f"hwstage_{name.replace('hwstage_', '', 1)}_{rb}.gif")])
if a.only != "hwstage":
    have = [(os.path.join(TR, f"eval_{m}.npz"), lab) for m, lab in METHODS if os.path.exists(os.path.join(TR, f"eval_{m}.npz"))]
    for rb in [r for r in a.robots if any(r in robots_in(h[0]) for h in have)]:
        jobs.append([py, anim, "compare", *[h[0] for h in have], "--labels", *[h[1] for h in have], "--robot", rb,
                     "--format", a.format, "--dpi", str(a.dpi), "--out", os.path.join(OUT, f"compare_transfer_{rb}.gif")])


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    lines = [l for l in (r.stdout + r.stderr).splitlines() if "frames ->" in l or "Error" in l or "Traceback" in l]
    return " | ".join(lines) or f"(no output) {cmd[3] if len(cmd) > 3 else ''} {cmd[-1]}"


print(f"{len(jobs)} animations, {a.jobs} at a time")
with ThreadPoolExecutor(a.jobs) as ex:
    for msg in ex.map(run, jobs):
        print(msg, flush=True)
