#!/usr/bin/env python3
"""
make_animations.py [--robots R ...] [--format mp4|gif] [--jobs N] [--only hwstage|eval|compare] [--method M ...]:
every video from the recorded jumps in ../trajectories, into ../figures/ilc2real/videos/, one folder per method:

  00_compare_all_methods/<robot>          every method's test-goal evaluation side by side, goal by goal
  <NN>_<method>/hardware_stage_<robot>    the method's real jumps (its hardware stage, or calibration / adaptation),
                                          batch by batch on its training goals
  <NN>_<method>/<when>_test_goals_<robot> the method's evaluation on the 8 reserved test goals (never trained on,
                                          never flown in the hardware stage), all 8 goals at once
                                          (<when>: zeroshot, after_24jumps, ...)

The table METHODS below says which recorded file goes where. MP4 (H.264, pausable and seekable in any player; needs
the imageio-ffmpeg wheel or a system ffmpeg) at real time, 17 fps. Renders in parallel (--jobs).
Example: python scripts/make_animations.py --robots real_s1 --method 01_ours
Replay any of these jumps in MuJoCo instead: scripts/replay_mujoco.py (see videos/README.md).
"""
import argparse, glob, json, os, subprocess, sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TR = os.path.join(HERE, "..", "trajectories")
OUT = os.path.join(HERE, "..", "figures", "ilc2real", "videos")
TP = ["blk15", "blk2", "crouch", "tall", "noseup", "nosedn", "mocapbad", "delay10"]
# folder: (label, hardware-stage file or None, [(when, eval file)])
METHODS = {
    "01_ours": ("ours", "gate_loose", [("zeroshot", "eval_ours_zeroshot"), ("after_24jumps", "eval_ours_24jumps")]),
    "02_ours_plus_dr_finetune": ("our learner + DR (A: DR fine-tune)", "hwstage_drA_hw24",
                                 [("zeroshot", "eval_learner_dr"), ("after_24jumps", "eval_dr_finetune_24jumps")]),
    "03_ours_plus_dr_scratch": ("our learner + DR (B: DR from scratch)", "hwstage_drB_hw24",
                                [("after_24jumps", "eval_dr_scratch_24jumps")]),
    **{f"04_ours_trained_under_perturbation/{p}": (f"ours, hardware stage under '{p}'", f"hwstage_tp_{p}",
                                                   [("after_24jumps", f"eval_tp_{p}")]) for p in TP},
    "05_ppo_dr": ("PPO + DR", None, [("zeroshot", "eval_ppo_dr")]),
    "06_rma": ("RMA", None, [("zeroshot", "eval_rma")]),
    "07_rma_crosstrial": ("RMA, cross-trial", "hwstage_rma_crosstrial_calibration", [("after_calibration", "eval_rma_crosstrial")]),
    "08_fada": ("FADA", "hwstage_fada_lora", [("after_adaptation", "eval_fada")]),
    "09_jumpilc_per_goal": ("per-goal ILC (JumpILC)", "hwstage_jumpilc_pergoal", []),   # its trials ARE on the test goals
}
WHEN = dict(zeroshot="zero-shot", after_24jumps="after 24 real jumps", after_calibration="after its 6 calibration jumps",
            after_adaptation="after its on-robot adaptation")
# the transfer comparison, in the figures' order
COMPARE = [("eval_ours_zeroshot", "ours, zero-shot"), ("eval_ours_24jumps", "ours, 24 real jumps"),
           ("eval_learner_dr", "our learner + DR (A), zero-shot"), ("eval_dr_finetune_24jumps", "our learner + DR (A), 24 real jumps"),
           ("eval_dr_scratch_24jumps", "our learner + DR (B), 24 real jumps"), ("eval_ppo_dr", "PPO + DR"), ("eval_rma", "RMA"),
           ("eval_rma_crosstrial", "RMA, cross-trial"), ("eval_fada", "FADA (adapted)")]

ap = argparse.ArgumentParser()
ap.add_argument("--robots", nargs="+", default=["real_r1", "real_s1", "real_r4", "real_r5", "real_r4m"],
                help="each file is rendered for the ones of these it has")
ap.add_argument("--format", default="mp4", choices=("mp4", "gif"))
ap.add_argument("--jobs", type=int, default=4)
ap.add_argument("--only", choices=("hwstage", "eval", "compare", "varied"))
ap.add_argument("--method", nargs="+", help="only these folders of METHODS")
ap.add_argument("--dpi", type=int, default=100)
a = ap.parse_args()

py, anim = sys.executable, os.path.join(HERE, "animate_hw_stage.py")
# an evaluation recorded with more episodes per goal (trajectories/eval4/: 4 per goal, the statistics' own seeds) is
# preferred over the 1-episode replay file of the same name
path = lambda name: next(p for p in (os.path.join(TR, "eval4", f"{name}.npz"), os.path.join(TR, f"{name}.npz"))
                         if os.path.exists(p) or p.endswith(f"{os.sep}{name}.npz") and os.sep + "eval4" + os.sep not in p)
robots_in = lambda f: {m["robot"] for m in json.loads(str(np.load(f, allow_pickle=True)["meta"]))}
common = ["--format", a.format, "--dpi", str(a.dpi)]
jobs, missing = [], []
for folder, (label, hw, evals) in METHODS.items():
    if a.method and folder not in a.method:
        continue
    d = os.path.join(OUT, folder)
    if hw and a.only in (None, "hwstage"):
        if os.path.exists(path(hw)):
            for rb in [r for r in a.robots if r in robots_in(path(hw))]:
                jobs.append([py, anim, "hwstage", path(hw), "--robot", rb, *common,
                             "--out", os.path.join(d, f"hardware_stage_{rb}.{a.format}")])
        else:
            missing.append(hw)
    for when, ev in evals if a.only in (None, "eval") else []:
        if not os.path.exists(path(ev)):
            missing.append(ev)
            continue
        for rb in [r for r in a.robots if r in robots_in(path(ev))]:
            jobs.append([py, anim, "eval", path(ev), "--robot", rb, "--label", f"{label}, {WHEN[when]}", *common,
                         "--out", os.path.join(d, f"{when}_test_goals_{rb}.{a.format}")])
if a.only in (None, "compare") and not a.method:
    have = [(path(f), lab) for f, lab in COMPARE if os.path.exists(path(f))]
    for rb in [r for r in a.robots if any(r in robots_in(h[0]) for h in have)]:
        jobs.append([py, anim, "compare", *[h[0] for h in have], "--labels", *[h[1] for h in have], "--robot", rb, *common,
                     "--out", os.path.join(OUT, "00_compare_all_methods", f"{rb}.{a.format}")])


# the varied recordings (fixbox/record_varied.sh -> trajectories/varied/): 18 held-out goals across the goal plane, and
# the 8 standard perturbations on the test goals -- per method and side by side
VARIED = {"ours_zeroshot": ("01_ours", "zeroshot", "ours, zero-shot"),
          "ours_24jumps": ("01_ours", "after_24jumps", "ours, 24 real jumps"),
          "learner_dr": ("02_ours_plus_dr_finetune", "zeroshot", "our learner + DR (A), zero-shot"),
          "dr_finetune_24jumps": ("02_ours_plus_dr_finetune", "after_24jumps", "our learner + DR (A), 24 real jumps"),
          "dr_scratch_24jumps": ("03_ours_plus_dr_scratch", "after_24jumps", "our learner + DR (B), 24 real jumps"),
          "ppo_dr": ("05_ppo_dr", "zeroshot", "PPO + DR"), "rma": ("06_rma", "zeroshot", "RMA"),
          "rma_crosstrial": ("07_rma_crosstrial", "after_calibration", "RMA, cross-trial"),
          "fada": ("08_fada", "after_adaptation", "FADA (adapted)")}
VD = os.path.join(TR, "varied")
KIND = dict(plane=("heldout_plane", "18 held-out goals across the goal plane (never trained on; incl. x 0.40 / 0.65 m and the 20 cm box)"),
            pert=("perturbed_test_goals", "the reserved test goals"))
if a.only in (None, "varied") and not a.method and os.path.isdir(VD):
    for kind, (stem, glab) in KIND.items():
        have = []
        for m, (folder, when, lab) in VARIED.items():
            f_ = os.path.join(VD, f"{kind}_{m}.npz")
            if not os.path.exists(f_):
                continue
            have.append((f_, lab))
            for rb in [r for r in a.robots if r in robots_in(f_)]:
                jobs.append([py, anim, "eval", f_, "--robot", rb, "--label", lab, "--goals-label", glab, *common,
                             "--out", os.path.join(OUT, folder, f"{when}_{stem}_{rb}.{a.format}")])
        for rb in [r for r in a.robots if any(r in robots_in(h[0]) for h in have)]:
            jobs.append([py, anim, "compare", *[h[0] for h in have], "--labels", *[h[1] for h in have], "--robot", rb,
                         *common, "--out", os.path.join(OUT, "00_compare_all_methods", f"{stem}_{rb}.{a.format}")])


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    lines = [l for l in (r.stdout + r.stderr).splitlines() if "frames ->" in l or "Error" in l or "Traceback" in l]
    return " | ".join(lines) or f"(no output) {cmd[3] if len(cmd) > 3 else ''} {cmd[-1]}"


if missing:
    print("not recorded (skipped):", " ".join(missing))
print(f"{len(jobs)} videos, {a.jobs} at a time")
with ThreadPoolExecutor(a.jobs) as ex:
    for msg in ex.map(run, jobs):
        print(msg.replace(os.path.abspath(OUT) + "/", "").replace(OUT + "/", ""), flush=True)
