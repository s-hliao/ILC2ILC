#!/usr/bin/env python3
"""
quad_eval_session.py RUN --policy TAG=DIR [TAG=DIR ...] --goals G [G ...] (2026-11-02): the hardware EVALUATION of the quad arms
(QUAD_DEPLOY_FL_1102.md section 7), served to the robot through the unchanged hw_session.py. On the GPU machine it writes one
REQUEST.json at a time into RUN/it<k>/ (as deploy.py --backend manual does), each = one policy x every goal x --reps jumps, waits
until hw_session.py has flown it (the episodes in RUN/it<k>/real), then writes the next. The policies alternate block by block
(order reversed every other round, so no policy always flies first or last), --rounds rounds: jumps per goal per policy =
rounds x reps. At the end it writes RUN/summary.json, which ends hw_session.py. No learning happens here.

    python3 quad_eval_session.py RUN --policy zs=.../fl2s/ppo_dr/it50/policy A=<RUN_A>/go1/it6/policy \
        --goals 0.4375 0.5625 0.4625,0.06 0.6125,0.07 --reps 2 --rounds 2 [--start tallh]
    # robot workstation:  python3 hw_session.py RUN --pose-topic ... (as for the stage)
    python3 quad_eval_session.py RUN --score          # per policy and goal: nerr, falls; change vs the first policy

--start NAME: every jump from that stance (hw_session.py's STARTS, written into each request's "starts", so the robot side needs no
flag). A physical perturbation (a block under the feet) is the operator's: name it with --note, it goes into REQUEST.md and the score.
nerr = max(|ex| / 5 cm, |ez| / 3 cm), a fall = 5 (the simulation's score).
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

STARTS = {"nominal": [0.0, 0.0, 0.0, 0.0], "crouch": [0.1, -0.2, 0.1, -0.2], "tall": [-0.1, 0.2, -0.1, 0.2],
          "noseup": [0.0, 0.0, 0.1, -0.2], "nosedn": [0.1, -0.2, 0.0, 0.0],
          "tallh": [-0.05, 0.1, -0.05, 0.1], "noseuph": [0.0, 0.0, 0.05, -0.1]}      # = hw_session.py's STARTS

ap = argparse.ArgumentParser()
ap.add_argument("run", help="the shared folder hw_session.py is pointed at (one per evaluation condition)")
ap.add_argument("--policy", nargs="+", default=[], metavar="TAG=DIR", help="policy folders (each with manifest.json, bank.json)")
ap.add_argument("--goals", nargs="+", default=[], help="x or x,h (m)")
ap.add_argument("--reps", type=int, default=2, help="jumps per goal in one block")
ap.add_argument("--rounds", type=int, default=2, help="blocks per policy")
ap.add_argument("--robot", default="go1")
ap.add_argument("--start", default="nominal", choices=tuple(STARTS))
ap.add_argument("--note", default="", help="the operator's physical perturbation, e.g. 'front-feet block 1.5 cm'")
ap.add_argument("--score", action="store_true")
a = ap.parse_args()


def gkey(g):
    g = np.atleast_1d(np.asarray(g, float))
    return (round(float(g[0]), 4), round(float(g[1]) if len(g) > 1 else 0.0, 4))


def episodes(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        if os.path.basename(f).startswith("ref_"):
            continue
        try:
            z = np.load(f, allow_pickle=True)
            info = np.asarray(z["info"], float)
            out.append(dict(goal=gkey(z["goal"]), ex=float(info[0]), ez=float(info[1]), pitch=float(info[2]), fell=bool(float(z["fell"]) > 0.5), file=f))
        except Exception:
            pass
    return out


def nerr(e):
    return 5.0 if e["fell"] else max(abs(e["ex"]) / 0.05, abs(e["ez"]) / 0.03)


if a.score:
    rows = {}
    for f in sorted(glob.glob(os.path.join(a.run, "it*", "REQUEST.json")), key=lambda p: int(p.split("/it")[-1].split("/")[0])):
        r = json.load(open(f))
        for e in episodes(os.path.join(os.path.dirname(f), "real")):
            rows.setdefault(r["tag"], {}).setdefault(e["goal"], []).append(e)
    if not rows:
        raise SystemExit("no episodes yet")
    tags = list(rows)
    note = json.load(open(glob.glob(os.path.join(a.run, "it0", "REQUEST.json"))[0])).get("condition", "")
    print(f"{a.run}: {note}")
    goals = sorted({g for t in tags for g in rows[t]}, key=lambda g: (g[1], g[0]))
    print("goal            " + "".join(f"| {t:>22s} " for t in tags))
    for g in goals:
        print(f"({g[0]:.4f}, {g[1]:.2f}) " + "".join(
            f"| n{len(rows[t].get(g, [])):2d} nerr {np.mean([nerr(e) for e in rows[t][g]]):4.2f} fell {sum(e['fell'] for e in rows[t][g]):d} "
            if g in rows[t] else "|" + " " * 24 for t in tags))
    base = tags[0]
    for t in tags:
        al = [nerr(e) for g in rows[t] for e in rows[t][g]]
        msg = f"{t}: {len(al)} jumps, nerr {np.mean(al):.2f}, falls {100 * np.mean([e['fell'] for g in rows[t] for e in rows[t][g]]):.0f} %"
        if t != base:
            common = [g for g in rows[t] if g in rows[base]]
            d = np.array([np.mean([nerr(e) for e in rows[t][g]]) - np.mean([nerr(e) for e in rows[base][g]]) for g in common])
            msg += f"; change vs {base}, mean over {len(common)} goals: {d.mean():+.2f} (goals better: {int((d < 0).sum())} of {len(d)})"
        print(msg)
    sys.exit(0)

if not a.policy or not a.goals:
    raise SystemExit("--policy and --goals are required (or --score)")
pols = []
for s in a.policy:
    t, d = s.split("=", 1)
    d = os.path.abspath(d)
    if not os.path.exists(os.path.join(d, "manifest.json")):
        raise SystemExit(f"{d}: no manifest.json (a policy folder: <RUN>/go1/it6/policy, or .../ppo_dr/it50/policy)")
    pols.append((t, d))
goals = [gkey([float(v) for v in g.split(",")]) for g in a.goals]
if os.path.exists(os.path.join(a.run, "summary.json")):
    raise SystemExit(f"{a.run} already finished (summary.json); use a new folder, or --score")
order = []
for k in range(a.rounds):
    order += pols if k % 2 == 0 else pols[::-1]
os.makedirs(a.run, exist_ok=True)
cond = a.start + (f" + {a.note}" if a.note else "")
for k, (tag, pdir) in enumerate(order):
    d = os.path.join(a.run, f"it{k}")
    rf = os.path.join(d, "REQUEST.json")
    if not os.path.exists(rf):
        os.makedirs(os.path.join(d, "real"), exist_ok=True)
        req = dict(robot=a.robot, it=k, policy=pdir, episode_dir=os.path.abspath(os.path.join(d, "real")),
                   goals=[list(g) for g in goals], reps=a.reps, tag=tag, condition=cond)
        if a.start != "nominal":
            req["starts"] = [[g[0], g[1], STARTS[a.start]] for g in goals]
        json.dump(req, open(rf + ".tmp", "w"), indent=1)
        os.replace(rf + ".tmp", rf)
        open(os.path.join(d, "REQUEST.md"), "w").write(
            f"# evaluation block {k + 1} of {len(order)}: policy {tag}\n\npolicy: {pdir}\ncondition: {cond}\n"
            f"goals ({a.reps} jumps each): {', '.join(f'({g[0]}, {g[1]})' for g in goals)}\n"
            + (f"\nOPERATOR: {a.note} -- before every jump of this block\n" if a.note else ""))
    print(f"block {k + 1}/{len(order)}: {tag}, {len(goals)} goals x {a.reps} -> {rf}", flush=True)
    while True:
        if os.path.exists(os.path.join(a.run, "ABORT")):
            raise SystemExit("ABORT")
        have = {}
        for e in episodes(os.path.join(d, "real")):
            have[e["goal"]] = have.get(e["goal"], 0) + 1
        if all(have.get(g, 0) >= a.reps for g in goals):
            break
        time.sleep(3)
    print(f"   flown: {sum(have.values())} jumps", flush=True)
json.dump(dict(evaluation=True, blocks=len(order), policies=dict(pols), goals=[list(g) for g in goals], condition=cond),
          open(os.path.join(a.run, "summary.json"), "w"), indent=1)
print(f"done; score with: python3 {os.path.basename(__file__)} {a.run} --score", flush=True)
