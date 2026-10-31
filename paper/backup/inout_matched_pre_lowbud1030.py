#!/usr/bin/env python3
"""inout_matched.py (2026-10-29): goal-space generalization (IN = the stage's 6 training goals; HULL / OUT = the
validation goals inside / outside their hull, flows_out/summary_extra.json) under the TABLES' protocol and metric only
(real_pogo.evaluate: 4 perturbed drops x 10 s per goal, the normalized error of table_cont.run_err), on the 6 sticks,
3 seeds, for three budget comparisons (paper/final/pogo_inout/inout_figs.py):
  matched sim hours  zero-shot at ~11 sim h: DistIL-C_sim (its frozen pick; the 40-iteration stage = 11.0 h) vs PPO + DR
                     and PPO (no DR) at their it010 snapshots (10.3 h)
  matched real       36 real rollouts per robot: the three DistIL-C_hw stages vs PPO + DR + PPO_hw (runs/ppo36 ep36);
                     FADA at its 24 (its recipe's budget)
  unlimited          each method at its full budget: the DistIL-C_hw stages (36), PPO_hw at 384, RMA (514 sim h, zero-shot),
                     FADA (+24)
    POGO_LEG=servo python inout_matched.py [--jobs 20]   -> runs/inout_matched/<arm>_s<k>.json ({goal_sets, runs})
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "runs")
OUT = os.path.join(R, "inout_matched")
sys.path.insert(0, HERE)
GS = json.load(open(os.path.join(R, "video", "flows_out", "summary_extra.json")))["goal_sets"]


def pick(arm, s):
    return os.path.join(HERE, json.load(open(f"{R}/v3/{arm}_s{s}_PF/args.json"))["policy"])


# arm -> per seed: a policy file (one network for every stick) or a directory of <stick>_policy.npz
ARMS = {"distilc_sim": lambda s: pick("nom", s),
        "distilc_dr_sim": lambda s: pick("drP", s),
        "ppo_dr_sim": lambda s: f"{R}/ppo_dr_s{s}/policy_final.npz",
        "ppo_dr_10h": lambda s: f"{R}/ppo_drc_s{s}/policy_it010.npz",
        "ppo_10h": lambda s: f"{R}/ppo_nodr_s{s}/policy_it010.npz",
        "distilc_hw36": lambda s: f"{R}/v3/nom_s{s}_PF",
        "distilc_dr_hw36": lambda s: f"{R}/v3/drP_s{s}_PF",
        "ppo_dr_distilc_hw36": lambda s: f"{R}/v3/ppo_dr_s{s}_PF",
        "ppo_dr_ppo_hw36": lambda s: f"{R}/ppo36/ppo_s{s}_ep36",
        "ppo_dr_ppo_hw384": lambda s: f"{R}/v3/ppo_s{s}_ep384",
        "fada_hw24": lambda s: f"{R}/v3/fadaF3h8_s{s}_hw24",
        "rma": lambda s: f"{R}/rmaR1full_s{s}/policy_final.npz"}


def run(job):
    arm, s = job
    out = os.path.join(OUT, f"{arm}_s{s}.json")
    if os.path.exists(out):
        return out
    import numpy as np
    import policies
    import real_pogo as rp
    src = ARMS[arm](s)
    rob = list(rp.DEFAULT_ROBOTS)
    pols = {r: policies.load_any(os.path.join(src, f"{r}_policy.npz") if os.path.isdir(src) else src) for r in rob}
    res = {}
    for g, goals in GS.items():
        res[g] = rp.evaluate(pols, np.array(goals, float), rob, runs=4, seconds=10.0)
    json.dump(dict(arm=arm, seed=s, source=src, goal_sets=GS, runs=res), open(out, "w"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    J = [(arm, s) for arm in ("rma", "ppo_dr_ppo_hw384", "fada_hw24") + tuple(k for k in ARMS if k not in ("rma", "ppo_dr_ppo_hw384", "fada_hw24"))
         for s in range(3)]                                   # RMA (slowest) first
    with Pool(a.jobs) as pool:
        for o in pool.imap_unordered(run, J):
            print("done", os.path.basename(o), flush=True)
    print("INOUT_MATCHED_DONE", flush=True)
