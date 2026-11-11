#!/usr/bin/env python3
"""budget_data.py (2026-10-29): the data of the pogo budget figures (paper/final/pogo_budget/budget_figs.py), per
coverage tier (coverage_table.py: A / B the 12 + 12 systematic robots of real_pogo.TIER_A / TIER_B (runs/tierab,
runs/tierab_ppo36, runs/simdata_test_ab; 2026-10-29, replacing nom / mass / spring and act / offset); C prog + the tier C
robots passing positivity),
on the tables' normalized error (table_cont: max(apex / 2 cm, |dv| / 0.1 m/s), a fall = 5; mean and 95 % cluster
bootstrap CI over (seed, robot, goal)), frozen test pass (seed 4301):
  sim     zero-shot vs sim hours: the snapshot runs (simdata_test.py: runs/simdata_test, runs/simdata_test_c) and the
          frozen RMA / FADA zero-shot points
  real    vs real rollouts per robot (0 = zero-shot): the frozen stages (12 / 24 / 36), PPO on the robot (24..384; 36 =
          the short runs), FADA's adaptation (24), RMA (0)
    python budget_data.py   -> ../../src/paper/final/pogo_budget/budget_data.json
"""
import glob
import json
import os
import re

import numpy as np

from coverage_table import TIERS
from table_cont import arm_rows, boot
from tierc_table import POS_MIN, ROB as TC_ROB, positivity
from ablation_table import BC


def teacher_h(run, K):
    """sim hours of a PPO run (runs/<run>/log.jsonl) after K iterations (simdata_ext.py's)"""
    P = [j for j in (json.loads(x) for x in open(f"runs/{run}/log.jsonl")) if j.get("event") == "ppo"]
    return P[-1]["sim_hours"] / (P[-1]["it"] + 1) * K

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "src", "paper", "final", "pogo_budget", "budget_data.json")
F, V, T, S, SC = "summary_test4301.json", "runs/v3", "runs/tierc", "runs/simdata_test", "runs/simdata_test_c"
SAB = "runs/simdata_test_ab"


def with_ab(p):
    """the tier A / B counterpart of a tier C path (tierab.sh writes the same layout)"""
    return (p.replace(f"{T}/", "runs/tierab/").replace("runs/tierc_ppo36/", "runs/tierab_ppo36/")
            .replace(f"{SC}/", f"{SAB}/"))


def tiers():
    pos = positivity()
    return dict(TIERS + [("C", ["real_prog"] + [r for r in TC_ROB if pos[r][0] >= POS_MIN])])


def stat(pats, robots, rng):
    pats = list(pats) + [with_ab(p) for p in pats if with_ab(p) != p]
    rows = [x for p in pats for x in arm_rows(p) if x[1] in robots and ("{s}" in p or x[0] == 0)]   # a seedless file once
    if not rows:
        return None
    m, lo, hi = boot(rows, rng)
    return [float(m), float(lo), float(hi)]


def sim_curves(TI, rng):
    its = range(5, 41, 5)
    ppo_its = (1, 2, 5, 10, 20, 50, 100, 150, 200, 250, 300, 400, 500)
    # (2026-10-30) + the continued chains (simdata_ext.py: nomX / drY segments until training stops improving)
    series = {"DistILC": ["bc"] + [f"ours_s{{s}}_it{i:03d}" for i in its] + [f"ours_s{{s}}_x{n:04d}" for n in range(50, 2001, 50)],
              "DistILC + DR": ["bc"] + [f"oursdr_s{{s}}_nom{i:03d}" for i in its] + [f"oursdr_s{{s}}_it{i:03d}" for i in its]
                              + [f"oursdr_s{{s}}_x{n:04d}" for n in range(25, 1001, 25)],
              "PPO": ["bc"] + [f"ppo_nodr_s{{s}}_it{i:03d}" for i in ppo_its],
              "PPO + DR": ["bc"] + [f"ppo_drc_s{{s}}_it{i:03d}" for i in ppo_its]}
    out = {}
    for arm, names in series.items():
        for tier, rob in TI.items():
            pts = []
            for nm in names:
                hrs = [json.load(open(f"{S}/{nm.format(s=s)}.json"))["hours"] for s in range(3)
                       if os.path.exists(f"{S}/{nm.format(s=s)}.json")]
                st = stat([f"{S}/{nm}.json", f"{SC}/{nm}.json"], rob, rng)
                if hrs and st:
                    pts.append([float(np.mean(hrs))] + st)
            out.setdefault(arm, {})[tier] = pts
    for arm, h, p, pc in (("RMA (DR)", 513.8, f"{V}/rmaR1_s{{s}}_zs/{F}", f"{T}/rma_s{{s}}_zs/summary.json"),):
        out[arm] = {tier: [[h] + stat([p, pc], rob, rng)] for tier, rob in TI.items()}
    # (2026-10-30) RMA across sim hours: teacher snapshot K + phase 2 for A iterations (rmaC_s<k>_t<K>_a<A>, simdata_ext.py), then the
    # final full-length point above
    rma = sorted({os.path.basename(f)[:-5].split("_", 2)[2] for f in glob.glob(f"{S}/rmaC_s*_t*_a*.json")})
    pts = {tier: [] for tier in TI}
    for nm in rma:
        hrs = [json.load(open(f"{S}/rmaC_s{s}_{nm}.json"))["hours"] for s in range(3) if os.path.exists(f"{S}/rmaC_s{s}_{nm}.json")]
        for tier, rob in TI.items():
            st_ = stat([f"{S}/rmaC_s{{s}}_{nm}.json", f"{SC}/rmaC_s{{s}}_{nm}.json"], rob, rng)
            if hrs and st_: pts[tier].append([float(np.mean(hrs))] + st_)
    for tier in TI:
        out["RMA (DR)"][tier] = sorted(pts[tier]) + out["RMA (DR)"][tier]
    # (2026-11-10, user: "align the fada training on pogos with our other graphs") FADA = the car's / quad's FADA: trained on exactly the
    # pogo RMA's phase-2 distillation data, oracle teacher (fada_pogo_tf.py, runs/fadaAlign/fadaRp_s<k>_it<K>_zs, all 38 robots). Hours =
    # RMA's teacher (513.8 h - its 200 x 512 x 300 x 0.01 s phase 2 = 428.47 h) + FADA's own K x 512 x 300 x 0.01 s; it200 = RMA's 513.8 h.
    # Replaces the fadaF3h8 / fadaC (fada.py) curve.
    out["FADA (DR)"] = {tier: [] for tier in TI}
    for K in (25, 50, 100, 200):
        for tier, rob in TI.items():
            st_ = stat([f"runs/fadaAlign/fadaRp_s{{s}}_it{K}_zs/{F}"], rob, rng)
            if st_: out["FADA (DR)"][tier].append([428.47 + K * 512 * 300 * 0.01 / 3600] + st_)
    return out


# the stopping-rule picks (sim_validate.py: policy_pick.npz of runs/nom_s<k>; drP: the frozen v3 stages' policy), i.e.
# the snapshots the zero-shot rows and every hardware stage start from
PICKS = {"DistILC": ["ours_s0_it015", "ours_s1_it015", "ours_s2_it015"],
         "DistILC + DR": ["oursdr_s0_it020", "oursdr_s1_it020", "oursdr_s2_it040"]}


def sim_picks(TI, rng):
    out = {}
    for arm, names in PICKS.items():
        for tier, rob in TI.items():
            rows, hrs = [], []
            for s, nm in enumerate(names):
                for d in (S, SC, SAB):
                    f = f"{d}/{nm}.json"
                    if os.path.exists(f):
                        rows += [(s,) + x[1:] for x in arm_rows(f) if x[0] == 0 and x[1] in rob]
                        hrs.append(json.load(open(f))["hours"])
            m, lo, hi = boot(rows, rng)
            out.setdefault(arm, {})[tier] = [float(np.mean(hrs)), float(m), float(lo), float(hi)]
    return out


def real_curves(TI, rng):
    stage = lambda a: [(0, f"{V}/{a}_s{{s}}_zs/{F}", f"{T}/{a}_s{{s}}_zs/summary.json"),
                       (12, f"{V}/{a}_s{{s}}_PF/it2/{F}", f"{T}/{a}_s{{s}}_PF/summary_it2.json"),
                       (24, f"{V}/{a}_s{{s}}_PF/it4/{F}", f"{T}/{a}_s{{s}}_PF/summary_it4.json"),
                       (36, f"{V}/{a}_s{{s}}_PF/{F}", f"{T}/{a}_s{{s}}_PF/summary.json")]
    ppo = [(0, f"{V}/ppo_dr_s{{s}}_zs/{F}", f"{T}/ppo_dr_s{{s}}_zs/summary.json"),
           (24, f"{V}/ppo_s{{s}}_ep24/{F}", f"{T}/ppo_s{{s}}_ep24/summary.json"),
           (36, "runs/ppo36/ppo_s{s}_ep36/summary.json", "runs/tierc_ppo36/ppo_s{s}_ep36/summary.json")] + \
          [(n, f"{V}/ppo_s{{s}}_ep{n}/{F}", f"{T}/ppo_s{{s}}_ep{n}/summary.json") for n in (48, 96, 192, 384)]
    series = {"DistILC": stage("nom"), "DistILC + DR": stage("drP"), "PPO + DR + DistILC": stage("ppo_dr"),
              "PPO + DR + PPO on robot": ppo,
              # (2026-11-10) aligned FADA (fada_pogo_tf.py + fada_real_pogo_tf.py, the car's / quad's T1-T3): no-teacher FADA from our
              # PPO + DR (fadaNPp) and from our RMA (fadaNRp), adapted with K real rollouts per robot
              "FADA (DR) + adaptation": [(K, f"runs/fadaAlign/fadaNPp_s{{s}}_ad_ep{K}/{F}", None) for K in (0, 12, 24, 36, 96, 384)],
              "FADA (RMA) + adaptation": [(K, f"runs/fadaAlign/fadaNRp_s{{s}}_ad_ep{K}/{F}", None) for K in (0, 12, 24, 36, 96, 384)],
              # (2026-11-10) our stage on the deployed RMA (hw_stage_rma.py: z_hat from the trial history, frozen adaptation module)
              "RMA + DistILC": [(0, f"{V}/rmaR1_s{{s}}_zs/{F}", f"{T}/rma_s{{s}}_zs/summary.json"),
                                (12, f"{V}/rmaR1_s{{s}}_PF/it2/{F}", f"{T}/rma_s{{s}}_PF/it2/{F}"),
                                (24, f"{V}/rmaR1_s{{s}}_PF/it4/{F}", f"{T}/rma_s{{s}}_PF/it4/{F}"),
                                (36, f"{V}/rmaR1_s{{s}}_PF/{F}", f"{T}/rma_s{{s}}_PF/{F}")],
              # (2026-11-10) GPS from PPO + DR (gps_pogo.py, MDGPS; 30 real rollouts per robot per iteration)
              "PPO + DR + GPS": [(30 * k, f"{V}/gps_s{{s}}_it{k}/{F}", f"{T}/gps_s{{s}}_it{k}/{F}") for k in (0, 1, 2, 4, 8, 12)],
              "RMA (DR), zero-shot": [(0, f"{V}/rmaR1_s{{s}}_zs/{F}", f"{T}/rma_s{{s}}_zs/summary.json")]}
    out = {}
    for arm, pts in series.items():
        for tier, rob in TI.items():
            out.setdefault(arm, {})[tier] = [[n] + st for n, p, pc in pts if (st := stat([q for q in (p, pc) if q], rob, rng))]
    return out


if __name__ == "__main__":
    os.chdir(HERE)
    TI = tiers()
    rng = np.random.default_rng(0)
    D = dict(tiers=TI, sim=sim_curves(TI, rng), sim_picks=sim_picks(TI, rng), real=real_curves(TI, rng),
             notes="frozen test pass (seed 4301), 3 seeds; each point [x, mean, lo, hi] of the normalized error "
                   "(max(apex / 2 cm, |dv| / 0.1 m/s), a fall = 5), 95 % cluster-bootstrap CI over (seed, robot, goal). "
                   "sim: x = sim hours (all arms from the 1.8 h Raibert-cloned start); sim_picks: the stopping-rule snapshots "
                   "(x = their mean hours over seeds); real: x = real rollouts per robot. "
                   "Made by ~/ilc_ws/log/pogo_servo/budget_data.py.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(D, open(OUT, "w"), indent=1)
    for k in ("sim", "real"):
        for arm, d in D[k].items():
            print(k, arm, {t: len(v) for t, v in d.items()})
    print("->", os.path.abspath(OUT))
