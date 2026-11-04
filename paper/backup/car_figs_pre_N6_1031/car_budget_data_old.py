#!/usr/bin/env python3
"""car_budget_data.py (2026-10-29): the data of the car budget figures (paper/final/car_budget/car_budget_figs.py), per
coverage tier (car_coverage_table.py: mb_car.TIER_A / B / C_CARS, 12 each), training tracks, on the tables' normalized
error (max(rms e_y / 10 cm, (1 - pace) / 0.1), a crash = 5; mean and 95 % cluster-bootstrap CI over (seed, car, plan)),
frozen test pass (EVAL_FREE_SEED 4301, car_tiers.sh / car_extra.sh):
  sim     zero-shot vs SIM LAPS (each run's log.jsonl sim_laps): DistIL-C_sim (freeNom5) and + DR (freeDR5): bc, it005..030
          (it030 = the final network, the frozen arm); PPO + DR (freePPO, 6 plans; freePPO5, 15 plans): it100 / 200 / 300
          (it300 = final), counted from 0 after its BC start (freeBC_s0 / freeBC5_s0 for every seed, whose laps are added)
  real    vs real laps per car (0 = zero-shot): the 10-lap stages, PPO on the car (10..320)
    python car_budget_data.py   -> ../../../src/paper/final/car_budget/car_budget_data.json
"""
import json
import os

import numpy as np

from car_coverage_table import TIERS, boot, rows

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "..", "..", "src", "ilc_f1tenth", "ilc2real", "runs")
OUT = os.path.join(HERE, "..", "..", "..", "src", "paper", "final", "car_budget", "car_budget_data.json")


def laps(run, event, it):
    """sim laps after training iteration `it` (1-based, = the snapshot policy_it<it>; 0 = the BC snapshot)"""
    L = [json.loads(x) for x in open(f"{RUNS}/{run}/log.jsonl")]
    if it == 0:
        return [j for j in L if j.get("event") == "bc"][-1]["sim_laps"]
    E = [(j["it"], j["sim_laps"]) for j in L if j.get("event") == event]        # PPO logs every 10th iteration
    return float(np.interp(it - 1, *zip(*E)))


def stat(pat, cars, rng):
    R = [x for x in rows(pat) if not x[2].startswith("mocap_squareH") and x[1] in cars]
    return [float(v) for v in boot(R, rng)] if R else None


def sim_curves(rng):
    ser = {"DistILC": ("freeNom5", "ilc", 0, [0, 5, 10, 15, 20, 25, 30]),
           "DistILC + DR": ("freeDR5", "ilc", 0, [0, 5, 10, 15, 20, 25, 30]),
           "PPO + DR": ("freePPO", "ppo", "freeBC", [100, 200, 300]),
           "PPO + DR (5 tracks)": ("freePPO5", "ppo", "freeBC5", [100, 200, 300])}
    out = {}
    for arm, (run, ev, bc, its) in ser.items():
        for tier, cars in TIERS:
            pts = []
            for it in its:
                tag = "bc" if it == 0 else f"it{it:03d}"
                x = np.mean([laps(f"{run}_s{s}", ev, it) + (laps(f"{bc}_s0", "bc", 0) if bc else 0) for s in range(3)])
                st = stat(f"simh_{run}_s{{s}}_{tag}_tiers_test", cars, rng)
                if st:
                    pts.append([float(x)] + st)
            out.setdefault(arm, {})[tier] = pts
    out["LQR on the plan"] = {t: [[0.0] + stat("lqr_tiers_test", c, rng)] for t, c in TIERS}
    return out


def real_curves(rng):
    ser = {"DistILC": [(0, "freeNom5_s{s}_tiers_testzs"), (10, "freeNom5_s{s}_tiers_test+stage")],
           "DistILC + DR": [(0, "freeDR5_s{s}_tiers_testzs"), (10, "freeDR5_s{s}_tiers_test+stage")],
           "PPO + DR + DistILC": [(0, "freePPO_s{s}_tiers_testzs"), (10, "freePPO_s{s}_tiers_test+stage")],
           "PPO + DR + PPO on car": [(0, "freePPO_s{s}_tiers_testzs")] +
                                    [(n, f"freePPO_s{{s}}_tiers_real_ep{n}_test") for n in (10, 20, 40, 80, 160, 320)],
           "LQR on the plan": [(0, "lqr_tiers_test")]}
    return {arm: {t: [[n] + st for n, p in pts if (st := stat(p, c, rng))] for t, c in TIERS} for arm, pts in ser.items()}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    D = dict(tiers={t: c for t, c in TIERS}, sim=sim_curves(rng), real=real_curves(rng),
             notes="car, frozen test pass (seed 4301, _mu80 plans, training tracks), 3 seeds; each point [x, mean, lo, hi] "
                   "of the normalized error (max(rms e_y / 10 cm, (1 - pace) / 0.1), a crash = 5), 95 % cluster-bootstrap "
                   "CI over (seed, car, plan). sim: x = sim laps (PPO: + its BC start); real: x = real laps per car. "
                   "Made by ~/ilc_ws/log/dilc/car_free/car_budget_data.py.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(D, open(OUT, "w"), indent=1)
    for k in ("sim", "real"):
        for arm, d in D[k].items():
            print(k, arm, {t: [(round(p[0]), round(p[1], 2)) for p in v] for t, v in d.items() if t == "A"})
    print("->", os.path.abspath(OUT))
