"""car_fig2_prac_tiers_data.py [SRC] [OUT.json] (2026-11-10, user: "why are A-C all in the same figure"): figure 2 with the coverage-tier
runs split into three panels A / B / C, as figure 1 shows them. Same runs, same scorer and lap counts as car_fig2_prac_data.py (imported,
unchanged); the only difference is that each panel's car mean keeps that tier's 12 cars (car keys tiera_* / tierb_* / tierc_*).
FIG2_REL=1: % change vs the run's own zero-shot, per car, as in car_fig2_prac_data.py."""
import json
import os
import sys

import numpy as np

sys.argv = sys.argv[:1] + sys.argv[1:]
import car_fig2_prac_data as m  # noqa: E402

OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(m.HERE, "car_fig2_prac_tiers.json")
PANELS = [["A", "A: inside DR", 12, "cars"], ["B", "B: extrapolated", 12, "cars"], ["C", "C: outside DR class", 12, "cars"]]


def make_cm(pre):
    def cm(e, k):
        cars = [c for c in e[k] if c.startswith(pre)]
        if m.REL:
            return float(np.mean([(e[k][c]["score"] / e["0"][c]["score"] - 1) * 100 for c in cars]))
        return float(np.mean([e[k][c]["score"] for c in cars]))
    return cm


def main():
    R = {a: {} for a in ("PPO + DR + DistILC", "RMA + DistILC", "DistILC", "DistILC + DR", "PPO + DR + PPO on robot",
                         "PPO + DR + GPS", "FADA + adaptation", "FADA (RMA) + adaptation")}
    for key, *_ in PANELS:
        m.cm = make_cm("tier" + key.lower() + "_")
        R["PPO + DR + DistILC"][key] = m.sb("tier", "freePPON")
        R["RMA + DistILC"][key] = m.sb("tier", "rmaN")
        R["DistILC"][key] = m.sb("tier", "ours")
        R["DistILC + DR"][key] = m.sb("tier", "oursdr")
        R["PPO + DR + PPO on robot"][key] = m.ppo("tier")
        R["PPO + DR + GPS"][key] = m.gps("tier")
        R["FADA + adaptation"][key] = m.fada("tier", "NP")
        R["FADA (RMA) + adaptation"][key] = m.fada("tier", "NR")
    D = {"panels": PANELS, "real": R, "real_ticks": [0, 2, 6, 24, 96, 384], "real_label": "laps on the deployment car (sim-to-sim)",
         "ylabel": "normalized error", "sharey": False, "ycap": {"A": 1.2, "B": 1.6, "C": 1.2}}
    if m.REL:
        D.update(ylabel="% change vs own zero-shot", sharey=True, ycap={"A": 45.0, "B": 45.0, "C": 45.0}, ylim=[-50, 47], ref_line=[0.0])
    json.dump(D, open(OUT, "w"), indent=1)
    for arm, per in R.items():
        print(f"{arm:26s}", " | ".join(f"{p}: " + " ".join(f"{x:.0f}:{v:.2f}" for x, v, lo, hi in pts) for p, pts in per.items()))
    print("->", OUT)


if __name__ == "__main__":
    main()
