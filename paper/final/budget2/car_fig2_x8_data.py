"""car_fig2_x8_data.py [OUT.json] (2026-11-10, user: "might need to run a couple extra laps on distILC improvement ... chose the best one it
converges to and stop there"): figure 2 (tiers A / B / C) with every DistILC-stage arm from the 8-iteration runs (ube sbX_one.sh / sbX_fix*.sh
-> fig2_src_x8/ube/SB_tier_<net>_s<s>_ss9001_x8, 3 network seeds x stage seed 9001; "ours" = the RETRAINED DistILC, not _preJ), scored on the
practised plan (sb_practised.json, its 0..8) exactly as car_fig2_prac_tiers_data.py (imported; baselines unchanged, from fig2_src_prac).
SELECTION RULE (written before the results were looked at): each DistILC curve stops at the iteration with the lowest mean score over all
36 tier cars (A + B + C pooled, mean over seeds); all three panels are cut at that same iteration; the last point is kept in "last"."""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "car_fig2_x8.json")
sys.argv = sys.argv[:1]
import car_fig2_prac_data as m           # noqa: E402
import car_fig2_prac_tiers_data as t     # noqa: E402

X8 = os.path.join(HERE, "fig2_src_x8", "ube")


def sb_x8(panel, net):
    runs = []
    for d in sorted(glob.glob(f"{X8}/SB_{panel}_{net}_s?_ss9001_x8")):
        if not os.path.exists(f"{d}/sb_practised.json"):
            print("  missing", os.path.basename(d)); continue
        e, s = m.ev(f"{d}/sb_practised.json"), json.load(open(f"{d}/summary.json"))
        per_trial = np.mean([s["real_laps"][c] / max(1, len(s["trials"][c])) for c in s["real_laps"]])
        runs.append({int(k): (int(k) * per_trial, m.cm(e, k)) for k in e})
    return m.curve(runs)


def pooled_best(net):
    """iteration with the lowest mean over all tier cars (and seeds)"""
    per_k = {}
    for d in sorted(glob.glob(f"{X8}/SB_tier_{net}_s?_ss9001_x8")):
        if os.path.exists(f"{d}/sb_practised.json"):
            e = m.ev(f"{d}/sb_practised.json")
            for k in e:
                per_k.setdefault(int(k), []).append(np.mean([v["score"] for v in e[k].values()]))
    means = {k: float(np.mean(v)) for k, v in per_k.items()}
    return min(means, key=means.get), means


def main():
    R = {a: {} for a in ("PPO + DR + DistILC", "RMA + DistILC", "DistILC", "DistILC + DR", "PPO + DR + PPO on robot",
                         "PPO + DR + GPS", "FADA + adaptation", "FADA (RMA) + adaptation")}
    NET = {"PPO + DR + DistILC": "freePPON", "RMA + DistILC": "rmaN", "DistILC": "ours", "DistILC + DR": "oursdr"}
    for key, *_ in t.PANELS:
        m.cm = t.make_cm("tier" + key.lower() + "_")
        for arm, net in NET.items():
            R[arm][key] = sb_x8("tier", net)
        R["PPO + DR + PPO on robot"][key] = m.ppo("tier")
        R["PPO + DR + GPS"][key] = m.gps("tier")
        R["FADA + adaptation"][key] = m.fada("tier", "NP")
        R["FADA (RMA) + adaptation"][key] = m.fada("tier", "NR")
    last, best = {}, {}
    for arm, net in NET.items():
        k, means = pooled_best(net)
        best[arm] = k
        last[arm] = {p: R[arm][p][-1] for p in R[arm]}
        for p in R[arm]:
            R[arm][p] = R[arm][p][:k + 1]
        print(f"{arm:20s} pooled means by iteration:", " ".join(f"{i}:{v:.3f}" for i, v in sorted(means.items())), f"-> stop at {k}")
    D = json.load(open(os.path.join(HERE, "car_fig2_prac_tiers_rel.json" if m.REL else "car_fig2_prac_tiers.json")))   # the staged layout keys
    D["real"], D["last"], D["best_iter"] = R, last, best
    json.dump(D, open(OUT, "w"), indent=1)
    for arm, per in R.items():
        print(f"{arm:26s}", " | ".join(f"{c}: " + " ".join(f"{x:.0f}:{y:.2f}" for x, y, lo, hi in pts) for c, pts in per.items()))
    print("->", OUT)


if __name__ == "__main__":
    main()
