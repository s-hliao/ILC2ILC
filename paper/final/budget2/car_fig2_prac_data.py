"""car_fig2_prac_data.py = car_fig2_data.py on the PRACTISED plans (2026-11-08, user: practice-only protocol, "fig2 is only sim2sim
transfer"): sb_practised / fig2_practised / gps_practised (ladder_eval, same scorer, plan = the panel's practice plan).
 [SRC] [OUT.json] (2026-11-07, paper/FIGURE_PLAN_1106.md figure 2): the car's few-shot curves -> budget_figs2.py's
"real" part. Error vs cumulative REAL laps per car on each panel's held-out peanut (ladder_eval: seed 701, 4 starts, 5 laps).
SRC (fig2_src/, staged from ube / Lambda on 2026-11-07):
  ube/SB_<panel>_<net>_s<s>_ss<k>/sb_heldout.json + summary.json   ours (SB stage) on freePPON / rmaN / ours / oursdr, its 0-3
  lam/fig2fadaN{P,R}_<panel>_s<s>/fig2_eval.json + fadaN_logs/      no-teacher FADA (PPO+DR- / RMA-taught), episodes 0 8 24 96 384
  lam/fig2ppo_<panel>_s<s>/fig2_eval.json + ppo_hw/*/summary.json   PPO+DR + PPO on the car (privileged critic), episodes 0 8 24 96 384
  lam/GPS{,2}_<panel>_freePPON_s<s>/gps_heldout.json + summary.json  PPO+DR + GPS (MDGPS), iterations 0 1 2 3 6 12 24
x = cumulative real laps per car (the run's own counts: SB real_laps / trials per iteration, GPS real_laps / 24 per iteration, FADA's
logged laps per budget, PPO's real_laps). y = mean over (run, car) of the score; 95 % CI by bootstrap over runs (the per-run car mean).
GPS: GPS2_* (the rerun) supersedes GPS_* for rubber / tier; plastic has only GPS_*."""
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "fig2_src_prac")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "car_fig2_prac.json")
PANELS = [["plastic", "plastic tires", 12, "cars"], ["rubber", "rubber tires", 2, "cars"], ["tier", "coverage tiers A-C", 36, "cars"]]
rng = np.random.default_rng(0)


REL = os.environ.get("FIG2_REL") == "1"               # (2026-11-08) y = % change vs the run's OWN zero-shot (per car, then car mean)


def ev(f):
    d = json.load(open(f))
    return d[list(d)[0]]                                 # the one held-out plan: {it: {car: {...}}}


def stat(per_run):
    """per_run: list of per-run car means -> mean, 95 % bootstrap CI over runs."""
    v = np.asarray(per_run, float)
    if len(v) == 1:
        return float(v[0]), float(v[0]), float(v[0])
    bs = rng.choice(v, (4000, len(v))).mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def curve(runs):
    """runs: list of {x: car-mean score}; x values shared within an arm up to small lap-count differences -> pooled by rank."""
    keys = sorted({k for r in runs for k in r})
    pts = []
    for k in keys:
        xs = [r[k][0] for r in runs if k in r]
        ys = [r[k][1] for r in runs if k in r]
        m, lo, hi = stat(ys)
        pts.append([float(np.mean(xs)), m, lo, hi])
    return pts


def cm(e, k):
    """the car mean at iteration k: the score, or (FIG2_REL) the % change vs the same car's iteration 0."""
    if REL:
        return float(np.mean([(e[k][c]["score"] / e["0"][c]["score"] - 1) * 100 for c in e[k]]))
    return float(np.mean([v["score"] for v in e[k].values()]))


def sb(panel, net):
    runs = []
    for d in sorted(glob.glob(f"{SRC}/ube/SB_{panel}_{net}_s?_ss*")):
        e, s = ev(f"{d}/sb_practised.json"), json.load(open(f"{d}/summary.json"))
        per_trial = np.mean([s["real_laps"][c] / max(1, len(s["trials"][c])) for c in s["real_laps"]])
        runs.append({int(k): (int(k) * per_trial, cm(e, k)) for k in ("0", "1", "2", "3") if k in e})
    return curve(runs)


def fada(panel, v):
    runs = []
    for s in range(3):
        e = ev(f"{SRC}/lam/fig2fada{v}_{panel}_s{s}/fig2_practised.json")
        laps = {0: [0.0]}
        for ln in open(f"{SRC}/lam/fadaN_logs/fig2_{v}_{panel}_s{s}.log"):
            if ln.startswith('{"car"'):
                j = json.loads(ln)
                for b in ("8", "24", "96", "384"):
                    laps.setdefault(int(b), []).append(j[b]["laps"])
        runs.append({int(k): (float(np.mean(laps[int(k)])), cm(e, k)) for k in e})
    return curve(runs)


def ppo(panel):
    runs = []
    for s in range(3):
        e = ev(f"{SRC}/lam/fig2ppo_{panel}_s{s}/fig2_practised.json")
        r = {}
        for k in e:
            lp = 0.0 if k == "0" else float(np.mean(list(json.load(open(f"{SRC}/lam/ppo_hw/fig2ppo_{panel}_s{s}_ep{k}/summary.json"))["real_laps"].values())))
            r[int(k)] = (lp, cm(e, k))
        runs.append(r)
    return curve(runs)


def gps(panel):
    runs = []
    for s in range(3):
        d = f"{SRC}/lam/GPS2_{panel}_freePPON_s{s}"
        d = d if os.path.isdir(d) else f"{SRC}/lam/GPS_{panel}_freePPON_s{s}"
        e, sm = ev(f"{d}/gps_practised.json"), json.load(open(f"{d}/summary.json"))
        per_it = np.mean(list(sm["real_laps"].values())) / int(sm["args"]["iters"])
        runs.append({int(k): (int(k) * per_it, cm(e, k)) for k in e})
    return curve(runs)


def main():
    R = {"PPO + DR + DistILC": {}, "RMA + DistILC": {}, "DistILC": {}, "DistILC + DR": {}, "PPO + DR + PPO on robot": {},
         "PPO + DR + GPS": {}, "FADA + adaptation": {}, "FADA (RMA) + adaptation": {}}
    for p, *_ in PANELS:
        R["PPO + DR + DistILC"][p] = sb(p, "freePPON")
        R["RMA + DistILC"][p] = sb(p, "rmaN")
        R["DistILC"][p] = sb(p, "ours")
        R["DistILC + DR"][p] = sb(p, "oursdr")
        R["PPO + DR + PPO on robot"][p] = ppo(p)
        R["PPO + DR + GPS"][p] = gps(p)
        R["FADA + adaptation"][p] = fada(p, "NP")
        R["FADA (RMA) + adaptation"][p] = fada(p, "NR")
    D = {"panels": PANELS, "real": R, "real_ticks": [0, 2, 6, 24, 96, 384], "real_label": "laps on the deployment car (sim-to-sim)",
         "ylabel": "normalized error", "sharey": False, "ycap": {"plastic": 4.0, "rubber": 1.2, "tier": 1.2}}
    if REL:
        D.update(ylabel="% change vs own zero-shot", sharey=True, ycap={"plastic": 45.0, "rubber": 45.0, "tier": 45.0}, ylim=[-50, 47], ref_line=[0.0])
    json.dump(D, open(OUT, "w"), indent=1)
    for arm, per in R.items():
        print(f"{arm:26s}", " | ".join(f"{p}: " + " ".join(f"{x:.0f}:{m:.2f}" for x, m, lo, hi in pts) for p, pts in per.items()))
    print("->", OUT)


if __name__ == "__main__":
    main()
