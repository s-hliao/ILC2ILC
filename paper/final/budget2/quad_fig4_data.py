"""quad_fig4_data.py [OUT.json] (2026-11-08, paper/FIGURE_PLAN_1106.md figure 4): the quadruped's simulation-training curves ->
budget_figs2.py's "sim" part. Zero-shot PAPER METRIC max(|ex| / 5 cm, |ez| / 3 cm), a fall = 5, held-out plan goals (planeval, seed
9001) from nominal and from perturbed stances, 4 sim robots (damping 0.1), every checkpoint scored by fig4_eval.sh (fig4/eval/<arm>).
x = simulated hours = simulated jumps x 1.9 s (0.9 s plan + 1.0 s landing; the GPU sim's episode), EXCLUDING the shared start network
mem/init_m0 (every arm starts from it):
  PPO / PPO + DR          1024 jumps per PPO iteration x it (25 / 100 / 200 / 300); the logs' "[N jumps" restart at each --resume, so
                          the cumulative count is 1024 x it (checked: the first block logs exactly 1024 per iteration)
  DistILC / + DR          plane_train.py --batch 96: 96 jumps per iteration, it 5 / 10 / 15 / 30 (members g0_plane_it<k>)
  RMA                     teacher fig4/rmat_s<s>: 1024 x its checkpoint (50 / 100 / 300) + the student's 25 x 1024
  FADA                    figure 4's FADA (fadaRq): teacher rmat it300 (1024 x 300) + 1024 per distillation iteration (1 / 5 / 10 / 25)
y = mean over (seed x robot) units of the per-unit mean score; 95 % CI by bootstrap over units. A checkpoint is plotted only when all
12 units are scored."""
import glob
import json
import os
import re
import sys

import numpy as np

P = "/home/henry/ilc_ws/log/dilc/plane/fig4"
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "quad_fig4.json")
ROB = ["real_r1_m500", "real_r4_m500", "real_r5_m500", "real_s1_m500"]
SEC = 1.9
rng = np.random.default_rng(0)
sc = lambda r: 5.0 if r["fell"] else max(abs(r["ex"]) / 0.05, abs(r["ez"]) / 0.03)


def jumps_at(log, it):
    for ln in open(log):
        m = re.match(rf"it {it}: .*\[(\d+) jumps", ln)
        if m:
            return int(m.group(1))
    return None


def point(evdir_fmt, seeds, x_of, sfx):
    vals, xs = [], []
    for s in seeds:
        for rb in ROB:
            f = f"{P}/eval/{evdir_fmt.format(s=s)}/{rb}/eval_planeval{sfx}_start.json"
            if not os.path.exists(f):
                return None
            R = json.load(open(f))
            vals.append(np.mean([sc(r) for r in R]))
        xs.append(x_of(s))
    if any(x is None for x in xs):
        return None
    v = np.array(vals)
    bs = rng.choice(v, (4000, len(v))).mean(1)
    return [float(np.mean(xs) * SEC / 3600), float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    S = range(3)
    R = {}
    for cond, sfx in (("nominal", ""), ("perturbed", "_rob")):
        for arm, key in (("PPO", "ppo"), ("PPO + DR", "ppodr")):
            R.setdefault(arm, {})[cond] = [point(f"{key}_s{{s}}_it{k}", S, lambda s, k=k: 1024 * k, sfx)
                                           for k in (25, 100, 200, 300)]
        for arm, key in (("DistILC", "ours"), ("DistILC + DR", "oursdr")):
            R.setdefault(arm, {})[cond] = [point(f"{key}_s{{s}}_it{k}", S, lambda s, k=k: 96 * k, sfx) for k in (5, 10, 15, 30)]
        R.setdefault("RMA", {})[cond] = [point(f"rma_s{{s}}_ck{c}", S, lambda s, c=c: 1024 * c + 25 * 1024, sfx)
                                         for c in (50, 100, 300)]
        R.setdefault("FADA", {})[cond] = [point(f"fadaRq_s{{s}}_it{k}", S, lambda s, k=k: 1024 * 300 + 1024 * k, sfx)
                                          for k in (1, 5, 10, 25)]
    for arm in R:
        for c in R[arm]:
            R[arm][c] = [p for p in R[arm][c] if p]
    D = {"panels": [["nominal", "held-out goals, nominal stance", 4, "robots"], ["perturbed", "held-out goals, perturbed stances", 4, "robots"]],
         "sim": R, "sharey": False, "ylabel": "normalized landing error (zero-shot)", "ref_line": [5.0]}
    json.dump(D, open(OUT, "w"), indent=1)
    for arm, per in R.items():
        print(f"{arm:14s}", " | ".join(f"{c}: " + " ".join(f"{x:.1f}h:{m:.2f}" for x, m, lo, hi in pts) for c, pts in per.items()))
    print("->", OUT)


if __name__ == "__main__":
    main()
