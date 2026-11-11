"""quad_fig5s_data.py [OUT.json] (2026-11-10, user: "the new hardware deployment uses srbd ... rerun") = quad_fig5_data.py with every
DistILC stage arm from the HARDWARE's update (stage_xfix32.sh: deploy_fm.py RS flags + --jac-at srbm --fall-mask 0.3, 8 iterations, 6 jumps
each; stage seed 9001 + ns, ns = 0 / 1 / 2):
  PPO + DR + DistILC        stage_fl/ppodr_SRB8_ns<ns>_sd<seed>   (deployed PPO + DR mem/ppo_m0/it300; 3 stage seeds x 4 robots)
  DistILC (+ DistILC_hw)    stage_fl/oursS_SRB8_ns<ns>_sd<seed>   (fig 4's from-scratch DistILC, network seed ns, member g0_plane_it30)
  DistILC + DR (+ hw)       stage_fl/oursSdr_SRB8_ns<ns>_sd<seed> (same with DR)
it 0 / 8 = the stage's start / final eval, it 2 / 4 / 6 = fig4/eval/fig5b_<net>_ns<ns>_it<k>. Baselines unchanged (below).
Then fig5_best_iter.py applies the selection rule written before the results (stop at the lowest mean nominal point).
--- quad_fig5_data.py's header:
quad_fig5_data.py [OUT.json] (2026-11-08, paper/FIGURE_PLAN_1106.md figure 5): the quadruped's few-shot curves -> budget_figs2.py's
"real" part. Error vs real jumps per robot, PAPER METRIC max(|ex| / 5 cm, |ez| / 3 cm), a fall = 5, on the held-out plan goals
(planeval, seed 9001) from nominal stances and from perturbed stances (planeval_rob); 4 sim robots (damping 0.1), memoryless network.
Arms (all from the deployed PPO+DR mem/ppo_m0/it300 unless noted):
  PPO + DR + DistILC        stage_fl/mem0k_RS_sd900{1,2,3} (the picked, masked stage): it 0 / 6 from the stage, it 2 / 4 from
                            fig4/eval/fig5_mem0k_RS_sd<sd>_it<k> (fig5_ours_mid.sh); 6 jumps per iteration
  PPO + DR + PPO on robot   fig5/eval_ppo_robot_s<s>_ck<K> (quad_real_ppo.py, privileged critic), K = 12 / 36 / 96 / 384 jumps
  FADA + adaptation         stage_fl/fadaNPq_s<s>/<robot>/ep<K> (no-teacher FADA from the same base), K = 0 / 6 / 12 / 24 / 36
  RMA, zero-shot            fig4/eval/rma_s<s>_ck300 (the RMA student, zero real jumps)
y = mean over units (seed or stage seed x robot) of the per-unit mean score; 95 % CI by bootstrap over units."""
import glob
import json
import os
import sys

import numpy as np

P = "/home/henry/ilc_ws/log/dilc/plane"
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "quad_fig5s.json")
ROB = ["real_r1_m500", "real_r4_m500", "real_r5_m500", "real_s1_m500"]
rng = np.random.default_rng(0)
sc = lambda r: 5.0 if r["fell"] else max(abs(r["ex"]) / 0.05, abs(r["ez"]) / 0.03)


def unit(f):
    if not os.path.exists(f):
        return None
    R = json.load(open(f))
    return float(np.mean([sc(r) for r in R])) if R else None


def pt(x, vals):
    v = np.array([u for u in vals if u is not None], float)
    if not len(v):
        return None
    bs = rng.choice(v, (4000, len(v))).mean(1)
    return [float(x), float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    R = {}
    for cond, sfx in (("nominal", ""), ("perturbed", "_rob")):
        for arm, net in (("PPO + DR + DistILC", "ppodr"), ("DistILC", "oursS"), ("DistILC + DR", "oursSdr")):
            a = R.setdefault(arm, {})[cond] = []
            for k in (0, 2, 4, 6, 8):
                if k in (0, 8):
                    fs = [f"{P}/stage_fl/{net}_SRB8_ns{ns}_sd{9001 + ns}/{rb}/eval_planeval{sfx}_{'start' if k == 0 else 'final'}.json" for ns in range(3) for rb in ROB]
                else:
                    fs = [f"{P}/fig4/eval/fig5b_{net}_ns{ns}_it{k}/{rb}/eval_planeval{sfx}_start.json" for ns in range(3) for rb in ROB]
                p = pt(6 * k, [unit(f) for f in fs])
                if p and sum(unit(f) is not None for f in fs) == len(fs):     # a point only when every unit is scored
                    a.append(p)
                elif p:
                    print(f"  {arm} {cond} it{k}: {sum(unit(f) is not None for f in fs)}/{len(fs)} units scored, point withheld")
        a = R.setdefault("PPO + DR + PPO on robot", {})[cond] = []
        a.append(pt(0, [unit(f"{P}/stage_fl/ppodr_SRB8_ns{ns}_sd{9001 + ns}/{rb}/eval_planeval{sfx}_start.json") for ns in range(3) for rb in ROB]))   # the same base network
        for K in (12, 36, 96, 384):
            a.append(pt(K, [unit(f"{P}/fig5/eval_ppo_robot_s{s}_ck{K}/{rb}/eval_planeval{sfx}_start.json") for s in range(3) for rb in ROB]))
        a = R.setdefault("FADA + adaptation", {})[cond] = []
        for K in (0, 6, 12, 24, 36):
            a.append(pt(K, [unit(f"{P}/stage_fl/fadaNPq_s{s}/{rb}/ep{K}/eval_planeval{sfx}.json") for s in range(3) for rb in ROB]))
        R.setdefault("RMA, zero-shot", {})[cond] = [pt(0, [unit(f"{P}/fig4/eval/rma_s{s}_ck300/{rb}/eval_planeval{sfx}_start.json") for s in range(3) for rb in ROB])]
    for arm in R:
        for c in R[arm]:
            R[arm][c] = [p for p in R[arm][c] if p]
    D = {"panels": [["nominal", "held-out goals, nominal stance", 4, "robots"]],   # (2026-11-11, user: perturbed stances belong in fig 6)
         "figw": 7.0, "legend_right": 0.64,
         "real": R, "real_ticks": [0, 6, 12, 24, 48, 96, 384], "real_label": "real jumps per robot", "sharey": False, "ycap": {"nominal": 7.0, "perturbed": 7.5}, "ref_line": [5.0],
         "ylabel": "normalized landing error"}
    json.dump(D, open(OUT, "w"), indent=1)
    for arm, per in R.items():
        print(f"{arm:26s}", " | ".join(f"{c}: " + " ".join(f"{x:.0f}:{m:.2f}" for x, m, lo, hi in pts) for c, pts in per.items()))
    print("->", OUT)


if __name__ == "__main__":
    main()
