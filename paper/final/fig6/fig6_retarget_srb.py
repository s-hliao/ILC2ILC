"""fig6_retarget_srb.py (2026-11-10, user: "the new hardware deployment uses srbd ... rerun") = fig6_retarget.py with every DistIL-C_hw row
from the HARDWARE's update (retarget_job_srb.sh: deploy_fm.py RS flags + --jac-at srbm --fall-mask 0.3, 6 iterations; stage_fl/rt_srb_<net>_<P>;
DistIL-C_sim rows = fig 4's from-scratch oursS_s0 / oursSdr_s0 it30). Writes paper/final/fig6_retarget.{pdf,png}.
--- fig6_retarget.py (2026-11-08, paper/FIGURE_PLAN_1106.md figure 6 "revised again"; stage_fl/RETARGET_1106.md): retargeting sim-trained
quadruped networks onto perturbed stances. Two heatmaps, columns = the 5 hardware-realizable stances P (front block 1.5 cm, crouched,
nose up, nose down, tall):
  left   the score FROM stance P: each network zero-shot vs after adapting with every practice jump flown from P
  right  the NOMINAL-stance score after adapting from P (the retargeting's cost), zero-shot nominal in the first column's header row
Rows: PPO + DR, RMA, DistIL-C_sim, DistIL-C_sim + DR, each zero-shot and + DistIL-C_hw (our stage, RS flags, masked: stage_fl/rt_rs_<net>_<P>);
FADA (in-regime, no-teacher from PPO + DR: fadaNPq_s0) zero-shot and + its own adaptation (stage_fl/rt_fadaNPq_<P>, fada_adapt_tf.py
--train-cond). Score = max(|ex| / 5 cm, |ez| / 3 cm), a fall = 5; stTr.py's pairing (per robot x goal, the mean of its repeats, the same seeds
before / after; held-out goals; 4 robots x 1 stage seed); adapted cells show the change vs the zero-shot, '*' = 95 % bootstrap CI over the
(robot, goal) cells excludes 0. Colour = absolute score (sequential). Writes paper/final/fig6_retarget.{pdf,png}."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import logging  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
ST = "/home/henry/ilc_ws/log/dilc/plane/stage_fl"
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for f in glob.glob("/usr/share/fonts/truetype/msttcorefonts/[Tt]imes*.ttf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Times New Roman", "TeX Gyre Termes", "Nimbus Roman", "STIXGeneral", "DejaVu Sans"],
                     "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42})
CONDS = ["blk15", "crouch", "noseup", "nosedn", "tall"]
CN = dict(blk15="front block\n1.5 cm", crouch="crouched", noseup="nose up", nosedn="nose down", tall="tall")
HW = r"DistIL-C$_\mathrm{hw}$"
NETS = [("PPO + DR", "srb_ppodr"), ("RMA", "srb_rma"), (r"DistIL-C$_\mathrm{sim}$", "srb_simS"), (r"DistIL-C$_\mathrm{sim}$ + DR", "srb_simSdr"),
        ("FADA", "fadaNPq")]
rng = np.random.default_rng(0)


def nerr(x):
    return 5.0 if x["fell"] else max(abs(x["ex"]) / .05, abs(x["ez"]) / .03)


def files(key, P, rb, rob):
    if key == "fadaNPq":
        d = f"{ST}/rt_fadaNPq_{P}/{rb}"
        return f"{d}/ep0/eval_planeval{'_rob' if rob else ''}.json", f"{d}/ep36/eval_planeval{'_rob' if rob else ''}.json"
    d = f"{ST}/rt_{key}_{P}/{rb}"
    return f"{d}/eval_planeval{'_rob' if rob else ''}_start.json", f"{d}/eval_planeval{'_rob' if rob else ''}_final.json"


def cells(key, P, rob):
    """paired (robot, goal) cells: (zero-shot, adapted) mean score; None if any robot is missing."""
    out = []
    for rb in ("real_r1_m500", "real_r4_m500", "real_r5_m500", "real_s1_m500"):
        a, b = files(key, P, rb, rob)
        if not (os.path.exists(a) and os.path.exists(b)):
            return None
        A, B = {}, {}
        for src, G in ((a, A), (b, B)):
            for x in json.load(open(src)):
                if x.get("mode", "policy") != "policy" or (rob and not x.get("cond", "").endswith("+" + P)):
                    continue
                G.setdefault(tuple(x["goal"]), []).append(nerr(x))
        out += [(np.mean(A[k]), np.mean(B[k])) for k in A if k in B]
    return np.array(out)


def summary(c):
    d = c[:, 1] - c[:, 0]
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return c[:, 0].mean(), c[:, 1].mean(), d.mean(), (hi < 0 or lo > 0)


def main():
    rows, labels = [], []
    for name, key in NETS:
        zs_s, ad_s, zs_n, ad_n = [], [], [], []
        for P in CONDS:
            cs, cn = cells(key, P, True), cells(key, P, False)
            zs_s.append(summary(cs) if cs is not None and len(cs) else None)
            zs_n.append(summary(cn) if cn is not None and len(cn) else None)
        rows.append((name, key, zs_s, zs_n))
    fig, ax0 = plt.subplots(1, 1, figsize=(4.6, 3.3))   # 2026-11-10 (user): the nominal-stance block is removed
    axs = [ax0]
    nr = 2 * len(rows)
    vmax = 7.0
    cmap = plt.get_cmap("Blues_r")
    for ax, which, title in ((axs[0], 2, "score from the practised stance"),):
        M = np.full((nr, len(CONDS)), np.nan)
        T = [["" for _ in CONDS] for _ in range(nr)]
        for i, (name, key, s_s, s_n) in enumerate(rows):
            S = s_s if which == 2 else s_n
            for j, v in enumerate(S):
                if v is None:
                    T[2 * i][j] = T[2 * i + 1][j] = "pending"
                    continue
                zs, ad, dd, sig = v
                M[2 * i, j], M[2 * i + 1, j] = zs, ad
                T[2 * i][j] = f"{zs:.2f}"
                T[2 * i + 1][j] = f"{ad:.2f}\n{dd:+.2f}{'*' if sig else ''}"
        ax.imshow(np.clip(M, 0, vmax), cmap=cmap.reversed(), vmin=0, vmax=vmax, aspect="auto")
        for i in range(nr):
            for j in range(len(CONDS)):
                v = M[i, j]
                ax.text(j, i, T[i][j], ha="center", va="center", fontsize=5.6,
                        color="white" if (not np.isnan(v) and v > 0.55 * vmax) else "#222222")
        for i in range(1, len(rows)):
            ax.axhline(2 * i - 0.5, color="white", lw=2.0)
        ax.set_xticks(range(len(CONDS)))
        ax.set_xticklabels([CN[c] for c in CONDS], fontsize=6.3)
        ax.xaxis.tick_top()
        ax.set_title(title, fontsize=7.5, pad=18)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        if ax is axs[0]:
            lab = []
            for name, key, _, _ in rows:
                lab += [f"{name}, zero-shot", f"{name} + {'adaptation' if key == 'fadaNPq' else HW}"]
            ax.set_yticks(range(nr))
            ax.set_yticklabels(lab, fontsize=6.3)
        else:
            ax.set_yticks([])
    fig.text(0.5, 0.01, "normalized landing error on held-out goals (lower is better; a fall = 5); adapted rows: change vs zero-shot, "
             "* = 95 % CI excludes 0;\n4 robots, 36 practice jumps", ha="center", fontsize=6, color="#555555")
    fig.subplots_adjust(left=0.31, right=0.99, top=0.84, bottom=0.07)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig6_retarget.{ext}"), dpi=300)
    print("->", OUT)


if __name__ == "__main__":
    main()
