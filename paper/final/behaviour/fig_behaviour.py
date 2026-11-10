"""fig_behaviour.py MERGED.json OUTSTEM (2026-11-08): the car's behaviour panel (practice-only protocol, sim). Same plans, same method:
plastic converges by sliding more, rubber stays in grip. Per cell (tire x practised plan) and network: (a) rear-out share = fraction of
corner time with signed sideslip beta*sign(r) < -3 deg (|r| > 0.5, v > 1), zero-shot (open) -> after the stage, iteration 3 (filled);
(b) % change of the normalized error, iteration 3 vs the run's own zero-shot. Per run = mean over the cell's cars; markers = mean over
runs, bars = 95 % bootstrap CI over runs. Data: collect_beh.py over runs/screen/PRACB_*/ladder_launch.json (launched scorer, 5 laps)."""
import json, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in glob.glob("/usr/share/fonts/truetype/msttcorefonts/[Tt]imes*.ttf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Times New Roman", "TeX Gyre Termes", "Nimbus Roman", "STIXGeneral", "DejaVu Sans"],
                     "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42})
INK, MUTED, GRID = "#222222", "#777777", "#e4e4e4"
NETS = [("freePPON", "PPO + DR + DistILC", "#1baf7a", "D"), ("rmaN", "RMA + DistILC", "#7a7a7a", "P")]
CELLS = [("plastic", "square", "square2fast"), ("plastic", "oval30", "oval a30"), ("plastic", "oval34", "oval a34"),
         ("plastic", "oval34grip", "oval a34, foam turn"), ("plastic", "fig8big34", "figure-eight R1.65 a34"), ("plastic", "ethz", "ETHZ"),
         ("plastic", "pnutdeep", "deep peanut"),
         ("rubber", "squarenom", "square2fast"), ("rubber", "square45", "square2fast a45"), ("rubber", "square33", "square2fast a33"),
         ("rubber", "oval30", "oval a30"), ("rubber", "oval34", "oval a34"), ("rubber", "oval34split", "oval a34, slick turn"),
         ("rubber", "fig8big34", "figure-eight R1.65 a34"), ("rubber", "ethz", "ETHZ"), ("rubber", "pnutdeep", "deep peanut")]
rng = np.random.default_rng(0)


def ci(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return v.mean(), v.mean(), v.mean()
    b = rng.choice(v, (4000, len(v))).mean(1)
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    runs = json.load(open(sys.argv[1])); stem = sys.argv[2]
    rows, table = [], []
    for prot, tr, lab in CELLS:
        per = {}
        for key, *_ in NETS:
            rs = [r for r in runs if r["prot"] == prot and r["track"] == tr and r["net"] == key]
            if not rs:
                continue
            cm = lambda r, k: np.mean([c[k] for c in r["cars"].values() if c[k] is not None])
            r0 = [100 * cm(r, "r0") for r in rs]; r3 = [100 * cm(r, "r3") for r in rs]
            pc = [np.mean([(c["s3"] / c["s0"] - 1) * 100 for c in r["cars"].values()]) for r in rs]
            per[key] = dict(n=len(rs), r0=ci(r0), r3=ci(r3), pc=ci(pc), imp=sum(p < 0 for p in pc))
            table.append(f"{prot:8s} {lab:24s} {key:9s} n={len(rs):2d} rear-out {per[key]['r0'][0]:5.1f} -> {per[key]['r3'][0]:5.1f} %"
                         f"  d-nerr {per[key]['pc'][0]:+6.1f} % [{per[key]['pc'][1]:+.1f}, {per[key]['pc'][2]:+.1f}]  improved {per[key]['imp']}/{len(rs)}")
        if per:
            rows.append((prot, lab, per))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 0.22 * len(rows) + 0.9), sharey=True, gridspec_kw=dict(wspace=0.08))
    y = np.arange(len(rows))[::-1]
    for i, (prot, lab, per) in enumerate(rows):
        for j, (key, name, col, mk) in enumerate(NETS):
            if key not in per:
                continue
            p = per[key]; yy = y[i] + (0.14 if j == 0 else -0.14)
            axs[0].annotate("", xy=(p["r3"][0], yy), xytext=(p["r0"][0], yy), arrowprops=dict(arrowstyle="->", color=col, lw=0.9))
            axs[0].plot(p["r0"][0], yy, mk, ms=3.6, mfc="white", mec=col, mew=0.8)
            axs[0].plot(p["r3"][0], yy, mk, ms=3.6, color=col, mec="white", mew=0.4)
            axs[0].plot([p["r3"][1], p["r3"][2]], [yy, yy], color=col, lw=0.6, alpha=0.6)
            axs[1].plot([p["pc"][1], p["pc"][2]], [yy, yy], color=col, lw=0.8)
            axs[1].plot(p["pc"][0], yy, mk, ms=3.6, color=col, mec="white", mew=0.4, label=name if i == 0 else None)
    for ax in axs:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=INK, labelsize=7, width=0.6, length=2.5)
        ax.grid(True, axis="x", color=GRID, lw=0.5)
    nplast = sum(r[0] == "plastic" for r in rows)
    for ax in axs:
        ax.axhline(y[nplast - 1] - 0.5, color=MUTED, lw=0.5, ls=":")
    axs[0].set_yticks(y, [f"{p}: {l}" for p, l, _ in rows])
    axs[0].set_xlabel("rear-out share of corner time (%)  (open: zero-shot, filled: after the stage)", color=INK)
    axs[1].axvline(0, color=MUTED, lw=0.5)
    axs[1].set_xlabel("normalized error, % change vs own zero-shot", color=INK)
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, labelcolor=INK, bbox_to_anchor=(0.55, 1.0))
    fig.savefig(stem + ".pdf", bbox_inches="tight"); fig.savefig(stem + ".png", dpi=300, bbox_inches="tight")
    open(stem + "_table.txt", "w").write("\n".join(table) + "\n"); print("\n".join(table)); print("->", stem)


if __name__ == "__main__":
    main()
