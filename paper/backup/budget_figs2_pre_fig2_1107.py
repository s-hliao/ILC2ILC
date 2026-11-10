"""budget_figs2.py DATA.json NAME (2026-11-06, paper/FIGURE_PLAN_1106.md figures 1 / 2 / 4 / 5): the pogo budget figures
(final/pogo_budget/budget_figs.py: style, colours, legend names) for the car and the quadruped, panels from the data file.
DATA.json: {"panels": [[key, title, n_units, unit_word], ...],
            "sim":  {arm: {panel: [[sim_hours, mean, lo, hi], ...]}},
            "real": {arm: {panel: [[real_episodes_per_unit, mean, lo, hi], ...]}},
            "real_ticks": [0, 2, 8, ...], "real_label": "real laps per car", "ylabel": "normalized error"}
Writes NAME_sim.{pdf,png} and NAME_real.{pdf,png} to paper/final/ (either part may be absent)."""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # paper/final
for f in glob.glob("/usr/share/fonts/truetype/msttcorefonts/[Tt]imes*.ttf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Times New Roman", "TeX Gyre Termes", "Nimbus Roman", "STIXGeneral", "DejaVu Sans"],
                     "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42})
INK, MUTED, GRID = "#222222", "#777777", "#e4e4e4"
# the pogo figures' colours / markers per arm (sim arms; the real-data arms reuse their source arm's colour)
STY = {"DistILC": ("#2a78d6", "o", "-"), "DistILC + DR": ("#eb6834", "s", "-"),
       "PPO + DR + DistILC": ("#1baf7a", "D", "-"), "RMA + DistILC": ("#7a7a7a", "P", "-"),
       "PPO + DR": ("#eda100", "^", "-"), "PPO + DR + PPO on robot": ("#eda100", "^", "-"),
       "PPO": ("#eda100", "v", "--"), "FADA": ("#6250d6", "h", "-"), "FADA + adaptation": ("#6250d6", "h", "-"),
       "RMA": (INK, "*", "-"), "RMA, zero-shot": (INK, "*", "none")}
SIM_N, HW_N = r"DistIL-C$_\mathrm{sim}$", r"DistIL-C$_\mathrm{hw}$"
LAB = {"sim": {"DistILC": SIM_N, "DistILC + DR": SIM_N + " + DR"},
       "real": {"DistILC": SIM_N + " + " + HW_N, "DistILC + DR": SIM_N + " + DR + " + HW_N, "PPO + DR + DistILC": "PPO + DR + " + HW_N,
                "RMA + DistILC": "RMA + " + HW_N, "PPO + DR + PPO on robot": r"PPO + DR + PPO$_\mathrm{hw}$"}}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK, labelsize=7, width=0.6, length=2.5)
    ax.grid(True, axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def draw(ax, arm, pts, xf, label):
    col, mk, ls = STY.get(arm, (MUTED, "o", "-"))
    a = np.array(sorted(pts), float)
    x = xf(a[:, 0])
    if len(a) > 1 and ls != "none":
        ax.fill_between(x, a[:, 2], a[:, 3], color=col, alpha=0.13, lw=0)
        ax.plot(x, a[:, 1], color=col, lw=1.3, ls=ls, marker=mk, ms=3.2, mec="white", mew=0.4, label=label)
    else:
        ax.errorbar(x, a[:, 1], yerr=[a[:, 1] - a[:, 2], a[:, 3] - a[:, 1]], color=col, marker=mk,
                    ms=6 if mk == "*" else 4, ls="none", capsize=1.5, lw=0.8, label=label)


def panel_fig(D, key, xf, ticks, xlabel, name):
    P = D["panels"]
    fig, axs = plt.subplots(1, len(P), figsize=(7.16, 2.15), sharey=True)
    axs = np.atleast_1d(axs)
    ymax = max(p[3] for per in D[key].values() for pts in per.values() for p in pts)
    for ax, (pk, title, n, unit) in zip(axs, P):
        style(ax)
        for arm, per in D[key].items():
            if per.get(pk):
                draw(ax, arm, per[pk], xf, LAB[key].get(arm, arm))
        ax.set_title(f"{title} ({n} {unit})", fontsize=7.5, color=INK, pad=3)
        ax.set_xticks(xf(np.array(ticks, float)))
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlabel(xlabel, color=INK)
        ax.set_ylim(0, ymax * 1.03)
    axs[0].set_ylabel(D.get("ylabel", "normalized error"), color=INK)
    h, l = [], []
    for ax in axs:                                          # every arm once, whichever panel has it
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, labelcolor=INK, handlelength=1.8, columnspacing=1.1,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    D = json.load(open(sys.argv[1]))
    name = sys.argv[2]
    if D.get("sim"):
        hrs = sorted({p[0] for per in D["sim"].values() for pts in per.values() for p in pts})
        cand = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
        ticks = [t for t in cand if hrs and hrs[0] * 0.7 <= t <= hrs[-1] * 1.4] or cand
        panel_fig(D, "sim", np.log10, ticks, "simulated hours (zero-shot)", f"{name}_sim")
    if D.get("real"):
        panel_fig(D, "real", lambda n: np.log2(1.0 + n), D.get("real_ticks", [0, 12, 36, 96, 384]),
                  D.get("real_label", "real episodes per robot"), f"{name}_real")
    print("->", OUT)
