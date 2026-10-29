"""Pogo goal generalization during the hardware stage: error on the goals flown (IN), on goals inside their convex hull
(HULL) and outside it (OUT), vs real trials per robot, for the stage arms, published FADA and zero-shot RMA.
  pogo_inout_task  fixed-point error |x* - g| (tolerance units: 2 cm apex, 0.1 m/s)
  pogo_inout_flow  mean apex-to-apex flow error |F(x) - g| over a tube of start states
Mean over 6 robots x 3 seeds with 95 % bootstrap CIs. HULL / OUT goals are validation goals (seed 301).

Data: inout_ci.json, a snapshot of ~/ilc_ws/log/pogo_servo/runs/video/flows_out/summary_extra.json
(log/pogo_servo/inout_fig.py aggregate, 2026-10-27; recompute there and copy it here to refresh).
Full text width (7.16 in), Times: Times New Roman text with STIX math, as method_diagram.py. Writes
pogo_inout_{task,flow}.{pdf,png} to paper/final/ (the generator lives in its own folder).
    python paper/final/pogo_inout/inout_figs.py      (any python with matplotlib; Times clones TeX Gyre Termes /
    Nimbus Roman stand in where Times New Roman is not installed)
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)                                          # paper/final

for f in glob.glob("/usr/share/fonts/truetype/msttcorefonts/[Tt]imes*.ttf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Times New Roman", "TeX Gyre Termes", "Nimbus Roman", "DejaVu Sans"], "mathtext.fontset": "stix",
                     "font.size": 7, "pdf.fonttype": 42})

INK = "#222222"
MUTED = "#777777"
GRID = "#e4e4e4"
TRIALS_PER_IT = 6
# start -> (label, color, marker); colors as the pogo data figures (fig_pogo_budget.png)
ARMS = {"ours": ("DistILC", "#2a78d6", "o"),
        "ours+DR": ("DistILC + DR", "#eb6834", "s"),
        "PPO+DR": ("PPO + DR + DistILC", "#1baf7a", "D"),
        "FADA": ("FADA (DR) + adaptation", "#6250d6", "P")}
SETS = [("IN", "goals flown"), ("HULL", "inside their hull"), ("OUT", "outside their hull")]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK, labelsize=7, width=0.6, length=2.5)
    ax.grid(True, axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def panel_fig(E, key, ylabel, name):
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.0), sharey=True)
    for ax, (g, title) in zip(axs, SETS):
        style(ax)
        for st, (lab, col, mk) in ARMS.items():
            ci = np.array(E["starts"][st][key][g])
            x = TRIALS_PER_IT * np.arange(len(ci)) if st != "FADA" else np.array([0, 24])
            ax.fill_between(x, ci[:, 1], ci[:, 2], color=col, alpha=0.15, lw=0)
            ax.plot(x, ci[:, 0], color=col, lw=1.4, marker=mk, ms=3.5, mec="white", mew=0.5, label=lab)
        r = np.array(E["starts"]["RMA"][key][g])[0]
        ax.errorbar([0], [r[0]], yerr=[[r[0] - r[1]], [r[2] - r[0]]], marker="*", ms=6, color=INK, ls="none",
                    capsize=1.5, lw=0.8, label="RMA (DR), zero-shot")
        if key == "task_err_ci":
            ax.axhline(1.0, color=INK, lw=0.6, ls=(0, (4, 3)))
            ax.text(36, 1.04, "tolerance", ha="right", va="bottom", fontsize=6.5, color=MUTED)
        ax.set_title(f"{title} ({len(E['goal_sets'][g])} goals)", fontsize=7.5, color=INK, pad=3)
        ax.set_xticks([0, 12, 24, 36])
        ax.set_xlim(-2, 38)
        ax.set_xlabel("real rollouts per robot", color=INK)
        ax.set_ylim(0, None)
    axs[0].set_ylabel(ylabel, color=INK)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, labelcolor=INK, handlelength=1.8, columnspacing=1.2,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    E = json.load(open(os.path.join(HERE, "inout_ci.json")))
    panel_fig(E, "task_err_ci", "fixed-point error\n(tolerance units)", "pogo_inout_task")
    panel_fig(E, "goal_err_ci", r"flow error $|F(x)-g|$" "\n(tolerance units)", "pogo_inout_flow")
    print("->", OUT)
