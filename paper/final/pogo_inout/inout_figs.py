"""Pogo goal generalization at matched budgets: normalized error on the goals the hardware stage flew (IN), on goals
inside their convex hull (HULL) and outside it (OUT), under the tables' protocol and metric (max(apex / 2 cm,
|dv| / 0.1 m/s), a fall = 5; 4 perturbed starts x 10 s per goal, 6 robots, 3 seeds, 95 % cluster-bootstrap CI over
(seed, robot, goal)). Three panels:
  matched simulation hours  zero-shot after ~11 simulated hours
  matched real rollouts     after 36 real rollouts per robot (FADA: its recipe's 24)
  no limits                 every method at its full budget
Data: inout_matched.json, made by ~/ilc_ws/log/pogo_servo/inout_matched_data.py (inout_matched.py runs). Replaces the
2026-10-29 pogo_inout_{task,flow} (apex-map protocol, flight speed: not the tables' metric).
Full text width (7.16 in), Times: Times New Roman text with STIX math, as method_diagram.py. Writes pogo_inout.{pdf,png}
to paper/final/ (the generator lives in its own folder).
    python paper/final/pogo_inout/inout_figs.py      (any python with matplotlib)
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
plt.rcParams.update({"font.family": ["Times New Roman", "TeX Gyre Termes", "Nimbus Roman", "DejaVu Sans"],
                     "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42})

INK = "#222222"
MUTED = "#777777"
GRID = "#e4e4e4"
S, H = r"DistIL-C$_\mathrm{sim}$", r"DistIL-C$_\mathrm{hw}$"
# arm -> (label, color); colors as the pogo budget figures
ARM = {"distilc_sim": (S + " (zero-shot)", "#9cc2ee"),                  # zero-shot sim arms: lighter tints
       "ppo_dr_10h": ("PPO + DR (zero-shot)", "#f5cd73"),
       "ppo_10h": ("PPO (zero-shot)", "#c4c4c4"),
       "distilc_hw36": (f"{S} + {H}", "#2a78d6"),
       "distilc_dr_hw36": (f"{S} + DR + {H}", "#eb6834"),
       "ppo_dr_distilc_hw36": (f"PPO + DR + {H}", "#1baf7a"),
       "ppo_dr_ppo_hw36": (r"PPO + DR + PPO$_\mathrm{hw}$", "#eda100"),
       "ppo_dr_ppo_hw384": (r"PPO + DR + PPO$_\mathrm{hw}$", "#eda100"),
       "fada_hw24": ("FADA (DR) + adaptation", "#6250d6"),
       "rma": ("RMA (DR)", INK)}
PANELS = [("matched sim hours (11 h)", ["distilc_sim", "ppo_dr_10h", "ppo_10h"], {}),
          ("matched real rollouts (36 per robot)", ["distilc_hw36", "distilc_dr_hw36", "ppo_dr_distilc_hw36", "ppo_dr_ppo_hw36",
                                                    "fada_hw24"], {"fada_hw24": "24"}),
          ("full budgets", ["distilc_hw36", "distilc_dr_hw36", "ppo_dr_distilc_hw36", "ppo_dr_ppo_hw384",
                                                     "fada_hw24", "rma"], {"ppo_dr_ppo_hw384": "384", "rma": "514 h"})]
SETS = [("IN", "flown"), ("HULL", "in hull"), ("OUT", "out of hull")]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK, labelsize=7, width=0.6, length=2.5)
    ax.grid(True, axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


if __name__ == "__main__":
    D = json.load(open(os.path.join(HERE, "inout_matched.json")))
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.9), sharey=True, gridspec_kw=dict(width_ratios=[3.6, 5, 6]))
    ymax = max(v[2] for a in D["arms"].values() for v in a.values())
    for ax, (title, arms, note) in zip(axs, PANELS):
        style(ax)
        arms = [a for a in arms if a in D["arms"]]
        w = 0.8 / max(len(arms), 1)
        for j, a in enumerate(arms):
            lab, col = ARM[a]
            v = np.array([D["arms"][a].get(g, [np.nan] * 4)[:3] for g, _ in SETS])
            x = np.arange(len(SETS)) + (j - (len(arms) - 1) / 2) * w
            ax.bar(x, v[:, 0], w * 0.9, color=col, alpha=0.85 if a != "rma" else 0.75, lw=0)
            ax.errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="none", ecolor=INK, lw=0.6, capsize=1.2)
            if a in note:
                for xi, yi in zip(x, v[:, 2]):
                    ax.text(xi, yi + 0.03 * ymax, note[a], ha="center", va="bottom", fontsize=5, color=MUTED, rotation=90)
        ax.set_xticks(np.arange(len(SETS)))
        ax.set_xticklabels([f"{t}\n({D['goal_sets'][g]})" for g, t in SETS], fontsize=6.5)
        ax.set_title(title, fontsize=7.5, color=INK, pad=3)
        ax.set_ylim(0, ymax * 1.15)
    axs[0].set_ylabel("normalized error", color=INK)
    seen, hs, ls = set(), [], []
    for _, arms, _ in PANELS:
        for a in arms:
            lab, col = ARM[a]
            if a in D["arms"] and lab not in seen:
                seen.add(lab)
                hs.append(plt.Rectangle((0, 0), 1, 1, color=col, alpha=0.85))
                ls.append(lab)
    fig.legend(hs, ls, loc="lower center", ncol=3, frameon=False, labelcolor=INK, handlelength=1.2, columnspacing=1.2,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.2, 1, 1), w_pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"pogo_inout.{ext}"), dpi=300)
    plt.close(fig)
    print("->", OUT)
