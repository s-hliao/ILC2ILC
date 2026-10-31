"""Pogo data budgets by dynamics coverage tier (A: inside the DR range; B: extrapolated along DR axes; C: outside the
model class, robots where positivity holds), normalized error max(apex / 2 cm, |dv| / 0.1 m/s) with a fall = 5,
mean and 95 % cluster-bootstrap CI over (seed, robot, goal), frozen test pass (seed 4301), 3 seeds:
  pogo_budget_sim   zero-shot error vs simulated hours of training data (every arm from the same 1.8 h Raibert clone)
  pogo_budget_real  error vs real rollouts per robot (0 = zero-shot)

Data: budget_data.json, made by ~/ilc_ws/log/pogo_servo/budget_data.py (rerun it, then this, to refresh).
Full text width (7.16 in), Times: Times New Roman text with STIX math, as method_diagram.py. Writes
pogo_budget_{sim,real}.{pdf,png} to paper/final/ (the generator lives in its own folder).
    python paper/final/pogo_budget/budget_figs.py      (any python with matplotlib)
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
plt.rcParams.update({"font.family": ["Times New Roman", "TeX Gyre Termes", "Nimbus Roman", "STIXGeneral", "DejaVu Sans"],
                     "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42})

INK = "#222222"
MUTED = "#777777"
GRID = "#e4e4e4"
TIERS = [("A", "A: inside the DR range"), ("B", "B: extrapolated along DR axes"), ("C", "C: outside the DR class")]
# arm -> (color, marker, linestyle); colors as the pogo data figures
STY = {"DistILC": ("#2a78d6", "o", "-"),
       "DistILC + DR": ("#eb6834", "s", "-"),
       "PPO + DR + DistILC": ("#1baf7a", "D", "-"),
       "PPO + DR": ("#eda100", "^", "-"),
       "PPO + DR + PPO on robot": ("#eda100", "^", "-"),
       "PPO": ("#eda100", "v", "--"),
       "FADA (DR)": ("#6250d6", "h", "-"),          # (2026-10-30, user: "connect the fada line")
       "FADA (DR) + adaptation": ("#6250d6", "h", "-"),   # (2026-10-30, user: FADA's bullets less weird: P -> h)
       "RMA (DR)": (INK, "*", "-"),                 # (2026-10-30: a curve over sim hours, simdata_ext.py)
       "RMA (DR), zero-shot": (INK, "*", "none")}
SIM_END = {"DistILC": 33.0, "DistILC + DR": 33.0}     # (2026-10-30, user: "end ours at around 30 sim hours")
# legend names: subscript sim = the simulation stage, hw = the hardware (real-robot) stage (the paper's \oursS / \oursH)
SIM_N, HW_N = r"DistIL-C$_\mathrm{sim}$", r"DistIL-C$_\mathrm{hw}$"
LAB = {"sim": {"DistILC": SIM_N, "DistILC + DR": SIM_N + " + DR"},
       "real": {"DistILC": SIM_N + " + " + HW_N, "DistILC + DR": SIM_N + " + DR + " + HW_N,
                "PPO + DR + DistILC": "PPO + DR + " + HW_N, "PPO + DR + PPO on robot": r"PPO + DR + PPO$_\mathrm{hw}$"}}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK, labelsize=7, width=0.6, length=2.5)
    ax.tick_params(which="minor", length=1.5, width=0.4, colors=MUTED)
    ax.grid(True, axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def draw(ax, arm, pts, xf, label):
    col, mk, ls = STY[arm]
    a = np.array(pts, float)
    x = xf(a[:, 0])
    if len(a) > 1 and ls != "none":
        ax.fill_between(x, a[:, 2], a[:, 3], color=col, alpha=0.13, lw=0)
        ax.plot(x, a[:, 1], color=col, lw=1.3, ls=ls, marker=mk, ms=3.2, mec="white", mew=0.4, label=label)
    else:
        ax.errorbar(x, a[:, 1], yerr=[a[:, 1] - a[:, 2], a[:, 3] - a[:, 1]], color=col, marker=mk,
                    ms=6 if mk == "*" else 4, ls="none", capsize=1.5, lw=0.8, label=label)


def panel_fig(D, key, xf, xticks, xlabel, name, ytop=None):
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.15), sharey=True)
    ymax = max(p[3] for arm in D[key].values() for pts in arm.values() for p in pts)
    for ax, (t, title) in zip(axs, TIERS):
        style(ax)
        for arm, per in D[key].items():
            pts = per.get(t)
            if key == "sim" and arm in ("RMA (DR)", "FADA (DR)") and pts and D["sim"].get("PPO", {}).get(t):
                # (2026-10-30, user: extend RMA, then FADA) RMA's teacher starts from the same 1.8 h Raibert clone
                # (runs/bc/policy_bc.npz) as every other arm, and FADA's hours are that clone + PPO + DR to its snapshot + its own
                # collection: both curves begin at that shared start point, as PPO's / PPO + DR's do
                pts = [D["sim"]["PPO"][t][0]] + pts
            if key == "sim" and arm in SIM_END and pts:
                pts = [p for p in pts if p[0] <= SIM_END[arm]]
            if pts:
                draw(ax, arm, pts, xf, LAB[key].get(arm, arm).replace(" (DR)", ""))   # (2026-10-30, user: no "(DR)" on RMA / FADA)
        ax.set_title(f"{title} ({len(D['tiers'][t])} robots)", fontsize=7.5, color=INK, pad=3)
        xt, xl = xticks
        ax.set_xticks(xf(np.array(xt, float)))
        ax.set_xticklabels(xl)
        ax.set_xlabel(xlabel, color=INK)
        ax.set_ylim(0, ytop or ymax * 1.03)   # ytop: curves above it run off the top edge, clipped at the frame
    axs[0].set_ylabel("normalized error", color=INK)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4 if key == "sim" else 3, frameon=False, labelcolor=INK, handlelength=1.8,
               columnspacing=1.1, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.15, 1, 1), w_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    D = json.load(open(os.path.join(HERE, "budget_data.json")))
    panel_fig(D, "sim", np.log10, ([2, 5, 10, 30, 100, 500], ["2", "5", "10", "30", "100", "500"]),
              "simulated hours (zero-shot)", "pogo_budget_sim", ytop=4.5)   # (2026-10-30, user: FADA's line runs out of frame; then "expand the range": 3.5 -> 4.5)
    real_x = lambda n: np.log2(1.0 + n)                       # 0 (zero-shot) at the left, rollouts on a log scale
    panel_fig(D, "real", real_x, ([0, 12, 36, 96, 384], ["0", "12", "36", "96", "384"]),
              "real rollouts per robot", "pogo_budget_real")
    print("->", OUT)
