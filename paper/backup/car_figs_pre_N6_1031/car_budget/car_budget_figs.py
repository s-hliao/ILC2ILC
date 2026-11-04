"""Car data budgets by dynamics coverage tier (A: inside the DR range; B: extrapolated along DR axes; C: outside the
model class: sensing, tire wear, stiffness imbalance; 12 cars each), normalized error max(rms e_y / 10 cm,
(1 - pace) / 0.1) with a crash = 5, mean and 95 % cluster-bootstrap CI over (seed, car, plan), training tracks, frozen
test pass (seed 4301), 3 seeds; LQR on the plan as a dotted reference line:
  car_budget_sim   zero-shot error vs simulated laps of training data (PPO: + its LQR-cloned BC start)
  car_budget_real  error vs real laps per car (0 = zero-shot)

Data: car_budget_data.json, made by ~/ilc_ws/log/dilc/car_free/car_budget_data.py (rerun it, then this, to refresh).
Style as the pogo budget figures (pogo_budget/budget_figs.py, imported). Writes car_budget_{sim,real}.{pdf,png} to
paper/final/.
    python paper/final/car_budget/car_budget_figs.py      (any python with matplotlib)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pogo_budget"))
import budget_figs as B  # noqa: E402  (fonts, colors, style, draw)
plt = B.plt

STY = dict(B.STY, **{"PPO + DR (5 tracks)": ("#eda100", "v", "--"),
                     "PPO + DR + PPO on car": B.STY["PPO + DR + PPO on robot"]})
B.STY.update(STY)
LAB = {"sim": dict(B.LAB["sim"]), "real": dict(B.LAB["real"], **{"PPO + DR + PPO on car": r"PPO + DR + PPO$_\mathrm{hw}$"})}
LQR = "LQR on the plan"


def panel_fig(D, key, xf, xticks, xlabel, name):
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.15), sharey=True)
    ymax = max(p[3] for arm in D[key].values() for pts in arm.values() for p in pts)
    for ax, (t, title) in zip(axs, B.TIERS):
        B.style(ax)
        for arm, per in D[key].items():
            if arm == LQR:
                ax.axhline(per[t][0][1], color=B.MUTED, ls=":", lw=1.0, label=LQR)
            elif per.get(t):
                B.draw(ax, arm, per[t], xf, LAB[key].get(arm, arm))
        ax.set_title(f"{title} ({len(D['tiers'][t])} cars)", fontsize=7.5, color=B.INK, pad=3)
        xt, xl = xticks
        ax.set_xticks(xf(np.array(xt, float)))
        ax.set_xticklabels(xl)
        ax.set_xlabel(xlabel, color=B.INK)
        ax.set_ylim(0, ymax * 1.03)
    axs[0].set_ylabel("normalized error", color=B.INK)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, labelcolor=B.INK, handlelength=1.8,
               columnspacing=1.1, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.15, 1, 1), w_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(B.OUT, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    D = json.load(open(os.path.join(HERE, "car_budget_data.json")))
    panel_fig(D, "sim", np.log10, ([1e3, 3e3, 1e4, 3e4, 1e5], ["1k", "3k", "10k", "30k", "100k"]),
              "simulated laps (zero-shot)", "car_budget_sim")
    real_x = lambda n: np.log2(1.0 + n)
    panel_fig(D, "real", real_x, ([0, 10, 40, 160, 320], ["0", "10", "40", "160", "320"]),
              "real laps per car", "car_budget_real")
    print("->", B.OUT)
