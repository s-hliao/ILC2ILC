"""fig7_car_hw.py (2026-11-10): figure 7, the car hardware stage. The six NO-braking 5-lap evaluations of 11-08 / 11-09
(car_results_stage/README.md on the go1 workstation; copy ~/hw_results/car_1108_1109; user 2026-11-10: "everything is supposed to be the
no-braking results"), re-scored with car_score.py (score = rms_ey / 10 cm + max(0, 1 - pace) / 10 %, lower is better). One panel per track:
each network's zero-shot evaluation and its evaluation after our stage (4 two-lap trials; 5 for PPO + DR on sq41free). One evaluation per
condition; on the plastic tracks the same network varied by about +-0.2-0.5 between flights (README). Writes paper/final/fig7_car_hw.{pdf,png}."""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

for f in glob.glob("/usr/share/fonts/truetype/msttcorefonts/[Tt]imes*.ttf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Times New Roman", "TeX Gyre Termes", "Nimbus Roman", "STIXGeneral", "DejaVu Sans"],
                     "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42})
INK, MUTED, GRID = "#222222", "#777777", "#e4e4e4"
HW = r"DistIL-C$_\mathrm{hw}$"
# (track title, [(network, colour, marker, zero-shot (req, score, rms_ey m, pace), stage (req, score, rms_ey m, pace))])
D = [("fig-8, rubber tire (grip)",
      [("PPO + DR", "#eda100", "^", ("00112", 2.70, .124, .85), ("00128", 1.91, .058, .87)),
       ("RMA", INK, "*", ("00113", 2.11, .124, .91), ("00133", 1.58, .073, .91))]),
     ("square, plastic tire (drift)",
      [("PPO + DR", "#eda100", "^", ("00083", 4.68, .278, .81), ("00084", 4.29, .213, .78)),
       ("RMA", INK, "*", ("00087", 4.42, .234, .79), ("00093", 3.83, .156, .77))]),
     ("oval with foam, plastic tire",
      [("PPO + DR", "#eda100", "^", ("00007", 5.75, .323, .75), ("00019", 5.06, .281, .77)),
       ("RMA", INK, "*", ("00008", 6.05, .346, .74), ("00020", 5.56, .314, .76))])]


def main():
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.3), sharey=True)
    for ax, (title, rows) in zip(axs, D):
        for name, col, mk, zs, st in rows:
            ax.plot([0, 1], [zs[1], st[1]], color=col, lw=1.3, marker=mk, ms=5 if mk == "*" else 3.6, mec="white", mew=0.4,
                    label=f"{name}  zero-shot $\\rightarrow$ + {HW}")
            ch = (st[1] / zs[1] - 1) * 100
            ax.annotate(f"{zs[1]:.2f}", (0, zs[1]), xytext=(-5, 0), textcoords="offset points", ha="right", va="center", fontsize=6, color=INK)
            ax.annotate(f"{st[1]:.2f} ({ch:+.0f} %)", (1, st[1]), xytext=(5, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=6, color=INK)
        ax.set_title(title, fontsize=7.5, color=INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["zero-shot", "after the stage"])
        ax.set_xlim(-0.55, 1.75)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=INK, labelsize=7, width=0.6, length=2.5)
        ax.grid(True, axis="y", color=GRID, lw=0.5)
    axs[0].set_ylim(0, 6.6)
    axs[0].set_ylabel("normalized error (5-lap eval)", color=INK)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, labelcolor=INK, handlelength=1.8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.8)
    out = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig7_car_hw.{ext}"), dpi=300)
    print("->", out)


if __name__ == "__main__":
    main()
