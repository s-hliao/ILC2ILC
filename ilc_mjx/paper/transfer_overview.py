#!/usr/bin/env python3
"""transfer_overview.py: sim-to-sim transfer of the Go1 jump, every method and ablation in one window.

Score = 0.01 x landing cost + 20 x fall (lower is better). "Unperturbed": nominal starts; "perturbed": the 8 start /
sensing perturbations per robot (front-foot blocks, crouch, tall, nose-up/down, bad mocap, 10 ms delay).
Sources: log/dilc/RESULTS_SUMMARY.md, deploy/NOTES.md, holdout/RESULTS.txt, budget/RESULTS.txt, modelsweep/; the
2026-10-05 queue (goal-range arms, stall guards, goal 0.575, few-shot baselines, per-perturbation errors) is read from
the eval files themselves (goals/, plots/pert_table.py's sources).
One graph per page in a single window: the < and > buttons (or the left/right arrow keys) page through them.
Run: ~/miniconda3/envs/ilcmjx/bin/python transfer_overview.py [--page N] [--save DIR]
(without a display, or with --save, every page is written to DIR, default src/paper/quadruped/transfer_overview/
NN_name.png -- inside the git repo, so the figures can be committed)."""
import glob
import json
import os
import sys

import matplotlib
import matplotlib.patches

if not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # ilc_ws/src
_LOG = os.path.normpath(os.path.join(_REPO, '..', 'log', 'dilc'))                                    # run records (not in git)
from matplotlib.widgets import Button  # noqa: E402

# reference palette (dataviz skill): categorical slots 1-3 (validated all-pairs), neutral for the oracle
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
C = {"ours": "#2a78d6", "abl": "#1baf7a", "base": "#eb6834", "oracle": "#7a7974"}
GROUP_NAME = {"ours": "Ours", "abl": "Ours, ablated", "base": "Baselines", "oracle": "Oracle (privileged)"}

# reserved test: 4 robots (real_r1, s1, r4, r5); 3 sim seeds unless noted. (unperturbed, perturbed or None)
RESERVED = [
    ("ours", "Ours: hardware stage from varied starts, cap 0.10", 2.71, 7.75),
    ("ours", "Ours: hardware stage from varied starts", 2.69, 8.08),
    ("ours", "Ours: hardware stage, nominal starts (E3)", 2.67, 10.76),
    ("abl", "half-wrong simulator (mass x0.93, motors x1.07)", 2.48, 9.40),
    ("abl", "safety gate on", 2.99, 11.06),
    ("abl", "random-direction steps (not the ILC step)", 3.33, None),
    ("abl", "feedback anchor (flow regularization)", 3.38, 11.64),
    ("abl", "no hardware stage (sim policy, zero-shot)", 3.48, 11.54),
    ("abl", "from scratch (no ILC-trained init)", 3.50, 14.60),
    ("abl", "network width 64 (from scratch)", 4.50, 14.00),
    ("abl", "4,800 sim episodes (from scratch)", 5.82, 14.00),
    ("abl", "network width 32 (from scratch)", 6.09, 16.70),
    ("abl", "no exploration in the sim stage (2 runs)", 1.62, 41.80),
    ("abl", "very wrong simulator (~30% thrust-to-weight)", 9.60, 32.50),
    ("base", "JumpILC (per-goal classical ILC)", 3.16, 28.60),
    ("base", "VG-SAC-FD (value-gradient critic)", 3.57, 10.70),
    ("base", "FADA-lite, DR teacher (our learner)", 5.29, 38.80),
    ("base", "DR (our learner)", 6.66, 15.50),
    ("base", "RMA-style student (our learner)", 6.89, 14.50),
    ("base", "PPO + DR (1 seed)", 8.38, 29.00),
    ("base", "RMA, PPO teacher (1 seed)", 9.86, 28.00),
    ("base", "FADA-lite, PPO teacher (1 seed)", 58.92, 64.88),
    ("oracle", "DR over the true robot family, CPU sim", 2.95, 6.77),
]

# frozen held-out test (real_r0, r2, r3; never used before): mean [95% CI], holdout/RESULTS.txt
HELDOUT = [
    ("ours", "Ours (E3), 6 seeds", (1.32, 1.03, 1.60), (7.88, 6.98, 8.80)),
    ("abl", "no hardware stage, 6 seeds", (1.60, 1.29, 1.93), (7.55, 6.74, 8.30)),
    ("abl", "from scratch, 3 seeds", (2.17, 1.15, 2.99), (10.12, 7.31, 12.33)),
    ("base", "per-goal JumpILC, matched budget", (0.76, np.nan, np.nan), None),
    ("base", "DR (our learner)", (3.98, 3.11, 4.92), (11.26, 10.29, 12.24)),
    ("base", "RMA-style student (our learner)", (3.88, 3.02, 4.84), (13.33, 12.21, 14.43)),
    ("base", "FADA-lite (our-learner teacher)", (4.00, 3.30, 4.82), (30.19, 27.81, 32.67)),
    ("base", "PPO + DR, 3 seeds", (9.83, 6.79, 13.61), (15.98, 11.28, 21.41)),
    ("base", "RMA (PPO), 3 seeds", (8.96, 2.88, 16.49), (15.38, 10.46, 19.08)),
    ("base", "FADA-lite (PPO teacher), 3 seeds", (39.13, 27.82, 56.61), (52.24, 48.64, 55.69)),
]

# jump budget (budget/RESULTS.txt), reserved test, 3 sim seeds x 4 robots
JUMPS = np.array([0, 3, 6, 9, 15, 21, 30])
B_UNP = np.array([3.48, 2.71, 2.74, 2.81, 2.71, 2.64, 2.67])
B_PER = np.array([11.54, 10.29, 10.54, 10.79, 10.98, 10.84, 10.76])

# model wrongness (modelsweep/, val, sim seeds ex1 + ex1s1): the Jacobians' model, the hardware stage's score
WRONG = [("x0.70", 1.725), ("x0.85", 1.700), ("right\nmodel", 1.590), ("x1.15", 1.655), ("x1.30", 1.730),
         ("sign\nflipped", 2.600)]
WRONG_ZERO_SHOT = 1.79


D = _LOG
SEEDS = ("ex1", "ex1s1", "ex1s2")
sc = lambda r: 0.01 * r["landing_cost"] + 20 * r["fell"]


def files(pat):
    return [f for f in glob.glob(os.path.join(D, pat)) if "invalid" not in f]


_JSON = {}


def load(f):
    if f not in _JSON:
        _JSON[f] = json.load(open(f))
    return _JSON[f]


def score(pat, keep=lambda r: True):
    """mean over files (robot x sim seed) of the per-file mean score; nan if none"""
    v = []
    for f in files(pat):
        rs = [r for r in load(f) if keep(r)]
        if rs:
            v.append(np.mean([sc(r) for r in rs]))
    return float(np.mean(v)) if v else np.nan


def arm(run):
    """a goal-range arm's run directories, all sim seeds"""
    return f"goals/{run}_ex1*/real_*"


# the 2026-10-05 arms: hardware stage on goals 0.40-0.55, 60 jumps (15 iterations x 4 goals), reserved test
ARMS = [("old stall guard (blind)", "g4055_blind"), ("no stall guard", "g4055"),
        ("saturation-aware guard", "g4055_smart"), ("fast steps (beta 0.6, cap 0.10)", "g4055_fast"),
        ("fast + saturation-aware guard", "g4055_smart_fast"), ("varied starts, cap 0.10", "g4055_vs")]
ARM_COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # categorical slots 1-6, fixed order
ARM_MK = ["o", "s", "D", "^", "v", "P"]
ARM_K = [0, 1, 2, 3, 5, 7, 10, 15]                          # checkpoints (iterations); 15 = the final policy


def arm_curve(run, inner=True):
    """% of the zero-shot score left vs real jumps, on the reserved test goals <= 0.55 (inside the training goals)"""
    keep = (lambda r: r["goal"][0] <= 0.55) if inner else (lambda r: True)
    z = score(f"{arm(run)}/eval_final_start.json", keep)
    out = []
    for k in ARM_K:
        tag = "start" if k == 0 else ("final" if k == 15 else f"it{k}")
        out.append(100 * score(f"{arm(run)}/eval_final_{tag}.json", keep) / z)
    return 4 * np.array(ARM_K), np.array(out)


def ext575(pat, last):
    """goal 0.575 after the stage, % of the zero-shot score there: (r1, s1, r4) and r5"""
    out = []
    for robots in (("real_r1", "real_s1", "real_r4"), ("real_r5",)):
        z = np.mean([score(pat.replace("real_*", r) + "/eval_ext575_it0.json") for r in robots])
        f = np.mean([score(pat.replace("real_*", r) + f"/eval_ext575_it{last}.json") for r in robots])
        out.append(100 * f / z)
    return out


EXT = [("E3: trained on 0.575", "deploy/e3fin_ex1*/real_*", 10, True),
       ("trained on 0.575, no guard", "goals/g3_nostall_ex1*/real_*", 20, True)] + \
      [(n, arm(r), 15, False) for n, r in ARMS]

# per-perturbation errors (plots/pert_table.py): reserved test, every method with perturbed evaluations
PERTS = ["blk15", "blk2", "crouch", "tall", "noseup", "nosedn", "mocapbad", "delay10"]
PERT_NAMES = ["block\n1.5 cm", "block\n2 cm", "crouch", "tall", "nose\nup", "nose\ndown", "bad\nmocap",
              "10 ms\ndelay"]
PERT_ROWS = [
    ("ours", "Ours: zero-shot", "deploy/e3fin_ex1*/real_*/eval_final_rob_start.json"),
    ("ours", "Ours: E3 (nominal starts)", "deploy/e3fin_ex1*/real_*/eval_final_rob_final.json"),
    ("ours", "Ours: varied starts", "vstart/vs1_cap10_ex1*/real_*/eval_final_rob_final.json"),
    ("abl", "sat-aware guard", "goals/g4055_smart_ex1*/real_*/eval_final_rob_final.json"),
    ("abl", "fast steps", "goals/g4055_fast_ex1*/real_*/eval_final_rob_final.json"),
    ("base", "JumpILC (per-goal ILC)", "final701/rob_jilc_real_*.json"),
    ("base", "DR (our learner)", "fada/eval_drplain/real_*/eval_final_rob_start.json"),
    ("base", "RMA-style student (our learner)", "fada/eval_student/real_*/eval_final_rob_start.json"),
    ("base", "PPO + DR", "ppo/eval_plain/real_*/eval_final_rob_start.json"),
    ("base", "RMA (PPO)", "ppo/eval_student/real_*/eval_final_rob_start.json"),
    ("base", "PPO + DR + our stage", "goals/fs_ppo_s*/real_*/eval_final_rob_final.json"),
    ("oracle", "Oracle (true-family DR)", "deploy/drtrue_ilc/real_*/eval_final_rob_start.json"),
]


def pert_matrix():
    M = np.full((len(PERT_ROWS), len(PERTS) + 1), np.nan)
    for i, (_, _, pat) in enumerate(PERT_ROWS):
        for j, pt in enumerate(PERTS):
            M[i, j] = score(pat, lambda r, pt=pt: r["cond"].endswith("+" + pt))
        M[i, -1] = np.nanmean(M[i, :-1])
    return M


def add_new_reserved_rows():
    """the 2026-10-05 queue's rows of the reserved-test panel, from the eval files"""
    rows = []
    for n, r in ARMS:
        rows.append(("abl", f"goals 0.40-0.55, 60 jumps: {n}", score(f"{arm(r)}/eval_final_final.json"),
                     score(f"{arm(r)}/eval_final_rob_final.json")))
    for n, pat in (("PPO + DR + our hardware stage (30 jumps)", "goals/fs_ppo_s*/real_*"),
                   ("RMA (PPO) + our hardware stage (30 jumps)", "goals/fs_rma_s*/real_*"),
                   ("DR (our learner) + our hardware stage", "goals/fs_dr/real_*")):
        rows.append(("base", n, score(f"{pat}/eval_final_final.json"), score(f"{pat}/eval_final_rob_final.json")))
    return rows


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def dumbbell(ax, rows, title, ci=False):
    """one row per method: unperturbed (filled) and perturbed (hollow) on one log score axis, grouped by color"""
    y, ylabels, ycolors, prev = 0.0, [], [], None
    ys = []
    for row in rows:
        g = row[0]
        if prev is not None and g != prev:
            y += 0.6                                      # a gap between groups
        ys.append(y)
        prev = g
        y += 1
    for yy, row in zip(ys, rows):
        g, name, u, p = row
        col = C[g]
        um = u[0] if ci else u
        pm = (p[0] if ci else p) if p is not None else None
        if pm is not None:
            ax.plot([um, pm], [yy, yy], color=col, lw=1.5, alpha=0.45, solid_capstyle="round", zorder=2)
        if ci:
            if np.isfinite(u[1]):
                ax.plot([u[1], u[2]], [yy, yy], color=col, lw=4, alpha=0.25, solid_capstyle="round", zorder=1)
            if p is not None:
                ax.plot([p[1], p[2]], [yy, yy], color=col, lw=4, alpha=0.25, solid_capstyle="round", zorder=1)
        ax.scatter([um], [yy], s=58, color=col, edgecolor=SURF, linewidth=2, zorder=4)
        if pm is not None:
            ax.scatter([pm], [yy], s=58, facecolor=SURF, edgecolor=col, linewidth=2, zorder=4)
        if g == "ours":                                   # selective direct labels: our rows only, beside the dots
            ax.annotate(f"{um:.2f}", (um, yy), xytext=(-9, 0), textcoords="offset points", ha="right", va="center",
                        fontsize=8.5, color=INK, fontweight="bold")
            if pm is not None:
                ax.annotate(f"{pm:.2f}", (pm, yy), xytext=(9, 0), textcoords="offset points", ha="left",
                            va="center", fontsize=8.5, color=INK)
        ylabels.append(name)
        ycolors.append(INK if g == "ours" else INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels(ylabels)
    for t, c, row in zip(ax.get_yticklabels(), ycolors, rows):
        t.set_color(c)
        if row[0] == "ours":
            t.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.55, 85)
    ax.set_xticks([1, 2, 3, 5, 10, 20, 50])
    ax.set_xticklabels(["1", "2", "3", "5", "10", "20", "50"])
    ax.set_xlabel("landing score (log scale, lower is better)", color=INK2, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")


TITLE = "Sim-to-sim transfer of a closed-loop dynamic skill (Go1 jump)"
SUB = ("Policies trained in a nominal GPU simulator, transferred to CPU-simulated 'real' robots (async timing, sensing "
       "noise, perturbed dynamics).\nScore = 0.01 x landing cost + 20 x fall; lower is better.")


def group_legend(fig, y=0.885):
    handles = [Line2D([], [], marker="o", ls="", ms=8, color=C[g], mec=SURF, label=GROUP_NAME[g]) for g in C]
    handles += [Line2D([], [], marker="o", ls="", ms=8, color=INK2, mec=SURF, label="unperturbed starts (filled)"),
                Line2D([], [], marker="o", ls="", ms=8, mfc=SURF, mec=INK2, mew=2, label="perturbed starts (hollow)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, y), ncol=6, frameon=False, fontsize=9.5,
               labelcolor=INK2, handletextpad=0.3, columnspacing=1.4)


def page_reserved(fig):
    ax = fig.add_axes(plot_rect(0.33, 0.08, 0.63, 0.74))
    style(ax)
    new = add_new_reserved_rows()
    grp = lambda rs, g: [r for r in rs if r[0] == g]
    rows = grp(RESERVED, "ours") + grp(RESERVED, "abl") + grp(new, "abl") + grp(RESERVED, "base") + grp(new, "base") \
        + grp(RESERVED, "oracle")
    dumbbell(ax, rows, "All methods and ablations (reserved test, 4 robots, 3 sim seeds unless noted)")
    group_legend(fig)


def page_heldout(fig):
    ax = fig.add_axes(plot_rect(0.25, 0.1, 0.71, 0.7))
    style(ax)
    dumbbell(ax, HELDOUT, "Frozen held-out test (3 never-used robots), 95% intervals", ci=True)
    group_legend(fig)


def page_jumps(fig):
    ax = fig.add_axes(plot_rect(0.07, 0.1, 0.7, 0.72))
    style(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ends = []
    for (n, r), col, mk in zip(ARMS, ARM_COL, ARM_MK):
        x, y = arm_curve(r)
        ax.plot(x, y, color=col, lw=2.2, marker=mk, ms=8, mec=SURF, mew=1.5, label=n, zorder=3)
        ends.append([y[-1], n, col])
    ends.sort(key=lambda e: e[0])
    yl = -1e9
    for e in ends:                                                 # direct labels at the line ends, kept apart
        yl = max(e[0], yl + 2.6)
        e.append(yl)
    for yv, n, col, yl in ends:
        ax.annotate(f"{yv:.0f}%  {n}", (60, yv), xytext=(61.5, yl), textcoords="data", va="center", fontsize=9,
                    color=INK)
    ax.set_xlim(-1, 61)
    ax.set_ylim(30, 103)
    ax.set_xticks(4 * np.array(ARM_K))
    ax.set_xlabel("real jumps on each robot (15 iterations x 4 training goals)", color=INK2, fontsize=10)
    ax.set_ylabel("% of zero-shot error left (reserved test goals <= 0.55)", color=INK2, fontsize=10)
    ax.set_title("Few-shot: error vs hardware jumps (training goals 0.40-0.55, 3 sim seeds x 4 robots)", loc="left",
                 color=INK, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, frameon=False, labelcolor=INK2, handlelength=2.4)


def page_ext(fig):
    ax = fig.add_axes(plot_rect(0.25, 0.1, 0.7, 0.72))
    style(ax)
    yy = np.arange(len(EXT))
    for i, (n, pat, last, trained) in enumerate(EXT):
        a, b = ext575(pat, last)
        ax.plot([a, b], [i, i], color=GRID, lw=2.5, zorder=1)
        ax.scatter([a], [i], s=75, color="#2a78d6", edgecolor=SURF, linewidth=1.5, zorder=3)
        ax.scatter([b], [i], s=85, marker="^", color="#eb6834", edgecolor=SURF, linewidth=1.2, zorder=3)
        ax.annotate(f"{b:.0f}%", (b, i), xytext=(9, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
        ax.annotate(f"{a:.0f}%", (a, i), xytext=(-9, 0), textcoords="offset points", va="center", ha="right",
                    fontsize=9, color=INK)
    ax.axvline(100, color=MUTED, lw=1.2, ls=(0, (3, 2)), zorder=0)
    ax.annotate("zero-shot", (100, -0.7), ha="center", fontsize=9, color=INK2)
    ax.axhline(1.5, color=GRID, lw=1)
    ax.annotate("trained on 0.575 (reference)", (1150, -0.45), ha="right", va="center", fontsize=9, color=INK2,
                style="italic")
    ax.annotate("trained on 0.40-0.55 only, 60 jumps", (1150, 1.85), ha="right", va="center", fontsize=9,
                color=INK2, style="italic")
    ax.set_ylim(len(EXT) - 0.5, -0.9)
    ax.set_yticks(yy)
    ax.set_yticklabels([n for n, *_ in EXT], fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlim(25, 1200)
    ax.set_xticks([30, 50, 100, 200, 500, 1000])
    ax.set_xticklabels(["30%", "50%", "100%", "200%", "500%", "1000%"])
    ax.set_xlabel("score at goal 0.575 after the hardware stage, % of the zero-shot score (log scale)", color=INK2,
                  fontsize=10)
    ax.set_title("Goal 0.575, outside the training goals: improves on 3 robots, fails on the heaviest", loc="left",
                 color=INK, fontsize=12, fontweight="bold")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=8, color="#2a78d6", mec=SURF, label="real_r1, s1, r4"),
                       Line2D([], [], marker="^", ls="", ms=8, color="#eb6834", mec=SURF, label="real_r5 (heaviest)")],
              loc="lower right", fontsize=9.5, frameon=False, labelcolor=INK2)


def page_pert(fig):
    ax = fig.add_axes(plot_rect(0.2, 0.06, 0.72, 0.7))
    M = pert_matrix()
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    im = ax.imshow(M, cmap=cmap, norm=LogNorm(2, 80), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9.5,
                    color=SURF if v > 12 else INK, fontweight="bold" if j == M.shape[1] - 1 else "normal")
    ax.axvline(M.shape[1] - 1.5, color=SURF, lw=3)
    for i in range(1, len(PERT_ROWS)):                       # surface gaps between method groups
        if PERT_ROWS[i][0] != PERT_ROWS[i - 1][0]:
            ax.axhline(i - 0.5, color=SURF, lw=3)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(PERT_NAMES + ["mean"], fontsize=9.5, color=INK2)
    ax.set_yticks(range(len(PERT_ROWS)))
    ax.set_yticklabels([r[1] for r in PERT_ROWS], fontsize=9.5)
    for t, r in zip(ax.get_yticklabels(), PERT_ROWS):
        t.set_color(INK if r[0] == "ours" else INK2)
        if r[0] == "ours":
            t.set_fontweight("bold")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.xaxis.tick_top()
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.set_label("score (log scale), lower is better", color=INK2, fontsize=9.5)
    cb.set_ticks([2, 5, 10, 20, 50])
    cb.set_ticklabels(["2", "5", "10", "20", "50"])
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=9, colors=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Perturbed transfer, per perturbation (reserved test, mean over robots and sim seeds)", loc="left",
                 color=INK, fontsize=12, fontweight="bold", pad=40)


def page_wrong(fig):
    ax = fig.add_axes(plot_rect(0.1, 0.12, 0.85, 0.68))
    style(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    x = np.arange(len(WRONG))
    cols = [C["ours"] if "right" in n else C["abl"] for n, _ in WRONG]
    ax.bar(x, [v for _, v in WRONG], width=0.55, color=cols, edgecolor=SURF, linewidth=2, zorder=3)
    for xi, (_, v) in zip(x, WRONG):
        ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, -16), textcoords="offset points", ha="center", fontsize=10,
                    color=SURF, fontweight="bold")
    ax.axhline(WRONG_ZERO_SHOT, color=MUTED, lw=1.2, ls=(0, (3, 2)), zorder=4)
    ax.annotate(f"dashed: no hardware stage ({WRONG_ZERO_SHOT:.2f})", (-0.4, 2.85), fontsize=9.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("\n", " ") for n, _ in WRONG], fontsize=10)
    ax.set_ylim(0, 3.0)
    ax.set_xlabel("the model the ILC's Jacobians come from (thrust-to-weight scaled)", color=INK2, fontsize=10)
    ax.set_ylabel("landing score after 30 jumps (val)", color=INK2, fontsize=10)
    ax.set_title("Only the model's sign matters (ILC Jacobians from a wrong model)", loc="left",
                 color=INK, fontsize=12, fontweight="bold")


# saved pages go into the git repo (src/) so they can be committed; log/ is not tracked
REPO_FIGS = os.path.join(_REPO, "paper", "quadruped", "transfer_overview")
# each page's summary column: what was run to make the graph, and what it shows
XS = 0.69                                                   # the graph's share of the (wider) figure, left
SUMMARY = {
    "all_methods": (
        "Every method on the reserved test: 4 CPU 'real' robots (real_r1, s1, r4, r5) x goals 0.4375 / 0.4875 / "
        "0.5125 / 0.5625 x 4 seeds (from 701); 'perturbed' adds 8 start / sensing perturbations per robot. Ours: "
        "policy from a nominal GPU sim (no domain randomization), then the few-shot hardware ILC stage (E3: 30 jumps; "
        "and from varied starts). Ablations change one component (sim model, safeguards, init, width, exploration) or "
        "the hardware stage's goals and steps (goals 0.40-0.55, 60 jumps). Baselines: per-goal JumpILC, VG-SAC-FD, DR / "
        "RMA / FADA with our learner and with PPO, and three of them given our hardware stage. Oracle: DR over the "
        "true robot family (privileged).",
        ["Ours is the best non-oracle method on both: 2.67-2.71 unperturbed, 7.75 perturbed (oracle 6.77).",
         "Training with DR transfers worse than a nominal sim (DR 6.66 / 15.5; PPO-DR 8.38 / 29.0).",
         "Our hardware stage improves the baselines (PPO+DR 8.88 -> 5.68) but they do not catch up: the sim stage "
         "matters, not only the ILC.",
         "Robustness comes from the hardware stage's varied starts (10.76 -> 7.75 perturbed), not from more jumps "
         "or stall guards.",
         "Exploration in the sim stage is what keeps it robust: without it 1.62 unperturbed but 41.8 perturbed."]),
    "heldout": (
        "The frozen final test, run once: 3 robots never used before (real_r0, r2, r3), 6 goals 0.4375-0.5625, seed "
        "1301, unperturbed and the 8 perturbations. Ours: 6 sim seeds (E3 hardware stage, 30 jumps per robot); "
        "baselines 1-3 seeds. Bars: 95% bootstrap intervals. Per-goal JumpILC flown with the same jump budget.",
        ["Our hardware stage beats its own zero-shot policy: 1.32 vs 1.60 (better in 15/18 robot x seed cells, "
         "p = 0.008); perturbed is unchanged (7.88 vs 7.55, n.s.).",
         "Every DR / RMA / FADA / PPO baseline is significantly worse (3.9 to 39).",
         "Per-goal JumpILC is better unperturbed (0.76 vs 1.04, also at fresh seeds) but it learns one goal at a time "
         "and collapses under perturbation (28.6 on the reserved test)."]),
    "error_vs_jumps": (
        "The hardware stage on training goals 0.40 / 0.45 / 0.50 / 0.55 for 15 iterations (60 jumps per robot), "
        "3 sim seeds x 4 robots. Each checkpoint is flown on the reserved test goals <= 0.55 (unperturbed); y is "
        "its score as % of the zero-shot policy's. Arms: the old stall guard (freezes any goal whose error stops "
        "falling); no guard; the saturation-aware guard (deploy.py --stall-sat 0.2 --sat-project: freezes only "
        "when the motors' torque-speed limits block >= 20% of the correction); fast steps (beta 0.6, step cap 0.10, "
        "no bold driver); fast + saturation-aware; varied starts (S = 1, cap 0.10).",
        ["The step-size safeguards were the bottleneck: fast steps leave 36% of the zero-shot error (a 64% "
         "reduction) vs 60-69%, and it is still falling at 60 jumps.",
         "The old stall guard plateaus at 69% after 8 jumps: it froze 74% of goal-iterations, most with the "
         "motors far from saturation.",
         "The saturation-aware guard freezes 6-7%, learns as well as no guard (60%) and has the best perturbed "
         "score of the nominal-start arms (10.38).",
         "Varied starts learn slower at first but keep improving (51%) and are the most robust arm (7.85)."]),
    "goal_0575": (
        "Graceful degradation: goal 0.575, outside the 0.40-0.55 training goals, flown 8 times per robot (seeds from "
        "701) with the zero-shot policy and after 60 jumps; the score after, as % of the zero-shot score at 0.575, for "
        "real_r1 / s1 / r4 (pooled) and real_r5, the heaviest robot. References: two runs trained on 0.575 itself "
        "(E3, 30 jumps; no guard, 60 jumps).",
        ["On 3 of 4 robots the training carries over: 0.575 improves to 49-80% of zero-shot (the fast arms most).",
         "On real_r5 it fails: it lands 17-28 cm short and nose-down, 1.5-6x its zero-shot score. Training on "
         "0.575 itself does not fix r5 either (124-265%).",
         "Cause: r5 is near its motor limits on long jumps, and the shared network carries its corrections across "
         "goals. The saturation guard cannot see this, because r5's training goals are not saturated.",
         "Next: anchor the policy beyond the training goals (to 0.60) so out-of-range behaviour stays zero-shot."]),
    "per_perturbation": (
        "Each method's final policy on the reserved test under the 8 perturbations per robot: 1.5 / 2 cm blocks under "
        "the front feet; crouched / tall / nose-up / nose-down stance; degraded mocap (120 Hz, 15 ms, noisier); +10 ms "
        "actuation delay. Goals 0.4375-0.5625 x 4 seeds; ours averaged over 3 sim seeds, baselines 1 seed (PPO + our "
        "stage: 3). Cells: mean score, darker is worse.",
        ["The varied-start hardware stage halves the block and nose-up errors without ever flying a block: "
         "7.7 mean, close to the oracle (6.8).",
         "Nose-down is unfixed in all our variants (~15, almost all real_r5). It is the one perturbation where PPO / "
         "RMA (11.7-13) beat us.",
         "JumpILC and the PPO-based methods collapse on blocks and nose-up (35-78).",
         "Timing and sensing perturbations (bad mocap, delay) are mild for ours (3-10)."]),
    "wrong_model": (
        "The E3 hardware stage (30 jumps, val goals) with the ILC's Jacobians taken from a deliberately wrong model: "
        "thrust-to-weight scaled x0.70 to x1.30, or the landing sensitivity's sign flipped. The policy and the robots "
        "are unchanged; sim seeds ex1 and ex1s1, 4 robots.",
        ["A model 30% wrong costs little (1.66-1.73 vs 1.59), and every scaled model still beats no hardware stage (1.79).",
         "Flipping the sign makes it worse than no stage at all (2.60).",
         "So the ILC step only needs the right direction from the nominal sim. Training the policy itself on a very "
         "wrong sim does fail (9.6, page 1)."]),
}


def plot_rect(l, b, w, h):
    """a graph's rectangle, given for a full-width page, inside the left XS of the figure"""
    return [l * XS, b, w * XS, h]


def summary(fig, name):
    import textwrap
    run, take = SUMMARY[name]
    x0, y, lh = XS + 0.025, 0.84, 0.0215                    # line height in figure units (9.5 pt on 9.5 in)
    fig.patches.append(matplotlib.patches.FancyBboxPatch((XS + 0.012, 0.075), 1 - XS - 0.022, 0.795,
                                                         boxstyle="round,pad=0,rounding_size=0.008",
                                                         transform=fig.transFigure, facecolor="#f1f0ec",
                                                         edgecolor=GRID, zorder=0))
    fig.text(x0, y, "What was run", fontsize=11, fontweight="bold", color=INK, va="top")
    y -= 0.035
    body = textwrap.fill(run, 68)
    fig.text(x0, y, body, fontsize=9.5, color=INK2, va="top", linespacing=1.45)
    y -= lh * (body.count("\n") + 1) + 0.03
    fig.text(x0, y, "Takeaways", fontsize=11, fontweight="bold", color=INK, va="top")
    y -= 0.035
    for t in take:
        lines = textwrap.fill(t, 64)
        fig.text(x0, y, "\u2022", fontsize=9.5, color=INK, va="top")
        fig.text(x0 + 0.012, y, lines, fontsize=9.5, color=INK, va="top", linespacing=1.45)
        y -= lh * (lines.count("\n") + 1) + 0.012


PAGES = [("all_methods", page_reserved), ("heldout", page_heldout), ("error_vs_jumps", page_jumps),
         ("goal_0575", page_ext), ("per_perturbation", page_pert), ("wrong_model", page_wrong)]


def draw(fig, i, nav=True):
    fig.clf()
    fig.set_facecolor(SURF)
    fig.text(0.012, 0.965, TITLE, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.93, SUB, fontsize=9.5, color=INK2, va="top")
    fig.text(0.988, 0.965, f"{i + 1} / {len(PAGES)}", fontsize=11, color=INK2, ha="right", va="top")
    PAGES[i][1](fig)
    summary(fig, PAGES[i][0])
    if nav:                                                       # the buttons are rebuilt with the page
        bp = Button(fig.add_axes([0.86, 0.012, 0.05, 0.04]), "<  prev", color=GRID, hovercolor="#cde2fb")
        bn = Button(fig.add_axes([0.915, 0.012, 0.05, 0.04]), "next  >", color=GRID, hovercolor="#cde2fb")
        bp.on_clicked(lambda _: go(fig, -1))
        bn.on_clicked(lambda _: go(fig, +1))
        fig._nav = (bp, bn)                                       # keep references, or the buttons go dead
    fig.canvas.draw_idle()


def go(fig, d):
    fig._page = (fig._page + d) % len(PAGES)
    draw(fig, fig._page)


def main():
    page = int(sys.argv[sys.argv.index("--page") + 1]) - 1 if "--page" in sys.argv else 0
    interactive = matplotlib.get_backend().lower() != "agg" and "--save" not in sys.argv
    fig = plt.figure(figsize=(21, 9.5), facecolor=SURF)
    if not interactive:
        out = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else \
            REPO_FIGS
        os.makedirs(out, exist_ok=True)
        for i, (name, _) in enumerate(PAGES):
            draw(fig, i, nav=False)
            fig.savefig(os.path.join(out, f"{i + 1:02d}_{name}.png"), dpi=130, facecolor=SURF)
        print(f"saved {len(PAGES)} pages to {out}")
        return
    fig._page = page
    fig.canvas.mpl_connect("key_press_event",
                           lambda e: go(fig, -1) if e.key == "left" else go(fig, +1) if e.key == "right" else None)
    draw(fig, page)
    plt.show()


if __name__ == "__main__":
    main()
