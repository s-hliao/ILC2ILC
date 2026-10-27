#!/usr/bin/env python3
"""
video_quad_goals.py (2026-10-26): goal conditioning -- ONE network π_k flies all 6 training jumps (3 flat distances,
3 boxes), iteration by iteration, no graphs: 6 side views of the same robot, each with its goal (star) and box,
earlier iterations' trunk paths as ghosts in their ramp colours; each iteration ends in a pause with every goal's
landing error and verdict (success: no fall, |ex| <= 5 cm, |ez| <= 3 cm). From the logged trials (no re-fly).
    python video_quad_goals.py RUN ROBOT [--procs 16] [--tag name]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iter_style as S                                        # noqa: E402
import video_quad_iters as V                                  # noqa: E402

GOALS = [(0.45, 0.0), (0.6, 0.0), (0.5, 0.05), (0.6, 0.1), (0.5, 0.15), (0.575, 0.15)]
D = None


def frame(job):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S.rc()
    i, k, t, phase, path = job
    K = D["K"]
    tk = D["T"][k] if t is None else t * V.SLOW / S.FPS
    fig = plt.figure(figsize=(16, 9), dpi=100)
    gs = fig.add_gridspec(2, 3, left=0.015, right=0.99, top=0.80, bottom=0.05, hspace=0.30, wspace=0.04)
    n_ok = 0
    for gi, g in enumerate(GOALS):
        ax = fig.add_subplot(gs[gi // 3, gi % 3])
        ax.set_xlim(-0.3, 1.0)
        ax.set_ylim(-0.03, 0.66)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.axhline(0, color=S.INK2, lw=1.2)
        if g[1] > 0:
            x0 = max(g[0] - 0.25, g[0] / 2 + 0.015)
            ax.add_patch(plt.Rectangle((x0, 0), 1.1 - x0, g[1], fc="#ecebe8", ec=S.INK2, lw=1.0, zorder=1))
        ax.plot([g[0]], [g[1] + 0.006], marker="*", ms=18, mfc="white", mec=S.INK, zorder=8)
        its = D["its"][g]
        if k >= len(its):
            ax.set_title(f"goal {gi + 1}: {g[0]:.3f} m" + (f", {100 * g[1]:.0f} cm box" if g[1] > 0 else ", flat"), loc="left")
            ax.text(0.5, 0.5, "frozen (held at its best)", transform=ax.transAxes, ha="center", color=S.INK2)
            continue
        for kk in range(k):
            pp = its[kk]["p"]
            ax.plot(pp[:, 0], pp[:, 2], color=S.color(kk, K), lw=1.3, alpha=0.8, zorder=2)
        cur = its[k]
        j = min(int(np.searchsorted(cur["t"], tk)), len(cur["t"]) - 1)
        ax.plot(cur["p"][:j + 1, 0], cur["p"][:j + 1, 2], color=S.color(k, K), lw=2.6, zorder=3)
        V.robot(ax, cur["p"][j], cur["pitch"][j], cur["q"][j], S.color(k, K))
        ax.set_title(f"goal {gi + 1}:  jump {g[0]:.3f} m" + (f" onto a {100 * g[1]:.0f} cm box" if g[1] > 0 else ", flat"),
                     loc="left", fontsize=12.5)
        if phase == "pause":
            ok = (not cur["fell"]) and abs(cur["ex"]) <= 5 and abs(cur["ez"]) <= 3
            n_ok += ok
            txt = "✗ fell" if cur["fell"] else (("✓ " if ok else "✗ ") + f"landed {cur['ex']:+.1f} cm forward, {cur['ez']:+.1f} cm up")
            ax.text(0.0, -0.03, txt, transform=ax.transAxes, ha="left", va="top", fontsize=12.5, weight="bold",
                    color=S.OK if ok else S.BAD)
            if cur["fell"]:
                ax.plot([cur["p"][j, 0]], [cur["p"][j, 2]], "x", color=S.BAD, ms=16, mew=3, zorder=9)
    res = f"one network π{S.sub(k)}, six jumps:  {n_ok} of 6 within 5 cm / 3 cm" if phase == "pause" else None
    S.header(fig, f"Quadruped jump (Go1): one goal-conditioned network jumps to every goal, iteration by iteration  "
                  f"({D['robot']}, sim-to-sim)", k, K, phase, res)
    fig.text(0.99, 0.845, f"star: the goal (landing CoM)     {V.SLOW:g}x real time", ha="right", va="center", fontsize=11,
             color=S.INK2)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    global D
    run, rob = sys.argv[1], sys.argv[2]
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 16
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
    its = {g: V.load(run, rob, g) for g in GOALS}
    K = max(len(v) for v in its.values()) - 1
    for g, v in its.items():
        print(g, ["fell" if q["fell"] else f"{np.hypot(q['ex'], q['ez']):.1f}" for q in v], flush=True)
    T = [max(v[k]["t"][-1] for v in its.values() if k < len(v)) for k in range(K + 1)]
    D = dict(its=its, K=K, T=T, robot=rob.replace("real_", "").replace("_m500", ""))
    os.makedirs(V.OUT, exist_ok=True)
    n_fly = [int(np.ceil(T[k] / V.SLOW * S.FPS)) + 1 for k in range(K + 1)]
    plan = S.frame_plan(n_fly, K)
    mp4 = f"{V.OUT}/quad_goals_{rob}{tag}.mp4"
    n = S.render(frame, plan, mp4, procs, stills=[(0, mp4[:-4] + "_pi0.png"), (K, mp4[:-4] + "_final.png")])
    print(mp4, n, "frames", flush=True)


if __name__ == "__main__":
    main()
