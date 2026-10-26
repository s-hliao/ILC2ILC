#!/usr/bin/env python3
"""
video_quad_iters.py (2026-10-26): the quad's hardware stage (Go1 jump, sim-to-sim), iteration by iteration, on ONE
robot and ONE training goal, from the LOGGED trials (no re-fly): <run>/<robot>/it<k>/real/*.npz (rec_true_pos / quat
/ q at 2 ms) and it<k>/real.json (the landing errors). Earlier iterations stay as ghosts in their ramp colours.
  left    side view: the robot (trunk + sagittal legs, Go1 FK) of the current iteration, every iteration's trunk path
  right   the landing point vs the goal, zoomed (tolerance box 5 cm x 3 cm; one dot per iteration) | the landing error
          per iteration
    python video_quad_iters.py RUN ROBOT GOALX GOALZ [--procs 16] [--tag name]
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iter_style as S                                        # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quad")
HIPX, L1, L2 = 0.1881, 0.213, 0.213
SLOW = 0.5                                                    # 0.5x real time
D = None


def load(run, robot, g):
    its = []
    for k in range(20):
        d = f"{run}/{robot}/it{k}"
        if not os.path.exists(f"{d}/real.json"):
            break
        R = {tuple(np.round(r["goal"], 3)): r for r in json.load(open(f"{d}/real.json"))}
        f = glob.glob(f"{d}/real/policy_g{g[0]:.3f}_{g[1]:.3f}_*.npz")
        r = R.get(tuple(np.round(g, 3)))
        if not f or r is None:
            break
        z = np.load(f[0])
        w, x, y, zq = z["rec_true_quat"].T
        pitch = np.arcsin(np.clip(2 * (w * y - zq * x), -1, 1))
        t = z["rec_t"] - z["rec_t"][0]
        its.append(dict(t=t, p=z["rec_true_pos"] - [z["rec_true_pos"][0, 0], 0, 0], pitch=pitch, q=z["rec_true_q"],
                        ex=100 * r["ex"], ez=100 * r["ez"], fell=bool(r["fell"])))
    return its


def legs(p, pitch, q):
    c, s = np.cos(pitch), np.sin(pitch)
    W = lambda bx, bz: (p[0] + c * bx + s * bz, p[2] - s * bx + c * bz)
    out = []
    for i, hx in enumerate([HIPX, HIPX, -HIPX, -HIPX]):
        t1, t2 = q[3 * i + 1], q[3 * i + 2]
        kx, kz = hx - L1 * np.sin(t1), -L1 * np.cos(t1)
        fx, fz = kx - L2 * np.sin(t1 + t2), kz - L2 * np.cos(t1 + t2)
        out.append((W(hx, 0.0), W(kx, kz), W(fx, fz)))
    return out


def robot(ax, p, pt, q, col, alpha=1.0, z=5):
    c, s = np.cos(pt), np.sin(pt)
    body = np.array([[0.19, 0.05], [0.19, -0.05], [-0.19, -0.05], [-0.19, 0.05], [0.19, 0.05]])
    bx = p[0] + c * body[:, 0] + s * body[:, 1]
    bz = p[2] - s * body[:, 0] + c * body[:, 1]
    for j, (h, kn, f) in enumerate(legs(p, pt, q)):
        far = j in (0, 2)
        ax.plot([h[0], kn[0], f[0]], [h[1], kn[1], f[1]], color=S.INK2 if far else S.INK, lw=1.8 if far else 2.6,
                alpha=(0.45 if far else 1.0) * alpha, solid_capstyle="round", zorder=z - 1)
    ax.fill(bx, bz, color=col, ec=S.INK, lw=1.0, alpha=alpha, zorder=z)


def frame(job):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S.rc()
    i, k, t, phase, path = job
    its, g, K = D["its"], D["goal"], len(D["its"]) - 1
    cur = its[k]
    tk = cur["t"][-1] if t is None else t * SLOW / S.FPS
    j = min(int(np.searchsorted(cur["t"], tk)), len(cur["t"]) - 1)
    fig = plt.figure(figsize=(16, 9), dpi=100)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1.0], left=0.04, right=0.985, top=0.79, bottom=0.07, hspace=0.38, wspace=0.12)
    ax, aL, aB = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    ax.set_xlim(-0.35, 1.05)
    ax.set_ylim(-0.03, 0.72)
    ax.set_aspect("equal")
    ax.axhline(0, color=S.INK2, lw=1.2)
    if g[1] > 0:
        x0 = max(g[0] - 0.25, g[0] / 2 + 0.015)
        ax.add_patch(plt.Rectangle((x0, 0), 1.2 - x0, g[1], fc="#ecebe8", ec=S.INK2, lw=1.0, zorder=1))
    ax.plot([g[0]], [g[1] + 0.006], marker="*", ms=20, mfc="white", mec=S.INK, zorder=8)
    for kk in range(k):
        pp = its[kk]["p"]
        ax.plot(pp[:, 0], pp[:, 2], color=S.color(kk, K), lw=1.6, alpha=0.9, zorder=2)
    ax.plot(cur["p"][:j + 1, 0], cur["p"][:j + 1, 2], color=S.color(k, K), lw=3.0, zorder=3)
    robot(ax, cur["p"][j], cur["pitch"][j], cur["q"][j], S.color(k, K))
    ax.set_title(f"robot '{D['robot']}'  -  goal: jump {g[0]:.3f} m forward" + (f" onto a {100 * g[1]:.0f} cm box" if g[1] > 0 else "")
                 + f"    (t = {tk:.2f} s, {SLOW:g}x real time)", loc="left")
    ax.set_xlabel("forward (m)")
    ax.set_ylabel("height (m)")

    # landing point vs goal
    done = k + 1 if phase == "pause" else k
    aL.add_patch(plt.Rectangle((-5, -3), 10, 6, fc=S.TOL, ec="none", zorder=0))
    aL.axhline(0, color=S.GRID, lw=0.8)
    aL.axvline(0, color=S.GRID, lw=0.8)
    aL.plot([0], [0], marker="*", ms=22, mfc="white", mec=S.INK, zorder=8)
    for kk in range(done):
        q = its[kk]
        if q["fell"]:
            continue
        aL.plot([q["ex"]], [q["ez"]], "o", color=S.color(kk, K), mec=S.INK if kk == done - 1 else "none", ms=14 if kk == done - 1 else 10, zorder=5)
        aL.text(q["ex"] + 0.6, q["ez"] + 0.5, f"π{S.sub(kk)}", fontsize=11, color=S.color(kk, K), zorder=6)
    if done > 1:
        pts = np.array([[its[kk]["ex"], its[kk]["ez"]] for kk in range(done) if not its[kk]["fell"]])
        aL.plot(pts[:, 0], pts[:, 1], color=S.INK2, lw=0.8, ls=":", zorder=4)
    L = D["lim"]
    aL.set_xlim(-L, L)
    aL.set_ylim(-L * 0.6, L * 0.6)
    aL.set_aspect("equal")
    aL.set_title("where it landed, relative to the goal (cm)", loc="left")
    aL.set_xlabel("forward landing error (cm)")
    aL.set_ylabel("height error (cm)")
    aL.text(4.8, -2.8, "success box\n±5 / ±3 cm", ha="right", va="bottom", fontsize=9.5, color=S.OK)

    for kk in range(K + 1):
        if kk < done:
            q = its[kk]
            e = np.hypot(q["ex"], q["ez"])
            if q["fell"]:
                aB.bar(kk, D["bmax"] * 0.9, color="none", ec=S.BAD, hatch="//", width=0.7)
                aB.text(kk, D["bmax"] * 0.45, "fell", ha="center", rotation=90, color=S.BAD)
            else:
                aB.bar(kk, e, color=S.color(kk, K), width=0.7)
                aB.text(kk, e + 0.3, f"{e:.1f}", ha="center", fontsize=10.5)
        else:
            aB.bar(kk, D["bmax"] * 0.9, color="none", ec=S.GRID, ls=":", width=0.7)
    aB.set_xticks(range(K + 1))
    aB.set_xticklabels([f"π{S.sub(kk)}" for kk in range(K + 1)])
    aB.set_ylim(0, D["bmax"])
    aB.set_ylabel("cm")
    aB.set_title("landing error per iteration", loc="left")
    aB.set_xlabel("iteration (one real jump per training goal each)")
    res = None
    if phase == "pause":
        res = (f"π{S.sub(k)}: FELL" if cur["fell"] else
               f"π{S.sub(k)}: landed {cur['ex']:+.1f} cm forward, {cur['ez']:+.1f} cm height from the goal"
               + ("  (success)" if abs(cur["ex"]) <= 5 and abs(cur["ez"]) <= 3 else ""))
    S.header(fig, f"Quadruped jump (Go1): DistILC hardware stage, iteration by iteration  ({D['robot']}, sim-to-sim)",
             k, K, phase, res)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    global D
    run, rob, gx, gz = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 16
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
    its = load(run, rob, (gx, gz))
    for k, q in enumerate(its):
        print(k, "fell" if q["fell"] else f"ex {q['ex']:+.1f} ez {q['ez']:+.1f}", flush=True)
    e = [np.hypot(q["ex"], q["ez"]) for q in its if not q["fell"]]
    D = dict(its=its, goal=(gx, gz), robot=rob.replace("real_", "").replace("_m500", ""),
             lim=max(8.0, 1.25 * max(max(abs(q["ex"]), abs(q["ez"]) / 0.6) for q in its if not q["fell"])),
             bmax=max(e) * 1.3)
    os.makedirs(OUT, exist_ok=True)
    n_fly = [int(np.ceil(q["t"][-1] / SLOW * S.FPS)) + 1 for q in its]
    plan = S.frame_plan(n_fly, len(its) - 1)
    mp4 = f"{OUT}/quad_iters_{rob}_g{gx:.3f}_{gz:.2f}{tag}.mp4"
    n = S.render(frame, plan, mp4, procs, stills=[(0, mp4[:-4] + "_pi0.png"), (len(its) - 1, mp4[:-4] + "_final.png")])
    print(mp4, n, "frames", flush=True)


if __name__ == "__main__":
    main()
