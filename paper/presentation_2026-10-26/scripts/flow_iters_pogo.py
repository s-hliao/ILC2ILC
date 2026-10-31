#!/usr/bin/env python3
"""
flow_iters_pogo.py (2026-10-26): the pogo's apex-to-apex flow after every stage iteration, re-converging from the
corrupted field (real stick, π₀) back toward the sim's (sim, π₀) -- the servo-leg stage of the iteration videos
(log/pogo_servo/runs/v3/nom_s0_curve: π₀ .. π₄). Portrait look (portrait_style): background = |V - V_sim,π₀| in
tolerance units, arrows = the field, orange = real runs and their fixed point, star / dashed box = goal / tolerance.
  columns  sim π₀ (the reference) | the real stick under π_k | over iterations: mean field deviation, fixed-point error
  rows     the offset stick, the mass stick
Each iteration holds HOLD_S seconds, then a XFADE_S crossfade (a blend of the two frames, no invented fields) to the
next; the header shows the iteration pills as in the stage videos.
    POGO_LEG=servo python flow_iters_pogo.py
"""
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FL = "/home/henry/ilc_ws/log/dilc/flows"
SERVO = "/home/henry/ilc_ws/log/pogo_servo"
sys.path.insert(0, FL)
sys.path.insert(0, f"{FL}/pogo")
sys.path.insert(0, HERE)
import portrait_style as ps                                       # noqa: E402
import flows_pogo as fp                                           # noqa: E402
import iter_style as S                                            # noqa: E402

HOLD_S, XFADE_S, FINAL_S = 2.2, 0.6, 3.5
ROBOTS = ("real_offset", "real_mass")
GOAL = np.array((0.8, 0.62))
N = 13


def paths(robot, arm="nom_s0"):
    R = f"{SERVO}/runs/v3/{arm}_curve"
    return [f"{SERVO}/runs/nom_s0/policy_pick.npz"] + [f"{R}/it{k}/{robot}_policy.npz" for k in (1, 2, 3)] + [f"{R}/{robot}_policy.npz"]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    assert fp.C.LEG == "servo"
    g = GOAL
    hs = np.linspace(g[1] - 0.08, g[1] + 0.08, N)
    vs = np.linspace(g[0] - 0.5, g[0] + 0.5, N)
    VV, HH = np.meshgrid(vs, hs)
    flat = np.stack([HH.ravel(), VV.ravel()], -1)
    Gl = np.repeat(g[None], len(flat), 0)
    X0, Gr = fp.drop_apexes(g[None])

    def field(p, system):
        F = fp.fly_hop(p, flat[:, 0], flat[:, 1], Gl, system)[0]
        return F - flat, fp.orbit(p, X0, Gr, system, hops=fp.N_HOPS)

    P0 = paths(ROBOTS[0])[0]
    Vs, os_ = field(P0, "sim")
    rl = ps.ref_len(Vs[:, 1], Vs[:, 0], vs[-1] - vs[0], hs[-1] - hs[0])
    K = 4
    real, dev, fpe, dev_near = {}, {}, {}, {}
    for r in ROBOTS:
        for k, p in enumerate(paths(r)):
            V, o = field(p, r)
            real[(r, k)] = (V, o)
            dV = np.linalg.norm((V - Vs) / fp.TOL, axis=1)
            dev[(r, k)] = float(np.nanmean(dV))
            near = (np.abs(flat[:, 0] - g[1]) <= 2 * fp.TOL[0]) & (np.abs(flat[:, 1] - g[0]) <= 2 * fp.TOL[1])
            dev_near[(r, k)] = float(np.nanmean(dV[near]))
            late = o[fp.FP_FROM:]
            fpe[(r, k)] = float(np.linalg.norm((late.reshape(-1, 2).mean(0) - g[::-1]) / fp.TOL[::-1])) \
                if np.isfinite(late).all() else np.nan
            print(r, k, f"mean deviation {dev[(r, k)]:.2f} tol, near the goal {dev_near[(r, k)]:.2f}, fixed point {fpe[(r, k)]:.2f} tol", flush=True)
    sim_dev = 0.0

    tmp = f"{HERE}/pogo/_flow_frames"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    keyframes = []
    for k in range(K + 1):
        ps.rc()
        S.rc()
        fig = plt.figure(figsize=(16, 9), dpi=100)
        gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 0.045, 0.32, 1.05], left=0.045, right=0.975, top=0.755, bottom=0.08,
                              hspace=0.36, wspace=0.16)
        for ri, r in enumerate(ROBOTS):
            for ci, (V, o, ttl) in enumerate([(Vs, os_, "sim, π₀  (the reference)"),
                                              (*real[(r, k)], f"real {r[5:]} stick, π{S.sub(k)}" + ("  (corrupted)" if k == 0 else ""))]):
                ax = fig.add_subplot(gs[ri, ci])
                if ci == 1:
                    ax.set_yticklabels([])
                dV = np.linalg.norm((V - Vs) / fp.TOL, axis=1).reshape(HH.shape)
                ps.background(ax, VV, HH, dV)
                ps.arrows_scaled(ax, VV, HH, V[:, 1].reshape(HH.shape), V[:, 0].reshape(HH.shape), vs[-1] - vs[0],
                                 hs[-1] - hs[0], N - 1, ref=rl)
                for j in range(o.shape[1]):
                    ps.runs(ax, o[:, j, 1], o[:, j, 0])   # (2026-10-30, user: no orange lines, the individual apex points)
                late = o[fp.FP_FROM:]
                if np.isfinite(late).all():
                    xb = late.reshape(-1, 2).mean(0)
                    ps.fixed_point(ax, xb[1], xb[0])
                ps.goal(ax, g[0], g[1])
                ax.add_patch(plt.Rectangle((g[0] - fp.TOL[1], g[1] - fp.TOL[0]), 2 * fp.TOL[1], 2 * fp.TOL[0], fill=False,
                                           ec=ps.INK, lw=1.0, ls=(0, (3, 2)), zorder=6))
                ax.set_xlim(vs[0], vs[-1])
                ax.set_ylim(hs[0], hs[-1])
                if ci == 1:
                    ttl += f"\nmean deviation from sim: {dV.mean():.2f} tol"
                    for s_ in ax.spines.values():
                        s_.set_visible(True)
                        s_.set_edgecolor(S.color(k, K))
                        s_.set_linewidth(3.0)
                ax.set_title(ttl, fontsize=12)
                if ci == 0:
                    ax.set_ylabel(f"apex height (m)")
                if ri == 1:
                    ax.set_xlabel("apex speed vx (m/s)")
        cax = fig.add_subplot(gs[:, 2])
        cax.set_position([cax.get_position().x0, 0.2, cax.get_position().width, 0.45])
        sm = plt.cm.ScalarMappable(norm=ps.NORM, cmap=ps.CMAP)
        cb = fig.colorbar(sm, cax=cax, extend="max")
        cb.set_label("deviation from the sim field |V - V_sim,π₀|  (tolerance units)", color=S.INK2)
        cb.outline.set_edgecolor(S.GRID)
        # over iterations
        for ri, (key, lab) in enumerate([(dev_near, "field deviation from sim near the goal (tol units)"),
                                         (fpe, "fixed point's distance to the goal (tol units)")]):
            ax = fig.add_subplot(gs[ri, 4])
            for r, mk, dy in zip(ROBOTS, ("o", "s"), (1, -1)):
                y = [key[(r, kk)] for kk in range(K + 1)]
                ax.plot(range(k + 1), y[:k + 1], "-", color=S.INK2, lw=1.2, zorder=2)
                for kk in range(k + 1):
                    ax.plot([kk], [y[kk]], mk, color=S.color(kk, K), mec=S.INK, ms=11 if kk == k else 8, zorder=3)
                ax.plot([], [], mk, color="white", mec=S.INK, ms=9, label=f"{r[5:]} stick")
            if ri == 1:
                ax.axhspan(0, 1.0, color=S.TOL, zorder=0)
                ax.text(-0.3, 0.5, "inside tolerance", ha="left", va="center", color=S.OK, fontsize=10)
            else:
                ax.axhline(sim_dev, color=S.INK, ls="--", lw=1.0)
                ax.text(-0.3, 0.08, "sim π₀ = 0   (over the 2 x 2 tolerance boxes around the goal)", ha="left", va="bottom", fontsize=9.5, color=S.INK2)
            ax.set_xticks(range(K + 1))
            ax.set_xticklabels([f"π{S.sub(kk)}" for kk in range(K + 1)])
            ax.set_xlim(-0.4, K + 0.4)
            ax.legend(loc="upper right", fontsize=10.5)
            ax.set_ylim(0, 1.15 * max(v for (r_, kk), v in key.items() if np.isfinite(v)))
            ax.set_title(lab, loc="left", fontsize=12)
            if ri == 1:
                ax.set_xlabel("iteration (6 real trials each)")
        res = (f"π{S.sub(k)}: field deviation near the goal offset {dev_near[(ROBOTS[0], k)]:.2f} / mass {dev_near[(ROBOTS[1], k)]:.2f} tol;  "
               f"fixed point offset {fpe[(ROBOTS[0], k)]:.2f} / mass {fpe[(ROBOTS[1], k)]:.2f} tol from the goal")
        S.header(fig, "Pogo: the apex-to-apex flow after each stage iteration, re-converging toward the sim's (ours, servo leg)",
                 k, K, "pause", res)
        f = f"{tmp}/key{k}.png"
        fig.savefig(f, dpi=100)
        plt.close(fig)
        keyframes.append(f)
    # hold + crossfade (ffmpeg blend of consecutive keyframes)
    from PIL import Image
    n = 0
    for k, f in enumerate(keyframes):
        im = Image.open(f).convert("RGB")
        for _ in range(int((FINAL_S if k == K else HOLD_S) * S.FPS)):
            im.save(f"{tmp}/f{n:05d}.png")
            n += 1
        if k < K:
            nxt = Image.open(keyframes[k + 1]).convert("RGB")
            m = int(XFADE_S * S.FPS)
            for i in range(1, m + 1):
                Image.blend(im, nxt, i / (m + 1)).save(f"{tmp}/f{n:05d}.png")
                n += 1
    out = f"{HERE}/pogo/flow_iters_pogo"
    S.encode(tmp, out + ".mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(S.FPS), "-i", f"{tmp}/f%05d.png", "-vf",
                    "fps=10,scale=1280:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse", out + ".gif"], check=True)
    shutil.copy(keyframes[0], out + "_pi0.png")
    shutil.copy(keyframes[-1], out + "_final.png")
    shutil.rmtree(tmp)
    print(out + ".mp4", n, "frames")


if __name__ == "__main__":
    main()
