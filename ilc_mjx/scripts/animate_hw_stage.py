#!/usr/bin/env python3
"""
animate_hw_stage.py: side-view animations (matplotlib, GIF) of recorded Go1 jumps (export_trajectories.py).

  hwstage  TRAJ.npz --robot R            the hardware stage batch by batch: one panel per training goal, one segment
                                         per iteration (the batch the policy flew after k updates); the earlier
                                         batches' base paths stay, faded -- the trials converging
  compare  A.npz B.npz ... --robot R     sim-to-sim transfer side by side: one panel per file (method), one segment
                                         per test goal
Options: --out FILE (default figures/ilc2real/anim/<mode>_<traj>_<robot>.gif), --fps 17, --stride 30 (2 ms ticks per frame:
real time at 17 fps), --dpi 100, --colors 48 (the GIF palette: few colours keep it small enough to commit),
--format gif|mp4 (mp4 needs ffmpeg). All of them at once:
make_animations.py. Poses come from the recorded ground truth through the menagerie
Go1's kinematics (no rendering backend needed).
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import traj_lib as tl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=("hwstage", "compare"))
ap.add_argument("traj", nargs="+")
ap.add_argument("--robot", required=True)
ap.add_argument("--labels", nargs="*", default=None, help="compare: a label per file")
ap.add_argument("--out")
ap.add_argument("--fps", type=int, default=17)
ap.add_argument("--stride", type=int, default=30, help="ticks (2 ms) per frame: 30 at 17 fps ~ real time")
ap.add_argument("--dpi", type=int, default=100)
ap.add_argument("--colors", type=int, default=48, help="GIF palette size")
ap.add_argument("--format", default="gif", choices=("gif", "mp4"), help="mp4 needs ffmpeg")
a = ap.parse_args()

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
SEQ = ["#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]       # iterations: light -> dark (one hue)
BOXC = "#d9a066"
LEGC = {"FR": "#2a78d6", "RR": "#2a78d6", "FL": "#1c5cab", "RL": "#1c5cab"}   # near legs lighter, far legs darker
Ts = [tl.load(p) for p in a.traj]
models = {}


def model(meta):
    b = tl.box_of(meta)
    key = (round(b[0], 4) if b[0] is not None else None, round(b[1], 4))
    if key not in models:
        models[key] = tl.go1_model(b, scene=False)
    return models[key]


# segments: a list of (title, [(panel title, T, i, [(T, i) earlier jumps to fade])]) ----------------------------
segments = []
if a.mode == "hwstage":
    T = Ts[0]
    idx = tl.select(T, robot=a.robot)
    goals = sorted({tuple(round(g, 4) for g in T["meta"][i]["goal"]) for i in idx}, key=lambda g: (g[1], g[0]))
    its = sorted({T["meta"][i]["iteration"] for i in idx})
    for it in its:
        panels = []
        for g in goals:
            cur = [i for i in tl.select(T, robot=a.robot, iteration=it, goal=g)]
            if not cur:
                continue
            prev = [(T, j) for k in its if k < it for j in tl.select(T, robot=a.robot, iteration=k, goal=g)]
            panels.append((f"goal x {g[0]:.3f} m" + (f", box {g[1]:.3f} m" if g[1] > 0.004 else " (flat)"), T, cur[0], prev))
        cond = T["meta"][idx[0]].get("train_cond", "nominal")
        segments.append((f"{a.robot}, hardware stage ({T['name']}, training under: {cond}) -- batch {it + 1} of {len(its)}: "
                         f"the policy after {it} update{'s' if it != 1 else ''}", panels))
else:
    labels = a.labels or [T["name"].replace("eval_", "") for T in Ts]
    goals = sorted({tuple(round(g, 4) for g in m["goal"]) for T in Ts for m in T["meta"] if m["robot"] == a.robot},
                   key=lambda g: (g[1], g[0]))
    for g in goals:
        panels = []
        for T, lab in zip(Ts, labels):
            cur = tl.select(T, robot=a.robot, goal=g)
            if cur:
                panels.append((lab, T, cur[0], []))
        segments.append((f"{a.robot}, sim-to-sim transfer, test goal x {g[0]:.3f} m" + (f", box {g[1]:.3f} m" if g[1] > 0.004 else " (flat)"), panels))

npan = max(len(p) for _, p in segments)
ncol = min(npan, 3)
nrow = int(np.ceil(npan / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 3.0 * nrow + 0.7), squeeze=False, facecolor=SURF)
axs = axs.ravel()
sup = fig.suptitle("", x=0.01, ha="left", fontsize=13, color=INK)
frames = []                                       # (segment, tick)
for s, (_, panels) in enumerate(segments):
    n = max(p[1]["meta"][p[2]]["n_ticks"] for p in panels)
    frames += [(s, k) for k in range(0, n, a.stride)] + [(s, n - 1)] * int(0.6 * a.fps)      # hold the landing


def base_path(T, i, k=None):
    n = T["meta"][i]["n_ticks"] if k is None else k + 1
    return T["pos"][i, :n, 0], T["pos"][i, :n, 2]


def draw(f):
    s, k = frames[f]
    title, panels = segments[s]
    sup.set_text(title)
    for ax in axs:
        ax.clear()
        ax.set_visible(False)
    for ax, (ptitle, T, i, prev) in zip(axs, panels):
        ax.set_visible(True)
        mt = T["meta"][i]
        m, d = model(mt)
        ax.set_facecolor(SURF)
        ax.axhline(0, color=INK2, lw=1)
        xf, h = tl.box_of(mt)
        if xf is not None and h >= 0.005:
            ax.add_patch(plt.Rectangle((xf, 0), 1.0, h, color=BOXC, zorder=1))
        x0 = float(T["pos"][i, 0, 0])
        ax.axvline(x0 + mt["goal"][0], color=INK2, lw=0.8, ls=":")
        for j, (Tp, ip) in enumerate(prev):                      # earlier batches, faded
            xs, zs = base_path(Tp, ip)
            ax.plot(xs, zs, color=SEQ[min(j, len(SEQ) - 1)], lw=1, alpha=0.5)
        xs, zs = base_path(T, i, min(k, mt["n_ticks"] - 1))
        ax.plot(xs, zs, color="#eb6834", lw=1.6)
        p = tl.side_points(m, d, T, i, k)
        for l in ("FL", "RL", "FR", "RR"):                        # far legs first
            ax.plot(*p["legs"][l].T, color=LEGC[l], lw=2.2, solid_capstyle="round", zorder=3)
        ax.plot([p["rear"][0], p["front"][0]], [p["rear"][1], p["front"][1]], color=INK, lw=5, solid_capstyle="round", zorder=4)
        done = k >= mt["n_ticks"] - 1 or k >= len(T["t"]) - a.stride
        res = (f"landing x {100 * mt['ex']:+.1f} cm, z {100 * mt['ez']:+.1f} cm" + (" -- FELL" if mt["fell"] else "")) if done else \
              f"t = {T['t'][min(k, len(T['t']) - 1)]:.2f} s"
        ax.set_title(f"{ptitle}\n{res}", fontsize=11, loc="left", color=INK)
        ax.set_xlim(x0 - 0.42, x0 + 1.35)
        ax.set_ylim(-0.02, 0.75)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=9, colors=INK2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    return []


out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "ilc2real", "anim",
                            f"{a.mode}_{os.path.splitext(os.path.basename(a.traj[0]))[0]}_{a.robot}.gif")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.text(0.01, 0.005, ("orange: this jump's trunk path; blue: the same goal's earlier batches (light = earliest); dotted: the goal; "
                      "black bar: trunk (hips); blue: legs (far side darker)") if a.mode == "hwstage" else
         "orange: the trunk path; dotted: the goal; black bar: trunk (hips); blue: legs (far side darker)", fontsize=10, color=INK2)
fig.tight_layout(rect=(0, 0.035, 1, 0.93))
if a.format == "mp4":
    from matplotlib.animation import FFMpegWriter
    out = os.path.splitext(out)[0] + ".mp4"
    FuncAnimation(fig, draw, frames=len(frames), blit=False).save(out, writer=FFMpegWriter(fps=a.fps, bitrate=2400), dpi=a.dpi)
else:
    # one palette for the whole GIF (from a sample of frames), no dithering: flat colours compress well and the
    # unchanged pixels between frames are written once (Pillow stores only each frame's changed box)
    from PIL import Image
    fig.set_dpi(a.dpi)
    rgb = []
    for f in range(len(frames)):
        draw(f)
        fig.canvas.draw()
        rgb.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
    sample = rgb[::max(1, len(rgb) // 12)]
    strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for j, im in enumerate(sample):
        strip.paste(im, (0, j * im.height))
    pal = strip.quantize(colors=a.colors, method=Image.Quantize.MEDIANCUT)
    ims = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in rgb]
    ims[0].save(out, save_all=True, append_images=ims[1:], duration=int(round(1000 / a.fps)), loop=0, optimize=False)
print(f"{len(frames)} frames -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
