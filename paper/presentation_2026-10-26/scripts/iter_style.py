"""
iter_style.py (2026-10-26): one look for the three "iteration by iteration" stage videos (pogo, car, quad).
Every iteration k has its own colour on one sequential blue ramp (light = early, dark = late), so the eye reads the
order without a legend; the iteration being flown is drawn on top in ink-dark blue, earlier ones stay as faded ghosts.
Each iteration ends in a pause: its result frozen with a banner ("π_k flown -> update -> π_k+1").
"""
import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"
OK, BAD = "#2e8540", "#d03b3b"
TOL = "#e3efe0"                                   # the tolerance band (status good, light)
RAMP = LinearSegmentedColormap.from_list("iters", ["#9cc0ec", "#6fa2e4", "#4f8fdb", "#256abf", "#163f73"])
FPS = 30
PAUSE_S = 1.6                                     # the hold after each iteration
FINAL_S = 3.0


def rc():
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.grid": False, "legend.frameon": False,
        "legend.fontsize": 10, "axes.spines.top": False, "axes.spines.right": False})


def sub(k):
    return "".join("₀₁₂₃₄₅₆₇₈₉"[int(c)] for c in str(k))


def color(k, K):
    """iteration k of 0..K on the ramp"""
    return RAMP(0.08 + 0.92 * k / max(K, 1))


def header(fig, title, k, K, phase, result=None):
    """the top band: the experiment, a row of iteration pills (done / now / to come), and what is happening"""
    fig.text(0.012, 0.965, title, fontsize=15, weight="bold", va="top")
    x0 = 0.012
    for j in range(K + 1):
        done, now = j < k or (j == k and phase == "pause"), j == k
        fc = color(j, K) if (done or now) else SURFACE
        ec = INK if now else (color(j, K) if done else GRID)
        fig.text(x0, 0.905, f" π{sub(j)} ", fontsize=13, va="center", color=("white" if (done or now) and j >= K * 0.45 else INK),
                 bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec=ec, lw=2.0 if now else 1.0))
        if j < K:
            fig.text(x0 + 0.043, 0.905, "→", fontsize=13, va="center", color=INK2)
        x0 += 0.062
    msg = {"fly": f"flying π{sub(k)}  (one trial per training goal)",
           "pause": (f"π{sub(k)} flown  →  update the network on its errors  →  π{sub(k + 1)}" if k < K
                     else f"π{sub(k)}: the stage's last network")}[phase]
    fig.text(x0 + 0.02, 0.905, msg, fontsize=13, va="center", color=INK if phase == "fly" else "#163f73",
             weight="bold" if phase == "pause" else "normal")
    if result:
        fig.text(0.012, 0.845, result, fontsize=13, va="center", ha="left", color=INK,
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID))


def frame_plan(n_fly, K):
    """(k, t-index or None for the pause, phase) for every frame: each iteration's flight, then its pause"""
    out = []
    for k in range(K + 1):
        for t in range(n_fly[k]):
            out.append((k, t, "fly"))
        for _ in range(int((FINAL_S if k == K else PAUSE_S) * FPS)):
            out.append((k, None, "pause"))
    return out


def encode(frames_dir, mp4):
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{frames_dir}/f%05d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", mp4],
                   check=True)


def render(frame_fn, plan, out_mp4, procs=16, stills=()):
    """frame_fn((i, k, t, phase, path)) draws one frame; identical pause frames are drawn once and copied."""
    import os
    import shutil
    from multiprocessing import Pool
    tmp = out_mp4[:-4] + "_frames"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    jobs, copies, first = [], [], {}
    for i, (k, t, ph) in enumerate(plan):
        p = f"{tmp}/f{i:05d}.png"
        if ph == "pause":
            if k in first:
                copies.append((first[k], p))
                continue
            first[k] = p
        jobs.append((i, k, t, ph, p))
    with Pool(procs) as pool:
        pool.map(frame_fn, jobs, chunksize=4)
    for a, b in copies:
        shutil.copy(a, b)
    encode(tmp, out_mp4)
    for k, name in stills:
        if k in first:
            shutil.copy(first[k], name)
    shutil.rmtree(tmp)
    return len(plan)
