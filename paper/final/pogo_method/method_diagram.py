"""Methodology schematic: the pogo flies a batch of trials at V goals, each trial's error drives one ILC step, the
corrected actions refit the policy, the new policy flies the goals again, ..., until every goal is inside tolerance.

The target drawn is only what the error rows score (common.errors): each flight's apex height against h* (band = the
2 cm tolerance) and its forward speed against v* (thin bar under the ground). No reference trajectory or hop period:
the task leaves the period free, so none is drawn.

Full text width (7.16 in), Times: Times New Roman text with STIX math (the LaTeX Times look; no usetex, the
matplotlib install lacks cm-super), DejaVu Sans only as the fallback for the check / cross glyphs.
Schematic only (hand-set hop errors, no run data). Writes method_pogo_ilc.{pdf,png} to paper/final/ (the generator
lives in its own folder).
    ~/miniconda3/envs/mj_quad/bin/python paper/final/pogo_method/method_diagram.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # paper/final

BLUE = "#2a6bcc"      # flown hop (as in the stage-flow videos)
INK = "#222222"
MUTED = "#777777"
BAND = "#e4e4e4"      # tolerance band (apex) / zone (speed)
TRACK = "#f0f0f0"     # speed bar track
ORANGE = "#e0663a"    # miss
GREEN = "#2e8b57"     # on target
BOX = "#f3f5f9"

# register Times New Roman directly (a matplotlib font cache built before the fonts were installed misses them)
for f in glob.glob("/usr/share/fonts/truetype/msttcorefonts/[Tt]imes*.ttf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Times New Roman", "DejaVu Sans"], "mathtext.fontset": "stix",
                     "font.size": 7, "pdf.fonttype": 42})
CHECK, CROSS = "\u2713", "\u2717"
FS, FS_SUB, FS_HEAD = 7.0, 6.4, 7.5   # body, formulas / legend, column headers

# ---- layout (data units; the axes is equal-aspect) ----
BW, PW, GAP = 15.6, 14.0, 1.4          # batch column, policy block, gap
PH, RGAP, YB = 6.2, 0.7, 3.0           # goal-row height, gap between rows, bottom of the lowest row
ROWS = [(YB + i * (PH + RGAP), YB + i * (PH + RGAP) + PH) for i in (2, 1, 0)]   # goal rows (bottom, top), g1 on top
x0 = 4.0
B0 = x0
P0 = B0 + BW + GAP
B1 = P0 + PW + GAP
P1 = B1 + BW + GAP
B2 = P1 + PW + GAP
DOTS = B2 + BW
BK = DOTS + 5.0
XMAX = BK + BW + 1.4
YLO, YHI = 0.3, ROWS[0][1] + 2.3

# ---- goals in panel units: apex height h* above ground, and the flight length a hop at v* covers ----
GROUND, YTD, DIP, LS = 1.15, 2.1, 0.6, 0.7
GOALS = [dict(h=4.1, lf=2.75, name="$g_1$"),
         dict(h=3.65, lf=3.3, name="$g_2$"),
         dict(h=3.2, lf=3.85, name="$g_3$")]
# per goal, the apex (panel units) and speed (fraction) error of pi_0; batch k scales it by its factor
E0 = [(-0.75, -0.20), (0.6, -0.24), (-0.5, 0.22)]
ITERS = [1.0, 0.42, 0.17, 0.02]
TOL_H, TOL_V = 0.22, 0.07              # the 2 cm / 0.1 m/s tolerance in these units
SBAR_Y, SBAR_H, SBAR_VSTAR = 0.42, 0.24, 0.72   # speed bar: height above panel bottom, thickness, v* as a fraction


def hop_path(h, lf, nhops=2):
    """COM path from a start apex: half flight down, then (stance, full flight) x nhops, then a stance half."""
    xs, ys = [], []

    def flight(xa, xb, ya, peak, half=False):
        t = np.linspace(0, 1, 40)
        y = ya - (ya - YTD) * t ** 2 if half else ya + 4 * (peak - ya) * t * (1 - t)
        return xa + (xb - xa) * t, y

    def stance(xa, frac=1.0):
        t = np.linspace(0, frac, 15)
        return xa + LS * t, YTD - DIP * np.sin(np.pi * t)

    seg = flight(0.0, lf / 2, h, h, half=True)
    xs.append(seg[0]); ys.append(seg[1])
    x = lf / 2
    for _ in range(nhops):
        sx, sy = stance(x); xs.append(sx); ys.append(sy); x += LS
        fx, fy = flight(x, x + lf, YTD, h); xs.append(fx); ys.append(fy); x += lf
    sx, sy = stance(x, 0.5)            # end at mid-stance, where the stick is drawn
    xs.append(sx); ys.append(sy)
    return np.concatenate(xs), np.concatenate(ys)


def zigzag(ax, xa, ya, xb, yb, n=5, w=0.24, **kw):
    t = np.linspace(0, 1, 2 * n + 3)
    px = xa + (xb - xa) * t
    py = ya + (yb - ya) * t
    nx, ny = -(yb - ya), (xb - xa)
    nn = np.hypot(nx, ny)
    off = np.zeros_like(t)
    off[2:-2:2], off[3:-2:2] = w, -w
    ax.plot(px + off * nx / nn, py + off * ny / nn, **kw)


def arrow(ax, xa, ya, xb, yb, color=INK, lw=0.8, ms=6, z=7):
    ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=z))


def speed_bar(ax, x, y, w, ev, label=True):
    """Thin bar under the ground: flown flight speed v (blue fill) against v* (tick, gray zone = tolerance); the gap
    to v* shaded when outside tolerance."""
    xs = x + w * SBAR_VSTAR
    xv = x + w * SBAR_VSTAR * (1 + ev)
    tol = w * SBAR_VSTAR * TOL_V
    ax.add_patch(Rectangle((x, y), w, SBAR_H, fc=TRACK, ec="none", zorder=2))
    ax.add_patch(Rectangle((xs - tol, y), 2 * tol, SBAR_H, fc=BAND, ec="none", zorder=2))
    miss = abs(ev) > TOL_V
    ax.add_patch(Rectangle((x, y), (min(xs, xv) if miss else xv) - x, SBAR_H, fc=BLUE, ec="none", zorder=3))
    if miss:                           # too slow: the missing speed; too fast: the excess
        ax.add_patch(Rectangle((min(xs, xv), y), abs(xv - xs), SBAR_H, fc=ORANGE, ec="none", alpha=0.6, zorder=3))
    ax.plot([xs, xs], [y - 0.18, y + SBAR_H + 0.18], color=INK, lw=0.7, zorder=4)
    if label:
        ax.text(x - 0.35, y + SBAR_H / 2, "$v$", ha="right", va="center", fontsize=FS_SUB, color=MUTED)


def batch_panel(ax, bx, row, g, eh, ev):
    yb, yt = row
    gy = yb + GROUND
    ax.add_patch(Rectangle((bx, yb), BW, yt - yb, fc="white", ec="#c9ced8", lw=0.6, zorder=0))
    ax.plot([bx + 0.4, bx + BW - 0.4], [gy] * 2, color=INK, lw=0.7, zorder=1)
    # target apex h* with its tolerance band
    ax.add_patch(Rectangle((bx + 0.4, gy + g["h"] - TOL_H), BW - 2.4, 2 * TOL_H, fc=BAND, ec="none", zorder=1))
    ax.plot([bx + 0.4, bx + BW - 2.0], [gy + g["h"]] * 2, ls=(0, (3, 2)), color=INK, lw=0.6, zorder=2)
    # flown hops (the drop starts at h*; the policy's hops then settle at its own apex and speed)
    ox = bx + 0.9
    lf = g["lf"] * (1 + ev)
    fx, fy = hop_path(g["h"] + eh, lf)
    fy = fy.copy()
    fy[:40] = (g["h"] - (g["h"] - YTD) * np.linspace(0, 1, 40) ** 2)   # start-up half flight from h*
    ax.plot(ox + fx, gy + fy, color=BLUE, lw=1.1, zorder=3)
    speed_bar(ax, bx + 1.5, yb + SBAR_Y, BW - 3.6, ev)
    # the stick at mid-stance: mass on a compressed spring leg
    mx, my = ox + fx[-1], gy + fy[-1]
    zigzag(ax, mx, my, mx, gy, color=INK, lw=0.6, zorder=4)
    ax.add_patch(Circle((mx, my), 0.36, fc=BLUE, ec=INK, lw=0.6, zorder=5))
    ok = abs(eh) <= TOL_H and abs(ev) <= TOL_V
    ax.text(bx + BW - 0.55, gy + g["h"], CHECK if ok else CROSS, ha="right", va="center",
            color=GREEN if ok else ORANGE, fontsize=FS, zorder=6)


def nn_glyph(ax, cx, cy, w=4.2, h=3.0):
    layers = [3, 4, 4, 2]
    pts = []
    for i, n in enumerate(layers):
        x = cx - w / 2 + w * i / (len(layers) - 1)
        ys = cy + (np.arange(n) - (n - 1) / 2) * (h / 3.6)
        pts.append([(x, y) for y in ys])
    for a, b in zip(pts[:-1], pts[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="#9aa6bb", lw=0.3, zorder=8)
    for layer in pts:
        for (x, y) in layer:
            ax.add_patch(Circle((x, y), 0.27, fc="white", ec=INK, lw=0.5, zorder=9))


def policy_block(ax, px, k):
    yb, yt = ROWS[-1][0], ROWS[0][1]
    ax.add_patch(FancyBboxPatch((px, yb), PW, yt - yb, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=BOX, ec="#8f9bb3", lw=0.7, zorder=1))
    cx = px + PW / 2
    ax.text(cx, yt - 1.0, "policy update", ha="center", va="top", fontsize=FS, fontweight="bold", color=INK,
            zorder=9)
    # numbered steps, left-aligned; the numbers are drawn (the math font has no circled digits)
    steps = [(yt - 3.5, "ILC step per trial", r"$\tilde u_v = u_v + \Delta u_v$"),
             (yt - 7.0, "replay buffer", r"$(g_v, \tilde u_v) \rightarrow \mathcal{D}$"),
             (yt - 10.5, r"refit $\pi_\theta$ on $\mathcal{D}$", "")]
    xn, xt = px + 1.7, px + 2.8
    for i, (y, head, sub) in enumerate(steps):
        ax.add_patch(Circle((xn, y), 0.62, fc="white", ec=INK, lw=0.5, zorder=9))
        ax.text(xn, y - 0.05, str(i + 1), ha="center", va="center", fontsize=5.5, color=INK, zorder=10)
        ax.text(xt, y, head, ha="left", va="center", fontsize=FS, color=INK, zorder=9)
        if sub:
            ax.text(xt, y - 1.5, sub, ha="left", va="center", fontsize=FS_SUB, color=MUTED, zorder=9)
    nn_glyph(ax, cx, yt - 14.0)
    ax.text(cx, yb + 1.9, rf"$\pi_\theta \rightarrow \pi_{k}$", ha="center", va="center", fontsize=FS_HEAD,
            color=INK, zorder=9)


def main():
    fig = plt.figure(figsize=(7.16, 7.16 * (YHI - YLO) / XMAX))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, XMAX)
    ax.set_ylim(YLO, YHI)
    ax.set_aspect("equal")
    ax.axis("off")

    cols = [B0, B1, B2, BK]
    heads = [r"batch 0: $\pi_0$ (sim-trained)", r"batch 1: $\pi_1$", r"batch 2: $\pi_2$",
             r"batch $K$: $\pi_K$ (converged)"]
    for c, rho, head in zip(cols, ITERS, heads):
        ax.text(c + BW / 2, ROWS[0][1] + 0.7, head, ha="center", va="bottom", fontsize=FS_HEAD, color=INK)
        for row, g, (eh, ev) in zip(ROWS, GOALS, E0):
            batch_panel(ax, c, row, g, rho * eh, rho * ev)

    for row, g in zip(ROWS, GOALS):
        ax.text(x0 - 1.0, (row[0] + row[1]) / 2, "goal " + g["name"], rotation=90, ha="center", va="center",
                fontsize=FS_HEAD, color=INK)

    # batch -> update (each goal's trial and error), update -> next batch
    for bx, px, k in [(B0, P0, 1), (B1, P1, 2)]:
        policy_block(ax, px, k)
        for row in ROWS:
            y = (row[0] + row[1]) / 2
            arrow(ax, bx + BW, y, px, y)
            arrow(ax, px + PW, y, px + PW + GAP, y)
    for row in ROWS:
        y = (row[0] + row[1]) / 2
        ax.text((DOTS + BK) / 2, y, r"$\cdots$", ha="center", va="center", fontsize=10, color=MUTED)

    # legend
    ly = 1.3
    lx = (XMAX - 58.5) / 2             # centered: the entries span ~58.5 units
    ax.plot([lx, lx + 2.4], [ly, ly], color=BLUE, lw=1.1)
    ax.text(lx + 3.0, ly, "flown hop", va="center", fontsize=FS_SUB)
    lx += 10.0
    ax.add_patch(Rectangle((lx, ly - TOL_H), 2.4, 2 * TOL_H, fc=BAND, ec="none"))
    ax.plot([lx, lx + 2.4], [ly, ly], color=INK, lw=0.6, ls=(0, (3, 2)))
    ax.text(lx + 3.0, ly, r"apex target $h^*\pm2$ cm", va="center", fontsize=FS_SUB)
    lx += 16.5
    speed_bar(ax, lx, ly - SBAR_H / 2, 3.2, -0.2, label=False)
    ax.text(lx + 3.8, ly, r"flight speed $v$ vs. target $v^*$", va="center", fontsize=FS_SUB)
    lx += 19.5
    ax.text(lx, ly, CROSS, color=ORANGE, va="center", fontsize=FS)
    ax.text(lx + 1.4, ly, "miss", va="center", fontsize=FS_SUB)
    ax.text(lx + 5.2, ly, CHECK, color=GREEN, va="center", fontsize=FS)
    ax.text(lx + 6.6, ly, "on target", va="center", fontsize=FS_SUB)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"method_pogo_ilc.{ext}"), dpi=300)
    print("wrote", os.path.join(OUT, "method_pogo_ilc.{pdf,png}"), f"({7.16:.2f} x {7.16 * (YHI - YLO) / XMAX:.2f} in)")


if __name__ == "__main__":
    main()
