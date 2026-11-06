#!/usr/bin/env python3
"""make_ethz_video.py [--out FILE] [--fps 20] (2026-11-05; user: "make the videos, and take videos of them throughout each part of the
stage, with a point tracking the flow under them"): the rubber car on the practised ETHZMobil track (x4.3, sim only), the median PPO+DR
stage run (ETHZ_rubber_freePPON_s0_ss9001, car ax:mu=1.667), one noise-free lap from the start line under pi_0 (zero-shot), pi_1, pi_2, pi_3
in sequence. Top: the car on the track (planned line, 0.5 m tube, the car's trail). Bottom: that iteration's flow strip (flow_car.py
ethz_resid: the stage's steering change vs pi_0, arrows = where the car is 1.0 s later, black = the line the flow pulls the car onto) with a
dot at the car's current (arc length, lateral offset). Real time (DT 0.025 s; every 2nd step at 20 fps). MP4 via imageio-ffmpeg's ffmpeg.
"""
import argparse
import math
import os
import sys

D = "/home/henry/ilc_ws/src/ilc_f1tenth/ilc2real"
os.environ.update(F1T_MU="0.6", F1T_PLANS=D + "/plans_nom06", F1T_TRACKS=D + "/../lifted_linear_tire_20261009/tracks_tight",
                  F1T_SAFETY_EY="0.5", F1T_DEV_TIME="1", NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1", JAX_PLATFORMS="cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "videos", "car_ethz_rubber_stage.mp4"))
ap.add_argument("--fps", type=int, default=20)
ap.add_argument("--variant", default="ethz_resid", help="flow_car.py variant: ethz_resid (lateral, policy residual) or speed (plastic peanut)")
ap.add_argument("--title", default="Rubber car on ETHZMobil (x4.3, practised, sim)")
args = ap.parse_args()
sys.argv = [sys.argv[0], args.variant]
sys.path.insert(0, HERE)
import flow_car as fc  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, D)
os.chdir(D)
import bank  # noqa: E402
import real_car as rc  # noqa: E402
import sim_jax_np as sn  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import imageio_ffmpeg  # noqa: E402
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

V = fc.V
pl = bank.load_plan(V["plan"])
P = sn.PlanNP(pl)
L = float(pl["length"])
pols = [fc.load_policy(f"{fc.RUN}/its/it{k}_{fc.CAR}_policy.npz") for k in range(4)]
laps = [rc.run(fc.CAR, pl, pols[k], laps=1.0, s0=0.0, pert=np.zeros(5), seed=0, noise=False) for k in range(4)]
LAT = V["var"] == "lat"
COL = 0 if LAT else 5                                                # err_true column: lateral offset, or speed error
F = np.load(os.path.join(HERE, f"{V.get('data', V['stem'])}_data.npz"))
ss, es, dS, dE = F["ss"], F["es"], F["dS"], F["dE"]
Rz = np.load(os.path.join(HERE, f"{V['resid']}.npz"))["U"] if "resid" in V else None
SS, EE = np.meshgrid(ss, es, indexing="ij")
xy = np.asarray(pl["x"])[:, :3]

def world(s, e):
    i = (np.round(np.mod(s, L) / P.ds).astype(int)) % P.M
    return xy[i, 0] - np.sin(xy[i, 2]) * e, xy[i, 1] + np.cos(xy[i, 2]) * e

def settle_line(dEk):
    out = np.full(len(ss), np.nan)
    for i in range(len(ss)):
        y = dEk[i]
        for j in range(len(es) - 1):
            if y[j] > 0 >= y[j + 1]:
                out[i] = es[j] + (es[j + 1] - es[j]) * y[j] / (y[j] - y[j + 1])
                break
        else:
            out[i] = es[-1] if y[-1] > 0 else (es[0] if y[0] < 0 else np.nan)
    return out

INK, MUT, RUN = "#0b0b0b", "#8a8984", "#eb6834"
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#5598e7", "#f0efec", "#ec8a89", "#c23a39"])
plt.rcParams.update({"font.size": 9, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
fig = plt.figure(figsize=(10, 7.2))
axm = fig.add_axes([0.05, 0.36, 0.90, 0.57])
axs = fig.add_axes([0.07, 0.07, 0.86, 0.22])
sl = np.linspace(0, L, 1500)
for e, st, c in ((0.0, "-", INK), (fc.TUBE, "--", MUT), (-fc.TUBE, "--", MUT)):
    X, Y = world(sl, np.full_like(sl, e)); axm.plot(X, Y, st, color=c, lw=0.9 if e == 0 else 0.7)
axm.set_aspect("equal"); axm.axis("off")
trail, = axm.plot([], [], color=RUN, lw=1.6, alpha=0.8)
car = Polygon(np.zeros((4, 2)), closed=True, color=INK, zorder=5)
axm.add_patch(car)
title = fig.text(0.05, 0.955, "", fontsize=11, ha="left")
info = fig.text(0.05, 0.925, "", fontsize=9, ha="left", color="#555555")
dot, = axs.plot([], [], "o", color=RUN, mec="white", mew=0.8, ms=8, zorder=8)
strip_art = []

def draw_strip(k):
    for a in strip_art:
        a.remove()
    strip_art.clear()
    if Rz is not None:
        strip_art.append(axs.pcolormesh(SS, EE, np.degrees(Rz[k, :, :, 0] - Rz[0, :, :, 0]), cmap=DIV,
                                        norm=TwoSlopeNorm(vcenter=0.0, vmin=-6.0, vmax=6.0), shading="gouraud"))
    sub = (slice(None, None, 2), slice(None, None, 2))
    k_len = 0.85 * (ss[1] - ss[0]) * 2 / np.nanquantile(np.hypot(dS[0], dE[0]), 0.9)
    strip_art.append(axs.quiver(SS[sub], EE[sub], k_len * dS[k][sub], k_len * dE[k][sub], angles="xy", scale_units="xy", scale=1.0,
                                color=INK, alpha=0.7, width=0.0018, headwidth=3.6, headlength=4.0, zorder=4))
    strip_art.extend(axs.plot(ss, settle_line(dE[k]), color=INK, lw=1.1, zorder=5))
    if LAT:
        axs.set_title("flow under this policy: shading = steering change vs zero-shot (red left, blue right); black = where the car is pulled",
                      fontsize=8, loc="left")
    else:
        axs.set_title("speed flow under this policy: arrows = where the speed error goes in 0.4 s; black = the speed the car is pulled to",
                      fontsize=8, loc="left")

axs.set_xlim(0, L)
if LAT:
    axs.set_ylim(-1.2 * fc.TUBE, 1.2 * fc.TUBE)
    for y in (-fc.TUBE, fc.TUBE):
        axs.axhline(y, color=MUT, lw=0.6, ls="--")
else:
    axs.set_ylim(es[0] - 0.08, es[-1] + 0.08)
axs.axhline(0, color=INK, lw=0.5, alpha=0.5)
axs.set_xlabel("arc length along the lap (m)", fontsize=8)
axs.set_ylabel("offset (m)" if LAT else "speed - plan's (m/s)", fontsize=8)
lpi = 1 if "ethz" in V["plan"] else 2                               # practice laps per iteration (ETHZMobil 1, mocap tracks 2)
names = ["zero-shot"] + [f"after {lpi * k} practice lap{'s' if lpi * k > 1 else ''}" for k in (1, 2, 3)]
frames = [(k, t) for k in range(4) for t in range(0, len(laps[k]["x_true"]), 2)]
cur = {"k": -1}

def update(fr):
    k, t = fr
    r = laps[k]
    if k != cur["k"]:
        cur["k"] = k
        draw_strip(k)
    x = np.asarray(r["x_true"])
    px, py, th = x[t, 0], x[t, 1], x[t, 2]
    c, s_ = math.cos(th), math.sin(th)
    body = np.array([[0.27, 0.14], [0.27, -0.14], [-0.27, -0.14], [-0.27, 0.14]])
    car.set_xy(np.c_[px + body[:, 0] * c - body[:, 1] * s_, py + body[:, 0] * s_ + body[:, 1] * c])
    trail.set_data(x[:t + 1, 0], x[:t + 1, 1])
    idx = np.asarray(r["idx"]); err = np.asarray(r["err_true"]); e = err[:, 0]
    sv = (idx[t] * P.ds) % L
    yv = err[t, COL]
    dot.set_data([sv], [np.clip(yv, -1.15 * fc.TUBE, 1.15 * fc.TUBE) if LAT else yv])
    rms = 100 * np.sqrt(np.mean(e[:t + 1] ** 2))
    out = float(np.sum(np.abs(e[:t + 1]) > fc.TUBE) * rc.DT)
    title.set_text(f"{args.title} -- policy {names[k]}")
    if LAT:
        info.set_text(f"t {t * rc.DT:5.1f} s   offset {100 * e[t]:+5.1f} cm   lap rms so far {rms:4.1f} cm   time off the track {out:.2f} s" if fc.TUBE != 0.5 else f"t {t * rc.DT:5.1f} s   offset {100 * e[t]:+5.1f} cm   lap rms so far {rms:4.1f} cm   time outside the tube {out:.2f} s")
    else:
        info.set_text(f"t {t * rc.DT:5.1f} s   speed error {yv:+5.2f} m/s   mean so far {np.mean(err[:t + 1, 5]):+5.2f} m/s   "
                      f"offset {100 * e[t]:+5.1f} cm")
    return car, trail, dot, title, info

os.makedirs(os.path.dirname(args.out), exist_ok=True)
w = FFMpegWriter(fps=args.fps, bitrate=2400, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
with w.saving(fig, args.out, dpi=110):
    for fr in frames:
        update(fr)
        w.grab_frame()
print("->", args.out, len(frames), "frames;", ", ".join(f"pi{k}: lap rms {100 * np.sqrt(np.mean(np.asarray(laps[k]['err_true'])[:, 0] ** 2)):.1f} cm, "
                                                          f"outside tube {laps[k]['dev_time']:.2f} s" for k in range(4)))
