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

D = os.environ.get("FLOW_D", "/home/henry/ilc_ws/src/ilc_f1tenth/ilc2real")   # (2026-11-11) FLOW_D: the fixed simulator
os.environ.update(F1T_MU="0.6", F1T_PLANS=D + "/plans_nom06", F1T_TRACKS=D + "/../lifted_linear_tire_20261009/tracks_tight",
                  F1T_SAFETY_EY="0.5", F1T_DEV_TIME="1", NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1", JAX_PLATFORMS="cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "videos", "car_ethz_rubber_stage.mp4"))
ap.add_argument("--fps", type=int, default=20)
ap.add_argument("--pause", type=float, default=2.0, help="(2026-11-06) seconds held at the end of each policy's laps")
ap.add_argument("--laps", type=float, default=2.0, help="(2026-11-06) full laps shown per policy (the stage trials: 2)")
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
pols = [fc.load_policy(f"{fc.RUN}/its/it{k}_{fc.CAR}_policy.npz") for k in fc.ITS]
# (2026-11-06, user: "make sure that the full 2 laps gets shown") NLAPS FULL laps per policy (the stage's 2-lap trials; --laps):
# real_car.run stops after laps x the plan's lap time, which a car slower than its plan (pace < 1, e.g. the rubber car on the peanut
# x1.5) does not finish in -- fly up to 2x that time and cut where the car's progress reaches NLAPS laps
NLAPS = args.laps
def full_lap(r):
    idx = np.asarray(r["idx"])
    d = (np.diff(idx) + P.M // 2) % P.M - P.M // 2
    prog = np.concatenate([[0.0], np.cumsum(d) * P.ds])
    n = int(np.argmax(prog >= NLAPS * L)) + 1 if np.any(prog >= NLAPS * L) else len(idx)
    if not np.any(prog >= NLAPS * L):
        print(f"WARNING: only {prog[-1] / L:.2f} of {NLAPS} laps in 2x the plan's time", flush=True)
    return {k_: (v[:n] if isinstance(v, np.ndarray) and len(v) >= len(idx) else v) for k_, v in r.items()}
laps = [full_lap(rc.run(fc.CAR, pl, pols[k], laps=NLAPS, s0=0.0, pert=np.zeros(5), seed=0, noise=False,
                        T_max=int(math.ceil(2.0 * NLAPS * pl["lap_time"] / rc.DT)))) for k in range(len(fc.ITS))]
for k in range(len(fc.ITS)):
    laps[k]["dev_time"] = float(np.sum(np.abs(np.asarray(laps[k]["err_true"])[:, 0]) > fc.TUBE) * rc.DT)
LAT = V["var"] == "lat"
COL = 0 if LAT else 5                                                # err_true column: lateral offset, or speed error
F = np.load(os.path.join(HERE, f"{V.get('data', V['stem'])}_data.npz"))
ss, es, dS, dE = F["ss"], F["es"], F["dS"], F["dE"]
_rf = os.path.join(HERE, f"{V['resid']}.npz") if "resid" in V else os.path.join(HERE, f"{V['stem']}_resid.npz")   # (2026-11-06) also <stem>_resid
Rz = np.load(_rf)["U"] if os.path.exists(_rf) else None
RCH = 1 if (os.environ.get("FLOW_RESID_CH") == "current" or not LAT) else 0   # (2026-11-11) FLOW_RESID_CH=current
CURL = LAT and RCH == 1
#                                               # the policy change shown: steering (deg) or motor current (A)
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
fig = plt.figure(figsize=(13, 7.2))
axm = fig.add_axes([0.0, 0.36, 0.47, 0.55])
axs = fig.add_axes([0.05, 0.07, 0.90, 0.22])
sl = np.linspace(0, L, 1500)
for e, st, c in ((0.0, "-", INK), (fc.TUBE, "--", MUT), (-fc.TUBE, "--", MUT)):
    X, Y = world(sl, np.full_like(sl, e)); axm.plot(X, Y, st, color=c, lw=0.9 if e == 0 else 0.7)
axm.set_aspect("equal"); axm.axis("off")
trail, = axm.plot([], [], color=RUN, lw=1.6, alpha=0.8)
prevp, = axm.plot([], [], ":", color="#3a6fb0", lw=1.4, alpha=0.9)   # (2026-11-06, user) the previous iteration's path
prevl = axm.text(0.0, 0.0, "", transform=axm.transAxes, fontsize=8, color="#3a6fb0")
car = Polygon(np.zeros((4, 2)), closed=True, color=INK, zorder=5)
axm.add_patch(car)
title = fig.text(0.05, 0.955, "", fontsize=11, ha="left")
info = fig.text(0.05, 0.925, "", fontsize=9, ha="left", color="#555555")
# (2026-11-06, user: "log the metrics for speed, lateral error, and full cost on the side after each run, and pause after each
# iteration"): a row per policy once its laps are done; J = the stage's own objective over those laps (hw_stage --objective task,
# --err-scale: lateral 0.10 m, speed 0.24 m/s): 1/2 sum_t (e_y / 0.10)^2 + (speed error / 0.24)^2
logt = fig.text(0.55, 0.88, "", fontsize=8.0, ha="left", va="top", family="monospace", color="#222222")
dot, = axs.plot([], [], "o", color=RUN, mec="white", mew=0.8, ms=8, zorder=8)
# (2026-11-06, user: "make the plastic much more visible"): a trace panel -- along the laps, the car's speed against the plan's (speed
# variants) or its lateral offset (lateral variants), for the plan, zero-shot, the previous iteration (dotted) and the current one
axt = fig.add_axes([0.53, 0.37, 0.44, 0.205])
def trace(k):
    r = laps[k]; idx = np.asarray(r["idx"]); x = np.asarray(r["x_true"]); err = np.asarray(r["err_true"])
    d = (np.diff(idx) + P.M // 2) % P.M - P.M // 2
    prog = np.concatenate([[0.0], np.cumsum(d) * P.ds])
    if LAT:
        return prog, 100 * err[:, 0], None
    zz = np.asarray(pl["z"]); vp = np.hypot(zz[idx % len(zz), 2], zz[idx % len(zz), 3])
    return prog, np.hypot(x[:, 3], x[:, 4]), vp
TR = [trace(k) for k in range(len(fc.ITS))]
allv = np.concatenate([t_[1] for t_ in TR] + ([t_[2] for t_ in TR] if not LAT else []))
axt.set_xlim(0, NLAPS * L); lo_, hi_ = np.nanpercentile(allv, 1), np.nanpercentile(allv, 99.5)
axt.set_ylim(lo_ - 0.08 * (hi_ - lo_), hi_ + 0.08 * (hi_ - lo_))
axt.set_xlabel("distance along the laps (m)", fontsize=7.5); axt.tick_params(labelsize=7)
axt.set_ylabel("lateral offset (cm)" if LAT else "speed (m/s)", fontsize=7.5)
if LAT:
    axt.axhline(0, color=INK, lw=0.8)
tr_cur, = axt.plot([], [], color=RUN, lw=1.6, zorder=5)
tr_art = []
def draw_trace(k):
    for a_ in tr_art:
        a_.remove()
    tr_art.clear()
    if not LAT:
        tr_art.extend(axt.plot(TR[k][0], TR[k][2], color=INK, lw=0.9, label="plan"))
    tr_art.extend(axt.plot(TR[0][0], TR[0][1], color="#9a9a9a", lw=1.0, label="zero-shot"))
    if k > 0:
        tr_art.extend(axt.plot(TR[k - 1][0], TR[k - 1][1], ":", color="#3a6fb0", lw=1.5, label=f"previous ({names[k - 1]})"))
    tr_art.append(axt.legend(handles=[*([] if LAT else [plt.Line2D([], [], color=INK, lw=0.9)]), plt.Line2D([], [], color="#9a9a9a"),
                                      *([plt.Line2D([], [], color="#3a6fb0", ls=":")] if k > 0 else []), plt.Line2D([], [], color=RUN)],
                             labels=[*([] if LAT else ["plan"]), "zero-shot", *([f"previous: {names[k - 1]}"] if k > 0 else []), f"now: {names[k]}"],
                             fontsize=6.5, loc="lower right", ncol=2, frameon=False))
    axt.set_title(("lateral offset from the plan along the laps" if LAT else "speed along the laps: the gap to the plan's speed is the speed error")
                  , fontsize=7.5, loc="left")
strip_art = []

def draw_strip(k):
    for a in strip_art:
        a.remove()
    strip_art.clear()
    if Rz is not None:
        dR = Rz[k, :, :, RCH] - Rz[0, :, :, RCH]
        dR = np.degrees(dR) if (LAT and not CURL) else dR
        lim = 6.0 if (LAT and not CURL) else max(1.0, float(np.nanmax(np.abs(Rz[3, :, :, 1] - Rz[0, :, :, 1]))))
        strip_art.append(axs.pcolormesh(SS, EE, dR, cmap=DIV, norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim), shading="gouraud"))
    sub = (slice(None, None, 2), slice(None, None, 2))
    k_len = 0.85 * (ss[1] - ss[0]) * 2 / np.nanquantile(np.hypot(dS[0], dE[0]), 0.9)
    strip_art.append(axs.quiver(SS[sub], EE[sub], k_len * dS[k][sub], k_len * dE[k][sub], angles="xy", scale_units="xy", scale=1.0,
                                color=INK, alpha=0.7, width=0.0018, headwidth=3.6, headlength=4.0, zorder=4))
    strip_art.extend(axs.plot(ss, settle_line(dE[k]), color=INK, lw=1.1, zorder=5))
    if CURL:
        axs.set_title("flow under this policy: shading = motor-current change vs zero-shot (red more, blue less); black = where the car is pulled",
                      fontsize=8, loc="left")
    elif LAT:
        axs.set_title("flow under this policy: shading = steering change vs zero-shot (red left, blue right); black = where the car is pulled",
                      fontsize=8, loc="left")
    else:
        axs.set_title("speed flow under this policy: shading = motor-current change vs zero-shot (red more, blue less); "
                      "arrows = where the speed error goes in 0.4 s; black = the speed the car is pulled to", fontsize=8, loc="left")

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
names = ["zero-shot"] + [f"after {lpi * k} practice lap{'s' if lpi * k > 1 else ''}" for k in fc.ITS[1:]]
def metrics(r):
    err = np.asarray(r["err_true"]); x = np.asarray(r["x_true"])
    ey, dv = err[:, 0], err[:, 5]
    return dict(ey=100 * np.sqrt(np.mean(ey ** 2)), v=float(np.mean(np.hypot(x[:, 3], x[:, 4]))), dv=float(np.mean(dv)),
                tr=len(ey) * rc.DT / (NLAPS * float(pl["lap_time"])), J=0.5 * float(np.sum((ey / 0.10) ** 2 + (dv / 0.24) ** 2)))
MET = [metrics(laps[k]) for k in range(len(fc.ITS))]
def log_text(n):
    hdr = (f"after each policy's {NLAPS:g} lap{'s' if NLAPS != 1 else ''}\n\n"
           f"{'policy':23s}{'lateral':>8s}{'speed':>7s}{'speed err':>10s}{'time':>7s}{'cost J':>8s}\n"
           f"{'':23s}{'rms cm':>8s}{'m/s':>7s}{'m/s':>10s}{'/plan':>7s}{'':>8s}\n")
    rows = "".join(f"{names[k]:23s}{MET[k]['ey']:8.1f}{MET[k]['v']:7.2f}{MET[k]['dv']:+10.2f}{MET[k]['tr']:7.2f}{MET[k]['J']:8.0f}\n"
                   for k in range(n))
    if n > 1:
        rows += f"\n{'change vs zero-shot':23s}{100 * (MET[n-1]['ey'] / MET[0]['ey'] - 1):+7.0f}%{'':7s}{'':10s}{'':7s}{100 * (MET[n-1]['J'] / MET[0]['J'] - 1):+7.0f}%\n"
    return hdr + rows + "\nJ = 1/2 sum_t (e_y/0.10 m)^2 + (speed err/0.24 m/s)^2\n    (the stage's own objective)"
HOLD = int(round(args.pause * args.fps))
frames = []
for k in range(len(fc.ITS)):
    T_k = len(laps[k]["x_true"])
    frames += [(k, t, False) for t in range(0, T_k, 2)] + [(k, T_k - 1, True)] * HOLD
cur = {"k": -1}

def update(fr):
    k, t, held = fr
    logt.set_text(log_text(k + 1 if held else k))
    r = laps[k]
    if k != cur["k"]:
        cur["k"] = k
        draw_strip(k)
        draw_trace(k)
        if k > 0:
            xp = np.asarray(laps[k - 1]["x_true"])
            prevp.set_data(xp[:, 0], xp[:, 1]); prevl.set_text(f"dotted: the path of the policy {names[k - 1]}")
        else:
            prevp.set_data([], []); prevl.set_text("")
    x = np.asarray(r["x_true"])
    px, py, th = x[t, 0], x[t, 1], x[t, 2]
    c, s_ = math.cos(th), math.sin(th)
    body = np.array([[0.27, 0.14], [0.27, -0.14], [-0.27, -0.14], [-0.27, 0.14]])
    car.set_xy(np.c_[px + body[:, 0] * c - body[:, 1] * s_, py + body[:, 0] * s_ + body[:, 1] * c])
    trail.set_data(x[:t + 1, 0], x[:t + 1, 1])
    tr_cur.set_data(TR[k][0][:t + 1], TR[k][1][:t + 1])
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
    return car, trail, dot, title, info, logt, prevp, prevl, tr_cur

os.makedirs(os.path.dirname(args.out), exist_ok=True)
w = FFMpegWriter(fps=args.fps, bitrate=2400, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
with w.saving(fig, args.out, dpi=110):
    for fr in frames:
        update(fr)
        w.grab_frame()
print("->", args.out, len(frames), "frames;", ", ".join(f"pi{k}: lap rms {100 * np.sqrt(np.mean(np.asarray(laps[k]['err_true'])[:, 0] ** 2)):.1f} cm, "
                                                          f"outside tube {laps[k]['dev_time']:.2f} s" for k in range(len(fc.ITS))))
