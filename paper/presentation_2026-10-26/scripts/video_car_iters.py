#!/usr/bin/env python3
"""
video_car_iters.py (2026-10-26): the car's hardware stage (F1TENTH, 25-degree drift, sim-to-sim), iteration by
iteration, on ONE car: every iteration's network flown 2 laps on the same figure-8 (presentation_2026-10-23
car_fly_fig8.py's re-flights, stage_trials_fig8_<run>.npz), earlier iterations kept as ghosts in their ramp colours.
  left    the whole figure-8 (reference line dashed) with every iteration's path, the car of the current one
  middle  a 1.2 m window following the car: the paths' differences at cm scale
  right   lateral error over time (every iteration, the current one growing) | RMS lateral error per iteration
    python video_car_iters.py CAR [--run v3drBtL_s0_ladR_val_its] [--speed 3] [--procs 16]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iter_style as S                                        # noqa: E402

SRC = "/home/henry/ilc_ws/log/dilc/presentation_2026-10-23/media/car"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "car")
D = None
SPEED = 3.0


def lateral(x, xy):
    """unsigned distance of each state's position to the plan's polyline (cm)"""
    P = xy[None, :, :]
    d = np.sqrt(((x[:, None, :2] - P) ** 2).sum(-1)).min(1)
    return 100 * d


def car_poly(X0, Y0, psi, L=0.30, W=0.16):
    body = np.array([[L / 2, W / 2], [L / 2, -W / 2], [-L / 2, -W / 2], [-L / 2, W / 2]])
    R = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
    return body @ R.T + [X0, Y0]


def frame(job):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    S.rc()
    i, k, t, phase, path = job
    its, K, dt = D["its"], len(D["its"]) - 1, D["dt"]
    cur = its[k]
    x = cur["x"]
    j = len(x) - 1 if t is None else min(int(t * SPEED / (S.FPS * dt)), len(x) - 1)
    fig = plt.figure(figsize=(16, 9), dpi=100)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 1.05], left=0.03, right=0.985, top=0.79, bottom=0.07,
                          hspace=0.35, wspace=0.16)
    axA, axZ = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])
    axE, axB = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])
    for ax in (axA, axZ):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(True)
            s_.set_color(S.GRID)
    xy = cur["plan"]
    for ax in (axA, axZ):
        ax.plot(np.r_[xy[:, 0], xy[:1, 0]], np.r_[xy[:, 1], xy[:1, 1]], color=S.INK, lw=1.2, ls="--", zorder=2)
        for kk in range(k):
            xp = its[kk]["x"]
            ax.plot(xp[:, 0], xp[:, 1], color=S.color(kk, K), lw=1.8 if ax is axZ else 1.2, alpha=0.9, zorder=3)
            if its[kk]["failed"]:
                ax.plot([xp[-1, 0]], [xp[-1, 1]], "x", color=S.BAD, ms=12, mew=2.5, zorder=7)
        ax.plot(x[:j + 1, 0], x[:j + 1, 1], color=S.color(k, K), lw=3.2 if ax is axZ else 2.2, zorder=4)
        b = car_poly(*x[j, :3])
        ax.fill(b[:, 0], b[:, 1], color=S.color(k, K), ec=S.INK, lw=1.0, zorder=6)
        if cur["failed"] and j == len(x) - 1:
            ax.plot([x[j, 0]], [x[j, 1]], "x", color=S.BAD, ms=16, mew=3, zorder=8)
    pad = 0.35
    axA.set_xlim(D["lim"][0] - pad, D["lim"][1] + pad)
    axA.set_ylim(D["lim"][2] - pad, D["lim"][3] + pad)
    w = 0.6
    axZ.set_xlim(x[j, 0] - w, x[j, 0] + w)
    axZ.set_ylim(x[j, 1] - w, x[j, 1] + w)
    axA.add_patch(Rectangle((x[j, 0] - w, x[j, 1] - w), 2 * w, 2 * w, fill=False, ec=S.INK2, lw=1.0, ls=":", zorder=9))
    axA.set_title(f"the figure-8, 2 laps   (t = {j * dt:.1f} s, {SPEED:g}x real time)", loc="left")
    axZ.set_title("zoom: 1.2 m around the car (dotted box)", loc="left")
    axZ.plot([x[j, 0] - w + 0.05, x[j, 0] - w + 0.15], [x[j, 1] - w + 0.06] * 2, color=S.INK, lw=2)
    axZ.text(x[j, 0] - w + 0.1, x[j, 1] - w + 0.08, "10 cm", ha="center", fontsize=10)

    for kk in range(k + 1):
        e = its[kk]["ey"]
        tt = np.arange(len(e)) * dt
        n = j + 1 if kk == k else len(e)
        axE.plot(tt[:n], e[:n], color=S.color(kk, K), lw=2.4 if kk == k else 1.3, alpha=1.0 if kk == k else 0.85)
        if its[kk]["failed"] and n == len(e):
            axE.plot([tt[-1]], [e[-1]], "x", color=S.BAD, ms=10, mew=2.5)
    axE.set_xlim(0, D["tmax"])
    axE.set_ylim(0, D["emax"])
    axE.set_title("distance from the reference line", loc="left")
    axE.set_xlabel("time (s)")
    axE.set_ylabel("cm")
    done = k + 1 if phase == "pause" else k
    for kk in range(K + 1):
        if kk < done:
            if its[kk]["failed"]:
                axB.bar(kk, D["bmax"] * 0.92, color="none", ec=S.BAD, hatch="//", width=0.7)
                axB.text(kk, D["bmax"] * 0.5, "crash", ha="center", rotation=90, color=S.BAD, fontsize=11)
            else:
                axB.bar(kk, its[kk]["rms"], color=S.color(kk, K), width=0.7)
                axB.text(kk, its[kk]["rms"] + 0.25, f"{its[kk]['rms']:.1f}", ha="center", fontsize=11)
        else:
            axB.bar(kk, D["bmax"] * 0.92, color="none", ec=S.GRID, ls=":", width=0.7)
    axB.set_xticks(range(K + 1))
    axB.set_xticklabels([f"π{S.sub(kk)}" for kk in range(K + 1)])
    axB.set_ylim(0, D["bmax"])
    axB.set_ylabel("cm")
    axB.set_title("RMS lateral error per iteration", loc="left")
    axB.set_xlabel("iteration (one 2-lap real trial each)")
    res = None
    if phase == "pause":
        res = (f"π{S.sub(k)}: crashed into the wall after {cur['laps']:.1f} laps" if cur["failed"]
               else f"π{S.sub(k)}: RMS lateral error {cur['rms']:.1f} cm over 2 laps")
        if cur["rung"]:
            res += f"  (plan eased: {cur['rung']})"
    S.header(fig, f"Drift car (F1TENTH): DistILC hardware stage, iteration by iteration  ({D['car']}, sim-to-sim)",
             k, K, phase, res)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    global D, SPEED
    car = sys.argv[1]
    run = sys.argv[sys.argv.index("--run") + 1] if "--run" in sys.argv else "v3drBtL_s0_ladR_val_its"
    SPEED = float(sys.argv[sys.argv.index("--speed") + 1]) if "--speed" in sys.argv else 3.0
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 16
    z = np.load(f"{SRC}/stage_trials_fig8_{run}.npz")
    meta = json.loads(str(z["meta"]))
    P = {k.split("__")[1]: z[k] for k in z.files if k.startswith("plan__")}
    its = []
    for i, m in sorted(((i, m) for i, m in enumerate(meta) if m["car"] == car), key=lambda q: q[1]["it"]):
        x = z[f"x_{i}"]
        xy = P[m["plan"]]
        ey = lateral(x, xy)
        rung = m["plan"].split("_b25")[1].lstrip("_")
        its.append(dict(x=x, plan=xy, ey=ey, failed=bool(m["failed"]), laps=m["laps"],
                        rms=100 * m["rms_ey"] if not m["failed"] else None,
                        rung={"": "", "mu80": "ladder rung friction 0.8", "mu60": "ladder rung friction 0.6"}[rung]))
        print(m["it"], m["plan"], "crash" if m["failed"] else f"{100 * m['rms_ey']:.1f} cm", flush=True)
    allxy = np.concatenate([P[m["plan"]] for m in meta if m["car"] == car])
    dt = float(z["dt"])
    D = dict(its=its, dt=dt, car=car.replace("real_", "").replace("ax:mu=", "friction x"),
             lim=(allxy[:, 0].min(), allxy[:, 0].max(), allxy[:, 1].min(), allxy[:, 1].max()),
             tmax=max(len(q["x"]) for q in its) * dt, emax=min(60.0, max(np.percentile(q["ey"], 99) for q in its) * 1.1),
             bmax=max(q["rms"] for q in its if q["rms"] is not None) * 1.3)
    os.makedirs(OUT, exist_ok=True)
    n_fly = [int(np.ceil(len(q["x"]) * dt * S.FPS / SPEED)) + 1 for q in its]
    plan = S.frame_plan(n_fly, len(its) - 1)
    tag = car.replace(":", "").replace("=", "")
    mp4 = f"{OUT}/car_iters_{tag}.mp4"
    n = S.render(frame, plan, mp4, procs, stills=[(0, mp4[:-4] + "_pi0.png"), (len(its) - 1, mp4[:-4] + "_final.png")])
    print(mp4, n, "frames", flush=True)


if __name__ == "__main__":
    main()
