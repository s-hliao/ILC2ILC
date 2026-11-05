"""Car: the hardware stage re-converging the closed-loop flow inside the 0.5 m tube around the plan (2026-11-05; user: "the
flow diagram of the car retargeting in the 0.5 tube around the reference trajectory similar to the pogo stick"; the car
version of pogo_flow/flow_strip.py).
  frame       x = arc length s along the lap (m), y = lateral offset e_y from the plan (m), the tube = the room's safety
              envelope |e_y| <= 0.5 m (env_tight.sh F1T_SAFETY_EY; net_runner.py safety_ey on the car)
  rows        the nominal car (mu 0.6, the friction the network and plans were made at) under pi_0 (the reference), then
              the plastic car n04_real_nom (every friction x 0.4 / 0.6) under pi_0 .. pi_3 of one plastic stage run
              (runs/screen/PLSL_freePPON_s0: PPO+DR network seed 0, stage seed 9001; 3 iterations x 2 laps on
              square2fast); shown on the HELD-OUT peanut (clockwise, _a33), never practised
  arrows      where the car is H = 0.4 s later when started at (s, e_y) on the plan's state shifted sideways (heading,
              speeds on the plan): (delta s, delta e_y), one length factor for every row
  background  |delta e_y - delta e_y(reference)| in tolerance units (0.10 m, lap_metrics ey_ok)
  points      four laterally perturbed starts (e_y0 = +-0.30, +-0.15 m at four places), every 0.2 s for one lap
Rollouts: real_car.run, no sensing noise, the plastic runs' exact environment (plastic_sl.sh). The fields are computed once
into flow_car_data.npz (delete it to recompute). Writes car_flow.{pdf,png} to paper/final/.
    ~/miniconda3/envs/f1t/bin/python paper/final/car_flow/flow_car.py
"""
import glob
import math
import os
import sys

D = "/home/henry/ilc_ws/src/ilc_f1tenth/ilc2real"
os.environ.update(F1T_MU="0.6", F1T_PLANS=D + "/plans_nom06", F1T_TRACKS=D + "/../lifted_linear_tire_20261009/tracks_tight",
                  F1T_SAFETY_EY="0.5", F1T_NO_CRASH="1", NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1", JAX_PLATFORMS="cpu")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)                                          # paper/final
# variants (2026-11-05, user: "do both"): the plastic and rubber cars' lateral flow; the plastic car's SPEED flow (the plastic stage's
# measured gain is lap pace, not lateral tracking)
VARIANTS = {
    "plastic": dict(plan="mocap_peanutM_b0_free_a33", run="PLSL_freePPON_s0", car="n04_real_nom", var="lat", stem="car_flow",
                    label="plastic car"),
    "rubber": dict(plan="mocap_peanutM_b0_free_x120", run="SL50x120r2_freePPON_s0", car="ax:mu=1.667", var="lat",
                   stem="car_flow_rubber", label="rubber car"),
    "speed": dict(plan="mocap_peanutM_b0_free_a33", run="PLSL_freePPON_s0", car="n04_real_nom", var="speed",
                  stem="car_flow_speed", label="plastic car"),
    # the largest held-out score drop among the cells with iteration policies on this machine (48 cells: PLSL_freePPON_s0, s1;
    # 1.04 -> 0.56); SELECTED as the best example -- disclose as such (the 216-cell median is -19 %)
    "speed_best": dict(plan="mocap_peanutM_b0_free_a33", run="PLSL_freePPON_s1", car="ax:mu=0.6000", var="speed",
                       stem="car_flow_speed_best", label="plastic car, friction x0.6 (best of 48)"),
}
V = VARIANTS[sys.argv[1] if len(sys.argv) > 1 else "plastic"]
DATA = os.path.join(HERE, f"{V['stem']}_data.npz")
PLAN, RUN, CAR = V["plan"], D + "/runs/screen/" + V["run"], V["car"]
REF_CAR = "real_nom"
TUBE, TOL, H, DT = 0.5, 0.10, 0.4, 0.025
NS, NE = 40, 11
if V["var"] == "lat":
    GRID = np.linspace(-0.45, 0.45, NE)                              # lateral offset e_y0 (m)
    STARTS = [(0.15, 0.30), (0.40, -0.30), (0.65, 0.15), (0.90, -0.15)]   # (lap fraction, e_y0)
else:
    GRID = np.linspace(-0.60, 0.30, NE)                              # speed error dV0 (m/s)
    STARTS = [(0.15, -0.45), (0.40, 0.20), (0.65, -0.25), (0.90, 0.0)]    # (lap fraction, dV0)


def load_policy(path):
    """hw_stage.load_policy (hw_stage parses argv on import)"""
    d = np.load(path)
    n = len([k for k in d.files if k.startswith("W")])
    return [(np.asarray(d[f"W{i}"]), np.asarray(d[f"b{i}"])) for i in range(n)]


def compute():
    sys.path.insert(0, D)
    os.chdir(D)
    import bank
    import real_car as rc
    import sim_jax_np as sn
    pl = bank.load_bank([PLAN])[0]
    P = sn.PlanNP(pl)
    L = float(pl["length"])
    pols = [load_policy(f"{RUN}/its/it{k}_{CAR}_policy.npz") for k in range(4)]
    panels = [(REF_CAR, pols[0])] + [(CAR, p) for p in pols]

    def start(s, e):
        i = int(round(s / P.ds)) % P.M
        x = pl["x"][i].copy()
        if V["var"] == "lat":                                        # sideways shift, everything else on the plan
            x[0] += -math.sin(x[2]) * e
            x[1] += math.cos(x[2]) * e
        else:                                                        # on the path, speed V + e (vx, vy, r, wheel scaled)
            f = (math.hypot(x[3], x[4]) + e) / math.hypot(x[3], x[4])
            x[3:7] *= f
        return x, i

    ss = np.linspace(0, L, NS, endpoint=False)
    es = GRID
    col = 0 if V["var"] == "lat" else 5                              # err_true: 0 = e_y, 5 = speed error dV
    T = int(round(H / DT))
    jobs, keys = [], []
    for k, (car, pol) in enumerate(panels):
        for i, s in enumerate(ss):
            for j, e in enumerate(es):
                x, i0 = start(s, e)
                jobs.append(dict(car_name=car, plan=pl, policy=pol, x_start=x, idx_start=i0, T_max=T + 1, noise=False))
                keys.append(("f", k, i, j))
        for m, (fr, e) in enumerate(STARTS):
            x, i0 = start(fr * L, e)
            jobs.append(dict(car_name=car, plan=pl, policy=pol, x_start=x, idx_start=i0,
                             T_max=int(round(float(pl["lap_time"]) / DT)), noise=False))
            keys.append(("p", k, m, 0))
    R = rc.run_many(jobs, 16)
    K = len(panels)
    dS, dE = np.full((K, NS, NE), np.nan), np.full((K, NS, NE), np.nan)
    orb = [[None] * len(STARTS) for _ in range(K)]
    for (kind, k, i, j), r, jb in zip(keys, R, jobs):
        ey = r["err_true"][:, col]
        if kind == "f":
            n = min(T, len(ey) - 1)
            dS[k, i, j] = ((r["idx"][n] - r["idx"][0] + P.M // 2) % P.M - P.M // 2) * P.ds
            dE[k, i, j] = ey[n] - ey[0]
        else:
            sabs = (r["idx"][0] * P.ds + np.cumsum(np.r_[0, ((np.diff(r["idx"]) + P.M // 2) % P.M - P.M // 2) * P.ds]))
            orb[k][i] = np.stack([sabs, ey], 1)
    O = np.full((K, len(STARTS), max(len(o) for row in orb for o in row), 2), np.nan)
    for k in range(K):
        for m in range(len(STARTS)):
            O[k, m, :len(orb[k][m])] = orb[k][m]
    vbar = float(np.mean(np.hypot(pl["x"][:, 3], pl["x"][:, 4])))
    np.savez(DATA, ss=ss, es=es, dS=dS, dE=dE, O=O, L=L, xy=np.asarray(pl["x"])[:, :3], vbar=vbar)


def main():
    if not os.path.exists(DATA):
        compute()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    sys.path[:0] = ["/home/henry/ilc_ws/log/dilc/flows"]
    import portrait_style as ps

    for f in glob.glob("/usr/share/fonts/truetype/msttcorefonts/[Tt]imes*.ttf"):
        font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family": ["Times New Roman", "TeX Gyre Termes", "Nimbus Roman", "STIXGeneral", "DejaVu Sans"],
                         "mathtext.fontset": "stix", "font.size": 7, "pdf.fonttype": 42})
    Dd = np.load(DATA)
    ss, es, dS, dE, O, L = Dd["ss"], Dd["es"], Dd["dS"], Dd["dE"], Dd["O"], float(Dd["L"])
    lat = V["var"] == "lat"
    tol = TOL if lat else 0.1 * float(Dd["vbar"])                    # 0.10 m, or 10 % of the plan's mean speed (pace_ok)
    ylim = (-0.6, 0.6) if lat else (es[0] - 0.08, es[-1] + 0.08)
    SS, EE = np.meshgrid(ss, es, indexing="ij")
    # one length factor for every row; in the speed panels the vertical axis is m/s, plotted on the same scale as m
    k_len = 0.85 * (ss[1] - ss[0]) * 2 / np.nanquantile(np.hypot(dS[0], dE[0]), 0.9)
    titles = [r"nominal car ($\mu$ 0.6), $\pi_0$ (reference)"] + \
             [V["label"] + rf", $\pi_{k}$" + (" (zero-shot)" if k == 0 else f" (after {2 * k} laps)") for k in range(4)]
    settle = []
    fig, axs = plt.subplots(len(titles), 1, figsize=(7.16, 5.6), sharex=True)
    for k, (ax, ttl) in enumerate(zip(axs, titles)):
        dev = np.abs(dE[k] - dE[0]) / tol
        im = ps.background(ax, SS, EE, dev)
        if lat:
            ax.axhspan(TUBE, 0.6, color="#e9e7e2", zorder=1)
            ax.axhspan(-0.6, -TUBE, color="#e9e7e2", zorder=1)
            for y in (-TUBE, TUBE):
                ax.axhline(y, color="#777777", lw=0.6, ls="--", zorder=2)
        ax.axhline(0, color=ps.INK, lw=0.5, alpha=0.5, zorder=2)
        sub = (slice(None, None, 2), slice(None, None, 2))
        q = ps.arrows(ax, SS[sub], EE[sub], k_len * dS[k][sub], k_len * dE[k][sub])
        q.width, q.headwidth, q.headlength = 0.0022, 3.6, 4.0
        q.set_alpha(0.8)
        tail = []
        for m in range(O.shape[1]):
            o = O[k, m]
            o = o[np.isfinite(o[:, 0])]
            out = np.nonzero(np.abs(o[:, 1]) > TUBE)[0] if lat else []
            if len(out):                                             # leaves the tube: on the car, the kill switch
                o = o[:out[0] + 1]
                ax.plot(np.mod(o[-1, 0], L), np.clip(o[-1, 1], -0.58, 0.58), "x", color=ps.RUN, ms=4, mew=1.0, zorder=7)
            tail.append(o[int(round(1.0 / DT)):, 1])                 # after 1 s: where the starts settle
            p = o[::8]                                               # every 0.2 s
            ax.plot(np.mod(p[:, 0], L), p[:, 1], "o", color=ps.RUN, mec="white", mew=0.3, ms=2.2, zorder=6)
        settle.append(float(np.mean(np.concatenate(tail))))
        if not lat:                                                  # the level the speed settles at
            ax.axhline(settle[-1], color=ps.RUN, lw=0.8, ls=":", zorder=5)
        ax.set_xlim(0, L)
        ax.set_ylim(*ylim)
        ax.set_yticks([-0.5, 0, 0.5] if lat else [-0.6, -0.3, 0, 0.3])
        ax.tick_params(labelsize=6, length=2, width=0.5, pad=1.5)
        ax.set_title(ttl, fontsize=7.5, loc="left", pad=2)
        extra = "" if lat else f", settles at {settle[-1]:+.2f} m/s"
        ax.text(1.0, 1.02, f"mean error {np.nanmean(dev):.2f} tol{extra}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6, color="#555555")
        for s_ in ax.spines.values():
            s_.set_linewidth(0.5)
            s_.set_color("#777777")
    axs[-1].set_xlabel("arc length along the held-out lap (m)", fontsize=7, labelpad=1.5)
    fig.supylabel("lateral offset from the plan (m)" if lat else "speed minus the plan's speed (m/s)", fontsize=7, x=0.012)
    hs, ls_ = [], []
    if lat:
        hs += [Line2D([], [], color="#777777", lw=0.6, ls="--")]
        ls_ += ["0.5 m tube (safety envelope)"]
    hs += [Line2D([], [], ls="none", marker="o", color=ps.RUN, mec="white", mew=0.3, ms=3)]
    ls_ += [("4 laterally perturbed" if lat else "4 speed-perturbed") + " starts, every 0.2 s"]
    if lat:
        hs += [Line2D([], [], ls="none", marker="x", color=ps.RUN, mew=1.0, ms=4)]
        ls_ += ["leaves the tube"]
    else:
        hs += [Line2D([], [], color=ps.RUN, lw=0.8, ls=":")]
        ls_ += ["where they settle (after 1 s)"]
    hs += [Line2D([], [], color=ps.INK, lw=0.8, marker=">", ms=2.5, alpha=0.8)]
    ls_ += [f"where the car is {H:.1f} s later"]
    fig.legend(hs, ls_, loc="lower center", bbox_to_anchor=(0.47, -0.005), ncol=4, frameon=False, fontsize=6, handlelength=1.4,
               columnspacing=1.2, handletextpad=0.4)
    fig.subplots_adjust(left=0.065, right=0.915, top=0.965, bottom=0.10, hspace=0.42)
    p0, p1 = axs[0].get_position(), axs[-1].get_position()
    cax = fig.add_axes([p0.x1 + 0.012, p1.y0, 0.009, p0.y1 - p1.y0])
    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.set_label(("lateral" if lat else "speed") + f" flow deviation from the reference (tol = {tol:.2f} " + ("m)" if lat else "m/s)"),
                 fontsize=6, labelpad=2)
    cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4)
    cb.outline.set_linewidth(0.4)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{V['stem']}.{ext}"), dpi=300)
    for k, t in enumerate(titles):                                   # the numbers the caption quotes
        dev = np.abs(dE[k] - dE[0]) / tol
        out = np.mean([np.any(np.abs(O[k, m][np.isfinite(O[k, m][:, 1]), 1]) > TUBE) for m in range(O.shape[1])]) if lat else 0.0
        print(f"{t:45s} mean dev {np.nanmean(dev):.2f} tol, perturbed starts settle at {settle[k]:+.3f}, leave the tube {out:.2f}")
    print("->", OUT, V["stem"])


if __name__ == "__main__":
    main()
