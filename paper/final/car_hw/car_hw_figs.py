"""Car hardware-protocol figures (2026-10-31; user: "use the pogo stick graphs for inspiration"). Headline: each network's
improvement over its OWN zero-shot (feedback_paper_claims). Simulation stand-in for the hardware session, same protocol as
HW_RUNBOOK_N6_1031.md: envelope 0.5 m, plastic practised on square2fast (_a45; figure-eight scored at _a33), rubber
practised on the R 1.00 m figure-eight (_x120), squareH held out; 3 stage iterations x 2-lap trials; networks PPO + DR
(freePPON) and RMA (rmaN), seeds 0 and 1. Normalized error max(rms e_y / 10 cm, (1 - pace) / 0.1), a crash = 5; mean and
95 % cluster-bootstrap CI over (seed, car) units (each unit: mean over its 4 x 5-lap validation runs).
  car_hw_laps     error vs real laps (0 = the network's own zero-shot; per-iteration networks runs/hw/<stage>_it<k>)
  car_hw_improve  stage minus own zero-shot per track, paired bootstrap (as pogo_inout)
  car_hw_track    the practised track before and after, path coloured by |e_y| (as pogo_flow), PPO + DR, one car
  car_hw_alonglap e_y, speed and steering along the lap, zero-shot vs stage, mean over the tire's cars
  car_hw_perlap   lap-by-lap error over the 5-lap run (the final multi-lap tracking test)
  car_hw_rho      the stage's positivity rho per iteration (supplementary)
Trajectories: car_traj_sim.npz (car_traj_data.py). Writes car_hw_*.{pdf,png} to paper/final/.
    python paper/final/car_hw/car_hw_figs.py
"""
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(OUT, "pogo_budget"))
import budget_figs as B  # noqa: E402  (fonts, colors, style)
plt = B.plt
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

E = "/home/henry/ilc_ws/log/dilc/car_free/eval_free/"
HW = "/home/henry/ilc_ws/src/ilc_f1tenth/ilc2real/runs/hw/"
SEEDS = (0, 1)
NETS = {"freePPON": "PPO + DR", "rmaN": "RMA"}
COL = {"freePPON": B.STY["PPO + DR + DistILC"][0], "rmaN": B.INK}
MRK = {"freePPON": "D", "rmaN": "o"}
LAPS_PER_IT, ITERS = 2, 3
ROLE = ("practised", "other track", "held out")
# tire -> role -> (track key in the eval files, [eval name for k = 0 .. ITERS] as a function of (net, seed))
EV = {"plastic": {
    "practised": ("square2fast", lambda n, s: [f"N6_{n}_s{s}_zsa45E5B"] + [f"E5B45_{n}_s{s}_pl_square2fast+it{k}" for k in (1, 2)]
                  + [f"E5B45_{n}_s{s}_pl_square2fast+stage"]),
    "other track": ("fig8R100", lambda n, s: [f"N6_{n}_s{s}_zsL33"] + [f"L33_{n}_s{s}_pl_square2fast+it{k}" for k in (1, 2)]
                    + [f"L33_{n}_s{s}_pl_square2fast+stage"]),
    "held out": ("squareH", lambda n, s: [f"N6_{n}_s{s}_zsa45E5B"] + [f"E5B45_{n}_s{s}_pl_square2fast+it{k}" for k in (1, 2)]
                 + [f"E5B45_{n}_s{s}_pl_square2fast+stage"])},
      "rubber": {r: (t, lambda n, s: [f"F8R100_rb12_{n}_s{s}_zs"] + [f"F8R100_rb12_{n}_s{s}+it{k}" for k in (1, 2)]
                     + [f"F8R100_rb12_{n}_s{s}+stage"])
                 for r, t in zip(ROLE, ("fig8R100", "square2fast", "squareH"))}}
TRACK_LAB = {"square2fast": "square2fast", "fig8R100": "figure-eight", "squareH": "squareH"}
STAGE_DIR = {"plastic": "E5A45_{n}_s{s}_pl_square2fast_stage", "rubber": "F8R100_rb12_{n}_s{s}_stage"}
rng = np.random.default_rng(0)


def nerr(r):
    return 5.0 if r["crash"] else max(r["rms_ey"] / 0.10, (1 - r["pace"]) / 0.1)


_cache = {}


def units(name, track):
    """{car: mean nerr over its runs on track} from eval_free/<name>.json, None if missing"""
    if name not in _cache:
        f = E + name + ".json"
        _cache[name] = json.load(open(f)) if os.path.exists(f) else None
    rows = _cache[name]
    if rows is None:
        return None
    o = collections.defaultdict(list)
    for r in rows:
        if r["plan"].split("_b0")[0].replace("mocap_", "") == track:
            o[r["car"]].append(nerr(r))
    return {c: float(np.mean(v)) for c, v in o.items()} or None


def series(tire, role, net):
    """k -> {(seed, car): nerr}; only iterations whose eval exists for both seeds"""
    track, names = EV[tire][role]
    out = {}
    for k in range(ITERS + 1):
        d = {}
        for s in SEEDS:
            u = units(names(net, s)[k], track)
            if u is None:
                break
            d.update({(s, c): v for c, v in u.items()})
        else:
            out[k] = d
    return out


def boot(v, n=4000):
    v = np.asarray(v, float)
    b = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def legend_below(fig, handles, ncol, y=0.0):
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, labelcolor=B.INK,
               handlelength=1.8, columnspacing=1.4)


def net_handles(dashed=True):
    h = [Line2D([], [], color=COL[n], marker=MRK[n], ms=3.5, mec="white", mew=0.4, lw=1.3, label=f"{NETS[n]} + {B.HW_N}")
         for n in NETS]
    if dashed:
        h += [Line2D([], [], color=B.MUTED, ls="--", lw=0.9, label="own zero-shot")]
    return h


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1. error vs real laps (pogo_budget_real, car_stage_laps)
def fig_laps():
    fig, axs = plt.subplots(2, 3, figsize=(7.16, 3.6), sharex=True)
    rep = {}
    for i, tire in enumerate(EV):
        for j, role in enumerate(ROLE):
            ax = axs[i, j]
            B.style(ax)
            top = 0.0
            for net in NETS:
                S = series(tire, role, net)
                if not S:
                    continue
                ks = sorted(S)
                cars = set.intersection(*[set(S[k]) for k in ks])
                # change of each (seed, car) against its OWN zero-shot, in % (the headline; paired, so the CI is of the change)
                m = np.array([boot([100 * (S[k][u] / S[0][u] - 1) for u in cars]) for k in ks])
                x = np.array(ks) * LAPS_PER_IT
                ax.fill_between(x, m[:, 1], m[:, 2], color=COL[net], alpha=0.13, lw=0)
                ax.plot(x, m[:, 0], "-", color=COL[net], marker=MRK[net], ms=3.5, mec="white", mew=0.4, lw=1.3)
                ax.annotate(f"{m[-1, 0]:+.0f} %", (x[-1], m[-1, 0]), xytext=(4, -4 if net == "freePPON" else 4), textcoords="offset points", va="center",
                            fontsize=6.5, color=COL[net])
                top = min(top, m[:, 1].min())
                rep[f"{tire} {role} {NETS[net]}"] = dict(laps=x.tolist(), pct=m[:, 0].round(1).tolist(),
                                                         zero_shot=round(float(np.mean([S[0][u] for u in cars])), 3))
            ax.axhline(0, color=B.MUTED, ls="--", lw=0.8)
            ax.set_ylim(min(-10, top * 1.15), 12)
            ax.set_xlim(-0.4, LAPS_PER_IT * ITERS + 1.4)
            ax.set_xticks(range(0, LAPS_PER_IT * ITERS + 1, LAPS_PER_IT))
            ax.set_title(f"{tire}, {role}: {TRACK_LAB[EV[tire][role][0]]}", fontsize=7.5, color=B.INK, pad=3)
            if j == 0:
                ax.set_ylabel("error vs own zero-shot (%)", color=B.INK)
            if i == 1:
                ax.set_xlabel("real laps on the practised track", color=B.INK)
    legend_below(fig, net_handles(dashed=False) + [Line2D([], [], color=B.MUTED, ls="--", lw=0.8, label="own zero-shot (0 %)")], 3)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.6, w_pad=0.8)
    save(fig, "car_hw_laps")
    return rep


# ---------------------------------------------------------------- 2. improvement over own zero-shot (pogo_inout)
def fig_improve():
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.3), sharey=True)
    rep = {}
    w = 0.34
    for ax, tire in zip(axs, EV):
        B.style(ax)
        ax.axhline(0, color=B.MUTED, lw=0.6)
        for j, role in enumerate(ROLE):
            for q, net in enumerate(NETS):
                S = series(tire, role, net)
                if 0 not in S or ITERS not in S:
                    continue
                cars = set(S[0]) & set(S[ITERS])
                d = [S[ITERS][u] - S[0][u] for u in cars]
                m, lo, hi = boot(d)
                xx = j + (q - 0.5) * w
                ax.bar(xx, m, w * 0.92, color=COL[net], alpha=0.85 if net == "freePPON" else 0.75, lw=0)
                ax.errorbar(xx, m, yerr=[[m - lo], [hi - m]], color=B.INK, capsize=1.8, lw=0.8, ls="none")
                rep[f"{tire} {role} {NETS[net]}"] = [round(m, 3), round(lo, 3), round(hi, 3), len(d)]
        ax.set_xticks(range(len(ROLE)))
        ax.set_xticklabels([f"{r}\n({TRACK_LAB[EV[tire][r][0]]})" for r in ROLE], color=B.INK)
        ax.set_title(f"{tire} tires", fontsize=7.5, color=B.INK, pad=3)
        ax.tick_params(axis="x", length=0)
    axs[0].set_ylabel("stage $-$ own zero-shot\n(normalized error; < 0 is better)", color=B.INK)
    legend_below(fig, [plt.Rectangle((0, 0), 1, 1, color=COL[n], lw=0, label=f"{NETS[n]} + {B.HW_N}") for n in NETS], 2)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.0)
    save(fig, "car_hw_improve")
    return rep


# ---------------------------------------------------------------- trajectories
def traj():
    f = os.path.join(HERE, "car_traj_sim.npz")
    if not os.path.exists(f):
        return None, None
    z = np.load(f, allow_pickle=True)
    return list(z["runs"]), z["plans"].item()


PRACTISED = {"plastic": "mocap_square2fast_b0_free_a45", "rubber": "mocap_fig8R100_b0_free_x120"}


def lap_of(r, L):
    """lap number of every sample (the plan index wraps each lap, so count the arclength wraps)"""
    return np.concatenate([[0], np.cumsum(np.diff(r["s"]) < -0.5 * L)])


def local_err(r, pl):
    """the score's terms per sample: max(|e_y| / 10 cm, speed shortfall vs the plan / 10 %), in tolerances"""
    vp = np.asarray(pl["v"])[r["idx"]]
    return np.maximum(np.abs(r["ey"]) / 0.10, np.clip(1 - r["v"] / np.maximum(vp, 0.1), 0, None) / 0.1)


def run_nerr(r):
    return 5.0 if r["crash"] else max(r["rms_ey"] / 0.10, (1 - r["pace"]) / 0.1)


def pick_car(runs, tire, net="freePPON"):
    """the car whose zero-shot -> stage change is the median of its tire's cars (no cherry-picking)"""
    d = {}
    for r in runs:
        if r["tire"] == tire and r["net"] == net and r["k"] in (0, ITERS):
            d.setdefault(r["car"], {})[r["k"]] = run_nerr(r)
    dd = sorted((v[ITERS] - v[0], c) for c, v in d.items() if len(v) == 2)
    return dd[len(dd) // 2][1]


# ---------------------------------------------------------------- 3. before / after on the track (pogo_flow)
EXAG, NB = {"plastic": 5.0, "rubber": 3.0}, 240   # lateral error drawn xEXAG about the plan (rubber errs ~3x larger); lap-mean over NB bins


def lap_mean_path(r, pl):
    """lap-mean e_y and local error per plan arc-length bin (laps after the launch); -> (bin plan index, e_y, err).
    e_y is to the plan's left: plan + e_y n_left reproduces the car's position (checked: 2-4 mm median)"""
    keep = lap_of(r, pl["length"]) >= 1
    n = len(pl["s"])
    b = (r["idx"][keep] * NB) // n
    cnt = np.bincount(b, minlength=NB)
    ey = np.bincount(b, r["ey"][keep], NB) / np.maximum(cnt, 1)
    er = np.bincount(b, local_err(r, pl)[keep], NB) / np.maximum(cnt, 1)
    ok = cnt > 0
    return ((np.arange(NB) + 0.5) * n / NB).astype(int)[ok], ey[ok], er[ok]


def drawn(pl, i, ey, exag):
    X, Y = pl["X"], pl["Y"]
    tx, ty = np.gradient(X), np.gradient(Y)
    nn = np.hypot(tx, ty)
    p = np.c_[X[i] - ty[i] / nn[i] * exag * ey, Y[i] + tx[i] / nn[i] * exag * ey]
    return np.r_[p, p[:1]]                                  # closed lap


def fig_track(runs, plans):
    fig, axs = plt.subplots(2, ITERS + 1, figsize=(7.16, 4.3), gridspec_kw=dict(height_ratios=[1, 1.45]))
    norm = plt.Normalize(0, 2)
    cmap = plt.get_cmap("RdYlGn_r")
    picked = {}
    for i, tire in enumerate(EV):
        car = pick_car(runs, tire)
        picked[tire] = car
        pl = plans[PRACTISED[tire]]
        ghost = None
        for k in range(ITERS + 1):
            ax = axs[i, k]
            r = next(r for r in runs if r["tire"] == tire and r["net"] == "freePPON" and r["car"] == car and r["k"] == k)
            ix, ey, er = lap_mean_path(r, pl)
            p = drawn(pl, ix, ey, EXAG[tire])
            ax.plot(np.r_[pl["X"], pl["X"][:1]], np.r_[pl["Y"], pl["Y"][:1]], color=B.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
            if ghost is not None:
                ax.plot(ghost[:, 0], ghost[:, 1], color="#9a9a9a", lw=1.0, alpha=0.75, zorder=2)
            seg = np.stack([p[:-1], p[1:]], 1)
            lc = LineCollection(seg, cmap=cmap, norm=norm, lw=2.6, capstyle="round", zorder=3)
            lc.set_array(np.r_[er, er[:1]][:-1])
            ax.add_collection(lc)
            if k == 0:
                ghost = p
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(False)
            pad = 0.2
            ax.set_xlim(min(pl["X"].min(), p[:, 0].min()) - pad, max(pl["X"].max(), p[:, 0].max()) + pad)
            ax.set_ylim(min(pl["Y"].min(), p[:, 1].min()) - pad, max(pl["Y"].max(), p[:, 1].max()) + pad)
            head = "zero-shot" if k == 0 else f"$\\pi_{k}$ ({k * LAPS_PER_IT} real laps)"
            sub = "crashed" if r["crash"] else f"error {run_nerr(r):.2f}\nrms $e_y$ {100 * r['rms_ey']:.1f} cm, pace {r['pace']:.2f}"
            ax.set_title(f"{head}\n{sub}", fontsize=7, color=B.INK, pad=2, linespacing=1.3)
            if k == 0:
                ax.text(-0.06, 0.5, f"{tire}, {TRACK_LAB[PRACTISED[tire].split('_b0')[0].replace('mocap_', '')]}\nlateral error $\\times${EXAG[tire]:.0f}",
                        transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=7.5, color=B.INK)
            if k < ITERS:
                ax.annotate("", xy=(1.1, 0.5), xytext=(0.98, 0.5), xycoords="axes fraction",
                            arrowprops=dict(arrowstyle="->", color=B.INK, lw=0.8))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=fig.add_axes([0.9, 0.2, 0.012, 0.6]), extend="max")
    cb.set_label("local error (tol), lap mean\nmax($|e_y|$/10 cm, speed shortfall/10 %)", color=B.INK, fontsize=6.5)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=6.5, width=0.5, length=2)
    legend_below(fig, [Line2D([], [], color=B.MUTED, lw=0.7, ls=(0, (3, 2)), label="plan"),
                       Line2D([], [], color=cmap(0.8), lw=2.6, label=f"PPO + DR + {B.HW_N}: lap-mean path, lateral error exaggerated"),
                       Line2D([], [], color="#9a9a9a", lw=1.0, label="zero-shot path (same scale)")], 3)
    fig.subplots_adjust(left=0.06, right=0.88, top=0.9, bottom=0.08, wspace=0.22, hspace=0.42)
    save(fig, "car_hw_track")
    return picked


# ---------------------------------------------------------------- 4. along the lap
def wheel(cmd, lim=0.34, k=60.0):
    """the wheel angle the model applies: car_model.sat, the smooth clamp of the command to +-0.34 rad (SAT_K 60)"""
    return cmd - np.logaddexp(0, k * (cmd - lim)) / k + np.logaddexp(0, k * (-lim - cmd)) / k


def binned(rs, key, edges, plan):
    out = []
    for r in rs:
        s = r["s"]
        m = lap_of(r, plan["length"]) >= 1             # drop the first lap (the launch)
        v = np.abs(r[key]) if key == "ey" else wheel(r[key]) if key == "steer" else r[key]
        h = np.histogram(s[m], edges, weights=v[m])[0] / np.maximum(np.histogram(s[m], edges)[0], 1)
        out.append(h)
    return np.array(out)


def fig_alonglap(runs, plans):
    fig, axs = plt.subplots(3, 2, figsize=(7.16, 3.8), sharex="col")
    for j, tire in enumerate(EV):
        pl = plans[PRACTISED[tire]]
        L = pl["length"]
        edges = np.linspace(0, L, 121)
        c = 0.5 * (edges[1:] + edges[:-1])
        kap = np.interp(c, pl["s"], np.abs(pl["kappa"]))
        sat_pct = {}
        for i, (key, lab) in enumerate((("ey", "$|e_y|$ (m)"), ("v", "speed (m/s)"), ("steer", "wheel angle (rad)"))):
            ax = axs[i, j]
            B.style(ax)
            tw = ax.twinx()
            tw.fill_between(c, 0, kap, color=B.GRID, lw=0, step="mid")
            tw.set_ylim(0, kap.max() * 3.2)
            tw.axis("off")
            ax.set_zorder(tw.get_zorder() + 1)
            ax.patch.set_visible(False)
            if key == "v":
                ax.plot(pl["s"], pl["v"], color=B.MUTED, lw=0.8, ls=(0, (3, 2)))
            if key == "steer":
                for sgn in (1, -1):
                    ax.axhline(0.34 * sgn, color="#c0392b", lw=0.6, ls=":")
                ax.plot(pl["s"], pl["steer"], color=B.MUTED, lw=0.8, ls=(0, (3, 2)))
            for net in NETS:
                for k, ls, a in ((0, "--", 0.8), (ITERS, "-", 1.0)):
                    rs = [r for r in runs if r["tire"] == tire and r["net"] == net and r["k"] == k and not r["crash"]]
                    if not rs:
                        continue
                    ax.plot(c, binned(rs, key, edges, pl).mean(0), color=COL[net], ls=ls, lw=1.0 if k else 0.8, alpha=a)
                    if key == "steer":
                        sat_pct[(net, k)] = 100 * np.mean([np.mean(np.abs(r["steer"]) > 0.34) for r in rs])
            if key == "steer":
                ax.text(0.99, 1.01, "steering command beyond the limit: " + ", ".join(
                    f"{NETS[n]} {sat_pct[(n, 0)]:.0f} $\\to$ {sat_pct[(n, ITERS)]:.0f} %" for n in NETS if (n, 0) in sat_pct and (n, ITERS) in sat_pct),
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=B.INK)
            if j == 0:
                ax.set_ylabel(lab, color=B.INK)
            if i == 0:
                ax.set_title(f"{tire}, practised: {TRACK_LAB[PRACTISED[tire].split('_b0')[0].replace('mocap_', '')]}",
                             fontsize=7.5, color=B.INK, pad=3)
            if i == 2:
                ax.set_xlabel("distance along the lap (m)", color=B.INK)
    h = [Line2D([], [], color=COL[n], lw=1.0, label=NETS[n]) for n in NETS]
    h += [Line2D([], [], color=B.INK, ls="--", lw=0.8, label="zero-shot"),
          Line2D([], [], color=B.INK, lw=1.0, label=f"+ {B.HW_N} (6 laps)"),
          Line2D([], [], color=B.MUTED, ls=(0, (3, 2)), lw=0.8, label="plan"),
          plt.Rectangle((0, 0), 1, 1, color=B.GRID, lw=0, label="|curvature|"),
          Line2D([], [], color="#c0392b", ls=":", lw=0.6, label="steering limit")]
    legend_below(fig, h, 7)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.9, w_pad=1.0)
    save(fig, "car_hw_alonglap")


# ---------------------------------------------------------------- 5. lap by lap (the multi-lap tracking test)
def per_lap(r, plan):
    """per complete lap after the launch lap (the last, partial one dropped): rms e_y (cm) and pace (mean speed over the
    plan's at the same arc lengths, as real_car.lap_metrics)"""
    lap = lap_of(r, plan["length"])
    vp = np.asarray(plan["v"])[r["idx"]]
    return [(100 * np.sqrt(np.mean(r["ey"][lap == q] ** 2)), r["v"][lap == q].mean() / vp[lap == q].mean())
            for q in range(1, int(lap.max()))]


def fig_perlap(runs, plans):
    fig, axs = plt.subplots(2, 2, figsize=(7.16, 3.0), sharex=True)
    for j, tire in enumerate(EV):
        pl = plans[PRACTISED[tire]]
        for net in NETS:
            for k, ls in ((0, "--"), (ITERS, "-")):
                L = [per_lap(r, pl) for r in runs if r["tire"] == tire and r["net"] == net and r["k"] == k and not r["crash"]]
                n = min(len(x) for x in L) if L else 0
                if not n:
                    continue
                A = np.array([x[:n] for x in L])                   # (car, lap, [rms e_y, pace])
                x = np.arange(2, n + 2)
                for i in range(2):
                    axs[i, j].plot(x, A[:, :, i].mean(0), color=COL[net], ls=ls, marker=MRK[net], ms=3, mec="white", mew=0.4,
                                   lw=1.0 if k else 0.8)
        for i in range(2):
            B.style(axs[i, j])
        axs[0, j].set_ylim(bottom=0)
        axs[1, j].axhline(1, color=B.MUTED, lw=0.6, ls=":")
        axs[1, j].set_xlabel("complete lap of a 5-plan-lap-time run (lap 1, the launch, dropped)", color=B.INK)
        axs[1, j].set_xticks(x)
        axs[0, j].set_title(f"{tire}, practised: {TRACK_LAB[PRACTISED[tire].split('_b0')[0].replace('mocap_', '')]}",
                            fontsize=7.5, color=B.INK, pad=3)
    axs[0, 0].set_ylabel("rms $e_y$ (cm)", color=B.INK)
    axs[1, 0].set_ylabel("pace (vs plan)", color=B.INK)
    h = [Line2D([], [], color=COL[n], marker=MRK[n], ms=3, lw=1.0, label=NETS[n]) for n in NETS]
    h += [Line2D([], [], color=B.INK, ls="--", lw=0.8, label="zero-shot"), Line2D([], [], color=B.INK, lw=1.0, label=f"+ {B.HW_N} (6 laps)")]
    legend_below(fig, h, 4)
    fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=0.4, w_pad=1.0)
    save(fig, "car_hw_perlap")


# ---------------------------------------------------------------- 6. positivity per iteration
def fig_rho():
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 1.8), sharey=True)
    for ax, tire in zip(axs, EV):
        B.style(ax)
        ax.axhline(0, color=B.MUTED, lw=0.6)
        for q, net in enumerate(NETS):
            for s in SEEDS:
                f = HW + STAGE_DIR[tire].format(n=net, s=s) + "/summary.json"
                if not os.path.exists(f):
                    continue
                for c, tr in json.load(open(f))["tr"].items():
                    for t in tr:
                        x = t["it"] + (q - 0.5) * 0.3 + rng.uniform(-0.08, 0.08)
                        ax.plot(x, np.clip(t["rho"], -1, 2), MRK[net], color=COL[net], ms=3, mec="white", mew=0.3, alpha=0.8)
        ax.set_xticks(range(1, ITERS))
        ax.set_xlim(0.5, ITERS - 0.5)
        ax.set_yticks([-1, 0, 1, 2])
        ax.set_yticklabels(["$\\leq$-1", "0", "1", "$\\geq$2"])
        ax.set_xlabel("stage update ($\\rho$ measured on the next trial; the last update has none)", color=B.INK)
        ax.set_title(f"{tire} tires", fontsize=7.5, color=B.INK, pad=3)
    axs[0].set_ylabel(r"$\rho$ = measured / predicted", color=B.INK)
    legend_below(fig, [Line2D([], [], color=COL[n], marker=MRK[n], ls="none", ms=3, label=NETS[n]) for n in NETS], 2)
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=1.0)
    save(fig, "car_hw_rho")


if __name__ == "__main__":
    rep = dict(laps=fig_laps(), improve=fig_improve())
    fig_rho()
    runs, plans = traj()
    if runs is not None:
        rep["track_car"] = fig_track(runs, plans)
        fig_alonglap(runs, plans)
        fig_perlap(runs, plans)
    json.dump(rep, open(os.path.join(HERE, "car_hw_data.json"), "w"), indent=1)
    for k, v in rep["improve"].items():
        print(f"{k:40s} {v}")
