"""Car: error vs real laps for the hardware stage on PPO + DR (2026-10-31; user: "how many [iterations] are really necessary ...
generate the relevant graphs"). One stage per practised track (square2fast, figfast; 3-lap trials, trust region, the score's
tolerances as the stage's error scales, slope anchor 1: runs/hw/freePPOV_s<k>_valS1wL3TRE8_<track>_stage), scored after k
iterations (its/it<k>, k = 0 .. 8: 3 k real laps per car per track) on its own track (PRACTISED, the two tracks pooled) and on
squareH (HELD-OUT, never driven by a stage; the two stages averaged per car). Validation protocol (eval_free.py: 14 cars, eval
seed 301, 4 x 5-lap runs), normalized error max(rms e_y / 10 cm, (1 - pace) / 0.1), a crash = 5, FEASIBLE (car, plan) pairs only
(car_v08_val_table.py's rule: some arm reached nerr <= 1), mean and 95 % cluster-bootstrap CI over (seed, car, plan), 3 seeds.
Baselines: PPO + DR and RMA zero-shot (lines), FADA + adaptation at its ~10 real laps per car (point).
    python paper/final/car_laps/car_laps_fig.py   -> paper/final/car_stage_laps.{pdf,png}
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(OUT, "pogo_budget"))
import budget_figs as B  # noqa: E402  (fonts, colors, style)
plt = B.plt

E = "/home/henry/ilc_ws/log/dilc/car_free/eval_free/"
RUB = ("ax:mu=4.5", "ax:mu=5.5,mass=1.1")
TR = ("square2fast", "figfast")
LAPS_PER_IT, KMAX = 3, 8
FADA_LAPS = 10


def nerr(r):
    return 5.0 if r["crash"] else max(r["rms_ey"] / 0.10, (1 - r["pace"]) / 0.1)


def load(n):
    f = os.path.join(E, n + ".json")
    if not os.path.exists(f):
        return None
    o = {}
    for r in json.load(open(f)):
        o.setdefault((r["car"], r["plan"].split("_b0")[0].replace("mocap_", "")), []).append(nerr(r))
    return {k: float(np.mean(v)) for k, v in o.items()}


def stage_name(s, t, k):
    if k == 0:
        return f"freePPOV_s{s}_valzs1m"
    return f"freePPOV_s{s}_valS1wL3TRE8{t}+stage" if k == KMAX else f"freePPOV_s{s}_valS1wL3TRE8it{k}{t}+stage"


def rows(src):
    """src(s) -> {track: eval name} or a single name; -> {(seed, car, plan, split): nerr}, None if any file is missing"""
    out = {}
    for s in range(3):
        n = src(s)
        ds = {t: load(x) for t, x in n.items()} if isinstance(n, dict) else {None: load(n)}
        if any(d is None for d in ds.values()):
            return None
        for t, d in ds.items():
            for (c, p), v in d.items():
                if (t is None and p in TR) or p == t:
                    out[(s, c, p, "practised")] = v
        for c in {c for d in ds.values() for (c, p) in d if p == "squareH"}:
            out[(s, c, "squareH", "held-out")] = float(np.mean([d[(c, "squareH")] for d in ds.values()]))
    return out


ARMS = {"PPO + DR": lambda s: f"freePPOV_s{s}_valzs1m", "RMA": lambda s: f"rmaV_s{s}_valzs1m",
        "FADA + adaptation": lambda s: {t: f"fadaV_s{s}_{t}+adapt" for t in TR}}
R = {a: rows(f) for a, f in ARMS.items()}
CURVE = {k: rows(lambda s, k=k: {t: stage_name(s, t, k) for t in TR}) for k in range(KMAX + 1)}
CURVE = {k: v for k, v in CURVE.items() if v is not None}
best = {}
for d in list(R.values()) + list(CURVE.values()):
    for (s, c, p, _), v in d.items():
        best[(c, p)] = min(best.get((c, p), 9), v)
feas = {k for k, v in best.items() if v <= 1}
rng = np.random.default_rng(0)


def ci(v):
    v = np.asarray(v)
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(3000)]
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


PANELS = [("practised tracks, plastic", "practised", False), ("practised tracks, rubber", "practised", True),
          ("held-out track, plastic", "held-out", False), ("held-out track, rubber", "held-out", True)]


def cell(d, split, rub):
    return [v for (s, c, p, sp), v in d.items() if sp == split and (c in RUB) == rub and (c, p) in feas]


if __name__ == "__main__":
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.0))
    report = {}
    for ax, (title, split, rub) in zip(axs, PANELS):
        B.style(ax)
        ks = sorted(CURVE)
        m = np.array([ci(cell(CURVE[k], split, rub)) for k in ks])
        x = np.array(ks) * LAPS_PER_IT
        col = B.STY["PPO + DR + DistILC"][0]
        ax.fill_between(x, m[:, 1], m[:, 2], color=col, alpha=0.15, lw=0)
        ax.plot(x, m[:, 0], "-", color=col, marker="D", ms=3, mec="white", mew=0.4, lw=1.3, label="PPO + DR + " + B.HW_N)
        for a, (col2, ls) in (("PPO + DR", (B.STY["PPO + DR"][0], "--")), ("RMA", (B.INK, ":"))):
            v = ci(cell(R[a], split, rub))
            ax.axhline(v[0], color=col2, ls=ls, lw=1.0, label=f"{a}, zero-shot")
            ax.axhspan(v[1], v[2], color=col2, alpha=0.08, lw=0)
        v = ci(cell(R["FADA + adaptation"], split, rub))
        ax.errorbar([FADA_LAPS], [v[0]], yerr=[[v[0] - v[1]], [v[2] - v[0]]], color=B.STY["FADA (DR)"][0], marker="h", ms=4,
                    capsize=1.5, lw=0.8, ls="none", label="FADA + adaptation")
        top = max(m[:, 2].max(), *[ci(cell(R[a], split, rub))[2] for a in ("PPO + DR", "RMA")]) * 1.15
        ax.set_ylim(0, top)                       # FADA's point runs off the top when it is above the rest (its value in the caption)
        if v[0] > top:
            ax.annotate(f"FADA {v[0]:.1f}", (FADA_LAPS, top), xytext=(FADA_LAPS, top * 0.9), ha="center", fontsize=6,
                        color=B.STY["FADA (DR)"][0], arrowprops=dict(arrowstyle="->", color=B.STY["FADA (DR)"][0], lw=0.7))
        ax.set_title(title, fontsize=7.5, color=B.INK, pad=3)
        ax.set_xticks(range(0, KMAX * LAPS_PER_IT + 1, 6))
        ax.set_xlabel("real laps per car per track", color=B.INK)
        report[title] = {"laps": x.tolist(), "stage": m.round(3).tolist()}
    axs[0].set_ylabel("normalized error", color=B.INK)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, labelcolor=B.INK, handlelength=1.8, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"car_stage_laps.{ext}"), dpi=300)
    json.dump(dict(report, iterations_present=sorted(CURVE), feasible_pairs=len(feas)), open(os.path.join(HERE, "car_laps_data.json"), "w"), indent=1)
    print("iterations present:", sorted(CURVE), "feasible pairs:", len(feas))
    for t, r in report.items():
        print(t, " ".join(f"{a}:{b[0]:.2f}" for a, b in zip(r["laps"], r["stage"])))
