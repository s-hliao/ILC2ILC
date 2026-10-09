#!/usr/bin/env python3
"""plane_grid.py: the sim networks over the 2D goal plane (grid_eval.py's maps, nominal GPU sim, one jump per goal):
per goal (x, h) the landing's |x error| as a colour, a fall as an X, goals outside the plane bank's valid hull grey;
success (no fall, |ex| <= 5 cm, |ez| <= 3 cm) counted over the hull goals only, the same goals for every network.
-> src/ilc_mjx/figures/plane/plane_grid.png (run with the ilcmjx env)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # ilc_ws/src
_LOG = os.path.normpath(os.path.join(_REPO, '..', 'log', 'dilc'))                                    # run records (not in git)

D = _LOG
OUT = os.path.join(_REPO, "ilc_mjx", "figures", "plane", "plane_grid.png")
PANELS = [("old box network (5 box plans, coilc)", "grid/box_pre_ex1s2_policy_g0_coilc.s0_e1200.json"),
          ("old flat + box network (coilc)", "grid/combo_pre_ex1_policy_g0_coilc.s0_e1200.json"),
          ("plane bank reference alone", "grid/plane_ref.json"),
          ("plane network v6 (it 30)", "grid/v6s1_it30.json"),
          ("plane network v7, c7 plans (it 10)", "grid/v7c45_it10.json"),
          ("v9: c7 plans, clearance, trust region,\nrobots' estimator in the loop (it 20)", "grid/v9s1_it20.json")]

# the valid region: the plane bank's plans' hull (barycentric weights all >= 0)
sys.path.insert(0, os.path.join(_REPO, "ilc_quad", "scripts"))
bank = json.load(open(os.path.join(D, "plane/bank.json")))
P = np.array([p["goal"] for p in bank["plans"]], float)


def in_hull(g):
    import itertools
    for t in itertools.combinations(range(len(P)), 3):
        A = np.column_stack([P[t[1]] - P[t[0]], P[t[2]] - P[t[0]]])
        if abs(np.linalg.det(A)) < 1e-12:
            continue
        l = np.linalg.solve(A, np.asarray(g) - P[t[0]])
        if l.min() >= -1e-9 and l.sum() <= 1 + 1e-9:
            return True
    return False


fig, axs = plt.subplots(1, len(PANELS), figsize=(4.2 * len(PANELS), 4.2), sharey=True)
hull_cache = {}
for ax, (title, f) in zip(axs, PANELS):
    rows = json.load(open(os.path.join(D, f)))["rows"]
    G = np.array([r["goal"] for r in rows])
    ins = np.array([hull_cache.setdefault(tuple(np.round(g, 4)), in_hull(g)) for g in G])
    ex = np.array([abs(r["ex"]) for r in rows]) * 100
    fell = np.array([r["fell"] for r in rows])
    ok = np.array([r["ok"] for r in rows])
    xs, hs = np.unique(G[:, 0]), np.unique(G[:, 1])
    M = np.full((len(hs), len(xs)), np.nan)
    for g, e, i_ in zip(G, ex, ins):
        if i_:
            M[np.searchsorted(hs, g[1]), np.searchsorted(xs, g[0])] = min(np.nan_to_num(e, nan=20.0), 20.0)
    dx, dh = (xs[1] - xs[0]) * 100, (hs[1] - hs[0]) * 100
    ax.set_facecolor("#d9d9d9")
    im = ax.imshow(M, origin="lower", cmap="viridis_r", vmin=0, vmax=20, aspect="auto",
                   extent=[xs[0] * 100 - dx / 2, xs[-1] * 100 + dx / 2, hs[0] * 100 - dh / 2, hs[-1] * 100 + dh / 2])
    F = G[fell & ins]
    ax.scatter(F[:, 0] * 100, F[:, 1] * 100, marker="x", c="#d62728", s=40, lw=2)
    ax.set_title(f"{title}\nok {ok[ins].mean():.0%}, fell {fell[ins].mean():.0%} ({ins.sum()} goals)", fontsize=9)
    ax.set_xlabel("goal x (cm)")
axs[0].set_ylabel("box height h (cm)")
cb = fig.colorbar(im, ax=axs, fraction=0.015, pad=0.01)
cb.set_label("|landing x error| (cm, capped at 20)")
fig.suptitle("One network over the 2D goal plane (nominal GPU sim): colour = landing error, X = fall, grey = outside "
             "the valid region", fontsize=11, y=1.06)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print(OUT)
