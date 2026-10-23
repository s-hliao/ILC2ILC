#!/usr/bin/env python3
"""
merge_flows.py (2026-10-22): the cross-domain flow figure from the domains' summary.json (SPEC.md schema: per start
policy, field_err / task_err per goal set as [iter][unit]).
- Each curve is the error normalized by its own iteration-0 mean, so the domains' different units share an axis.
- One panel per goal set (IN / HULL / OUT); rows: field error, task error.
- Domains with no summary.json yet are skipped and named in the printout.
    python merge_flows.py [--out fig_flows_all.png] [--starts-like "ours" ...]
Field-error reference per domain: pogo = sim pi_0 (no plan; summary_extra / the goal-flow figure has the
goal-referenced one), car = its report's choice (FLOWS.md), quad = the plan's flow (goal-referenced). See
../../../src/paper/HANDOFF_2026-10-22.md §2 for why the goal-referenced error is the defensible one.
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOMAINS = ("pogo", "car", "quad")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "fig_flows_all.png"))
    ap.add_argument("--starts-like", nargs="*", default=None, help="only start policies containing one of these")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = {}
    for d in DOMAINS:
        f = os.path.join(HERE, d, "summary.json")
        if os.path.exists(f):
            S[d] = json.load(open(f))
        else:
            print(f"{d}: no summary.json yet (run its run_remaining.sh / flows_quad.py aggregate)")
    fig, ax = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    rows = []
    for di, (d, s) in enumerate(S.items()):
        for si, (name, st) in enumerate(s["starts"].items()):
            if a.starts_like and not any(t in name for t in a.starts_like):
                continue
            for r, key in enumerate(("field_err", "task_err")):
                for c, G in enumerate(("IN", "HULL", "OUT")):
                    v = np.asarray(st[key].get(G, []), float)                 # [iter][unit]
                    if v.size == 0 or not np.isfinite(v).any():
                        continue
                    m = np.nanmean(v, 1)
                    if not np.isfinite(m[0]) or m[0] == 0:
                        continue
                    ax[r, c].plot(st["iters"][:len(m)], m / m[0], marker="os^"[di], ls="-:-"[di] if di < 3 else "-",
                                  label=f"{d}: {name}")
                    if key == "field_err":
                        rows.append((d, name, G, m[0], m[-1], m[-1] / m[0]))
    for c, G in enumerate(("IN", "HULL", "OUT")):
        ax[0, c].set_title(G)
        ax[1, c].set_xlabel("hardware-stage iteration")
    ax[0, 0].set_ylabel("field error / iteration 0")
    ax[1, 0].set_ylabel("task error / iteration 0")
    for x in ax.ravel():
        x.axhline(1.0, color="k", lw=0.5)
    ax[0, 2].legend(fontsize=6, ncol=1)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}")
    print(f"{'domain':6s} {'start':34s} {'set':5s} {'it0':>8s} {'final':>8s} {'ratio':>6s}")
    for d, n, G, m0, m1, r in rows:
        print(f"{d:6s} {n[:34]:34s} {G:5s} {m0:8.3f} {m1:8.3f} {r:6.2f}")


if __name__ == "__main__":
    main()
