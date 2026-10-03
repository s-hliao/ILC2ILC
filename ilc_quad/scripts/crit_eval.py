#!/usr/bin/env python3
"""
crit_eval.py SETS... --critic LABEL=PATH[:MEMBER] ...: how well each gradient model predicts the measured
directional derivatives (random directions only) of value_bench.py's sets (TAG -> TAG_base/dirs/pert.npz),
pooled -- correlation and slope, with a bootstrap over base points. Models: the SRB closed-loop co-states
(always), and each critic (critic_truegrad.py npz, or a trainer checkpoint with :MEMBER).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from critic_bench import critic_grad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--critic", nargs="*", default=[])
    ap.add_argument("--gamma", type=float, default=0.99)
    a = ap.parse_args()
    preds, meas, pid = {}, [], []
    for si, tag in enumerate(a.sets):
        b, d, p = np.load(f"{tag}_base.npz"), np.load(f"{tag}_dirs.npz"), np.load(f"{tag}_pert.npz")
        lab = [str(x) for x in d["labels"]]
        rj = [j for j, l in enumerate(lab) if l.startswith("rand")]
        D = d["D"][:, rj]
        m = ((p["Jp"] - p["Jm"]) / (2 * float(p["eps"])))[:, rj]
        Nc = b["act"].shape[1]
        und = (a.gamma ** (Nc - 1 - np.arange(Nc)))[None, :, None]
        models = {"srb": 0.01 * b["g_srb"] / und}
        for spec in a.critic:
            label, _, rest = spec.partition("=")
            path, _, mem = rest.partition(":") if not rest.endswith(".npz") else (rest, "", "")
            models[label] = critic_grad(path, mem, b["obs"], b["act"], a.gamma)
        for k, g in models.items():
            preds.setdefault(k, []).append(np.einsum("pkj,pdkj->pd", g, D))
        meas.append(m)
        pid.append(np.repeat(np.arange(len(m))[:, None] + 1000 * si, m.shape[1], 1))
    M = np.concatenate([x.ravel() for x in meas])
    I = np.concatenate([x.ravel() for x in pid])
    ok = np.isfinite(M)
    rng = np.random.default_rng(0)
    pts = np.unique(I[ok])
    print(f"{len(pts)} base points, {ok.sum()} measured directional derivatives")
    for k, v in preds.items():
        P = np.concatenate([x.ravel() for x in v])
        r = np.corrcoef(P[ok], M[ok])[0, 1]
        boots = []
        for _ in range(200):
            s = rng.choice(pts, len(pts))
            sel = np.concatenate([np.flatnonzero(ok & (I == q)) for q in s])
            boots.append(np.corrcoef(P[sel], M[sel])[0, 1])
        slope = P[ok] @ M[ok] / max(P[ok] @ P[ok], 1e-12)
        print(f"  {k:14s} r {r:.3f} [{np.percentile(boots, 5):.2f}, {np.percentile(boots, 95):.2f}]  slope {slope:.2f}")


if __name__ == "__main__":
    main()
