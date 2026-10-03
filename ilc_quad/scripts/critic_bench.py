#!/usr/bin/env python3
"""
critic_bench.py dirs|score ...: value_bench.py's host side -- the critics' action gradients along the
policy's base jumps, the directions to fly, and the scores.

  dirs   BASE.npz OUT_DIRS.npz --critic CKPT:MEMBER[,...]: per base point the directions
         [SRB model, critic 1, ..., random x --n-random], each normalized to rms --rms per action
         element (contact samples only); the models' predicted dJ/da saved alongside
  score  BASE.npz DIRS.npz PERT.npz: for every model, predicted against measured dJ/d eps along the
         random directions (correlation, slope -- the gradient's accuracy where nothing favours it)
         and along its own direction (does stepping against it lower J, by how much)

Units: J the landing score (0.01 r_scale e'Qe e, the eval score); a the normalized action. The
critic's Q at sample k is -gamma^(Nc-1-k) J (the landing reward on the last transition), the SRB's
closed-loop co-states are discounted the same way; both are undiscounted here.
"""
import argparse
import json
import os
import sys

import numpy as np


def critic_grad(ckpt, member, obs, act, gamma):
    import torch
    if ckpt.endswith(".npz"):                         # critic_truegrad.py's twins
        z = np.load(ckpt)
        W = [torch.tensor(np.stack([z[f"W{i}_0"], z[f"W{i}_1"]])) for i in range(3)]
        B = [torch.tensor(np.stack([z[f"b{i}_0"], z[f"b{i}_1"]]))[:, None] for i in range(3)]
        mi = 0
    else:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        names = [n for n, _ in ck["variants"]]
        mi = names.index(member)
        W = [ck["critic"][f"ls.{i}.W"] for i in range(3)]
        B = [ck["critic"][f"ls.{i}.b"] for i in range(3)]
    P, Nc1, od = obs.shape
    Nc = act.shape[1]
    o = torch.tensor(obs[:, :Nc], dtype=torch.float32)
    o[..., 9:] = 0.0                                   # the members without history see zeros there
    a = torch.tensor(act, dtype=torch.float32, requires_grad=True)
    x = torch.cat([o, a], -1).reshape(-1, od + act.shape[-1])
    qs = []
    for t in (2 * mi, 2 * mi + 1):
        h = torch.tanh(x @ W[0][t] + B[0][t][0])
        h = torch.tanh(h @ W[1][t] + B[1][t][0])
        qs.append((h @ W[2][t] + B[2][t][0])[:, 0])
    q = torch.minimum(qs[0], qs[1]).reshape(P, Nc)
    g = torch.autograd.grad(q.sum(), a)[0].numpy()        # dQ/da
    k = np.arange(Nc)
    return -g / (gamma ** (Nc - 1 - k))[None, :, None]   # dJ/da, undiscounted


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("dirs", "score"))
    ap.add_argument("base")
    ap.add_argument("dirs")
    ap.add_argument("pert", nargs="?")
    ap.add_argument("--critic", nargs="*", default=[], help="LABEL=CKPT:MEMBER")
    ap.add_argument("--bank", default="/home/henry/ilc_ws/log/dilc/off10/bank.json")
    ap.add_argument("--n-random", type=int, default=4)
    ap.add_argument("--rms", type=float, default=0.5)
    a = ap.parse_args()
    bank = json.load(open(a.bank))
    gamma = float(bank.get("gamma", 0.99))
    b = np.load(a.base)
    Nc = b["act"].shape[1]
    Ndc = bank["phases"][0]
    mask = np.ones((Nc, 4))
    mask[Ndc:, :2] = 0.0
    k = np.arange(Nc)
    srb = 0.01 * b["g_srb"] / (gamma ** (Nc - 1 - k))[None, :, None]   # dJ/da, undiscounted
    if a.mode == "dirs":
        models = {"srb": srb * mask}
        if "g_sec" in b.files:
            models["srb_sec"] = 0.01 * b["g_sec"] / (gamma ** (Nc - 1 - k))[None, :, None] * mask
        for spec in a.critic:
            label, _, rest = spec.partition("=")
            ck, _, mem = rest.partition(":")
            models[label] = critic_grad(ck, mem, b["obs"], b["act"], gamma) * mask
        rng = np.random.default_rng(0)
        P = b["act"].shape[0]
        n_el = mask.sum()
        norm = lambda d: d * a.rms * np.sqrt(n_el) / max(np.linalg.norm(d), 1e-12)
        D, labels = [], list(models) + [f"rand{i}" for i in range(a.n_random)]
        for p in range(P):
            ds = [norm(models[m][p]) for m in models]
            ds += [norm(rng.standard_normal((Nc, 4)) * mask) for _ in range(a.n_random)]
            D.append(np.stack(ds))
        np.savez(a.dirs, D=np.stack(D), labels=np.array(labels),
                 **{f"grad_{m}": v for m, v in models.items()})
        print(f"dirs: {P} points x {len(labels)} directions {labels}")
        return
    z = np.load(a.dirs)
    pr = np.load(a.pert)
    D, labels = z["D"], list(z["labels"])
    meas = (pr["Jp"] - pr["Jm"]) / (2 * float(pr["eps"]))             # dJ/d eps measured
    models = [l for l in labels if not l.startswith("rand")]
    rand = [j for j, l in enumerate(labels) if l.startswith("rand")]
    print(f"{b['cond']}: {D.shape[0]} base points (J0 mean {np.nanmean(b['J0']):.2f}), eps {float(pr['eps'])}")
    print(f"{'model':10s} {'r (random dirs)':>16s} {'slope':>7s} {'r (all dirs)':>13s} | own direction: measured dJ/deps, "
          f"J0 -> J(-eps d)")
    for m in models:
        G = z[f"grad_{m}"]
        pred = np.einsum("pkj,pdkj->pd", G, D)
        ok = np.isfinite(meas[:, rand])
        x, y = pred[:, rand][ok], meas[:, rand][ok]
        r = np.corrcoef(x, y)[0, 1]
        slope = float(x @ y / max(x @ x, 1e-12))
        okall = np.isfinite(meas)
        r_all = np.corrcoef(pred[okall], meas[okall])[0, 1]
        j = labels.index(m)
        own = meas[:, j]
        drop = b["J0"] - pr["Jm"][:, j]
        print(f"{m:10s} {r:16.2f} {slope:7.2f} {r_all:13.2f} | {np.nanmean(own):+.2f} (>0 for {np.mean(own[np.isfinite(own)] > 0):.0%}), "
              f"J {np.nanmean(b['J0']):.2f} -> {np.nanmean(pr['Jm'][:, j]):.2f} (drop {np.nanmean(drop):+.2f})")
    rd = np.nanmean(b["J0"][:, None] - pr["Jm"][:, rand])
    print(f"{'random':10s} {'':16s} {'':7s} {'':13s} | J {np.nanmean(b['J0']):.2f} -> {np.nanmean(pr['Jm'][:, rand]):.2f} (drop {rd:+.2f})")


if __name__ == "__main__":
    main()
