#!/usr/bin/env python3
"""
interp_check.py: does one hardware-stage update inform the trajectories of NEARBY goals? (2026-10-12: "the step should inform other
(nearby) trajectories, since that is what the entire method relies on, interpolation. check that this is occurring")

For each recorded update of a hardware-stage arm (the per-robot policies before / after it: it<k> -> it<k+1>), deterministic GPU-sim
rollouts at every valid goal of the plane grid, at the robot's own dynamics parameters (dyn_from_cond: the sim cannot model its
delays / sensing -- the update's EFFECT is what is measured here, not the robot's error), no trial noise:
  action level   along each goal's (before-policy) trajectory, the network's change d_pi(g) = pi_after(o, g) - pi_before(o, g)
                 over the contact samples; cos(d_pi(g), d_pi(f)) and |d_pi(g)| / |d_pi(f)| against the nearest FLOWN goal f
  outcome level  the landing change d_e(g) = e_after - e_before (x, z, pitch); sign agreement of d_e_x / d_e_z with f's, and
                 the cosine of the 3-vectors
binned by the goal-space distance to f (cm; x and box height), split by same / other contact mode (flat: height < 1 cm).
-> log/dilc/plane/interp_check.json, src/paper/quadruped/figures/15_interpolation_check.png, printed tables.
    ~/miniconda3/envs/ilcmjx/bin/python interp_check.py --arms fixbox/drB_hw24 [--gpu 1]
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--arms", nargs="+", default=["fixbox/drB_hw24"])
ap.add_argument("--robots", nargs="+", default=["real_r1", "real_s1", "real_r4"])
ap.add_argument("--step", type=float, default=0.025)
ap.add_argument("--every", type=int, default=1, help="use every k-th update (it<k> -> it<k+1>) of each run")
ap.add_argument("--gpu", default="1")
ap.add_argument("--out", default="/home/henry/ilc_ws/log/dilc/plane/interp_check.json")
a = ap.parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "ilc_quad", "scripts"))

import jax                                       # noqa: E402
import jax.numpy as jnp                          # noqa: E402
import numpy as np                               # noqa: E402

import dilc_execute as dx                        # noqa: E402
from ilc_mjx.jump import Config, JumpEnv         # noqa: E402

P = "/home/henry/ilc_ws/log/dilc/plane"
mode = lambda g: "flat" if g[1] < 0.01 else "box"
KEYS = ("W0", "b0", "W1", "b1", "W_mu", "b_mu")


def weights(pdir):
    d = np.load(os.path.join(pdir, "policy.npz"))
    return {k: jnp.asarray(d[k], jnp.float32) for k in KEYS}


def arm_goals(arm):
    s = json.load(open(f"{arm}/summary.json"))
    s = s[0] if isinstance(s, list) else s
    g = s.get("args", s)["goals"]
    g = json.loads(g) if isinstance(g, str) else g
    return [tuple(float(v) for v in (x if isinstance(x, (list, tuple)) else (x, 0.0))) for x in g]


envs = {}


def env_for(pdir):
    bank = json.load(open(os.path.join(pdir, "bank.json")))
    key = json.dumps(bank.get("phases")) + str(len(json.dumps(bank)))
    if key not in envs:
        envs[key] = JumpEnv(bank, Config(est_window=0.025))
    return envs[key]


def run(env, w, goals, dyn):
    ref = env.references(goals)
    o = env.rollout(w, ref, jax.random.PRNGKey(0), stochastic=False, dyn=dyn)
    xr = np.asarray(ref["x_ref"])
    N = env.N
    tg = xr[:, N].copy()
    tg[:, :2] = xr[:, 0, :2] + goals
    tg[:, 2] = 0.0
    e = np.asarray(o["X"])[:, N, :3] - tg[:, :3]
    return dict(e=e, obs=np.asarray(o["obs"]), fell=np.asarray(o["fell"]).astype(bool))


def main():
    rows = []
    actor = jax.jit(jax.vmap(jax.vmap(lambda w, x: JumpEnv.actor(w, x), in_axes=(None, 0)), in_axes=(None, 0)))
    for arm_rel in a.arms:
        arm = os.path.join(P, arm_rel)
        flown = arm_goals(arm)
        for r in a.robots:
            its = sorted(int(d[2:]) for d in os.listdir(f"{arm}/{r}") if d.startswith("it") and os.path.exists(f"{arm}/{r}/{d}/policy/policy.npz"))
            if len(its) < 2:
                continue
            env = env_for(f"{arm}/{r}/it{its[0]}/policy")
            G = env.plane_grid(a.step)
            F = np.array(flown)
            allg = np.concatenate([F, G])
            dyn = env.dyn_arrays(np.tile(np.asarray(JumpEnv.dyn_from_cond(None, dx.NAMED[r]), float), (len(allg), 1)))
            mask = np.ones((env.Nc, 4), np.float32)
            mask[env.Ndc:, :2] = 0.0
            for k0, k1 in list(zip(its[:-1], its[1:]))[::a.every]:
                wb, wa = weights(f"{arm}/{r}/it{k0}/policy"), weights(f"{arm}/{r}/it{k1}/policy")
                b, aa = run(env, wb, allg, dyn), run(env, wa, allg, dyn)
                ob = jnp.asarray(b["obs"][:, :env.Nc], jnp.float32)
                dpi = (np.asarray(actor(wa, ob)) - np.asarray(actor(wb, ob))) * mask       # (B, Nc, 4)
                de = aa["e"] - b["e"]
                ok = ~(b["fell"] | aa["fell"])
                nf = len(F)
                for j in range(nf, len(allg)):
                    g = tuple(allg[j])
                    dists = np.hypot(F[:, 0] - g[0], F[:, 1] - g[1])
                    for split, cand in (("nearest", np.arange(nf)),
                                        ("same_mode", np.array([i for i in range(nf) if mode(flown[i]) == mode(g)]))):
                        if len(cand) == 0:
                            continue
                        f = int(cand[np.argmin(dists[cand])])
                        v, vf = dpi[j].ravel(), dpi[f].ravel()
                        nv, nvf = np.linalg.norm(v), np.linalg.norm(vf)
                        if nvf < 1e-9 or not (ok[j] and ok[f]):
                            continue
                        rows.append(dict(arm=arm_rel, robot=r, update=f"{k0}->{k1}", goal=list(g), flown=list(flown[f]), split=split,
                                         d=float(100 * dists[f]), cross_mode=mode(g) != mode(flown[f]), mode=mode(g),
                                         cos_pi=float(v @ vf / (nv * nvf + 1e-12)), mag=float(nv / nvf),
                                         sign_x=float(np.sign(de[j, 0]) == np.sign(de[f, 0])),
                                         sign_z=float(np.sign(de[j, 1]) == np.sign(de[f, 1])),
                                         cos_e=float(de[j] @ de[f] / (np.linalg.norm(de[j]) * np.linalg.norm(de[f]) + 1e-12)),
                                         de_x=float(100 * de[j, 0]), de_x_f=float(100 * de[f, 0])))
                print(f"{arm_rel} {r} update {k0}->{k1}: {len(G)} grid goals", flush=True)
    edges = [0, 3, 6, 10, 15, 25]
    out = dict(rows=rows, tables={})
    print("\nper bin of distance to the nearest flown goal (cm): mean cos(d_pi), |d_pi| ratio, landing-change sign agreement x / z, cos(d_e)")
    for split, sel in (("nearest flown goal, same mode", lambda r: r["split"] == "nearest" and not r["cross_mode"]),
                       ("nearest flown goal, OTHER mode", lambda r: r["split"] == "nearest" and r["cross_mode"]),
                       ("nearest flown goal of the same mode", lambda r: r["split"] == "same_mode")):
        tab = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = [r for r in rows if sel(r) and lo <= r["d"] < hi]
            if not s:
                continue
            t = dict(bin=f"[{lo}, {hi})", n=len(s), cos_pi=float(np.mean([r["cos_pi"] for r in s])),
                     mag=float(np.median([r["mag"] for r in s])), sign_x=float(np.mean([r["sign_x"] for r in s])),
                     sign_z=float(np.mean([r["sign_z"] for r in s])), cos_e=float(np.mean([r["cos_e"] for r in s])))
            tab.append(t)
        out["tables"][split] = tab
        print(f"  {split}:")
        for t in tab:
            print(f"    {t['bin']:>9s} cm  n={t['n']:4d}  cos(d_pi) {t['cos_pi']:+.2f}  |d_pi| ratio {t['mag']:.2f}  "
                  f"sign agree x {t['sign_x']:.0%} z {t['sign_z']:.0%}  cos(d_e) {t['cos_e']:+.2f}")
    json.dump(out, open(a.out, "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.4))
    for split, col in (("nearest flown goal, same mode", "#2b6cb0"), ("nearest flown goal, OTHER mode", "#c05621")):
        tab = out["tables"].get(split, [])
        xs = [0.5 * (float(t["bin"][1:-1].split(",")[0]) + float(t["bin"][1:-1].split(",")[1])) for t in tab]
        for k, (key, lab) in enumerate((("cos_pi", "cos(network change, flown goal's)"), ("sign_x", "landing x change: same sign as flown"),
                                        ("mag", "|network change| / flown goal's"))):
            ax[k].plot(xs, [t[key] for t in tab], "o-", color=col, label=split.replace("nearest flown goal, ", ""))
            ax[k].set_xlabel("distance to nearest flown goal (cm)")
            ax[k].set_ylabel(lab)
    ax[0].axhline(0, color="0.6", lw=0.8)
    ax[1].axhline(0.5, color="0.6", lw=0.8, ls="--")
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Does one hardware-stage update inform nearby goals? (deterministic sim at each robot's parameters)", fontsize=9)
    fig.tight_layout()
    fig.savefig("/home/henry/ilc_ws/src/paper/quadruped/figures/15_interpolation_check.png", dpi=150)


if __name__ == "__main__":
    main()
