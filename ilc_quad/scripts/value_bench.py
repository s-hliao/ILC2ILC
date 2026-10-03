#!/usr/bin/env python3
"""
value_bench.py: does a value function know the problem's local structure? (container side)

For a deterministic policy pi, adding eps d_k to its action at every sample k (the policy reacting to
the states that follow) changes the jump's landing cost J by, to first order,
    dJ/d eps = sum_k dJ/da_k . d_k = -sum_k grad_a Q(s_k, a_k) . d_k       (Q = -cost, normalized a)
so any model of the action gradient along the policy's own jump can be checked against flown jumps.

  base     the policy jumps at each (goal, seed): its states and actions (the episode's obs, act),
           its landing score J0, and the SRB model's prediction: the closed-loop co-state action
           gradient of the Stage III (landing) cost along the jump (ilc_labels' mca_cl[:, 2], the
           gradient the ILC and the Sobolev labels are built on)              -> OUT.npz
  perturb  each direction of DIRS.npz (per base point, normalized action units) flown at +eps and
           -eps with the base point's seed (common random numbers): J+ and J-, and the landing errors
           E+ and E- (x_N - target, 6)                                           -> OUT.npz

  lin      the base jumps flown again (deterministic: same seeds, same offsets) for the approximate
           model along them: the SRB's A_k, B_k (normalized state error / action, as the episodes
           store them) and the ballistic flight map from takeoff to landing, Phi_n = d e_land /
           d err_Nc (6 x 6, the landing error per normalized takeoff state error)   -> OUT.npz

--offset S (base): the state coverage a nominal policy's jumps lack -- each base jump flies the policy
plus a smooth random action offset (rms S per action element, contact samples; saved as "off"), so its
states and landing errors range as far as a mismatched robot's would. perturb flies the same offset.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


SECANT = ""


def _jump(job):
    import dilc_execute as dx
    mode, policy, member, g, cond, seed, d, eps = job
    os.environ["ROS_DOMAIN_ID"] = str(1 + os.getpid() % 98)
    bank, actor, _, _ = dx._cached(("policy", policy, member), lambda: dx.load_policy(policy, member))
    if d is not None:
        base_actor = actor

        def actor(obs, rng=None):              # the policy plus eps d_k at sample k
            a = base_actor(obs, rng)
            k = int(round((obs[6] + 1.0) * bank.Nc / 2.0))
            return a + eps * d[k] if k < len(d) else a
    ref, box, extrap = bank.reference(g)
    drv, log, rec, fell, ref, _ = dx.run_jump(bank, g, cond, seed, dict(actor=actor, window=bank.cfg["est_window"]),
                                              land_time=1.5, reference=(ref, box, extrap))
    sc, e = dx.landing_score(log["X"], ref["x_ref"], g, fell, np.asarray(bank.cfg["qe"]), bank.cfg["r_scale"])
    out = dict(J=sc, fell=bool(fell), e=np.asarray(e, float))   # e: the landing error x_N - target
    if mode == "lin":
        maker = dx._cached(("maker", policy), lambda: dx.EpisodeMaker(bank))
        ep = maker.make(g, ref, drv.X, drv.U, log["X"], fell, None)
        Af = maker.srb.f_lin(np.zeros(6), np.zeros(4), np.zeros(2), np.zeros(2), bank.dt)[0].full()
        Phi = np.linalg.matrix_power(Af, bank.N - bank.Nc)
        out.update(obs=ep["obs"], A=ep["A"], B=ep["B"], Phi_n=Phi * np.asarray(bank.sx, float)[None, :])
    if mode == "base":
        maker = dx._cached(("maker", policy), lambda: dx.EpisodeMaker(bank))
        ep = maker.make(g, ref, drv.X, drv.U, log["X"], fell, None)
        lab = maker.ilc_labels(ref, drv.U, log["X"], g, ref)
        out.update(obs=ep["obs"], act=ep["act"], g_srb=lab["mca_cl"][:, 2])
        if SECANT:                              # the model corrected by the sim's own trial pairs
            import copy
            b2 = copy.copy(bank)
            b2.cfg = dict(bank.cfg, secant_C=SECANT, secant_full=1)
            m2 = dx.EpisodeMaker(b2)
            out["g_sec"] = m2.ilc_labels(ref, drv.U, log["X"], g, ref)["mca_cl"][:, 2]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("base", "perturb", "lin"))
    ap.add_argument("--policy", required=True)
    ap.add_argument("--member", required=True)
    ap.add_argument("--cond", default="nominal")
    ap.add_argument("--goals", type=float, nargs="+", default=[0.425, 0.475, 0.525, 0.575])
    ap.add_argument("--seeds", type=int, nargs="+", default=[601, 602])
    ap.add_argument("--base", help="perturb: the base npz")
    ap.add_argument("--dirs", help="perturb: npz with D (n_points, n_dirs, Nc, 4)")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--secant", default="", help="base: also the secant-corrected model's gradient")
    ap.add_argument("--offset", type=float, default=0.0, help="base: smooth random action offsets, rms")
    ap.add_argument("--offset-seed", type=int, default=0)
    a = ap.parse_args()
    global SECANT
    SECANT = a.secant
    import dilc_execute as dx
    from multiprocessing import Pool
    if os.path.isfile(a.cond):                    # a conds file (name|args): its first line
        cond = dx.conditions(a.cond)[0][1]
    else:
        cond = dict(dx.conditions(a.cond))[a.cond]
    if a.mode == "base":
        pts = [(g, s) for g in a.goals for s in a.seeds]
        offs = [None] * len(pts)
        if a.offset > 0:
            bank = dx.GoalBank(json.load(open(os.path.join(a.policy, "bank.json"))))
            Nc, Ndc = bank.Nc, bank.Ndc
            rng = np.random.default_rng(a.offset_seed)
            ker = np.exp(-0.5 * (np.arange(-12, 13) / 4.0) ** 2)
            mask = np.ones((Nc, 4)); mask[Ndc:, :2] = 0.0
            offs = []
            for _ in pts:
                z = np.stack([np.convolve(rng.standard_normal(Nc + 24), ker, "valid") for _ in range(4)], 1)[:Nc]
                z = z * mask
                offs.append(z * a.offset / np.sqrt((z ** 2).sum() / mask.sum()))
        jobs = [("base", a.policy, a.member, np.array([g, 0.0]), cond, s, o, 1.0) for (g, s), o in zip(pts, offs)]
        with Pool(a.jobs, maxtasksperchild=4) as pool:
            res = pool.map(dx_safe, jobs)
        np.savez(a.out, goal=np.array([p[0] for p in pts]), seed=np.array([p[1] for p in pts]),
                 J0=np.array([r["J"] for r in res]), fell=np.array([r["fell"] for r in res]),
                 e0=np.stack([r["e"] for r in res]),
                 obs=np.stack([r["obs"] for r in res]), act=np.stack([r["act"] for r in res]),
                 g_srb=np.stack([r["g_srb"] for r in res]), cond=a.cond,
                 **({"off": np.stack(offs)} if a.offset > 0 else {}),
                 **({"g_sec": np.stack([r["g_sec"] for r in res])} if a.secant else {}))
        print(f"base: {len(pts)} jumps, J0 mean {np.mean([r['J'] for r in res]):.3f}")
    elif a.mode == "lin":
        b = np.load(a.base)
        off = b["off"] if "off" in b.files else [None] * len(b["goal"])
        jobs = [("lin", a.policy, a.member, np.array([float(g), 0.0]), cond, int(s_), o, 1.0)
                for g, s_, o in zip(b["goal"], b["seed"], off)]
        with Pool(a.jobs, maxtasksperchild=4) as pool:
            res = pool.map(dx_safe, jobs)
        obs = np.stack([r["obs"] for r in res])
        print(f"lin: {len(res)} jumps, max |obs - base obs| {np.abs(obs - b['obs']).max():.2e}")
        np.savez(a.out, A=np.stack([r["A"] for r in res]), B=np.stack([r["B"] for r in res]),
                 Phi_n=np.stack([r["Phi_n"] for r in res]))
    else:
        b = np.load(a.base)
        D = np.load(a.dirs)["D"]                                  # (P, nd, Nc, 4)
        off = b["off"] if "off" in b.files else np.zeros(D[:, 0].shape)
        jobs, idx = [], []
        for p in range(D.shape[0]):
            for j in range(D.shape[1]):
                for sgn in (1.0, -1.0):
                    jobs.append(("perturb", a.policy, a.member, np.array([float(b["goal"][p]), 0.0]), cond,
                                 int(b["seed"][p]), off[p] + sgn * a.eps * D[p, j], 1.0))
                    idx.append((p, j, sgn))
        with Pool(a.jobs, maxtasksperchild=4) as pool:
            res = pool.map(dx_safe, jobs)
        Jp = np.full(D.shape[:2], np.nan)
        Jm = np.full(D.shape[:2], np.nan)
        Ep = np.full(D.shape[:2] + (6,), np.nan)
        Em = np.full(D.shape[:2] + (6,), np.nan)
        for (p, j, sgn), r in zip(idx, res):
            (Jp if sgn > 0 else Jm)[p, j] = r["J"] if not r["fell"] else np.nan
            (Ep if sgn > 0 else Em)[p, j] = r["e"] if not r["fell"] else np.nan
        np.savez(a.out, Jp=Jp, Jm=Jm, Ep=Ep, Em=Em, eps=a.eps)
        print(f"perturb: {len(jobs)} jumps")


def dx_safe(job):
    try:
        return _jump(job)
    except Exception as err:                       # a failed jump is a missing point, not a crash
        print(f"jump failed: {err!r}", file=sys.stderr)
        return dict(J=np.nan, fell=True, obs=None, act=None, g_srb=None, e=np.full(6, np.nan))


if __name__ == "__main__":
    main()
