#!/usr/bin/env python3
"""
quad_real_ars.py --policy DIR_TEMPLATE --out OUT: Augmented Random Search (Mania et al. 2018, ARS-V1t) on each CPU robot itself --
the exact-dynamics RL oracle when PPO will not hold a good policy (every PPO variant degraded rma_exact7's 37.5 % checkpoint:
it optimizes the stochastic policy, the score flies its mean). ARS optimizes the DETERMINISTIC policy directly:

  per iteration and robot  K antithetic perturbations delta_i of the output layer (W_mu, b_mu; N(0, sigma^2)), each flown
                           deterministically (+delta_i and -delta_i on the SAME m random goals of the training box, never a
                           test goal); return of a jump = -10 log(1 + landing cost / 50) - 40 fall + 10 success (the
                           objective of the PPO runs: rma_exact7's --log-cost 50 --fall-pen 40 --success-bonus 10)
  update                   the top-b directions by max(r+, r-): theta += lr / (b sigma_R) sum (r+ - r-) delta (sigma_R: their
                           returns' std)
Rollouts: the executor's serve workers on the robot's own condition string (serve_workers.check_cond asserts it).
Checkpoints at --checkpoints jumps per robot -> OUT/ck<k>/<robot>/policy (quad_real_ppo.py's layout: its --eval-only scores
them on the reserved test goals).
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "ilc_quad", "scripts"))
import dilc_execute as dx                                       # noqa: E402
from ilc_mjx.serve_workers import Workers, check_cond           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True, help="deploy dir, may hold {robot}")
ap.add_argument("--out", required=True)
ap.add_argument("--robots", nargs="+", default=["real_r1", "real_s1", "real_r5"])
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--dirs", type=int, default=8)
ap.add_argument("--top", type=int, default=4)
ap.add_argument("--goals-per", type=int, default=4)
ap.add_argument("--sigma", type=float, default=0.01)
ap.add_argument("--lr", type=float, default=0.02)
ap.add_argument("--goal-box", type=float, nargs=4, default=[0.45, 0.65, 0.0, 0.20])
ap.add_argument("--checkpoints", type=int, nargs="+", default=[768, 1536, 3072, 5000])
ap.add_argument("--overlay", default="/ilc_ws/log/dilc/plane/fixbox/overlay")
ap.add_argument("--seed", type=int, default=2018)
a = ap.parse_args()
TEST = [(0.4875, 0.0), (0.6125, 0.0), (0.52, 0.12), (0.54, 0.14), (0.57, 0.11), (0.51, 0.17), (0.4875, 0.035), (0.6375, 0.09)]
PDIR = lambda r: a.policy.format(robot=r)
KEYS = ("W0", "b0", "W1", "b1", "W_mu", "b_mu")


def load(r):
    d = dict(np.load(os.path.join(PDIR(r), "policy.npz")))
    return {k: np.asarray(d[k], np.float64) for k in KEYS}


def arrays(w):
    arr = dict(w)
    arr.update(W_ls=np.zeros_like(w["W_mu"]), b_ls=np.full(w["W_mu"].shape[0], np.log(1e-6)), n_hidden=np.array(2))
    return arr


def export(w, d, r):
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "policy.npz"), **arrays(w))
    shutil.copy(os.path.join(PDIR(r), "bank.json"), d)
    json.dump(dict(best="policy", eval_goals=None, dyn_dim=0, members=[dict(name="policy", group=0, variant="policy", tag="policy",
                                                                             episodes=0, updates=0, score=None)]),
              open(os.path.join(d, "manifest.json"), "w"))


def ret(ep):
    rc = np.asarray(ep["rc"], np.float64)
    fell = float(ep["fell"])
    ex, ez = float(ep["info"][0]), float(ep["info"][1])
    ok = (not fell) and abs(ex) <= 0.05 and abs(ez) <= 0.03
    return -10.0 * np.log1p(float(rc[:, 2].sum()) / 50.0) - 40.0 * fell + 10.0 * ok, ok, fell


def main():
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    rng = np.random.default_rng(a.seed)
    theta = {r: load(r) for r in a.robots}
    for r in a.robots:
        export(theta[r], os.path.join(a.out, "ck0", r, "policy"), r)
    wk = Workers(a.workers, os.path.join(PDIR(a.robots[0]), "bank.json"), a.overlay)
    logf = open(os.path.join(a.out, "train.log"), "a")
    jumps, done, it, t0 = {r: 0 for r in a.robots}, {0}, 0, time.time()
    try:
        while min(jumps.values()) < max(a.checkpoints):
            jobs, meta = [], []
            plan = {}
            for r in a.robots:
                goals = [(rng.uniform(a.goal_box[0], a.goal_box[1]), rng.uniform(a.goal_box[2], a.goal_box[3]))
                         for _ in range(a.goals_per)]
                assert not set(goals) & set(TEST)
                deltas = [{k: rng.normal(0.0, a.sigma, theta[r][k].shape) for k in ("W_mu", "b_mu")} for _ in range(a.dirs)]
                plan[r] = deltas
                for i, dl in enumerate(deltas):
                    for sgn in (+1, -1):
                        w = dict(theta[r])
                        for k in dl:
                            w[k] = theta[r][k] + sgn * dl[k]
                        arr = arrays(w)
                        for g in goals:
                            jobs.append((dict(goal=list(g), seed=int(rng.integers(1, 1 << 30)), cond=r, stochastic=False,
                                              id=len(jobs)), arr))
                            meta.append((r, i, sgn))
            res = wk.run(jobs)
            if not any(h is not None and h.get("ok") for h, _ in res):
                raise RuntimeError(f"iteration {it}: every rollout failed (workers dead?)")
            line = dict(it=it + 1, t=round(time.time() - t0))
            for r in a.robots:
                R = {(i, s): [] for i in range(a.dirs) for s in (+1, -1)}
                succ, falls = [], 0
                for (h, ep), (rr, i, s) in zip(res, meta):
                    if rr != r or h is None or not h.get("ok") or ep is None:
                        continue
                    check_cond(h, dx.NAMED[r])
                    v, ok, fell = ret(ep)
                    R[(i, s)].append(v)
                    succ.append(ok)
                    falls += int(fell)
                rp = np.array([np.mean(R[(i, +1)]) if R[(i, +1)] else -1e3 for i in range(a.dirs)])
                rm = np.array([np.mean(R[(i, -1)]) if R[(i, -1)] else -1e3 for i in range(a.dirs)])
                top = np.argsort(-np.maximum(rp, rm))[:a.top]
                sr = float(np.std(np.r_[rp[top], rm[top]])) + 1e-8
                for k in ("W_mu", "b_mu"):
                    step = sum((rp[i] - rm[i]) * plan[r][i][k] for i in top)
                    theta[r][k] = theta[r][k] + a.lr / (a.top * sr) * step
                jumps[r] += sum(1 for m in meta if m[0] == r)
                line[r] = dict(jumps=jumps[r], ret=round(float(np.mean(np.r_[rp, rm])), 2), succ=round(float(np.mean(succ)), 3),
                               falls=falls)
            it += 1
            print(json.dumps(line), flush=True)
            logf.write(json.dumps(line) + "\n")
            logf.flush()
            for k in a.checkpoints:
                if k not in done and min(jumps.values()) >= k:
                    for r in a.robots:
                        export(theta[r], os.path.join(a.out, f"ck{k}", r, "policy"), r)
                    done.add(k)
    finally:
        wk.close()
    json.dump(dict(jumps=jumps, seconds=time.time() - t0), open(os.path.join(a.out, "train_summary.json"), "w"))


if __name__ == "__main__":
    main()
