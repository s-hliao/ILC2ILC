#!/usr/bin/env python3
"""
fada_collect_cpu.py --source DEPLOY_DIR --robot NAME --out DIR: FADA's planner / IDM data collected ON THE CPU ROBOT itself
(the exact-dynamics oracle, user 2026-10-11: "make true dynamics the true robot for rma and fada, i want EXACT MATCHES to
the transferred dynamics"; runbook log/dilc/plane/match/EXACT_ORACLES_RUNBOOK.md, step Q2). fada_train.py's GPU collection
draws the same data at only the robot's 8 dynamics parameters; here every jump is flown by the executor's serve workers on
the robot's own condition string (delays, drops, mocap, sensors, compliant ground, the controller's estimator), and each
rollout's echoed condition is asserted.

  planner data  the source flown deterministically, a stance exploration per jump (the planar joints [F thigh, F calf,
                R thigh, R calf] offset ~ N(0, explore[0]) rad, the executor's stand_offset -- jump.JumpEnv.explore_noise's
                q_offset)
  IDM data      the same plus the source's tanh-Gaussian action noise, sigma explore[1] (pre-tanh, i.i.d. per sample: the
                serve protocol has no smooth offsets, which the GPU version uses)
  goals         the 16-goal training grid of quad_fada_true.sh, cycled (never a reserved test goal)
-> DIR/data.npz (Po Pt Io It Ia Im: what fada_train.py --load-data reads) and DIR/collect.json.
Run from src/ilc_mjx with the ilcmjx env; the ilc_quad container must be up.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "ilc_quad", "scripts"))
import dilc_execute as dx                                       # noqa: E402  (no ROS at import)
from ilc_mjx.serve_workers import Workers, check_cond           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, help="deploy dir (policy.npz + bank.json) of the privileged source on this robot")
ap.add_argument("--robot", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--jumps", type=int, default=6400, help="half for the planner, half for the IDM")
ap.add_argument("--workers", type=int, default=28)
ap.add_argument("--batch", type=int, default=224)
ap.add_argument("--explore", type=float, nargs=2, default=[0.08, 0.15])
ap.add_argument("--horizon", type=int, default=1)
ap.add_argument("--goals", nargs="+", default="0.45 0.50 0.55 0.60 0.50,0.05 0.55,0.05 0.60,0.05 0.50,0.10 0.55,0.10 "
                "0.60,0.10 0.50,0.15 0.55,0.15 0.60,0.15 0.50,0.20 0.55,0.20 0.60,0.20".split())
ap.add_argument("--overlay", default="/ilc_ws/log/dilc/plane/fixbox/overlay")
ap.add_argument("--seed", type=int, default=7070)
a = ap.parse_args()

OD = 31
TEST = [(0.4875, 0.0), (0.6125, 0.0), (0.52, 0.12), (0.54, 0.14), (0.57, 0.11), (0.51, 0.17), (0.4875, 0.035), (0.6375, 0.09)]
gvec = lambda g: tuple(float(v) for v in g.split(",")) if "," in g else (float(g), 0.0)
GOALS = [gvec(g) for g in a.goals]
assert not set(GOALS) & set(TEST), "a training goal is a reserved test goal"
BANK = json.load(open(os.path.join(a.source, "bank.json")))
NDC, NSC = BANK["phases"][:2]
NC = NDC + NSC
MASK = np.ones((NC, 4), np.float32)
MASK[NDC:, :2] = 0.0
COND = dx.NAMED[a.robot]


def actor_arrays(sigma):
    d = dict(np.load(os.path.join(a.source, "policy.npz")))
    arr = {k: np.asarray(d[k], np.float64) for k in ("W0", "b0", "W1", "b1", "W_mu", "b_mu")}
    assert arr["W0"].shape[1] == OD, f"source reads {arr['W0'].shape[1]} inputs, expected the deployable {OD} (fold z first)"
    arr.update(W_ls=np.zeros_like(arr["W_mu"]), b_ls=np.full(arr["W_mu"].shape[0], np.log(sigma)), n_hidden=np.array(2))
    return arr


def main():
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    rng = np.random.default_rng(a.seed)
    wk = Workers(a.workers, os.path.join(a.source, "bank.json"), a.overlay)
    data = {k: [] for k in ("Po", "Pt", "Io", "It", "Ia", "Im")}
    stats = dict(robot=a.robot, cond=COND, flown={}, kept={}, falls={}, short_obs=0, failed_rollouts=0)
    t0, n_goal = time.time(), 0
    try:
        for kind in ("planner", "idm"):
            arr = actor_arrays(a.explore[1] if kind == "idm" else 1e-6)
            n = a.jumps // 2
            stats["flown"][kind] = stats["kept"][kind] = stats["falls"][kind] = 0
            while stats["flown"][kind] < n:
                jobs = []
                for k in range(min(a.batch, n - stats["flown"][kind])):
                    g = GOALS[n_goal % len(GOALS)]
                    n_goal += 1
                    q = rng.normal(0.0, a.explore[0], 4)
                    cond = COND + " --param stand_offset:=[" + ",".join(f"{x:.4f}" for x in q) + "]"
                    jobs.append((dict(goal=list(g), seed=int(rng.integers(1, 1 << 30)), cond=cond,
                                      stochastic=kind == "idm", id=len(jobs)), arr))
                res = wk.run(jobs)
                if not any(h is not None and h.get('ok') for h, _ in res):
                    raise RuntimeError('every rollout failed (workers dead?)')
                stats["flown"][kind] += len(jobs)
                for h, ep in res:
                    if h is None or not h.get("ok") or ep is None:
                        stats["failed_rollouts"] += 1
                        continue
                    check_cond(h, COND, prefix=True)
                    if "first_header" not in stats:
                        stats["first_header"] = h
                    if float(ep["fell"]):
                        stats["falls"][kind] += 1
                        continue
                    o = np.asarray(ep["obs"], np.float32)[:, :OD]
                    act = np.asarray(ep["act"], np.float32)[:NC]
                    if not (np.isfinite(o).all() and np.isfinite(act).all()) or len(act) < NC:
                        stats["failed_rollouts"] += 1
                        continue
                    last = len(o) - 1
                    if last < NC:
                        stats["short_obs"] += 1           # no landing row: the horizon index is clamped to the last row
                    o_k = o[:NC]
                    e_n = o[np.minimum(np.minimum(np.arange(NC) + a.horizon, NC), last), :6]
                    if kind == "planner":
                        data["Po"].append(o_k)
                        data["Pt"].append(e_n)
                    else:
                        data["Io"].append(o_k)
                        data["It"].append(e_n)
                        data["Ia"].append(np.clip(act, -0.999, 0.999))
                        data["Im"].append(MASK)
                    stats["kept"][kind] += 1
                print(json.dumps(dict(kind=kind, flown=stats["flown"][kind], kept=stats["kept"][kind],
                                      falls=stats["falls"][kind], t=round(time.time() - t0))), flush=True)
    finally:
        wk.close()
    np.savez(os.path.join(a.out, "data.npz"), **{k: np.concatenate(v).astype(np.float32) for k, v in data.items()})
    stats["seconds"] = time.time() - t0
    json.dump(stats, open(os.path.join(a.out, "collect.json"), "w"), indent=1)
    print(json.dumps(dict(done=True, **{k: v for k, v in stats.items() if k != "first_header"})), flush=True)


if __name__ == "__main__":
    main()
