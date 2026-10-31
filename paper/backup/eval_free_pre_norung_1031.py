#!/usr/bin/env python3
"""
eval_free.py NAME POLICY [NAME POLICY ...] (2026-10-26): the car evaluation for the sideslip-free study -- every network
on every car (the 8 current + the 6 harsh), on the free goals (the two training tracks' _b0_free plans) and the held-out
track (mocap_squareH_b0_free), 4 runs per (car, plan): spread starts (s0 = k/4 of the lap), a start perturbation
N(0, 0.5), 5 chained laps, sensing noise on, the validation seed 301 (+ k). POLICY: one npz for every car, a directory
of <car>_policy.npz (an adapted arm: each car its own network), or 'lqr' (the plans' LQR).
Ladder rungs (2026-10-27): an adapted arm whose directory has a summary.json with 'rungs' flies each car's plans on
the rung its stage settled on (per track; the held-out track on the more conservative of the two), and pace is scored
against that rung's plan -- the car's ladder protocol (reeval.py). POLICY@STAGEDIR flies POLICY on STAGEDIR's rungs (a
baseline given our rungs). EVAL_FREE_NO_RUNGS=1 restores the old scoring (every arm on the full-speed plans).
Success as everywhere on the car: no crash, RMS e_y <= 10 cm, pace >= 90 %. -> eval_free/<NAME>.json
    source env_tight.sh; NUMBA_NUM_THREADS=1 F1T_PROCS=12 python eval_free.py ...
"""
import json
import os
import sys

import numpy as np

CAR = "/home/henry/ilc_ws/src/ilc_f1tenth/ilc2real"
sys.path.insert(0, CAR)
import bank                                                    # noqa: E402
import real_car as rc                                          # noqa: E402
import policy_runtime as pr                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CARS = ["real_nom", "real_mass", "real_mu", "real_act", "real_lag", "ax:mu=0.9", "ax:mu=0.7", "ax:mu=0.6"]
HARSH = ["harsh_combo", "harsh_steer", "harsh_sense", "harsh_drift"]      # harsh_grip == ax:mu=0.6; harsh_tire dropped (calib1.txt)
# EVAL_FREE_PLAN_SUFFIX (2026-10-27; user: "just make the plan actually driveable"): fly and score one plan variant per
# track for every car, e.g. _mu60 (planned at 0.6 x the nominal friction: drivable by every car of the suite)
SFX = os.environ.get("EVAL_FREE_PLAN_SUFFIX", "")
PLANS = [p + SFX for p in ("mocap_square2fast_b0_free", "mocap_figfast_b0_free", "mocap_squareH_b0_free")]
RUNS, LAPS = 4, 5.0
# EVAL_FREE_SEED (2026-10-29): the evaluation seed (default the validation seed 301; the frozen test pass sets a fresh one);
# EVAL_FREE_CARS: a space-separated car list replacing CARS + HARSH (e.g. mb_car's coverage tiers)
SEED = int(os.environ.get("EVAL_FREE_SEED", "301"))
if os.environ.get("EVAL_FREE_CARS"):
    CARS, HARSH = os.environ["EVAL_FREE_CARS"].split(), []


def load_pol(p):
    # (2026-10-30) policy_runtime's loader: an MLP as before, or RMA (mode rma + rma_adapt.npz) / FADA, which real_car.run flies
    return pr.load_policy(p)


def policy_for(spec, car):
    if spec == "lqr":
        return None
    if os.path.isdir(spec):
        f = os.path.join(spec, f"{car}_policy.npz")
        return load_pol(f) if os.path.exists(f) else None
    return load_pol(spec)


ORDER = ["", "_mu80", "_mu60"]


def rungs_for(spec):
    """{car: {track: suffix}} from spec's stage summary, or None."""
    if os.environ.get("EVAL_FREE_NO_RUNGS") == "1":
        return None
    d = spec.split("@")[1] if "@" in spec else spec
    f = os.path.join(d, "summary.json")
    if not os.path.isdir(d) or not os.path.exists(f):
        return None
    return json.load(open(f)).get("rungs")


def rung_of(rungs, car, pn):
    if rungs is None or car not in rungs:
        return ""
    r = rungs[car]
    track = pn.split("_b0")[0]
    if track in r:
        return r[track]
    return max(r.values(), key=ORDER.index) if r else ""       # held-out track: the more conservative rung


def main():
    args = sys.argv[1:]
    arms = list(zip(args[0::2], args[1::2]))
    ALL = PLANS + ([p + sfx for p in PLANS for sfx in ORDER[1:]] if not SFX else [])
    plans = {p["name"]: p for p in bank.load_bank(ALL)}
    jobs, meta = [], []
    rng = np.random.default_rng(SEED)
    perts = rng.normal(size=(len(CARS + HARSH), len(PLANS), RUNS, 5)) * 0.5
    for name, spec in arms:
        for ci, c in enumerate(CARS + HARSH):
            pol = policy_for(spec.split("@")[0], c)
            rungs = rungs_for(spec)
            if spec.split("@")[0] != "lqr" and pol is None:
                continue
            for pi_, pn0 in enumerate(PLANS):
                pn = pn0 + rung_of(rungs, c, pn0)
                L = float(plans[pn]["length"])
                for k in range(RUNS):
                    jobs.append(dict(car_name=c, plan=plans[pn], policy=pol, laps=LAPS, s0=k / RUNS * L,
                                     pert=perts[ci, pi_, k], seed=SEED + k))
                    meta.append(dict(arm=name, car=c, plan=pn0, rung=pn[len(pn0):], run=k))
    res = rc.run_many(jobs, procs=int(os.environ.get("F1T_PROCS", "12")))
    os.makedirs(os.path.join(HERE, "eval_free"), exist_ok=True)
    for name, _ in arms:
        rows = []
        for m, r in zip(meta, res):
            if m["arm"] != name:
                continue
            lm = rc.lap_metrics(r, plans[m["plan"] + m["rung"]])
            rows.append(dict(m, success=bool(lm.get("success")), failed=bool(lm.get("failed")), crash=lm.get("crash", ""),
                             rms_ey=lm.get("rms_ey"), pace=lm.get("pace"), laps=lm.get("laps")))
        json.dump(rows, open(os.path.join(HERE, "eval_free", f"{name}.json"), "w"), indent=1)
        for grp, cs in (("current 8", CARS), ("harsh 5", HARSH)):
            for held in (False, True):
                rs = [r for r in rows if r["car"] in cs and (r["plan"] == PLANS[2]) == held]
                if rs:
                    print(f"{name:22s} {grp:9s} {'held-out' if held else 'train   '} success {100 * np.mean([r['success'] for r in rs]):5.1f} % "
                          f"crashes {100 * np.mean([r['failed'] for r in rs]):5.1f} %  (n {len(rs)})", flush=True)


if __name__ == "__main__":
    main()
