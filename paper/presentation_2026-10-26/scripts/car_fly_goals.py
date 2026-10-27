#!/usr/bin/env python3
"""
car_fly_goals.py (2026-10-26): goal conditioning on the car -- ONE car's network of every stage iteration (its/it<k>)
flown on 6 goals: the two tracks (square, figure-8) x drift angle 0 / 18 / 25 degrees (the stage trained on the 25-degree
plans; the others are the anchor plans). 2.05 laps from s0 = 0, no start perturbation, seed 9001 + 100 k.
-> car/goals_trials_<car>.npz (x_true per run + plan geometry) for video_car_goals.py.
    source env_tight.sh; ~/miniconda3/envs/f1t/bin/python car_fly_goals.py CAR [RUN]
"""
import json
import os
import sys

import numpy as np

CAR = "/home/henry/ilc_ws/src/ilc_f1tenth/ilc2real"
sys.path.insert(0, CAR)
import bank                                                    # noqa: E402
import real_car as rc                                          # noqa: E402

car = sys.argv[1]
RUN = sys.argv[2] if len(sys.argv) > 2 else "v3drBtL_s0_ladR_val_its"
D = f"/home/henry/ilc_ws/log/dilc/flows/car/runs/{RUN}"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "car")
GOALS = ["mocap_square2fast_b0", "mocap_square2fast_b18", "mocap_square2fast_b25",
         "mocap_figfast_b0", "mocap_figfast_b18", "mocap_figfast_b25"]


def load_pol(p):
    d = np.load(p)
    n = len([k for k in d.files if k.startswith("W")])
    return [(np.asarray(d[f"W{i}"]), np.asarray(d[f"b{i}"])) for i in range(n)]


def main():
    S = json.load(open(f"{D}/summary.json"))
    its = json.load(open(f"{D}/its/its.json"))
    plans = {p["name"]: p for p in bank.load_bank(GOALS)}
    jobs, meta = [], []
    for k in sorted(int(q) for q in its):
        f = f"{D}/its/it{k}_{car}_policy.npz"
        if not os.path.exists(f):
            continue
        pol = load_pol(f)
        for gname in GOALS:
            jobs.append(dict(car_name=car, plan=plans[gname], policy=pol, laps=S["args"]["trial_laps"] + 0.05, s0=0.0,
                             pert=np.zeros(5), seed=S["args"]["seed"] + 100 * k))
            meta.append(dict(car=car, it=k, plan=gname))
    res = rc.run_many(jobs, procs=int(os.environ.get("F1T_PROCS", "12")))
    out = {}
    for i, (m, r) in enumerate(zip(meta, res)):
        lm = rc.lap_metrics(r, plans[m["plan"]]) if not r["failed"] else {}
        m.update(failed=bool(r["failed"]), crash=r.get("crash", ""), laps=float(r["laps"]), rms_ey=lm.get("rms_ey"))
        out[f"x_{i}"] = np.asarray(r["x_true"])[:, :6]
        print(m["it"], m["plan"], "crash" if m["failed"] else f"{100 * m['rms_ey']:.1f} cm", flush=True)
    for n, p in plans.items():
        out[f"plan__{n}__xy"] = np.asarray(p["x"])[:, :2]
    out["meta"] = json.dumps(meta)
    out["dt"] = 0.025
    tag = car.replace(":", "").replace("=", "")
    np.savez(f"{OUT}/goals_trials_{tag}.npz", **out)
    print("saved", f"{OUT}/goals_trials_{tag}.npz")


if __name__ == "__main__":
    main()
