#!/usr/bin/env python3
"""inout_matched_data.py (2026-10-29): aggregates inout_matched.py (runs/inout_matched/<arm>_s<k>.json) into the data of
paper/final/pogo_inout/inout_figs.py: per arm and goal set (IN / HULL / OUT) the mean normalized error (table_cont.run_err)
with a 95 % cluster-bootstrap CI over (seed, stick, goal), as table_cont.boot.
    python inout_matched_data.py   -> ../../src/paper/final/pogo_inout/inout_matched.json
"""
import json
import os

import numpy as np

from table_cont import boot, run_err

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "runs", "inout_matched")
OUT = os.path.join(HERE, "..", "..", "src", "paper", "final", "pogo_inout", "inout_matched.json")


def rows(arm, gset):
    out = []
    for s in range(3):
        f = os.path.join(SRC, f"{arm}_s{s}.json")
        if not os.path.exists(f):
            continue
        for stick, rs in json.load(open(f))["runs"][gset].items():
            for r in rs:
                n = run_err(r)[0]
                out.append((s, stick, tuple(r["goal"]), n))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    arms = sorted({f.rsplit("_s", 1)[0] for f in os.listdir(SRC) if f.endswith(".json")})
    D = {"goal_sets": {}, "arms": {}}
    for arm in arms:
        for g in ("IN", "HULL", "OUT"):
            rs = rows(arm, g)
            if rs:
                m, lo, hi = boot(rs, rng)
                D["arms"].setdefault(arm, {})[g] = [float(m), float(lo), float(hi), len({r[0] for r in rs})]
    some = json.load(open(os.path.join(SRC, os.listdir(SRC)[0])))
    D["goal_sets"] = {g: len(v) for g, v in some["goal_sets"].items()}
    D["notes"] = ("table protocol (real_pogo.evaluate, 4 perturbed drops x 10 s), normalized error max(apex / 2 cm, |dv| / 0.1 "
                  "m/s), a fall = 5; 6 sticks, 3 seeds; [mean, lo, hi, seeds]. Made by ~/ilc_ws/log/pogo_servo/inout_matched_data.py.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(D, open(OUT, "w"), indent=1)
    for a, d in D["arms"].items():
        print(f"{a:22s} " + "  ".join(f"{g} {v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}] ({v[3]} seeds)" for g, v in d.items()))
    print("->", os.path.abspath(OUT))
