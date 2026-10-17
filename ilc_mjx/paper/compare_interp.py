#!/usr/bin/env python3
"""
compare_interp.py: does a hardware-stage update inform nearby goals -- the recipe, its controls and the neighbour-target fix,
side by side (2026-10-17). Arms (log/dilc/plane/fixbox/, r1 / s1 / r4, DR-B start, the pre-stage evaluation = drB_hw24's):
    drB_hw24    the recipe (6 goals x 4 iterations)          ctl_null    --step-rms 0 (the refit alone)
    ctl_random  a random-direction step, same size           nbr_r05     neighbour targets within 5 cm (same mode)
    sg_*        one flown goal, 12 jumps (clean attribution)
Per arm: test success before -> after (8 test goals x 4 episodes per robot), the test goals' |e_x| gain, and the interpolation
metrics of interp_check.py (cos of the network change vs the nearest flown goal's, landing-change sign agreement, cos of the
landing changes) at 0-3 and 3-6 cm. Runs interp_check.py on the arms that have no result yet.
-> log/dilc/plane/compare_interp.json, src/paper/quadruped/figures/16_interp_compare.png, a printed table.
"""
import json
import os
import subprocess

import numpy as np

P = "/home/henry/ilc_ws/log/dilc/plane"
HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.expanduser("~/miniconda3/envs/ilcmjx/bin/python")
ROB = ("real_r1", "real_s1", "real_r4")
ok = lambda x: (not x["fell"]) and abs(x["ex"]) <= 0.05 and abs(x["ez"]) <= 0.03
ARMS = [("recipe (drB_hw24)", ["drB_hw24"], f"{P}/interp_check.json"),
        ("single goal (sg_*)", ["sg_f045", "sg_f060", "sg_b050_15", "sg_b060_10"], f"{P}/interp_check_sg.json"),
        ("null: no step", ["ctl_null"], f"{P}/interp_check_null.json"),
        ("null: random step", ["ctl_random"], f"{P}/interp_check_random.json"),
        ("neighbour targets", ["nbr_r05"], f"{P}/interp_check_nbr.json")]


def load(p):
    js = json.load(open(p))
    return js.get("episodes", js) if isinstance(js, dict) else js


def test_scores(arm):
    out = {}
    for r in ROB:
        f0, f1 = f"{P}/fixbox/drB_hw24/{r}/eval_planefinal_start.json", f"{P}/fixbox/{arm}/{r}/eval_planefinal_final.json"
        if not (os.path.exists(f0) and os.path.exists(f1)):
            continue
        e0, e1 = load(f0), load(f1)
        ex0 = np.mean([abs(x["ex"]) for x in e0 if not x["fell"]])
        ex1 = np.mean([abs(x["ex"]) for x in e1 if not x["fell"]])
        out[r] = dict(before=100 * np.mean([ok(x) for x in e0]), after=100 * np.mean([ok(x) for x in e1]), dabs_ex=100 * (ex0 - ex1))
    return out


def interp_metrics(path):
    t = json.load(open(path))["tables"].get("nearest flown goal of the same mode", [])
    return {x["bin"]: x for x in t}


def main():
    res = {}
    for name, arms, ipath in ARMS:
        if not os.path.exists(ipath) and all(os.path.exists(f"{P}/fixbox/{a_}") for a_ in arms):
            subprocess.run([PY, os.path.join(HERE, "interp_check.py"), "--arms", *[f"fixbox/{a_}" for a_ in arms], "--gpu", "2",
                            "--out", ipath], check=False)
        sc = {}
        for a_ in arms:
            for r, v in test_scores(a_).items():
                sc.setdefault(r, []).append(v)
        sc = {r: {k: float(np.mean([v[k] for v in vs])) for k in vs[0]} for r, vs in sc.items()}
        im = interp_metrics(ipath) if os.path.exists(ipath) else {}
        res[name] = dict(test=sc, interp=im)
    print(f"{'arm':22s} | {'test before -> after (r1/s1/r4 mean)':36s} | {'|e_x| gain':10s} | "
          f"{'0-3 cm: cos(d_pi) / sign x / cos(d_e)':40s} | {'3-6 cm: cos(d_pi) / sign x / cos(d_e)':40s}")
    for name, v in res.items():
        t = v["test"]
        b = np.mean([x["before"] for x in t.values()]) if t else np.nan
        a_ = np.mean([x["after"] for x in t.values()]) if t else np.nan
        g = np.mean([x["dabs_ex"] for x in t.values()]) if t else np.nan
        cell = lambda k: (f"{v['interp'][k]['cos_pi']:+.2f} / {v['interp'][k]['sign_x']:.0%} / {v['interp'][k]['cos_e']:+.2f}"
                          f" (n={v['interp'][k]['n']})" if k in v["interp"] else "n/a")
        print(f"{name:22s} | {b:5.1f} -> {a_:5.1f} %  ({', '.join(f'{x['after']:.0f}' for x in t.values())}) | {g:+.2f} cm   | "
              f"{cell('[0, 3)'):40s} | {cell('[3, 6)'):40s}")
    json.dump(res, open(f"{P}/compare_interp.json", "w"), indent=1, default=float)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    names = list(res)
    for k, (key, lab) in enumerate((("cos_pi", "cos(network change, flown goal's)"), ("sign_x", "landing x moves with the flown goal"),
                                    ("cos_e", "cos(landing change, flown goal's)"))):
        for j, b in enumerate(("[0, 3)", "[3, 6)")):
            ax[k].bar(np.arange(len(names)) + 0.38 * j, [res[n]["interp"].get(b, {}).get(key, np.nan) for n in names], 0.38,
                      label=f"{b} cm", color=("#2b6cb0", "#90cdf4")[j])
        ax[k].set_xticks(np.arange(len(names)) + 0.19)
        ax[k].set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=7)
        ax[k].set_ylabel(lab, fontsize=8)
        ax[k].axhline(0.5 if key == "sign_x" else 0.0, color="0.5", lw=0.8, ls="--")
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Does one update inform nearby goals? recipe vs controls vs neighbour targets (same-mode nearest flown goal)", fontsize=9)
    fig.tight_layout()
    fig.savefig("/home/henry/ilc_ws/src/paper/quadruped/figures/16_interp_compare.png", dpi=150)


if __name__ == "__main__":
    main()
