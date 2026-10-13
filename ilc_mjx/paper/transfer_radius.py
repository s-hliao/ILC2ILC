#!/usr/bin/env python3
"""
transfer_radius.py: how far does a hardware-stage correction carry across goals? (the "nonlinear goal space" criticism, 2026-10-12)

QUADRUPED. Every hardware-stage arm under log/dilc/plane/fixbox/<arm>/<robot>/ with a paired before / after evaluation on the 8
reserved TEST goals (eval_planefinal_start.json / _final.json: 4 episodes per goal, the same seeds before and after) and its
training goals (summary.json args). Per (arm, robot, test goal):
    gain      success after - before (4 paired episodes), |e_x| before - after (cm), landing-cost ratio
    distance  to the nearest FLOWN (training) goal in goal space (x, box height) [m]; the nearest of the SAME contact mode
              (flat: height < 1 cm; box otherwise); the height difference to it
    direction the test goal's signed e_x change vs the nearest flown goal's (its own training jumps, first vs last iteration)
Reference: the flown goals' own change over the stage (first vs last iteration's training jump).
CAR. Every (zero-shot arm, hardware-stage arm) pair from the same network (runs/hw/*_zs vs *_hw24 / *_lap*, no ladder arms):
per car and evaluation plan, the success / RMS e_y change vs the sideslip distance (deg) to the nearest trained plan of the
SAME track, and for the other track.
-> src/paper/quadruped/figures/14_transfer_radius.png, src/paper/car/figures/fig19_transfer_radius.png,
   log/dilc/plane/transfer_radius.json (all rows + the binned tables), printed tables.
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WS = "/home/henry/ilc_ws"
P = f"{WS}/log/dilc/plane"
C = f"{WS}/src/ilc_f1tenth/ilc2real"
ROB = ("real_r1", "real_s1", "real_r4", "real_r5")
ok = lambda x: (not x["fell"]) and abs(x["ex"]) <= 0.05 and abs(x["ez"]) <= 0.03
mode = lambda g: "flat" if g[1] < 0.01 else "box"


def load_eval(p):
    js = json.load(open(p))
    return js.get("episodes", js) if isinstance(js, dict) else js


def arm_goals(arm):
    s = json.load(open(f"{arm}/summary.json"))
    s = s[0] if isinstance(s, list) else s
    g = s.get("args", s).get("goals")
    g = json.loads(g) if isinstance(g, str) else g
    return [tuple(float(v) for v in (x if isinstance(x, (list, tuple)) else (x, 0.0))) for x in g]


def train_jumps(rdir):
    """{goal: [(iteration, ex), ...]} from the stage's own training jumps (it<k>/real/*.npz)."""
    out = {}
    for d in sorted(glob.glob(f"{rdir}/it*/real")):
        k = int(d.split("/it")[-1].split("/")[0])
        for f in glob.glob(f"{d}/*.npz"):
            try:
                z = np.load(f, allow_pickle=True)
                g = tuple(round(float(v), 4) for v in z["goal"])
                out.setdefault(g, []).append((k, float(z["info"][0]), bool(z["info"][3])))
            except Exception:
                pass
    return out


def start_of(arm):
    s = json.load(open(f"{arm}/summary.json"))
    a = (s[0] if isinstance(s, list) else s)
    a = a.get("args", a)
    return (str(a.get("policy")), str(a.get("member")))


def quad():
    rows = []
    starts = {}                                         # (policy, member, robot) -> a pre-stage test evaluation (deterministic,
    for arm in sorted(glob.glob(f"{P}/fixbox/*")):      # the same seeds: shared by every arm that starts from that network)
        try:
            key = start_of(arm)
        except Exception:
            continue
        for r in ROB:
            for p in (f"{arm}/{r}/eval_planefinal_start.json", f"{arm}/{r}/it0/eval_planefinal.json"):
                if os.path.exists(p):
                    starts.setdefault(key + (r,), p)
    for arm in sorted(glob.glob(f"{P}/fixbox/*")):
        try:
            goals = arm_goals(arm)
        except Exception:
            continue
        for r in ROB:
            a0, a1 = f"{arm}/{r}/eval_planefinal_start.json", f"{arm}/{r}/eval_planefinal_final.json"
            if not os.path.exists(a0):               # the pre-stage test evaluation is also it0's
                a0 = f"{arm}/{r}/it0/eval_planefinal.json"
            if not os.path.exists(a0):               # or that of another arm from the same network
                a0 = starts.get(start_of(arm) + (r,), a0)
            if not (os.path.exists(a0) and os.path.exists(a1)):
                continue
            e0, e1 = load_eval(a0), load_eval(a1)
            if len(e0) != len(e1):
                continue
            tj = train_jumps(f"{arm}/{r}")
            tdx = {}                                    # flown goal -> signed e_x change (first vs last iteration)
            for g, js in tj.items():
                js = [j for j in js if not j[2]]
                if len(js) >= 2:
                    js.sort()
                    tdx[g] = js[-1][1] - js[0][1]
                    rows.append(dict(kind="flown", arm=os.path.basename(arm), robot=r, goal=g, d=0.0, d_same=0.0, dh=0.0,
                                     dabs_ex=100 * (abs(js[0][1]) - abs(js[-1][1])), dex=100 * tdx[g], mode=mode(g)))
            by = {}
            for x0, x1 in zip(e0, e1):
                by.setdefault(tuple(round(v, 4) for v in x0["goal"]), []).append((x0, x1))
            for g, pairs in by.items():
                dists = [np.hypot(g[0] - t[0], g[1] - t[1]) for t in goals]
                same = [np.hypot(g[0] - t[0], g[1] - t[1]) for t in goals if mode(t) == mode(g)]
                near = goals[int(np.argmin(dists))]
                near_same = [t for t in goals if mode(t) == mode(g)]
                ns = near_same[int(np.argmin(same))] if near_same else None
                s0 = np.mean([ok(p[0]) for p in pairs])
                s1 = np.mean([ok(p[1]) for p in pairs])
                ex0 = np.mean([p[0]["ex"] for p in pairs if not p[0]["fell"]] or [np.nan])
                ex1 = np.mean([p[1]["ex"] for p in pairs if not p[1]["fell"]] or [np.nan])
                rows.append(dict(kind="test", arm=os.path.basename(arm), robot=r, goal=g, mode=mode(g),
                                 flown_modes="+".join(sorted({mode(t) for t in goals})),
                                 d=float(min(dists)), d_same=float(min(same)) if same else np.nan,
                                 dh=float(abs(g[1] - ns[1])) if ns else np.nan,
                                 dsucc=100 * (s1 - s0), dabs_ex=100 * (abs(ex0) - abs(ex1)), dex=100 * (ex1 - ex0),
                                 near_dex=100 * tdx.get(tuple(round(v, 4) for v in near), np.nan)))
    return rows


def car():
    rows = []
    FIVE = ("real_nom", "real_mass", "real_mu", "real_act", "real_lag")
    beta = lambda n: int(n.split("_b")[-1].split("_")[0])
    track = lambda n: n.split("_b")[0]
    for zs in sorted(glob.glob(f"{C}/runs/hw/*_zs")):
        base = os.path.basename(zs)[:-3]
        if "lad" in base or "fb" in base or base.startswith("quick") or "probe" in base:
            continue
        for hw in sorted(glob.glob(f"{C}/runs/hw/{base}_hw24") + glob.glob(f"{C}/runs/hw/{base}_lap*")):
            if any(t in hw for t in ("_pt", "lad", "_s2", "_l10")):
                continue
            p0, p1 = f"{zs}/eval_big.json", f"{hw}/eval_big.json"
            if not (os.path.exists(p0) and os.path.exists(p1)):
                continue
            args = json.load(open(f"{hw}/args.json"))
            if args.get("fallback_suffix"):
                continue
            goals = args.get("goals") or [f"mocap_{t}_b{b}" for t in ("square2fast", "figfast") for b in (10, 18, 25)]
            e0, e1 = json.load(open(p0)), json.load(open(p1))
            for c in FIVE:
                if c not in e0 or c not in e1:
                    continue
                for n, m1 in e1[c].items():
                    if n not in e0[c] or not isinstance(m1, dict):
                        continue
                    m0 = e0[c][n]
                    same = [abs(beta(n) - beta(g)) for g in goals if track(g) == track(n)]
                    ey0, ey1 = m0.get("rms_ey"), m1.get("rms_ey")
                    rows.append(dict(arm=os.path.basename(hw), car=c, plan=n, track_trained=bool(same),
                                     dbeta=float(min(same)) if same else np.nan, flown=bool(same) and min(same) == 0,
                                     dsucc=100 * (m1["success"] - m0["success"]),
                                     dey=(100 * (ey0 - ey1)) if ey0 is not None and ey1 is not None else np.nan))
    return rows


def binned(rows, key, edges, val):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        v = [r[val] for r in rows if lo <= r[key] < hi and np.isfinite(r[val])]
        if v:
            out.append(dict(bin=f"[{lo:g}, {hi:g})", n=len(v), mean=float(np.mean(v)), sem=float(np.std(v) / np.sqrt(len(v))),
                            improved=float(np.mean(np.asarray(v) > 0))))
    return out


def main():
    Q = quad()
    test = [r for r in Q if r["kind"] == "test" and r["robot"] != "real_r5"]
    flown = [r for r in Q if r["kind"] == "flown" and r["robot"] != "real_r5"]
    edges = [0, 0.03, 0.06, 0.10, 0.20]
    res = dict(quad={})
    print(f"QUADRUPED: {len({r['arm'] for r in test})} arms, {len(test)} (arm, robot, test goal) rows (r1/s1/r4; r5 is capability-bound, apart)")
    print(f"  flown goals (reference): |e_x| gain {np.nanmean([r['dabs_ex'] for r in flown]):+.2f} cm (n={len(flown)}), improved "
          f"{np.mean([r['dabs_ex'] > 0 for r in flown]):.0%}")
    for split, sub in (("all test goals", test), ("flat test goals", [r for r in test if r["mode"] == "flat"]),
                       ("box test goals", [r for r in test if r["mode"] == "box"])):
        for val, lab in (("dsucc", "success gain (points)"), ("dabs_ex", "|e_x| gain (cm)")):
            b = binned(sub, "d", edges, val)
            res["quad"][f"{split} / {lab} / by distance to nearest flown goal"] = b
            print(f"  {split:16s} {lab:22s} by distance to nearest flown goal: " +
                  "  ".join(f"{x['bin']} {x['mean']:+.1f}±{x['sem']:.1f} (n={x['n']}, {x['improved']:.0%} up)" for x in b))
    box = [r for r in test if r["mode"] == "box" and np.isfinite(r["dh"])]
    b = binned(box, "dh", [0, 0.02, 0.05, 0.10, 0.20], "dabs_ex")
    res["quad"]["box test goals / |e_x| gain / by height difference to nearest flown box"] = b
    print("  box test goals |e_x| gain by height difference to the nearest flown BOX: " +
          "  ".join(f"{x['bin']} {x['mean']:+.1f}±{x['sem']:.1f} (n={x['n']})" for x in b))
    pr = [(r["dex"], r["near_dex"]) for r in test if np.isfinite(r["dex"]) and np.isfinite(r["near_dex"])]
    if pr:
        a = np.asarray(pr)
        for split in ("flat", "box"):
            s = np.asarray([(r["dex"], r["near_dex"]) for r in test if r["mode"] == split and np.isfinite(r["dex"]) and np.isfinite(r["near_dex"])])
            if len(s) > 3:
                res["quad"][f"direction r ({split})"] = float(np.corrcoef(s.T)[0, 1])
        res["quad"]["direction r (all)"] = float(np.corrcoef(a.T)[0, 1])
        print(f"  direction transfer: corr(test-goal e_x change, nearest flown goal's e_x change) = {res['quad']['direction r (all)']:+.2f} "
              f"(flat {res['quad'].get('direction r (flat)', np.nan):+.2f}, box {res['quad'].get('direction r (box)', np.nan):+.2f}; n={len(a)})")
    xm = [r for r in test if r["mode"] == "box" and r.get("flown_modes") == "flat"]
    if xm:
        res["quad"]["cross-mode (box test goals, flat goals flown)"] = dict(n=len(xm), dsucc=float(np.mean([r['dsucc'] for r in xm])),
                                                                          dabs_ex=float(np.nanmean([r['dabs_ex'] for r in xm])))
        print(f"  CROSS-MODE: box test goals when only FLAT goals were flown: success gain {np.mean([r['dsucc'] for r in xm]):+.1f} points, "
              f"|e_x| gain {np.nanmean([r['dabs_ex'] for r in xm]):+.2f} cm (n={len(xm)}, arms {sorted({r['arm'] for r in xm})})")
        fm = [r for r in test if r["mode"] == "flat" and r.get("flown_modes") == "flat"]
        print(f"  same-mode in those arms: flat test goals: success gain {np.mean([r['dsucc'] for r in fm]):+.1f}, |e_x| gain "
              f"{np.nanmean([r['dabs_ex'] for r in fm]):+.2f} cm (n={len(fm)})")
    r5 = [r for r in Q if r["kind"] == "test" and r["robot"] == "real_r5"]
    print(f"  r5 (capability-bound): test success gain {np.mean([r['dsucc'] for r in r5]):+.1f} points (n={len(r5)})")

    Cr = car()
    res["car"] = {}
    print(f"CAR: {len({r['arm'] for r in Cr})} hardware-stage arms vs their zero-shot, {len(Cr)} (arm, car, plan) rows")
    for lab, sub in (("same track", [r for r in Cr if r["track_trained"]]),):
        for val, vl in (("dsucc", "success gain (points)"), ("dey", "RMS e_y gain (cm)")):
            b = binned(sub, "dbeta", [0, 1, 5, 12, 30], val)
            res["car"][f"{lab} / {vl} / by sideslip distance (deg)"] = b
            print(f"  {lab:10s} {vl:22s} by sideslip distance to nearest trained plan: " +
                  "  ".join(f"{x['bin']} {x['mean']:+.1f}±{x['sem']:.1f} (n={x['n']})" for x in b))
    ot = [r for r in Cr if not r["track_trained"]]
    if ot:
        print(f"  untrained track: success gain {np.nanmean([r['dsucc'] for r in ot]):+.1f}, RMS e_y gain {np.nanmean([r['dey'] for r in ot]):+.2f} cm (n={len(ot)})")

    # figures
    os.makedirs(f"{WS}/src/paper/quadruped/figures", exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for split, col in (("flat", "#2b6cb0"), ("box", "#c05621")):
        sub = [r for r in test if r["mode"] == split]
        for k, (val, lab) in enumerate((("dsucc", "test success gain (points)"), ("dabs_ex", "test |e_x| gain (cm)"))):
            b = binned(sub, "d", edges, val)
            xs = [0.5 * (edges[i] + edges[i + 1]) * 100 for i in range(len(edges) - 1)][:len(b)]
            ax[k].errorbar(xs, [x["mean"] for x in b], yerr=[x["sem"] for x in b], marker="o", color=col, label=f"{split} test goals", capsize=3)
            ax[k].set_xlabel("distance to nearest flown goal (cm, goal space x / box height)")
            ax[k].set_ylabel(lab)
            ax[k].axhline(0, color="0.6", lw=0.8)
    ax[1].axhline(np.nanmean([r["dabs_ex"] for r in flown]), color="0.2", ls="--", lw=1, label="flown goals themselves")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].legend(frameon=False, fontsize=8)
    if pr:
        a = np.asarray(pr)
        ax[2].scatter(a[:, 1], a[:, 0], s=8, alpha=0.4, color="0.3")
        ax[2].set_xlabel("nearest flown goal: e_x change (cm)")
        ax[2].set_ylabel("test goal: e_x change (cm)")
        ax[2].set_title(f"direction transfer r = {res['quad']['direction r (all)']:+.2f}", fontsize=9)
    fig.suptitle("Quadruped: how far a hardware-stage correction carries (49 arms, r1/s1/r4, paired before/after, 4 episodes per goal)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{WS}/src/paper/quadruped/figures/14_transfer_radius.png", dpi=150)
    if Cr:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        sub = [r for r in Cr if r["track_trained"]]
        for k, (val, lab) in enumerate((("dsucc", "success gain (points)"), ("dey", "RMS e_y gain (cm)"))):
            b = binned(sub, "dbeta", [0, 1, 5, 12, 30], val)
            ax[k].errorbar(range(len(b)), [x["mean"] for x in b], yerr=[x["sem"] for x in b], marker="o", color="#2b6cb0", capsize=3)
            ax[k].set_xticks(range(len(b)))
            ax[k].set_xticklabels([x["bin"] for x in b])
            ax[k].set_xlabel("sideslip distance to nearest trained plan (deg)")
            ax[k].set_ylabel(lab)
            ax[k].axhline(0, color="0.6", lw=0.8)
        fig.suptitle("Car: hardware-stage gain vs distance to the trained plans (same track)", fontsize=9)
        fig.tight_layout()
        fig.savefig(f"{WS}/src/paper/car/figures/fig19_transfer_radius.png", dpi=150)
    res["rows_quad"] = [{k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()} for r in Q]
    res["rows_car"] = Cr
    json.dump(res, open(f"{P}/transfer_radius.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
