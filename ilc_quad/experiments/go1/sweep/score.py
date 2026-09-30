"""score.py OUT: every run in OUT (<run>.json + <run>/trial_*.npz) scored per trial on the
TRUE state (robust_report.trial: whole-body CoM/pitch at t = N dt, falls = tilt past
0.6 rad after the jump window), next to what the ILC saw and did. Writes OUT/trials.csv
(one row per trial) and OUT/runs.csv (one per run), and prints a line per run.

A run "passes" as in the paper's Table I on its last trial (no falls anywhere, |ex|<=1 cm,
|ey|<=2 cm, |eθ|<=2°); "recovers" if it fell at least once and its last trial passes with no fall in
the final 2 -- the paper's behaviour this sweep is about. STILL_FALLING: a fall in the
last 5; SETTLED_MISS: falls only earlier, but the last trial misses; NOFALL_MISS."""
import csv, glob, json, math, os, sys
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "robust"))
sys.argv, _argv = sys.argv[:1], sys.argv          # robust_report runs its report on import
import robust_report as rr                         # noqa: E402
sys.argv = _argv


def score_run(jf):
    run = jf[:-5]; name = os.path.basename(run)
    try:
        hist = json.load(open(jf))["history"]
    except Exception:
        hist = []
    files = sorted(glob.glob(f"{run}/trial_*.npz"))
    rows = []
    for k, p in enumerate(files):
        try:
            t = rr.trial(p)
        except Exception as e:                      # noqa: BLE001
            t = dict(e=(np.nan,) * 3, fell=True, nose=np.nan, rear=np.nan, creep=np.nan)
        h = hist[k] if k < len(hist) else {}
        sol = h.get("solver") or {}
        rows.append(dict(run=name, trial=k + 1, stage=h.get("stage"),
                         ex=t["e"][0], ey=t["e"][1], eth=t["e"][2], fell=int(bool(t["fell"])),
                         ilc_fell=int(bool(h.get("fell"))), ilc_pos=100 * h.get("pos_err", np.nan),
                         ilc_th=math.degrees(h.get("theta_err", np.nan)),
                         measured_until=h.get("measured_until"), clearance=100 * h.get("clearance", np.nan),
                         rejected=int(bool(h.get("rejected"))), qu3_scale=h.get("qu3_scale"),
                         qp_ok=int(bool(sol.get("success", True))), slack=sol.get("slack_max"),
                         nose=t["nose"], rear=t["rear"]))
    return name, rows


def summarize(name, rows):
    n = len(rows)
    if not n:
        return dict(run=name, n=0, falls=0, verdict="NO_TRIALS")
    fell = [r["trial"] for r in rows if r["fell"]]
    last = rows[-1]
    err = lambda r: max(abs(r["ex"]), abs(r["ey"]) / 2, abs(r["eth"]))      # cm-ish, Table I scaled
    ok = lambda r: not r["fell"] and abs(r["ex"]) <= 1 and abs(r["ey"]) <= 2 and abs(r["eth"]) <= 2
    good = lambda r: not r["fell"] and abs(r["ex"]) <= 3 and abs(r["ey"]) <= 3 and abs(r["eth"]) <= 3
    tail = rows[-5:]
    passed = not fell and ok(last)
    # fell, then learned: the last trial passes and nothing fell in the final 2 (a run that
    # converged stops early, so the tail can be short)
    recovers = bool(fell) and ok(last) and not any(r["fell"] for r in rows[-2:])
    first_ok = next((r["trial"] for r in rows if ok(r)), None)
    verdict = ("PASS" if passed else "RECOVER" if recovers else
               "STILL_FALLING" if fell and fell[-1] > n - 5 else "NOFALL_MISS" if not fell else "SETTLED_MISS")
    return dict(run=name, n=n, falls=len(fell), first_fall=fell[0] if fell else "",
                last_fall=fell[-1] if fell else "", falls_last5=sum(r["fell"] for r in tail),
                first_ok=first_ok or "", t1_err=err(rows[0]), last_err=err(last),
                best_err=min(err(r) for r in rows if not r["fell"]) if len(fell) < n else np.nan,
                ex=last["ex"], ey=last["ey"], eth=last["eth"],
                rejected=sum(r["rejected"] for r in rows), qp_fail=sum(1 - r["qp_ok"] for r in rows),
                max_qu3=max([r["qu3_scale"] or 1 for r in rows]),
                passed=int(passed), recovers=int(recovers), verdict=verdict)


if __name__ == "__main__":
    out = sys.argv[1]
    jfs = sorted(f for f in glob.glob(f"{out}/*.json"))
    with Pool(min(len(jfs), os.cpu_count() or 1) or 1) as pool:
        res = pool.map(score_run, jfs)
    trials, runs = [], []
    for name, rows in res:
        trials += rows; runs.append(summarize(name, rows))
    for fn, data in (("trials.csv", trials), ("runs.csv", runs)):
        if data:
            keys = list(dict.fromkeys(k for d in data for k in d))
            with open(f"{out}/{fn}", "w", newline="") as f:
                w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(data)
    for s in runs:
        if not s["n"]:
            print(f"{s['run']:44s} NO TRIALS"); continue
        print(f"{s['run']:44s} {s['verdict']:12s} n={s['n']:2d} falls={s['falls']:2d} "
              f"(first {s['first_fall']!s:>2}, last {s['last_fall']!s:>2}, last5 {s['falls_last5']}) "
              f"t1 {s['t1_err']:5.1f} best {s['best_err']:5.1f} last {s['last_err']:5.1f} | "
              f"{s['ex']:+5.1f} {s['ey']:+5.1f} {s['eth']:+5.1f}° rej={s['rejected']} qpfail={s['qp_fail']}")
    v = [s["verdict"] for s in runs]
    print(f"{len(runs)} runs: " + ", ".join(f"{k} {v.count(k)}" for k in sorted(set(v))))
