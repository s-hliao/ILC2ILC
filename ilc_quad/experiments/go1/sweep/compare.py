"""compare.py BASE_OUT BASE_KEY CAND_OUT CAND_KEY: task x condition comparison of two sweeps
over the conds.txt set -- falls (base>cand, [late falls in the last 5], P/p = passes Table I
or recovers to it, cand/base) and sticking (worst unstick_ms over the last 5, stick.py)."""
import csv, glob, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ns = {}; exec(open(f"{HERE}/stick.py").read().split("\nfor out in sys.argv")[0], ns)


def load(d, key):
    R = {}
    for r in csv.DictReader(open(f"{d}/runs.csv")):
        if f"__{key}__" not in r["run"]:
            continue
        t, c = r["run"].split(f"__{key}__")
        tr = [ns["trial"](p) for p in sorted(glob.glob(f"{d}/{r['run']}/trial_*.npz"))]
        t5 = [x for x in tr[-5:] if not x["fell"]]
        R[(t, c)] = dict(falls=int(r["falls"]), late=int(r["falls_last5"]),
                         ok=r["verdict"] in ("PASS", "RECOVER"),
                         un=max((x["un"] for x in t5), default=np.nan))
    return R


bd, bk, cd, ck = sys.argv[1:5]
B, C = load(bd, bk), load(cd, ck)
for name, R in (("BASE", B), ("CAND", C)):
    v = list(R.values()); u = [x["un"] for x in v]
    print(f"{name}: falls {sum(x['falls'] for x in v)}, runs with falls {sum(x['falls'] > 0 for x in v)}, "
          f"late falls {sum(x['late'] for x in v)} in {sum(x['late'] > 0 for x in v)} runs, "
          f"Table I pass {sum(x['ok'] for x in v)}/{len(v)}, unstick median {np.nanmedian(u):.0f} ms, "
          f"p90 {np.nanpercentile(u, 90):.0f} ms, >300 ms in {sum(x > 300 for x in u)} runs, "
          f"no clean trial in last 5: {sum(np.isnan(x) for x in u)}")
conds = [l.split("|")[0] for l in open(f"{HERE}/../robust/conds.txt") if l.strip()]
tasks = sorted({t for t, _ in B})
print(f"\n{'falls':12s}" + "".join(f"{c[:9]:>11s}" for c in conds))
for t in tasks:
    print(f"{t:12s}" + "".join(
        f"{B[(t, c)]['falls']:>3d}>{C[(t, c)]['falls']:<2d}{('[' + str(C[(t, c)]['late']) + ']') if C[(t, c)]['late'] else '':3s}"
        f"{'P' if C[(t, c)]['ok'] else ' '}{'p' if B[(t, c)]['ok'] else ' '}" for c in conds))
print(f"\n{'unstick ms':12s}" + "".join(f"{c[:9]:>11s}" for c in conds))
for t in tasks:
    print(f"{t:12s}" + "".join(f"{B[(t, c)]['un']:>5.0f}>{C[(t, c)]['un']:<5.0f}" for c in conds))
