"""report.py OUT KEY CONDS_FILE [OUT2 KEY2]: task x condition table for runs <task>__KEY__<cond>.
Per cell: falls [late = in the last 5 trials], P if the last trial passes Table I (true state,
no fall in the final 2), the rear-unloaded ms worst over the last 5 clean trials, and the last
landing error. With a second sweep, the totals of both."""
import csv, glob, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ns = {}; exec(open(f"{HERE}/stick.py").read().split("\nfor out in sys.argv")[0], ns)


def load(d, key, conds, tasks):
    runs = {r["run"]: r for r in csv.DictReader(open(f"{d}/runs.csv"))}
    R = {}
    for t in tasks:
        for c in conds:
            r = runs.get(f"{t}__{key}__{c}")
            if r is None:
                continue
            tr = [ns["trial"](p) for p in sorted(glob.glob(f"{d}/{t}__{key}__{c}/trial_*.npz"))]
            v = [x["rear"] for x in tr[-5:] if not x["fell"]]
            R[(t, c)] = dict(f=int(r["falls"]), late=int(r["falls_last5"]), ok=r["verdict"] in ("PASS", "RECOVER"),
                             rear=max(v) if v else np.nan, ex=float(r["ex"]), ey=float(r["ey"]), eth=float(r["eth"]),
                             after5=sum(x["fell"] for x in tr[5:]))
    return R


def totals(name, R):
    v = list(R.values()); rr = np.array([x["rear"] for x in v])
    print(f"{name}: {len(v)} runs | Table I pass {sum(x['ok'] for x in v)} | falls {sum(x['f'] for x in v)} "
          f"(runs {sum(x['f'] > 0 for x in v)}), after trial 5 {sum(x['after5'] for x in v)}, late (last 5) {sum(x['late'] for x in v)} "
          f"in {sum(x['late'] > 0 for x in v)} runs | rear-unloaded worst-of-last-5: median {np.nanmedian(rr):.0f} ms, "
          f">100 ms in {int(np.sum(rr > 100))} runs, no clean trial {int(np.sum(np.isnan(rr)))} | "
          f"median |ex| {np.median([abs(x['ex']) for x in v]):.1f} cm, |eθ| {np.median([abs(x['eth']) for x in v]):.1f}°")


d, key, cf = sys.argv[1:4]
conds = [l.split("|")[0].strip() for l in open(cf) if l.strip()]
tasks = [l.split()[0] for l in open(f"{HERE}/../tasks.txt")]
R = load(d, key, conds, tasks)
totals(key, R)
if len(sys.argv) > 5:
    totals(sys.argv[5], load(sys.argv[4], sys.argv[5], conds, tasks))
print(f"\n{'':12s}" + "".join(f"{c[:10]:>15s}" for c in conds))
for t in tasks:
    row = f"{t:12s}"
    for c in conds:
        x = R.get((t, c))
        row += "              -" if x is None else \
            f"{x['f']:>3d}{('[' + str(x['late']) + ']') if x['late'] else '':3s}{'P' if x['ok'] else ' '}{x['rear']:4.0f}ms{abs(x['eth']):3.0f}°"
    print(row)
