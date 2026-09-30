"""rear.py VARIANT_DIR:KEY ...: per variant, over the given sweeps' runs named <task>__<KEY>__<cond>:
rear-foot unloaded time (stick.py rear_ms, last 5 clean trials: median / p90 / max), the share
of those trials with the rear feet never unloaded (< 10 ms), falls, and per cell."""
import glob, os, sys, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ns = {}; exec(open(f"{HERE}/stick.py").read().split("\nfor out in sys.argv")[0], ns)
cells = None
for spec in sys.argv[1:]:
    d, key = spec.split(":")
    per = {}
    for rd in sorted(glob.glob(f"{d}/*__{key}__*/")):
        run = os.path.basename(rd[:-1]); t, c = run.split(f"__{key}__")
        tr = [ns["trial"](p) for p in sorted(glob.glob(rd + "trial_*.npz"))]
        per[(t, c)] = (sum(r["fell"] for r in tr), [r["rear"] for r in tr[-5:] if not r["fell"]])
    if cells is None:
        cells = sorted(per)
    per = {k: v for k, v in per.items() if k in cells}
    allr = np.array([x for _, v in per.values() for x in v])
    falls = sum(f for f, _ in per.values())
    print(f"{key:10s} runs {len(per):2d} falls {falls:3d} | rear unloaded ms median {np.nanmedian(allr):4.0f} "
          f"p90 {np.nanpercentile(allr, 90):4.0f} max {np.nanmax(allr):4.0f} | rear never unloaded {np.mean(allr < 10):4.0%} | "
          + " ".join(f"{t[:6]}/{c[:4]}:{np.nanmax(v) if v else float('nan'):3.0f}{'F'+str(f) if f else ''}" for (t, c), (f, v) in sorted(per.items())))
