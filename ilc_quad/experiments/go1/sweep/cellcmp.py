"""cellcmp.py CELLS DIR:KEY ...: for the cells (lines "task cond"), per variant: falls (total, after
trial 5), rear-foot unloaded ms over the last 5 clean trials (median / max, share never
unloaded), Table I pass count, and per cell falls/max-rear-ms."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ns = {}; exec(open(f"{HERE}/stick.py").read().split("\nfor out in sys.argv")[0], ns)
cells = [tuple(l.split()) for l in open(sys.argv[1]) if l.strip()]
for spec in sys.argv[2:]:
    d, key = spec.split(":")
    rear, falls, late, per = [], 0, 0, []
    for t, c in cells:
        rd = f"{d}/{t}__{key}__{c}/"
        tr = [ns["trial"](p) for p in sorted(glob.glob(rd + "trial_*.npz"))]
        f = sum(r["fell"] for r in tr); falls += f; late += sum(r["fell"] for r in tr[5:])
        v = [r["rear"] for r in tr[-5:] if not r["fell"]]; rear += v
        per.append(f"{t[:6]}/{c[:5]}:{f}/{max(v) if v else float('nan'):.0f}")
    r = np.array(rear)
    print(f"{key:9s} falls {falls:3d} (after trial 5: {late:2d}) | rear ms median {np.nanmedian(r):3.0f} max {np.nanmax(r):4.0f} "
          f"never {np.mean(r < 10):4.0%} | " + " ".join(per))
