#!/usr/bin/env python3
"""score_episodes.py DIR [DIR ...]: the jumps policy_jump_node.py saved (or any dilc_execute episodes), scored as every
test here: 0.01 x landing cost + 20 x fall, per goal and overall, with the mean landing error (x, z, pitch)."""
import glob
import os
import sys
from collections import defaultdict

import numpy as np

for d in sys.argv[1:]:
    rows = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        if os.path.basename(f).startswith("ref_"):
            continue
        z = np.load(f)
        sc = 0.01 * float(np.asarray(z["rc"])[:, 2].sum()) + 20.0 * float(z["fell"])
        rows[round(float(z["goal"][0]), 4)].append((sc, *np.asarray(z["info"], float)[:3], float(z["fell"])))
    print(f"{d}:")
    allv = []
    for g, v in sorted(rows.items()):
        v = np.array(v)
        allv.extend(v[:, 0])
        print(f"  goal {g:.4f}: score {v[:, 0].mean():6.2f} (n {len(v)}, falls {int(v[:, 4].sum())}), landing error "
              f"x {v[:, 1].mean() * 100:+.1f} cm, z {v[:, 2].mean() * 100:+.1f} cm, pitch {np.degrees(v[:, 3].mean()):+.1f} deg")
    if allv:
        print(f"  all: {np.mean(allv):.2f} over {len(allv)} jumps")
