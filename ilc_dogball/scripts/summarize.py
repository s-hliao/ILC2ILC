"""Budget curves from stage logs: mean and 95 % CI (t over seeds) per budget point, per arm.

    python scripts/summarize.py runs --x parks --metrics held_pos_err held_psi_err held_parked
    python scripts/summarize.py runs --x goals --metrics val_time_to_reach_s val_wall_hits_per_goal ...   (ilc_dogball)
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

T95 = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 8: 2.36, 10: 2.26}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs')
    ap.add_argument('--x', required=True)
    ap.add_argument('--metrics', nargs='+', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    arms = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in sorted(glob.glob(os.path.join(a.runs, 'stage_*', 'log.jsonl'))):
        tag = os.path.basename(os.path.dirname(path))
        arm = re.sub(r'_s\d+$', '', tag)
        for line in open(path):
            d = json.loads(line)
            if a.metrics[0] in d:
                for k in a.metrics:
                    arms[arm][d[a.x]][k].append(d[k])
    table = {}
    for arm, pts in arms.items():
        print(f'== {arm}')
        print(f'{a.x:>6s} ' + ' '.join(f'{k:>26s}' for k in a.metrics) + '   n')
        table[arm] = {}
        for x in sorted(pts):
            cells, row = [], {}
            for k in a.metrics:
                v = np.asarray(pts[x][k], float)
                ci = T95.get(len(v), 2.0) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float('nan')
                cells.append(f'{v.mean():>16.3f} +- {ci:<7.3f}')
                row[k] = dict(mean=float(v.mean()), ci95=float(ci), n=int(len(v)), values=v.tolist())
            table[arm][x] = row
            print(f'{x:>6d} ' + ' '.join(cells) + f'   {len(pts[x][a.metrics[0]])}')
    if a.out:
        json.dump(table, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
