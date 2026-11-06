"""rescore_table.py ROOT PATTERN NEW_FILE PLAN [OLD_FILE] (2026-11-06): the car table's cell (car_final_score.py's statistic: per run the
mean over its cars of the % score change, it 3 vs its own it 0; bootstrap CI over runs) from a fixed-simulator re-score (ladder_eval.py
output: {plan: {it: {car: {score, rms_ey, pace, dev_time, fail}}}}), with failures (spin-outs) per iteration, rms e_y, pace, and, if
OLD_FILE is given (the old simulator's screen_eval_701.json: {plan: {it: {car: score}}} or a ladder_eval file), the same statistic from
it over the same runs."""
import glob
import json
import sys

import numpy as np

root, pat, fnew, plan = sys.argv[1:5]
fold = sys.argv[5] if len(sys.argv) > 5 else None
rng = np.random.default_rng(0)


def score(v):
    return v['score'] if isinstance(v, dict) else float(v)


def cell(files):
    R, cells, fails, ey, pace, n = [], [], {'0': [], '3': []}, {'0': [], '3': []}, {'0': [], '3': []}, 0
    for f in files:
        e = json.load(open(f))
        if plan not in e or '3' not in e[plan]:
            continue
        p = e[plan]
        d = [100 * (score(p['3'][c]) / score(p['0'][c]) - 1) for c in p['0'] if c in p['3']]
        R.append(np.mean(d)); cells += d; n += 1
        for k in ('0', '3'):
            for c, v in p[k].items():
                if isinstance(v, dict):
                    fails[k].append(v['fail']); ey[k].append(v['rms_ey']); pace[k].append(v['pace'])
    if not R:
        return None
    R = np.array(R)
    bs = R[rng.integers(0, len(R), (4000, len(R)))].mean(1)
    s = (f'{R.mean():+6.1f} % [{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}]  runs better {int((R < 0).sum())}/{len(R)}  '
         f'cells better {int((np.array(cells) < 0).sum())}/{len(cells)}')
    if fails['0']:
        s += (f'  spin-out share it0 {100 * np.mean(fails["0"]):.1f} % -> it3 {100 * np.mean(fails["3"]):.1f} %  rms e_y '
              f'{100 * np.nanmean(ey["0"]):.1f} -> {100 * np.nanmean(ey["3"]):.1f} cm  pace {np.nanmean(pace["0"]):.3f} -> {np.nanmean(pace["3"]):.3f}')
    return s


for net, nn in (('freePPON', 'PPO+DR + stage'), ('rmaN', 'RMA + stage')):
    fs = sorted(glob.glob(f'{root}/{pat}_{net}_s*/{fnew}'))
    r = cell(fs)
    if r:
        print(f'{nn:15s} {plan:28s} fixed sim ({len(fs)} runs): {r}')
    if fold:
        fo = [f.replace(fnew, fold) for f in fs]
        r = cell([f for f in fo if glob.os.path.exists(f)])
        if r:
            print(f'{"":15s} {"":28s} old sim   (same runs): {r}')
