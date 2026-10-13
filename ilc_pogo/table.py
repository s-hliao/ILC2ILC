#!/usr/bin/env python3
"""table.py [arm ...]: the results table of the pogo-stick arms (runs/hw/*/summary.json).
Columns: success / falls / RMS apex error (cm) / |speed error| (m/s) over all robots; success on the robots inside the
DR family (real_nom, real_mass, real_spring) and outside it (real_act, real_offset, real_prog); per-robot success;
real trials per robot; positivity (fraction of trial pairs with cos > 0, median cos)."""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IN_FAM, OUT_FAM = ('real_nom', 'real_mass', 'real_spring'), ('real_act', 'real_offset', 'real_prog')
ROB = IN_FAM + OUT_FAM


def load(arm):
    f = os.path.join(HERE, 'runs/hw', arm, 'summary.json')
    if not os.path.exists(f):
        return None
    s = json.load(open(f))
    if not isinstance(s.get('runs'), dict) and isinstance(s.get('per_run'), dict):   # ppo_real.py's layout
        s['runs'] = s['per_run']
    return s


def row(arm, s):
    ev = s['eval']
    rs = [r for r in ROB if r in ev]
    allm = [m for r in rs for m in s['runs'][r]]
    ok = [m for m in allm if not m['failed']]
    sr = lambda group: np.mean([ev[r]['success'] for r in group if r in ev]) * 100 if any(r in ev for r in group) else np.nan
    pos = [q['cos'] for r in rs for q in s.get('positivity', {}).get(r, [])]
    tr = s.get('real_trials', {})
    return dict(arm=arm, success=100 * np.mean([m['success'] for m in allm]), fail=100 * np.mean([m['failed'] for m in allm]),
                apex=100 * np.mean([m['rms_apex'] for m in ok if m['rms_apex'] is not None]) if ok else np.nan,
                verr=np.mean([abs(m['v_err']) for m in ok if m['v_err'] is not None]) if ok else np.nan,
                infam=sr(IN_FAM), outfam=sr(OUT_FAM), per={r: 100 * ev[r]['success'] for r in rs},
                trials=int(np.mean([tr.get(r, 0) for r in rs])) if rs else 0,
                pos=(100 * np.mean(np.array(pos) > 0), np.median(pos)) if pos else None, n=len(allm))


def main():
    arms = sys.argv[1:] or sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(HERE, 'runs/hw/*/summary.json')))
    hdr = f'{"arm":<26}| succ% | fall% | apex cm | |dv| m/s | in-DR% | out-DR% | ' + ' | '.join(r[5:9] for r in ROB) + ' | trials | pos% cos'
    print(hdr)
    print('-' * len(hdr))
    for arm in arms:
        s = load(arm)
        if s is None or 'eval' not in s:
            continue
        q = row(arm, s)
        per = ' | '.join(f'{q["per"].get(r, np.nan):4.0f}' for r in ROB)
        pos = f'{q["pos"][0]:4.0f} {q["pos"][1]:+.2f}' if q['pos'] else '   -'
        print(f'{arm:<26}| {q["success"]:5.1f} | {q["fail"]:5.1f} | {q["apex"]:7.2f} | {q["verr"]:8.3f} | {q["infam"]:6.1f} | '
              f'{q["outfam"]:7.1f} | {per} | {q["trials"]:6d} | {pos}')


if __name__ == '__main__':
    main()
