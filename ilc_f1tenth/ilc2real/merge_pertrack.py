#!/usr/bin/env python3
"""
merge_pertrack.py OUT SQUARE_ARM FIG_ARM: the per-track ablation's combined result. Two networks, one per track, each
evaluated on its own track's plans only (reeval.py with the arm's eval_plans) -> runs/hw/OUT/ with eval_big.json (the
square arm's square plans + the figfast arm's figfast plans: the same 8 plans, runs and starts as a shared network's
evaluation), summary.json (real laps per car = the two arms' sum) and args.json, so results tables and figures read it
like any other arm.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
out, sq, fg = sys.argv[1:4]
D = lambda a: os.path.join(HERE, 'runs/hw', a)
ev = {}
for arm, track in ((sq, 'square2fast'), (fg, 'figfast')):
    e = json.load(open(os.path.join(D(arm), 'eval_big.json')))
    for car, plans in e.items():
        for p, m in plans.items():
            if f'mocap_{track}_' in p:
                ev.setdefault(car, {})[p] = m
s_sq, s_fg = (json.load(open(os.path.join(D(a), 'summary.json'))) for a in (sq, fg))
laps = {c: s_sq.get('real_laps', {}).get(c, 0) + s_fg.get('real_laps', {}).get(c, 0) for c in ev}
crashes = {c: (s_sq.get('crashes', {}).get(c) or []) + (s_fg.get('crashes', {}).get(c) or []) for c in ev}
os.makedirs(D(out), exist_ok=True)
json.dump(ev, open(os.path.join(D(out), 'eval_big.json'), 'w'), indent=1)
json.dump(dict(real_laps=laps, crashes=crashes, merged_from=[sq, fg], safety_ey=s_sq.get('safety_ey')),
          open(os.path.join(D(out), 'summary.json'), 'w'), indent=1)
json.dump(dict(merged_from=[sq, fg], per_track=True, cars=list(ev)), open(os.path.join(D(out), 'args.json'), 'w'), indent=1)
print(out, '<-', sq, '+', fg, {c: round(v) for c, v in laps.items()})
