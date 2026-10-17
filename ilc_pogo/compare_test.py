#!/usr/bin/env python3
"""compare_test.py: paired cluster-bootstrap differences (percentage points, 95 % CI) of the headline arms on the
frozen test pass (summary_test.json; seeds pooled; the same robot x goal clusters for both arms)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ci                                       # noqa: E402

R = 'runs/v3'
S = lambda pat: [json.load(open(os.path.join(R, pat.format(s=s), 'summary_test.json'))) for s in (0, 1, 2)]
ARMS = {
    'ours nominal @36': S('nom_s{s}_curve/it6'), 'ours nominal @24': S('nom_s{s}_curve/it4'),
    'ours+DR @24': S('drLK20_s{s}_curve/it4'), 'ours+DR @36': S('drLK20_s{s}_curve/it6'),
    'PPO+DR + stage @36': S('ppo_dr_s{s}_curve/it6'), 'PPO+DR zs': S('ppo_dr_s{s}_zs'),
    'RMA (fixed) zs': S('rmaR1_s{s}_zs'), 'FADA @24': S('fadaF2_s{s}_hw24'),
    'PPO on robots @48': S('ppo_s{s}_ep48'), 'PPO on robots @384': S('ppo_s{s}_ep384'),
}
PAIRS = [('ours nominal @36', 'PPO+DR zs'), ('ours nominal @36', 'RMA (fixed) zs'), ('ours nominal @24', 'FADA @24'),
         ('ours nominal @36', 'PPO on robots @48'), ('ours nominal @36', 'PPO on robots @384'),
         ('ours+DR @24', 'PPO+DR zs'), ('ours+DR @36', 'ours nominal @36'), ('PPO+DR + stage @36', 'RMA (fixed) zs'),
         ('PPO+DR + stage @36', 'ours nominal @36'), ('FADA @24', 'PPO+DR zs')]
for a, b in PAIRS:
    d, lo, hi = ci.paired_ci(ARMS[a], ARMS[b])
    print(f'{a:>20s} - {b:<20s} {100 * d:+6.1f} pts [{100 * lo:+6.1f}, {100 * hi:+6.1f}]' + ('' if lo <= 0 <= hi else '  *'))
