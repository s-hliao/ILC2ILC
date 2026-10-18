import json, os, sys, numpy as np
sys.path.insert(0, '.')
import ci
arms = sys.argv[1:]
for net in ('nom', 'drLK20', 'ppo_dr'):
    print(f'== {net}')
    for arm in arms:
        row = []
        for f in ('summary_it2.json', 'summary_it4.json', 'summary.json'):
            ss = [json.load(open(f'runs/v4/{arm}_{net}_s{s}/{f}')) for s in (0, 1, 2) if os.path.exists(f'runs/v4/{arm}_{net}_s{s}/{f}')]
            if not ss: row.append('   -    '); continue
            m, lo, hi = ci.cluster_ci(ss)
            row.append(f'{100*m:5.1f} [{100*lo:4.0f},{100*hi:4.0f}]')
        full = [json.load(open(f'runs/v4/{arm}_{net}_s{s}/summary.json')) for s in (0, 1, 2) if os.path.exists(f'runs/v4/{arm}_{net}_s{s}/summary.json')]
        seeds = [round(100*np.mean([v['success'] for v in d['eval'].values()])) for d in full]
        rhos = [q['rho'] for d in full for r in d.get('tr', {}) for q in d['tr'][r]]
        rb = np.mean([sum(len(v) for v in d['rollbacks'].values()) for d in full]) if full else np.nan
        re = [q['realized'] for d in full for r in d['realize'] for q in d['realize'][r]]
        print(f'  {arm:10s} @12 {row[0]}  @24 {row[1]}  @36 {row[2]}  seeds {seeds}  rho med {np.median(rhos) if rhos else float("nan"):.2f}  rollbacks/run {rb:.1f}  realized {np.median(re) if re else float("nan"):.2f}')
