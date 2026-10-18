import json, os, sys, numpy as np
TR = ('mocap_square2fast', 'mocap_figfast')
def score(d):
    ev, rungs = d['eval'], d.get('rungs', {})
    b25, allp, crash = [], [], []
    for c, v in ev.items():
        for t in TR:
            suf = (rungs.get(c) or {}).get(t, '')
            keys = [k for k in v if k.startswith(t + '_b') and (k.endswith(suf) if suf else '_mu' not in k)]
            for k in keys:
                allp.append(v[k]['success']); crash.append(v[k].get('fail', 0))
                if '_b25' in k: b25.append(v[k]['success'])
    return 100*np.mean(b25), 100*np.mean(allp), 100*np.mean(crash)
for arm in sys.argv[1:]:
    for s in (0, 1, 2):
        f = f'runs/tr/{arm}_s{s}/summary.json'
        if os.path.exists(f):
            d = json.load(open(f)); b, a_, c = score(d)
            rhos = [q['rho'] for cc in d.get('tr', {}) for q in d['tr'][cc]]
            print(f'{arm:11s} s{s}  b25 {b:5.1f}  all {a_:5.1f}  crash {c:4.1f}  rollbacks {sum(len(v) for v in d["rollbacks"].values()):2d}  '
                  f'rho n {len(rhos)} med {np.median(rhos) if rhos else float("nan"):.2f}  rungs ' +
                  ' '.join(f'{c[:9]}:{"/".join((d.get("rungs",{}).get(c) or {}).get(t,"") or "-" for t in TR)}' for c in d['eval']))
