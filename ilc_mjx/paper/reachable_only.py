#!/usr/bin/env python3
"""reachable_only.py: the headline quadruped success rates over all goals and over the REACHABLE goals only (robots r1 s1 r5;
a goal on a robot is "impossible" when per-goal ILC's best landing is > 5 cm off AND no method ever succeeded on it:
3 real_r5 goal pairs, log/dilc/plane/unreachable_goals.json). A secondary column: the headline stays the full test set.
Run from log/dilc/plane/fixbox:  python3 ~/ilc_ws/src/ilc_mjx/paper/reachable_only.py | tee ../reachable_only.txt"""
import json,glob,os
ok=lambda x:(not x['fell']) and abs(x['ex'])<=0.05 and abs(x['ez'])<=0.03
U=json.load(open('/home/henry/ilc_ws/log/dilc/plane/unreachable_goals.json'))
key=lambda x: (round(x['goal'][0],3), round(x['goal'][1],3))
# every evaluated jump by any method, any arm (start / final / snapshots)
ever=set()
files=glob.glob('*/real_*/eval_planefinal*.json')+glob.glob('*/*/real_*/eval_planefinal*.json')+glob.glob('*/real_*/it*/eval_planefinal.json')+glob.glob('../match/drb_snap/*.json')
for f in files:
    rb=[p for p in f.split('/') if p.startswith('real_')] or [os.path.basename(f).split('_it')[1].split('_',1)[1].replace('.json','')]
    rb=rb[0].replace('.json','')
    try: d=json.load(open(f))
    except Exception: continue
    for x in d if isinstance(d,list) else []:
        if isinstance(x,dict) and 'goal' in x and ok(x): ever.add((rb,key(x)))
imp=set()
for rb,gs in U.items():
    for g in gs:
        k=(rb,(round(g[0],3),round(g[1],3)))
        if k not in ever: imp.add(k)
print('evaluated files:', len(files)); print('IMPOSSIBLE (per-goal ILC never within 5 cm AND no method ever succeeded):', sorted(imp))
R3=['real_r1','real_s1','real_r5']
def score(src, tag):
    rs=[]
    for rb in R3:
        f=f'{src}/{rb}/eval_planefinal_{tag}.json'
        if os.path.exists(f): rs+= [dict(x,rb=rb) for x in json.load(open(f))]
    if not rs: return None
    keep=[x for x in rs if (x['rb'],key(x)) not in imp]
    return 100*sum(map(ok,rs))/len(rs), len(rs), 100*sum(map(ok,keep))/len(keep), len(keep)
rows=[('ours nominal, zero-shot','hw6g16_v10s2it20','start'),('ours nominal + 24','gate_loose','final'),('ours nominal + 96','hw6g16_v10s2it20','final'),
      ('ours + DR (A), zero-shot','drA_hw24','start'),('ours + DR (A) + 24','drA_hw24','final'),
      ('ours + DR (B), zero-shot','drB_hw24','start'),('ours + DR (B) + 24','drB_hw24','final'),
      ('PPO+DR, zero-shot','base/eval_ppo_plain','start'),('RMA, zero-shot','base/eval_rma','start')]
print(f"\n{'':28s} {'all goals':>14s}   {'reachable only':>16s}")
for lab,src,tag in rows:
    r=score(src,tag)
    if r: print(f"{lab:28s} {r[0]:5.1f}% (n={r[1]:3d})   {r[2]:5.1f}% (n={r[3]:3d})")
    else: print(f"{lab:28s} (no data at {src})")
