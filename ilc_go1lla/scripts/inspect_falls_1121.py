"""2026-11-21: replay the goal walks that fell after stage (early arm) with the zero-shot and stage policies, signal by signal.
usage: inspect_falls_1121.py RESULT_JSON   (a results/goalnvE_final_*.json)"""
import json, pickle, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ilc_go1lla import walkenv as WE, srb as S
import jax
R = json.load(open(sys.argv[1])); A = R['args']; c = R['cases'][-1]
kind, vals = c['case'].split(':'); v = tuple(float(x) for x in vals.split(','))
pert = dict(foam=v) if kind == 'foam' else dict(payload=v)
th = np.radians(np.random.default_rng(300).uniform(-15, 15, A['walks'])); sp = c['speed']
goals = [np.array([A['dist'] * np.cos(t), A['dist'] * np.sin(t), sp], np.float32) for t in th]
t_sw = float(WE.ref_end_time(goals[0])) + WE.STAND_DELAY; n = int((t_sw + A['settle']) / 0.02)
zs = WE.Walker(A['params'], stand_at_goal=True, **pert)
st = WE.Walker(A['params'], stand_at_goal=True, **pert); pk = pickle.load(open(A['policy'], 'rb'))
st.params = (pk['norm'], pk['pol']) + tuple(st.params[2:]); st._roll = jax.jit(st._rollout, static_argnames=('n',))
KIN = S.LegKin(zs.env.mj_model)
fell = [i for i, r in enumerate(c['rows']) if r['fell']]; ok = [i for i, r in enumerate(c['rows']) if not r['fell']][:2]
def rpy(q):
    w, x, y, z = q.T
    return np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))), np.degrees(np.arcsin(np.clip(2*(w*y-z*x), -1, 1)))
print(c['case'], 'speed', sp, 'fell idx', fell, 'ok idx', ok)
for i in fell + ok:
    for tag, w in (('zs', zs), ('stage', st)):
        o = w.rollout(goals[i], n=n); kf = S.fall_index_feet(o, KIN); roll, pitch = rpy(o['quat'])
        k_end = kf if kf is not None else n
        an = np.linalg.norm(o['a'], axis=1); tau = np.abs(o['tau']).max(axis=1); vx = o['v'][:, 0]
        print(f"walk {i:2d} th {np.degrees(th[i]):6.1f} {tag:5s} fall_t {None if kf is None else kf*0.02}")
        for t in np.arange(0.5, min(k_end*0.02, 4.0) + 1e-6, 0.25):
            k = min(int(t / 0.02), k_end - 1); sl = slice(max(0, k - 12), k + 1)
            print(f"   t {k*0.02:4.2f} z {o['p'][k,2]:.3f} roll {roll[k]:6.1f} pitch {pitch[k]:6.1f} |a| {an[sl].mean():5.2f} max|a| {an[sl].max():5.2f} taumax {tau[sl].max():5.1f} vx {vx[k]:5.2f} cmd {o['cmd'][k,0]:5.2f} contact {o['contact'][k].astype(int).tolist()}")
