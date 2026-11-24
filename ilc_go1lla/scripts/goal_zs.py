"""Zero-shot goal-to-goal metrics of the nominal walker on perturbed robots, with the standing controller at the goal.

Task: start at rest at the origin, goal 2.5 m away within +-15 deg (seed 300 goals), reference 0.8 m/s; the walker hands
over to the standing controller (joint PD to the home pose) STAND_DELAY s after the reference reaches the goal; the
walk lasts --settle s more. Per perturbation:
  falls walking / falls standing   before / after the hand-over
  time to goal   first time the trunk is within --arrive m of the goal (fallen walks never arrive)
  path error     RMS distance from the straight start->goal line until arrival (or the fall)
  final error    distance to the goal at the end (upright walks)
  sink           nominal minus perturbed standing trunk height at the end (cm; foam: how deep the feet sink)
A case is a perturbation: name=foam:tc,width[,dampratio] | payload:m,dy,dz | weak:leg,scale | <name in zeroshot_cases>.

    ~/venvs/pg0/bin/python scripts/goal_zs.py --cases nominal foam:0.2,0.1 payload:4,-0.10,0.08 --out results/goal_zs.json
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ilc_go1lla import walkenv as WE, srb as S        # noqa: E402
from zeroshot_cases import CASES                       # noqa: E402
import jax                                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--params', default='runs/walker_nom_s0/params.pkl')
ap.add_argument('--cases', nargs='+', required=True)
ap.add_argument('--walks', type=int, default=20)
ap.add_argument('--speeds', type=float, nargs='+', default=[0.8])
ap.add_argument('--dist', type=float, default=2.5)
ap.add_argument('--settle', type=float, default=0.3, help='s scored after the hand-over')
ap.add_argument('--arrive', type=float, default=0.15)
ap.add_argument('--policy', default=None, help='stage policy pickle used on the perturbed (non-nominal) cases')
ap.add_argument('--out', required=True)
a = ap.parse_args()


def parse(c):
    if c in CASES:
        return c, dict(CASES[c])
    kind, vals = c.split(':')
    v = [float(x) for x in vals.split(',')]
    if kind == 'foam':
        return c, dict(foam=tuple(v))
    if kind == 'payload':
        return c, dict(payload=tuple(v))
    if kind == 'weak':
        return c, dict(weak_leg=(vals.split(',')[0], float(vals.split(',')[1])))
    raise ValueError(c)


th = np.radians(np.random.default_rng(300).uniform(-15, 15, a.walks))
KIN = None


def metrics(out, g, n, t_sw):
    kf = S.fall_index_feet(out, KIN)
    end = n if kf is None else kf
    p = out['p'][:end, :2]
    d = np.linalg.norm(p - g[:2], axis=1)
    arr = np.where(d < a.arrive)[0]
    u = g[:2] / np.linalg.norm(g[:2])
    cross = np.abs(u[0] * p[:, 1] - u[1] * p[:, 0])
    k_end = arr[0] + 1 if len(arr) else end
    return dict(fell=kf is not None, t_fall=None if kf is None else kf * 0.02,
                fell_walking=kf is not None and kf * 0.02 < t_sw, fell_standing=kf is not None and kf * 0.02 >= t_sw,
                t_goal=None if (kf is not None or not len(arr)) else float(arr[0] * 0.02),
                final_cm=None if kf is not None else float(100 * d[-1]),
                path_rms_cm=float(100 * np.sqrt(np.mean(cross[:k_end] ** 2))), path_max_cm=float(100 * cross[:k_end].max()),
                z_end=None if kf is not None else float(out['p'][-10:, 2].mean()))


def summarize(rows, z_nom=None):
    up = [r for r in rows if not r['fell']]
    arr = [r['t_goal'] for r in rows if r['t_goal'] is not None]
    med = lambda xs: float(np.median(xs)) if xs else None
    z = med([r['z_end'] for r in up])
    return dict(walks=len(rows), falls_walking=sum(r['fell_walking'] for r in rows),
                falls_standing=sum(r['fell_standing'] for r in rows), arrived=len(arr),
                t_goal_mean=float(np.mean(arr)) if arr else None, t_goal_max=float(np.max(arr)) if arr else None,
                path_rms_cm=float(np.mean([r['path_rms_cm'] for r in rows if not r['fell_walking']] or [np.nan])),
                final_cm_median=med([r['final_cm'] for r in up]), z_end=z,
                sink_cm=None if (z is None or z_nom is None) else 100 * (z_nom - z))


res = dict(args=vars(a), cases=[])
z_nom = {}
for c in (['nominal'] if 'nominal' not in a.cases else []) + a.cases:
    name, pert = parse(c)
    w = WE.Walker(a.params, stand_at_goal=True, **pert)
    if a.policy and name != 'nominal':
        import pickle
        pk = pickle.load(open(a.policy, 'rb'))
        w.params = (pk['norm'], pk['pol']) + tuple(w.params[2:])
        w._roll = jax.jit(w._rollout, static_argnames=('n',))
    if KIN is None:
        KIN = S.LegKin(w.env.mj_model)
    for v in a.speeds:
        goals = [np.array([a.dist * np.cos(t), a.dist * np.sin(t), v], np.float32) for t in th]
        t_sw = float(WE.ref_end_time(goals[0])) + WE.STAND_DELAY
        n = int((t_sw + a.settle) / 0.02)
        rows = [metrics(w.rollout(g, n=n), g, n, t_sw) for g in goals]
        s = summarize(rows, z_nom.get(v))
        if name == 'nominal':
            z_nom[v] = s['z_end']
        res['cases'].append(dict(case=name, speed=v, t_switch=t_sw, perturb={k: list(v_) for k, v_ in pert.items()}, summary=s, rows=rows))
        print(name, v, json.dumps(s), flush=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(res, open(a.out, 'w'), indent=1)
print('wrote', a.out)
