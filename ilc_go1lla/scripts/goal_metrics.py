"""Goal-to-goal metrics on held-out walks: time to goal, final error, error along the path, falls.

Each walk: start at the origin at rest, goal 2.5 m away within +-15 deg (seed 300 goals; the first 6 are the stage's
evaluation goals), reference speed 0.8 m/s, 8 s horizon so slow walkers can still arrive.
  time to goal  first time the trunk is within --arrive m of the goal (a fall or no arrival = did not arrive)
  final error   distance to the goal at the end of the horizon (upright walks only)
  path error    RMS distance from the straight start->goal line, from the start until arrival (or the fall)
  max path err  the largest such distance
Ideal time to goal (the reference itself) is printed for comparison.

    ~/venvs/pg0/bin/python scripts/goal_metrics.py --out results/goal_metrics_1119.json
"""
import argparse, glob, json, os, pickle, re, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ilc_go1lla import walkenv as WE, srb as S        # noqa: E402
import jax                                             # noqa: E402
import jax.numpy as jnp                                # noqa: E402
from zeroshot_cases import CASES                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--params', default='runs/walker_nom_s0/params.pkl')
ap.add_argument('--walks', type=int, default=20)
ap.add_argument('--speed', type=float, default=0.8)
ap.add_argument('--dist', type=float, default=2.5)
ap.add_argument('--horizon', type=float, default=8.0)
ap.add_argument('--arrive', type=float, default=0.15)
ap.add_argument('--out', required=True)
a = ap.parse_args()

GROUPS = [   # (label, case, run dirs)
    ('foam', 'foam', ['stage4_foam_v08_s0', 'stage4_foam_v08_s1', 'stage4_foam_v08_s2']),
    ('2 kg right', 'payload2R', [f'stagev_payload2R_s{s}' for s in range(3)]),
    ('3 kg left', 'payload3L', [f'stagev_payload3L_s{s}' for s in range(3)]),
    ('3 kg right', 'payload3R', [f'stagev_payload3R_s{s}' for s in range(3)]),
    ('4 kg right (curriculum)', 'payload4R', ['stage5_payload4R_s0', 'stage5_payload4R_s1'] + [f'stagevc_payload4R_s{s}' for s in range(3)]),
    ('4 kg right (fixed 0.8)', 'payload4R', [f'stagev_payload4R_s{s}' for s in range(3)]),
    ('5 kg right', 'payload5R', [f'stagev_payload5R_s{s}' for s in range(3)]),
]
n = int(a.horizon / 0.02)
th = np.radians(np.random.default_rng(300).uniform(-15, 15, a.walks))
goals = [np.array([a.dist * np.cos(t), a.dist * np.sin(t), a.speed], np.float32) for t in th]
ts = np.arange(n) * 0.02
p_ref = np.stack([np.asarray(WE.reference(t, goals[0])[0]) for t in ts])
t_ideal = float(ts[np.argmax(np.linalg.norm(p_ref - goals[0][:2], axis=1) < a.arrive)])


def last_policy(d):
    f = sorted(glob.glob(f'runs/{d}/policy_it*.pkl'), key=lambda s: int(re.findall(r'it(\d+)', s)[0]))[-1]
    return f, pickle.load(open(f, 'rb'))


def metrics(out, g):
    kf = S.fall_index(out)
    end = n if kf is None else kf
    p = out['p'][:end, :2]
    d = np.linalg.norm(p - g[:2], axis=1)
    arr = np.where(d < a.arrive)[0]
    t_arr = None if (kf is not None or not len(arr)) else float(arr[0] * 0.02)
    u = g[:2] / np.linalg.norm(g[:2])
    cross = np.abs(u[0] * p[:, 1] - u[1] * p[:, 0])
    k_end = arr[0] + 1 if len(arr) else end
    return dict(fell=kf is not None, t_fall=None if kf is None else kf * 0.02, t_goal=t_arr,
                final_cm=None if kf is not None else float(100 * d[-1]),
                path_rms_cm=float(100 * np.sqrt(np.mean(cross[:k_end] ** 2))), path_max_cm=float(100 * cross[:k_end].max()))


def summarize(rows):
    up = [r for r in rows if not r['fell']]
    arr = [r['t_goal'] for r in rows if r['t_goal'] is not None]
    m = lambda xs: float(np.mean(xs)) if xs else None
    return dict(walks=len(rows), falls=sum(r['fell'] for r in rows), arrived=len(arr),
                t_goal_mean=m(arr), t_goal_median=float(np.median(arr)) if arr else None,
                final_cm_mean=m([r['final_cm'] for r in up]), final_cm_median=float(np.median([r['final_cm'] for r in up])) if up else None,
                path_rms_cm=m([r['path_rms_cm'] for r in rows]), path_max_cm=m([r['path_max_cm'] for r in rows]))


res = dict(args=vars(a), t_ideal=t_ideal, groups=[])
nominal = WE.Walker(a.params)
res['nominal'] = summarize([metrics(nominal.rollout(g, n=n), g) for g in goals])
print('ideal time to goal', t_ideal, '| nominal robot', json.dumps(res['nominal']), flush=True)
walkers = {}
for label, case, dirs in GROUPS:
    if case not in walkers:
        w = WE.Walker(a.params, **CASES[case])
        base = w.params

        def roll(params, goal, du, key, w=w):
            w.params = params
            return w._rollout(goal, du, key, n)
        walkers[case] = (w, base, jax.jit(roll))
    w, base, roll = walkers[case]

    def run(params):
        rows = []
        for g in goals:
            out = jax.tree.map(np.asarray, roll(params, jnp.asarray(g), jnp.zeros((n, 12)), jax.random.PRNGKey(0)))
            rows.append(metrics(out, g))
        return rows
    zs = run(base)
    G = dict(label=label, case=case, zero_shot=summarize(zs), zero_shot_rows=zs, runs=[])
    print(label, 'zero-shot', json.dumps(G['zero_shot']), flush=True)
    for d in dirs:
        f, pk = last_policy(d)
        rows = run((pk['norm'], pk['pol']) + tuple(base[2:]))
        G['runs'].append(dict(run=d, policy=f, stage=summarize(rows), rows=rows))
        print(' ', d, os.path.basename(f), json.dumps(G['runs'][-1]['stage']), flush=True)
    res['groups'].append(G)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(res, open(a.out, 'w'), indent=1)
print('wrote', a.out)
