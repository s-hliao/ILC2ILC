"""(2026-11-24) Evaluate a goal-conditioned walker (goalwalk.py) in sim, nominal Go1, observation noise OFF.

Goals at 8 bearings (0, +-45, +-90, +-135, 180 deg from the start heading) x distances; one walk each from a fixed start.
Reports per bearing: reached (within ARRIVE and held 1 s), time to reach, final distance, path length / straight
distance, total heading turned, falls.

    ~/venvs/pg0/bin/python scripts/eval_goalwalker.py runs/goalwalk/vel_s0 [--dist 2.5 3.5] [--T 12]
"""
import argparse
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('run')
ap.add_argument('--dist', type=float, nargs='+', default=[2.0, 3.5])
ap.add_argument('--T', type=float, default=12.0)
a = ap.parse_args()

import jax                                                       # noqa: E402
import jax.numpy as jp                                            # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ilc_go1lla import goalwalk as GW                             # noqa: E402
from ilc_go1lla import walkenv as WE                              # noqa: E402

meta = json.load(open(os.path.join(a.run, 'meta.json')))
cfg = GW.default_config(meta['args'].get('variant', 'vel'))
cfg.noise_config.level = 0.0
env = GW.GoalWalker(config=cfg)
params, net = WE.load_policy(env, os.path.join(a.run, 'params.pkl'))
pol = jax.jit(lambda o: WE.policy_apply(net, params, o))
n = int(a.T / env.dt)


@jax.jit
def walk(goal_rel):
    st = env.reset(jax.random.PRNGKey(0))
    q = env._init_q                                              # fixed start: origin, yaw 0, at rest
    data = st.data.replace(qpos=q, qvel=jp.zeros_like(st.data.qvel))
    info = dict(st.info, goal=goal_rel, d_prev=jp.linalg.norm(goal_rel))
    st = st.replace(data=data, info=info, obs=env._get_obs(data, info))

    def body(s, _):
        s2 = env.step(s, pol(s.obs))
        g = goal_rel                                             # keep ONE goal (no resampling) for the eval. (11-24 fix: s.info
        s2.info['goal'] = g                                      # is the dict env.step mutates, so it held the NEW goal)
        s2.info['d_prev'] = jp.linalg.norm(g - s2.data.qpos[:2])
        return s2, (s2.data.qpos[:2], WE.yaw_of(s2.data.qpos[3:7]), env.get_upvector(s2.data)[-1])
    _, (p, yaw, up) = jax.lax.scan(body, st, None, length=n)
    return p, yaw, up


rows = []
for dist in a.dist:
    for bdeg in [0, 45, -45, 90, -90, 135, -135, 180]:
        b = np.radians(bdeg)
        g = jp.array([dist * np.cos(b), dist * np.sin(b)], jp.float32)
        p, yaw, up = map(np.asarray, walk(g))
        d = np.linalg.norm(p - np.asarray(g), axis=1)
        fell = bool((up < 0.3).any())
        kf = int(np.argmax(up < 0.3)) if fell else n
        inside = d[:kf] < GW.ARRIVE
        hold = int(round(GW.HOLD_S / env.dt))
        reach = next((k for k in range(kf - hold) if inside[k:k + hold].all()), None)
        path = float(np.linalg.norm(np.diff(p[:kf], axis=0), axis=1).sum())
        dyaw = np.unwrap(yaw[:kf])
        rows.append(dict(dist=dist, bearing=bdeg, reached=reach is not None,
                         t_reach=None if reach is None else round(reach * env.dt, 2),
                         d_final=round(float(d[kf - 1]), 3), path_ratio=round(path / dist, 2),
                         turned_deg=round(float(np.degrees(np.abs(dyaw - dyaw[0]).max())), 1), fell=fell))
        print(json.dumps(rows[-1]), flush=True)
s = dict(run=a.run, reached=sum(r['reached'] for r in rows), n=len(rows), falls=sum(r['fell'] for r in rows),
         median_t=float(np.median([r['t_reach'] for r in rows if r['reached']] or [np.nan])))
print('SUMMARY', json.dumps(s))
json.dump(dict(summary=s, rows=rows), open(os.path.join(a.run, 'eval_goals.json'), 'w'), indent=1)
