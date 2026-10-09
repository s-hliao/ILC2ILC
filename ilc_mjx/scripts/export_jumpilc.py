#!/usr/bin/env python3
"""
export_jumpilc.py [--seed 718] [--out DIR]: the per-goal ILC baseline's jumps (JumpILC, scratch, on the 8 reserved test
goals x 4 robots: log/dilc/plane/jilc_v3/scratch_g<x>_<h>_<robot>_s<seed>/trial_NNN.npz) in export_trajectories.py's
format, as one "hardware stage" file -- iteration = the trial number, so animate_hw_stage.py / make_animations.py show
each robot's 8 goals trial by trial (hwstage_jumpilc_pergoal_<robot>.gif).
JumpILC's executor records the robot's measured base pose and joints (rec_pos / rec_quat / rec_q, mocap-level
accuracy), not the policy recorder's ground-truth fields; foot contacts are not recorded (all False). The landing error
is the executor's (jilc_budget.py: the last logged state against the plan's landing offset by the goal).
-> DIR/hwstage_jumpilc_pergoal.npz + .json. Run from src/ilc_mjx with the ilcmjx env.
"""
import argparse
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--seed', default='718')
ap.add_argument('--src', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'log', 'dilc',
                                              'plane', 'jilc_v3'))
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trajectories'))
ap.add_argument('--name', default='hwstage_jumpilc_pergoal')
a = ap.parse_args()

CANON = ['FL', 'FR', 'RL', 'RR']                 # the recorder's leg order
MODEL = ['FR', 'FL', 'RR', 'RL']                 # go1.xml's qpos order
PERM = np.concatenate([[3 * CANON.index(l) + j for j in range(3)] for l in MODEL])

runs = sorted(d for d in glob.glob(os.path.join(a.src, 'scratch_g*')) if d.endswith(f'_s{a.seed}'))
items = []
for d in runs:
    meta = json.load(open(os.path.join(d, 'meta.json')))
    prm = meta['params']
    base = os.path.basename(d)
    robot = 'real_' + base.split('_real_')[1].rsplit('_s', 1)[0]
    gx, gh = float(base.split('_g')[1].split('_')[0]), float(base.split('_g')[1].split('_')[1])
    for f in sorted(glob.glob(os.path.join(d, 'trial_*.npz'))):
        z = np.load(f)
        xr, lx = np.asarray(z['x_ref'], float), np.asarray(z['log_X'], float)
        target = xr[-1].copy()
        target[:2] = xr[0, :2] + np.array([gx, gh])
        target[2] = 0.0
        e = lx[-1] - target
        items.append((z, dict(robot=robot, cond=robot, iteration=int(os.path.basename(f)[6:9]), goal=[gx, gh],
                              box_x_front=prm['box_x_front'] if prm['box_height'] > 0 else None,
                              box_height=float(prm['box_height']), ex=float(e[0]), ez=float(e[1]),
                              eth=float(e[2]) if len(e) > 2 else 0.0, fell=bool(z['log_fell']), J=0.0,
                              train_cond='per-goal ILC (JumpILC)', file=os.path.relpath(f, a.src))))
T = max(len(z['rec_t']) for z, _ in items)
pad = lambda x: np.concatenate([x, np.repeat(x[-1:], T - len(x), 0)]) if len(x) < T else x[:T]
pos = np.stack([pad(np.asarray(z['rec_pos'], np.float32)) for z, _ in items])
quat = np.stack([pad(np.asarray(z['rec_quat'], np.float32)) for z, _ in items])
q = np.stack([pad(np.asarray(z['rec_q'], np.float32)[:, PERM]) for z, _ in items])
con = np.zeros(pos.shape[:2] + (4,), bool)
meta = [dict(m, n_ticks=int(len(z['rec_t']))) for z, m in items]
dt = float(np.median(np.diff(items[0][0]['rec_t'])))
t = (np.arange(T) * dt).astype(np.float32)
os.makedirs(a.out, exist_ok=True)
out = os.path.join(a.out, f'{a.name}.npz')
np.savez_compressed(out, t=t, pos=pos, quat=quat, q=q, contacts=con, meta=np.array(json.dumps(meta)))
json.dump(dict(run=a.name, source=os.path.abspath(a.src), n=len(meta), ticks=int(T), dt=dt,
               joint_order='go1.xml qpos: FR FL RR RL x (hip, thigh, calf)', quat='w x y z',
               recipe=dict(method='per-goal ILC (JumpILC, Nguyen et al. 2024), scratch on each test goal', seed=a.seed),
               jumps=meta), open(os.path.join(a.out, f'{a.name}.json'), 'w'), indent=1)
print(f'{a.name}: {len(meta)} jumps ({len(runs)} goal runs) x {T} ticks -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)')
