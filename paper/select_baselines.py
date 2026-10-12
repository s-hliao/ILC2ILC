#!/usr/bin/env python3
"""select_baselines.py: the RMA and FADA baselines the paper shows -- for each robot, of each method, the BEST ORACLE and the
BEST NON-ORACLE variant (user, 2026-10-11: "those should be the only two baselines for rma and FADA"), picked from every
variant we ran by its best success at any real-data budget (the variant's most favourable checkpoint). -> paper/baselines.json,
read by every figure script. Plain python + json (runs in either env); rerun after a candidate run finishes.

  oracle      the variant uses privileged knowledge of the deployment: the robot's / car's TRUE dynamics (latent), and / or an
              ORACLE planner fit to our own controller's real trials
  non-oracle  the method as published: its own sim-trained components, adapted (or not) with its own real data

Scores: quadruped = success on the reserved test goals (8 goals x 4 jumps, robots r1 s1 r5); car = success over the 5 cars x 8
evaluation plans (reeval.py's eval_big.json).
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
QP = '/home/henry/ilc_ws/log/dilc/plane'
CAR = os.path.join(HERE, '..', 'ilc_f1tenth', 'ilc2real', 'runs', 'hw')
R3 = ['real_r1', 'real_s1', 'real_r5']
ok = lambda x: (not x['fell']) and abs(x['ex']) <= 0.05 and abs(x['ez']) <= 0.03


def q_rate(files):
    rows = [r for f in files for r in json.load(open(f))]
    return (100 * sum(map(ok, rows)) / len(rows), len(rows)) if rows else None


def q_point(run, tag):
    """A quadruped single evaluation (fixbox/<run>/<robot>/eval_planefinal_<tag>.json) on r1 s1 r5."""
    fs = [os.path.join(QP, 'fixbox', run, rb, f'eval_planefinal_{tag}.json') for rb in R3]
    return q_rate(fs) if all(map(os.path.exists, fs)) else None


PER_ITER = {'fada_real': 24, 'fada_real_oracle': 24, 'fada_real_oracle16': 64, 'fada_real_true': 32}


def q_curve(name):
    """A quadruped data-matching run: {real jumps per robot: (success, n)} from its per-checkpoint evaluations."""
    pts = {}
    for f in glob.glob(os.path.join(QP, 'match', name, 'ck*', 'real_*', 'eval_planefinal.json')):
        pts.setdefault(int(re.findall(r'/ck(\d+)/', f)[0]), []).append(f)
    for f in glob.glob(os.path.join(QP, 'match', name, 'real_*', 'it*', 'eval_planefinal.json')) if name in PER_ITER else []:
        pts.setdefault(int(re.findall(r'/it(\d+)/', f)[0]) * PER_ITER[name], []).append(f)
    robot = lambda f: next(p for p in f.split('/') if p.startswith('real_'))
    return {k: q_rate(fs) for k, fs in sorted(pts.items()) if {robot(f) for f in fs} >= set(R3)}   # all 3 robots scored


def c_eval(arm):
    f = os.path.join(CAR, arm, 'eval_big.json')
    if not os.path.exists(f):
        return None
    e = json.load(open(f))
    v = [x for c in e.values() for x in c.values() if isinstance(x, dict)]
    return (100 * sum(x['success'] for x in v) / len(v), sum(x.get('n', 0) for x in v))


def c_curve(prefix):
    pts = {}
    for d in glob.glob(os.path.join(CAR, f'{prefix}_ep*')):
        m = re.search(r'_ep(\d+)$', d)
        r = c_eval(os.path.basename(d)) if m else None
        if r:
            pts[int(m.group(1))] = r
    return dict(sorted(pts.items()))


# (role, label, kind, source): kind 'point' = one evaluation at a fixed real budget (budget given); 'curve' = a run over budgets
CANDIDATES = {
    'quadruped': {
        'rma_nonoracle': [('RMA (adaptation module), zero-shot', 'point', ('base/eval_rma', 'start'), 0),
                          ('RMA, cross-trial (6 calibration jumps)', 'point', ('base/eval_rma_ctx', 'final'), 6)],
        'rma_oracle': [('RMA teacher with the TRUE dynamics, PPO on the robot', 'curve', 'rma_real', None),
                       ('RMA teacher with the TRUE dynamics, PPO on the robot (reward fixed)', 'curve', 'rma_real_v2', None)],
        'fada_nonoracle': [('FADA, LoRA on the robot (48 jumps)', 'point', ('base/eval_fada', 'final'), 48),
                           ('FADA, LoRA on the robot, its own planner', 'curve', 'fada_real', None)],
        'fada_oracle': [('FADA, ORACLE planner (6 goals)', 'curve', 'fada_real_oracle', None),
                        ('FADA, ORACLE planner (16 goals)', 'curve', 'fada_real_oracle16', None),
                        ('FADA, TRUE dynamics + ORACLE planner, trained on the test goals', 'curve', 'fada_real_true', None)],
    },
    'car': {
        'rma_nonoracle': [('RMA (adaptation module), zero-shot', 'point', 'rma_zs', 0)],
        'rma_oracle': [('RMA teacher with each car\'s TRUE latent, fine-tuned on the car', 'curve', 'rma_real', None)],
        'fada_nonoracle': [('FADA (IDM with actuator states), zero-shot', 'point', 'fada_hw24_zs', 0),
                           ('FADA (v2 IDM, horizon 4), zero-shot', 'point', 'fada_v2_h4_hw24_zs', 0),
                           ('FADA (v2 IDM, horizon 4), 24 laps of its adaptation', 'point', 'fada_v2_h4_hw24', 24),
                           ('FADA, planner gain 0.5, on the cars', 'curve', 'fada_real_l05', None),
                           ('FADA, planner gain 0.8, on the cars', 'curve', 'fada_real_l08', None),
                           ('FADA, planner gain 0.95, on the cars', 'curve', 'fada_real_l095', None),
                           ('FADA (v2 IDM), its own planner, on the cars', 'curve', 'fada_real_v2h4', None)],
        'fada_oracle': [('FADA, ORACLE planner (refit on our real laps)', 'curve', 'fada_real_oracle', None),
                        ('FADA, ORACLE planner (our stopping-rule network), 14 plans', 'curve', 'fada_real_oracle14', None),
                        ('FADA (v2 IDM), ORACLE planner (our stopping-rule network), 14 plans', 'curve', 'fada_real_v2h4_oracle14',
                         None)],
    },
}


# The real-data budgets every figure reads (user, 2026-10-11: a "low amount" where most of the ILC's value is already got, and
# an "asymptotic amount" where it plateaus). Quadruped: 24 jumps per robot carry 81 % of the 96-jump gain (nominal headline
# 18.8 -> 32.3 -> 35.4 %). Car: LOW = 10 laps, the user's cap on our method's real laps (it carries ~1/3 of the tracking-error
# gain: mean RMS e_y of the 4 non-friction cars 4.41 -> 4.11 cm, vs 3.53 at 48); ASYM = 48, where the stopping-rule network
# plateaus (car_budget.sh, runs/hw/v3drBt_stop_lap*: 48 vs 96 laps 3.53 vs 3.54 cm, success 84.2 vs 84.4 %).
BUDGETS = {'quadruped': {'low': 24, 'asym': 96, 'unit': 'jumps per robot'},
           'car': {'low': 10, 'asym': 48, 'unit': 'laps per car'}}


def score(robot, kind, src, budget):
    """-> (curve {budget: (success, n)}, best budget, best success) or None."""
    if robot == 'quadruped':
        cur = {budget: q_point(*src)} if kind == 'point' else q_curve(src)
    else:
        cur = {budget: c_eval(src)} if kind == 'point' else c_curve(src)
    cur = {k: v for k, v in cur.items() if v}
    if not cur:
        return None
    b = max(cur, key=lambda k: (cur[k][0], -k))      # the best checkpoint (ties: the smaller budget)
    return cur, b, cur[b][0]


def main():
    out = {}
    for robot, roles in CANDIDATES.items():
        out[robot] = {}
        for role, cands in roles.items():
            scored = []
            for lab, kind, src, budget in cands:
                s = score(robot, kind, src, budget)
                if s:
                    scored.append(dict(label=lab, kind=kind, source=src, best_budget=s[1], best_success=round(s[2], 1),
                                       best_n=s[0][s[1]][1], oracle=role.endswith('_oracle'),
                                       curve={str(k): [round(v[0], 2), v[1]] for k, v in s[0].items()}))
            if not scored:
                continue
            best = max(scored, key=lambda c: (c['best_success'], -c['best_budget']))
            out[robot][role] = dict(best=best, candidates=[dict(label=c['label'], best_success=c['best_success'],
                                                                best_budget=c['best_budget']) for c in scored])
            print(f"{robot:9s} {role:15s} -> {best['label']}: {best['best_success']:.1f} % at {best['best_budget']} "
                  f"(of {len(scored)} variants)")
    out['budgets'] = BUDGETS
    json.dump(out, open(os.path.join(HERE, 'baselines.json'), 'w'), indent=1)
    print('->', os.path.join(HERE, 'baselines.json'))


if __name__ == '__main__':
    main()
