#!/usr/bin/env python3
"""reeval.py ARM [ARM ...] --runs 12: re-evaluate finished arms with more runs (tighter confidence intervals), from the
per-car networks the arm saved (zero-shot arms: its starting policy; the LQR arm: the plans' LQR; the per-track ILC:
not handled here). Same protocol as hw_stage.evaluate (spread starts, perturbation, chained laps, reserved seeds from
--seed, distinct from the hardware stage's own 701). -> runs/hw/ARM/eval_big.json
  --tag T        write eval_big<T>.json / eval_big_adaptive<T>.json (another seed without overwriting: e.g. --seed 301 --tag _val)
  --rungs-from A a baseline without a ladder of its own flies and is scored on the plan rungs ladder arm A settled on per
                 car (the idealized reference: what a perfect rung choice would give it, with no real laps)"""
import argparse
import json
import os
import sys

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bank                                      # noqa: E402
import real_car as rc                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = [f'mocap_{t}_b{b}' for t in ('square2fast', 'figfast') for b in (25, 21, 14, 0)]


def load(path):
    if path == 'lqr':
        return None
    d = np.load(path)
    n = len([k for k in d.files if k.startswith('W')])
    pol = [(d[f'W{i}'], d[f'b{i}']) for i in range(n)]
    if 'mode' in d.files and str(d['mode']) == 'fada' or 'lam' in d.files:
        orc = {k[len('oracle__'):]: np.asarray(d[k]) for k in d.files if k.startswith('oracle__')}
        return dict(fada=pol, lam=float(d['lam']), oracle=orc or None,
                    dims=np.asarray(d['dims']) if 'dims' in d.files else None,
                    horizon=int(d['horizon']) if 'horizon' in d.files else 1)
    ad = os.path.join(os.path.dirname(path), 'rma_adapt.npz')
    if 'mode' in d.files and str(d['mode']) == 'rma' and os.path.exists(ad):
        e = np.load(ad)
        m = len([k for k in e.files if k.startswith('W')])
        return dict(actor=pol, adapt=[(e[f'W{i}'], e[f'b{i}']) for i in range(m)], hist=int(e['hist']))
    return pol


def adaptive(d, summ, tag='', rungs_from=None):
    """Goal-fallback arms: each evaluation plan scored on the variant the car's hardware stage settled on for the
    nearest trained goal of the same track (by target sideslip): eval_big_adaptive.json."""
    args = json.load(open(os.path.join(d, 'args.json')))
    sfx = args.get('fallback_suffix') or ''
    rungs = rungs_from or (json.load(open(os.path.join(d, 'summary.json'))).get('rungs') if sfx else None)
    if not sfx and rungs is None:
        return
    if rungs is not None:                             # a ladder arm: each plan on its track's rung (all sideslips)
        adap = {c: {n: v.get(n + rungs.get(c, {}).get(n.split('_b')[0], ''), v.get(n)) for n in EVAL if n in v}
                for c, v in summ.items()}
        json.dump(adap, open(os.path.join(d, f'eval_big_adaptive{tag}.json'), 'w'), indent=1)
        return
    sw = json.load(open(os.path.join(d, 'summary.json'))).get('switched', {})
    beta = lambda n: int(n.split('_b')[-1])
    track = lambda n: n.split('_b')[0]
    adap = {}
    for c, v in summ.items():
        adap[c] = {}
        for n in EVAL:
            goals = [g for g in args['goals'] if track(g) == track(n)]
            g = min(goals, key=lambda g: abs(beta(g) - beta(n))) if goals else None
            use_fb = g is not None and g in sw.get(c, {}) and (n + sfx) in v and beta(n) > 0
            adap[c][n] = v[n + sfx] if use_fb else v[n]
    json.dump(adap, open(os.path.join(d, f'eval_big_adaptive{tag}.json'), 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arms', nargs='+')
    ap.add_argument('--runs', type=int, default=12)
    ap.add_argument('--laps', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=1701)
    ap.add_argument('--tag', default='')
    ap.add_argument('--rungs-from', default='')
    a = ap.parse_args()
    rf = json.load(open(os.path.join(HERE, 'runs/hw', a.rungs_from, 'summary.json')))['rungs'] if a.rungs_from else None
    for arm in a.arms:
        d = os.path.join(HERE, 'runs/hw', arm)
        out_f = os.path.join(d, f'eval_big{a.tag}.json')
        if os.path.exists(out_f):
            adaptive(d, json.load(open(out_f)), a.tag, rf)
            continue
        args = json.load(open(os.path.join(d, 'args.json')))
        sfx = args.get('fallback_suffix') or ''
        ev_names = [n for n in EVAL if n in (args.get('eval_plans') or EVAL)]   # a per-track network: its own track's plans
        rungs = rf or (json.load(open(os.path.join(d, 'summary.json'))).get('rungs') if sfx else None)
        if rf is not None:
            sfx = sfx or ','.join(sorted({x for c in rf.values() for x in c.values() if x}))
        ladder = [x for x in sfx.split(',') if x]
        exists = lambda n: os.path.exists(os.path.join(bank.PLAN_DIR, n + '.npz'))
        if rungs is not None:                          # a ladder arm: per car, the plans and its own rung's variants
            per_car = {c: list(dict.fromkeys(ev_names + [n + rungs[c].get(n.split('_b')[0], '') for n in ev_names
                                                         if exists(n + rungs[c].get(n.split('_b')[0], ''))]))
                       for c in args['cars']}
        else:
            names = ev_names + [n + sfx for n in ev_names if sfx and exists(n + sfx)]
            per_car = {c: names for c in args['cars']}
        plans = {p['name']: p for p in bank.load_bank(list(dict.fromkeys(sum(per_car.values(), []))))}
        base = lambda n: next((n[:-len(x)] for x in ladder if n.endswith(x)), n)
        jobs = []
        for car in args['cars']:
            f = os.path.join(d, f'{car}_policy.npz')
            pol = load(f) if os.path.exists(f) else load(args['policy'])
            for name in per_car[car]:
                pl = plans[name]
                rs = np.random.default_rng(a.seed + EVAL.index(base(name)))
                for k in range(a.runs):
                    jobs.append(dict(car_name=car, plan=pl, policy=pol, laps=a.laps, s0=(k / a.runs) * float(pl['length']),
                                     pert=rs.uniform(-0.5, 0.5, 5), seed=a.seed * 3 + k))
        res = rc.run_many(jobs)
        ev = {}
        for j, r in zip(jobs, res):
            m = rc.lap_metrics(r, j['plan'])
            m.pop('per_lap_ey', None)
            ev.setdefault(j['car_name'], {}).setdefault(j['plan']['name'], []).append(m)
        summ = {c: {n: dict(success=float(np.mean([m['success'] for m in ms])), fail=float(np.mean([m['failed'] for m in ms])),
                            wall=float(np.mean([m.get('crash') == 'wall' for m in ms])),
                            peak_ey=float(np.max([m.get('peak_ey', 0.0) for m in ms])),
                            rms_ey=float(np.mean([m['rms_ey'] for m in ms if not m['failed']])) if any(not m['failed'] for m in ms) else None,
                            rms_dbeta=float(np.mean([m['rms_dbeta'] for m in ms if not m['failed']])) if any(not m['failed'] for m in ms) else None,
                            pace=float(np.mean([m['pace'] for m in ms if not m['failed']])) if any(not m['failed'] for m in ms) else None,
                            mean_abs_beta=float(np.mean([m['mean_abs_beta'] for m in ms if not m['failed']])) if any(not m['failed'] for m in ms) else None,
                            n=len(ms)) for n, ms in v.items()} for c, v in ev.items()}
        json.dump(summ, open(out_f, 'w'), indent=1)
        adaptive(d, summ, a.tag, rf)
        print(arm, 'done', flush=True)


if __name__ == '__main__':
    main()
