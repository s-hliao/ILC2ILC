"""fig1_eval_prac.py OUT.json (= fig1_eval.py on the practised plans) | POLICY [POLICY ...] (2026-11-06, paper/FIGURE_PLAN_1106.md figure 1): each policy file scored zero-shot, as
ladder_eval.py scores the main tables (4 starts at k/4 of the lap, perturbation N(0, 0.5) from seed 701, sensing noise; score max(rms
e_y / 0.1, (1 - pace) / 0.1), a spin-out = 5; F1T_DEV_TIME=1; set F1T_SPIN_RHO=3 for the yaw-rate spin rule), on the figure's panels:
  A / B / C   mb_car.py's pre-registered coverage tiers (2026-10-29; 12 cars each), the held-out peanut at the plan speed
  plastic     the 12 plastic cars of the main tables, held-out peanut _a33
  rubber      the 2 rubber cars, held-out peanut _x120
FIG1_LAPS (default 5) laps per run, F1T_PROCS workers. Existing entries in OUT.json are kept (resumable).
-> OUT.json {policy path: {panel: {car: {score, rms_ey, pace, fail}}}}, each the mean over the 4 runs."""
import json
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))
os.environ['F1T_DEV_TIME'] = '1'
for k, v in dict(F1T_MU='0.6', F1T_PLANS=D + '/plans_nom06', F1T_TRACKS=D + '/../lifted_linear_tire_20261009/tracks_tight',
                 F1T_SAFETY_EY='0.5', NUMBA_NUM_THREADS='1', OMP_NUM_THREADS='1', JAX_PLATFORMS='cpu').items():
    os.environ[k] = v
sys.path.insert(0, D)
import numpy as np  # noqa: E402

import bank  # noqa: E402
import mb_car  # noqa: E402
import policy_runtime as pr  # noqa: E402
import real_car as rc  # noqa: E402

SEED = 701
LAPS = float(os.environ.get('FIG1_LAPS', '5'))
TIER = {t: sorted(k for k in mb_car.REAL_CARS if k.startswith(f'tier{t.lower()}_')) for t in 'ABC'}
PANELS = {**{t: (cs, 'mocap_square2fast_b0_free') for t, cs in TIER.items()},          # (fig1_eval_prac.py, 2026-11-08: the PRACTISED
          'plastic': (list(mb_car.PLASTIC_N04), 'mocap_square2fast_b0_free_a45'),          # plans of figure 2's panels, practice-only
          'rubber': (list(mb_car.RUBBER_N06), 'mocap_fig8R100_b0_free_x120')}              # protocol; was the held-out peanut)


def main():
    out = sys.argv[1]
    res = json.load(open(out)) if os.path.exists(out) else {}
    plans = {p['name']: p for p in bank.load_bank(sorted({pn for _, pn in PANELS.values()}))}
    for f in sys.argv[2:]:
        if f in res:
            continue
        pol = pr.load_policy(f)
        jobs, meta = [], []
        for panel, (cars, pn) in PANELS.items():
            L = float(plans[pn]['length'])
            perts = np.random.default_rng(SEED).normal(size=(len(cars), 1, 4, 5)) * 0.5
            for ci, c in enumerate(cars):
                for r in range(4):
                    jobs.append(dict(car_name=c, plan=plans[pn], policy=pol, laps=LAPS, s0=r / 4 * L, pert=perts[ci, 0, r], seed=SEED + r))
                    meta.append((panel, pn, c))
        rs = rc.run_many(jobs, procs=int(os.environ.get('F1T_PROCS', '8')))
        o = {}
        for (panel, pn, c), r in zip(meta, rs):
            lm = rc.lap_metrics(r, plans[pn])
            v = 5.0 if lm.get('failed') else max(lm['rms_ey'] / 0.1, (1 - lm['pace']) / 0.1)
            o.setdefault(panel, {}).setdefault(c, []).append(dict(score=v, rms_ey=lm.get('rms_ey', np.nan), pace=lm.get('pace', np.nan),
                                                                  fail=float(lm.get('failed', True))))
        res[f] = {p: {c: {q: float(np.nanmean([x[q] for x in v])) for q in v[0]} for c, v in cs.items()} for p, cs in o.items()}
        json.dump(res, open(out + '.tmp', 'w'))
        os.replace(out + '.tmp', out)
        print(f, {p: round(float(np.mean([v['score'] for v in cs.values()])), 3) for p, cs in res[f].items()}, flush=True)


if __name__ == '__main__':
    main()
