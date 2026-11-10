"""orca_eval.py DIR [DIR ...] (2026-11-05): the ORCA stage runs (orca_stage.sh) scored as the car table's screen_eval.py
(4 runs per car and plan, starts at k/4 of the lap, perturbation N(0, 0.5) from seed 701, sensing noise; score
max(rms e_y / 0.1, (1 - pace) / 0.1), a spin-out / stall = 5) on the practised orca43 (1 lap = 77 m) and the held-out
peanut (5 laps), its 0 (zero-shot) and 3 (6 trial laps), with F1T_DEV_TIME=1: leaving the 0.5 m envelope is timed (dev_time,
s; dev_frac of the run), not stopped. -> DIR/orca_eval.json {plan: {it: {car: {score, rms_ey, pace, dev_time, dev_frac, fail}}}}
(each the mean over the 4 runs)."""
import json
import os
import sys

D = '/home/henry/ilc_ws/src/ilc_f1tenth/ilc2real'
os.environ['F1T_DEV_TIME'] = '1'
for k, v in dict(F1T_MU='0.6', F1T_PLANS=D + '/plans_nom06', F1T_TRACKS=D + '/../lifted_linear_tire_20261009/tracks_tight',
                 F1T_SAFETY_EY='0.5', NUMBA_NUM_THREADS='1', OMP_NUM_THREADS='1', JAX_PLATFORMS='cpu').items():
    os.environ[k] = v                       # the plastic runs' environment (plastic_sl.sh / env_tight.sh), never the caller's
sys.path.insert(0, D)
import numpy as np  # noqa: E402

import bank  # noqa: E402
import policy_runtime as pr  # noqa: E402
import real_car as rc  # noqa: E402

CARS = ('n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense '
        'n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000').split()
PL = {'orca43_b0_free_a33': 1.0, 'mocap_peanutM_b0_free_a33': 5.0}      # plan: laps
SEED = 701


def main():
    plans = {p['name']: p for p in bank.load_bank(list(PL))}
    perts = np.random.default_rng(SEED).normal(size=(len(CARS), 1, 4, 5)) * 0.5
    for d in sys.argv[1:]:
        out = os.path.join(d, 'orca_eval.json')
        if os.path.exists(out):
            continue
        jobs, meta = [], []
        for k in (0, 3):
            for ci, c in enumerate(CARS):
                f = f'{d}/its/it{k}_{c}_policy.npz'
                if not os.path.exists(f):
                    continue
                pol = pr.load_policy(f)
                for pn, laps in PL.items():
                    L = float(plans[pn]['length'])
                    for r in range(4):
                        jobs.append(dict(car_name=c, plan=plans[pn], policy=pol, laps=laps, s0=r / 4 * L, pert=perts[ci, 0, r],
                                         seed=SEED + r))
                        meta.append((pn, k, c))
        res = rc.run_many(jobs, procs=int(os.environ.get('F1T_PROCS', '8')))
        o = {}
        for (pn, k, c), r in zip(meta, res):
            lm = rc.lap_metrics(r, plans[pn])
            v = 5.0 if lm.get('failed') else max(lm['rms_ey'] / 0.1, (1 - lm['pace']) / 0.1)
            o.setdefault(pn, {}).setdefault(str(k), {}).setdefault(c, []).append(
                dict(score=v, rms_ey=lm.get('rms_ey', np.nan), pace=lm.get('pace', np.nan), dev_time=lm.get('dev_time', 0.0),
                     dev_frac=lm.get('dev_frac', 0.0), fail=float(lm.get('failed', True))))
        o = {pn: {k: {c: {q: float(np.nanmean([x[q] for x in v])) for q in v[0]} for c, v in kk.items()} for k, kk in ks.items()}
             for pn, ks in o.items()}
        json.dump(o, open(out, 'w'), indent=1)
        print(d, 'done', flush=True)


if __name__ == '__main__':
    main()
