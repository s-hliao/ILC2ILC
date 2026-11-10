#!/usr/bin/env python3
"""ladder_eval_launch.py RUN_DIR... (2026-11-08 ~01:30, written from matcha): ladder_eval.py with the STAGE TRIALS' / HARDWARE's start --
a slow rolling launch (hw_stage.py --sim-launch 1 --launch 1.0 8.0: v0 0.6 m/s, the launch controller until 1.0 m/s, current 8, then
the network) and --skip-below 0.5 (scored from the first sample whose measured speed > 0.5 m/s; was --skip-launch until 11-08 ~02:10). Why: ladder_eval drops the car AT PLAN
SPEED at s0 = r/4 L; on mocap_fig8R100_b0_free s0 = 0 is the plan's fastest point (3.63 m/s) on a curvature reversal, which spins every
mu <= 0.4 plastic car within 0.75 s (fig8_trace.py) -- a start no trial and no real flight ever has. Same 4 starts (s0 = r/4 L, the
lateral / heading part of the seed-701 perturbations), laps, cars, iterations, score (nerr, crash 5) as ladder_eval; adds the behaviour
metrics: signed sideslip beta * sign(r) in corners (|r| > 0.5, v > 1): p50, share < -3 deg (rear out), share < -10 deg.
Env as ladder_eval: LADDER_PLANS, LADDER_LAPS, LADDER_ITS, LADDER_CARS, LADDER_OUT (default ladder_launch.json), F1T_PROCS."""
import json
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))
os.environ['F1T_DEV_TIME'] = '1'
os.environ.setdefault('F1T_SPIN_RHO', '3')   # (2026-11-08 ~03:00) the stages' spin rule (prac/yaw/fig8x scripts export it); a bare caller had none
for k, v in dict(F1T_MU='0.6', F1T_PLANS=D + '/plans_nom06', F1T_TRACKS=D + '/../lifted_linear_tire_20261009/tracks_tight',
                 F1T_SAFETY_EY='0.5', NUMBA_NUM_THREADS='1', OMP_NUM_THREADS='1', JAX_PLATFORMS='cpu').items():
    os.environ[k] = v
sys.path.insert(0, D)
import numpy as np  # noqa: E402

import bank  # noqa: E402
import policy_runtime as pr  # noqa: E402
import real_car as rc  # noqa: E402

CARS = os.environ.get('LADDER_CARS', 'ax:mu=1.667|ax:mu=1.833,mass=1.1').split('|')
PL = {p: float(os.environ.get('LADDER_LAPS', '1')) for p in os.environ['LADDER_PLANS'].split()}
SEED = 701
LAUNCH = dict(v0=0.6, v_hand=1.0, current=8.0)
PER_STEP = ('x', 'x_true', 'obs', 'obs_true', 'mu', 'a', 'err', 'err_true', 'idx', 'applied')


def trim(r, skip=0.5):
    """As hw_stage.run_jobs (sim_launch) + launch_cut with --skip-below 0.5 (user 2026-11-08 "new mask is good"; every stage, sim and
    hardware): the network's part from the hand-over, then from the first sample whose MEASURED speed exceeds skip."""
    h = max(int(r.get('handover', 0)), 0)
    r = dict(r)
    for n in PER_STEP:
        if n in r:
            r[n] = np.asarray(r[n])[h:]
    xm = np.asarray(r['x'])
    ok = np.nonzero(xm[:, 3] > skip)[0] if len(xm) else []
    t0 = int(ok[0]) if len(ok) else 0
    for n in PER_STEP:
        if n in r:
            r[n] = r[n][t0:]
    return r


def slip(r):
    x = np.asarray(r['x_true'])
    if len(x) < 2:
        return dict(sb_p50=np.nan, rear3=np.nan, rear10=np.nan)
    v = np.hypot(x[:, 3], x[:, 4]); m = (v > 1.0) & (np.abs(x[:, 5]) > 0.5)
    b = np.degrees(np.arctan2(x[:, 4], np.maximum(x[:, 3], 0.05))) * np.sign(x[:, 5])
    if not m.any():
        return dict(sb_p50=np.nan, rear3=np.nan, rear10=np.nan)
    return dict(sb_p50=float(np.median(b[m])), rear3=float(np.mean(b[m] < -3)), rear10=float(np.mean(b[m] < -10)))


def main():
    plans = {p['name']: p for p in bank.load_bank(list(PL))}
    perts = np.random.default_rng(SEED).normal(size=(len(CARS), 1, 4, 5)) * 0.5
    for d in sys.argv[1:]:
        out = os.path.join(d, os.environ.get('LADDER_OUT', 'ladder_launch.json'))
        if os.path.exists(out):
            continue
        jobs, meta = [], []
        for k in tuple(int(x) for x in os.environ.get('LADDER_ITS', '0 3').split()):
            for ci, c in enumerate(CARS):
                f = f'{d}/its/it{k}_{c}_policy.npz'
                if not os.path.exists(f):
                    continue
                pol = pr.load_policy(f)
                for pn, laps in PL.items():
                    L = float(plans[pn]['length'])
                    for r in range(4):
                        jobs.append(dict(car_name=c, plan=plans[pn], policy=pol, laps=laps, s0=r / 4 * L, pert=perts[ci, 0, r],
                                         seed=SEED + r, launch=LAUNCH))
                        meta.append((pn, k, c))
        res = rc.run_many(jobs, procs=int(os.environ.get('F1T_PROCS', '8')))
        o = {}
        for (pn, k, c), r in zip(meta, res):
            r = trim(r)
            lm = rc.lap_metrics(r, plans[pn]) if len(r['err_true']) >= 2 else dict(failed=True)
            v = 5.0 if lm.get('failed') else max(lm['rms_ey'] / 0.1, (1 - lm['pace']) / 0.1)
            o.setdefault(pn, {}).setdefault(str(k), {}).setdefault(c, []).append(
                dict(score=v, rms_ey=lm.get('rms_ey', np.nan), pace=lm.get('pace', np.nan), dev_time=lm.get('dev_time', 0.0),
                     dev_frac=lm.get('dev_frac', 0.0), fail=float(lm.get('failed', True)), **slip(r)))
        o = {pn: {k: {c: {q: float(np.nanmean([x[q] for x in v])) for q in v[0]} for c, v in kk.items()} for k, kk in ks.items()}
             for pn, ks in o.items()}
        json.dump(o, open(out, 'w'), indent=1)
        print(d, 'done', flush=True)


if __name__ == '__main__':
    main()
