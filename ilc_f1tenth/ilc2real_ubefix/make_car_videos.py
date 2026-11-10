#!/usr/bin/env python3
"""
make_car_videos.py [--root DIR] [--jobs N]: every car video from the recorded trajectories in DIR/trajectories
(record_car.py) into src/paper/car/videos/ (another --root: DIR/videos/), one folder per method (the quadruped's make_animations.py layout):

  00_compare_all_methods/<car>_<plan>   every method side by side on one plan, with the sideslip against the plan's
  <NN>_<method>/hardware_stage_<car>    the method's hardware stage, iteration by iteration (one panel per plan)
  <NN>_<method>/<when>_eval_<car>       the method on the 8 evaluation plans at once (<when>: zeroshot, after_2laps / after_10laps,
                                        after_10laps)
MP4 (H.264, pausable), real time. Plus videos/README.md.
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--root', default=HERE)
ap.add_argument('--jobs', type=int, default=6)
ap.add_argument('--cars', nargs='+', default=['real_nom', 'real_mass', 'real_mu', 'real_act', 'real_lag'])
ap.add_argument('--only', choices=('hwstage', 'eval', 'compare'))
a = ap.parse_args()
TR = os.path.join(os.path.abspath(a.root), 'trajectories')
OUT = (os.path.normpath(os.path.join(HERE, '..', '..', 'paper', 'car', 'videos')) if os.path.abspath(a.root) == HERE
       else os.path.join(os.path.abspath(a.root), 'videos'))   # src/paper
# folder: (label, hardware-stage recording or None, [(when, eval recording)]). The paper's final rows (2026-10-11): our
# headline = the stopping-rule network (zero-shot, the LOW 10 and the asymptotic 48 real laps); the RMA / FADA rows = only
# the best non-oracle and the best oracle of each (src/paper/baselines.json, at their own best real budget)
import json
_BJ = json.load(open(os.path.join(HERE, '..', '..', 'paper', 'baselines.json')))['car']
_pk = lambda r: (f"{r.split('_')[0].upper()}, best {'oracle' if _BJ[r]['best']['oracle'] else 'non-oracle'}"
                 f" ({_BJ[r]['best']['best_budget'] or 'zero-shot, 0'} real laps)")
METHODS = {
    '01_ours_stopping_rule': ('ours + DR (task objective, stopping rule)', 'hwstage_ours_stop',
                              [('zeroshot', 'eval_ours_stop_zeroshot'), ('after_10laps', 'eval_ours_stop_10laps'),
                               ('after_48laps', 'eval_ours_stop_48laps')]),
    '02_ours_plus_dr_scratch': ('our learner + DR (B)', 'hwstage_ours_dr_b',
                                [('zeroshot', 'eval_ours_dr_b_zeroshot'), ('after_10laps', 'eval_ours_dr_b_10laps')]),
    '03_ours_plus_dr_finetune': ('our learner + DR (A)', 'hwstage_ours_dr_a',
                                 [('zeroshot', 'eval_ours_dr_a_zeroshot'), ('after_10laps', 'eval_ours_dr_a_10laps')]),
    '04_ours_nominal_sim': ('ours, nominal sim', 'hwstage_ours', [('zeroshot', 'eval_ours_zeroshot'),
                                                                 ('after_10laps', 'eval_ours_10laps')]),
    '05_ppo_dr': ('PPO+DR', 'hwstage_ppo_dr', [('zeroshot', 'eval_ppo_dr_zeroshot'), ('after_10laps', 'eval_ppo_dr_10laps')]),
    '06_rma_best_nonoracle': (_pk('rma_nonoracle'), None, [('best', 'eval_rma_nonoracle')]),
    '07_rma_best_oracle': (_pk('rma_oracle'), None, [('best', 'eval_rma_oracle')]),
    '08_fada_best_nonoracle': (_pk('fada_nonoracle'), None, [('best', 'eval_fada_nonoracle')]),
    '09_fada_best_oracle': (_pk('fada_oracle'), None, [('best', 'eval_fada_oracle')]),
    '10_plan_lqr': ('plan LQR', None, [('zeroshot', 'eval_plan_lqr')]),
}
WHEN = dict(zeroshot='zero-shot', after_2laps='after 2 real laps', after_10laps='after 10 real laps (low budget)',
            after_48laps='after 48 real laps (asymptotic budget)', best='at its best real-data budget')
COMPARE = [('eval_ours_stop_10laps', 'ours (stopping rule) + 10'), ('eval_ours_dr_b_10laps', 'ours+DR (B) + 10'),
           ('eval_ours_dr_a_10laps', 'ours+DR (A) + 10'), ('eval_ours_10laps', 'ours, nominal + 10'),
           ('eval_ppo_dr_10laps', 'PPO+DR + 10 (our stage)'), ('eval_rma_nonoracle', 'RMA, best non-oracle'),
           ('eval_rma_oracle', 'RMA, best oracle'), ('eval_fada_nonoracle', 'FADA, best non-oracle'),
           ('eval_fada_oracle', 'FADA, best oracle'), ('eval_plan_lqr', 'plan LQR')]
f = lambda n: os.path.join(TR, n + '.npz')
anim = [sys.executable, os.path.join(HERE, 'animate_car.py')]
jobs, missing = [], []
for folder, (label, hw, evals) in METHODS.items():
    d = os.path.join(OUT, folder)
    if hw and a.only in (None, 'hwstage'):
        if os.path.exists(f(hw)):
            jobs += [anim + ['hwstage', f(hw), '--car', c, '--stride', '3', '--out', os.path.join(d, f'hardware_stage_{c}.mp4')]
                     for c in a.cars]
        else:
            missing.append(hw)
    for when, e in evals if a.only in (None, 'eval') else []:
        if os.path.exists(f(e)):
            jobs += [anim + ['eval', f(e), '--car', c, '--label', f'{label}, {WHEN[when]}', '--stride', '3',
                             '--out', os.path.join(d, f'{when}_eval_{c}.mp4')] for c in a.cars]
        else:
            missing.append(e)
if a.only in (None, 'compare'):
    have = [(f(n), lab) for n, lab in COMPARE if os.path.exists(f(n))]
    for c in a.cars:
        for p in ('mocap_square2fast_b25', 'mocap_figfast_b25'):
            jobs.append(anim + ['compare', *[h[0] for h in have], '--labels', *[h[1] for h in have], '--car', c,
                                '--plan', p, '--stride', '2',
                                '--out', os.path.join(OUT, '00_compare_all_methods', f'{c}_{p.replace("mocap_", "")}.mp4')])


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = [l for l in (r.stdout + r.stderr).splitlines() if 'frames ->' in l or 'Error' in l or 'Traceback' in l]
    return ' | '.join(out).replace(OUT + '/', '') or f'(no output) {cmd[-1]}'


if missing:
    print('not recorded (skipped):', ' '.join(missing))
print(f'{len(jobs)} videos, {a.jobs} at a time', flush=True)
with ThreadPoolExecutor(a.jobs) as ex:
    for m in ex.map(run, jobs):
        print(m, flush=True)
os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, 'README.md'), 'w').write('''# F1TENTH ILC2Real car videos

Top-down views of the recorded runs (`src/ilc_f1tenth/ilc2real/trajectories/`, record_car.py) on the five multi-body "real" cars, MP4 (H.264;
pause and scrub in any player), real time. Grey: the plan; the path is coloured by |sideslip| (light 0 to dark 35 deg);
blue: the car and its heading; red: its velocity (the angle between them is the sideslip). Regenerate:
`python make_car_videos.py --root <results root>` after `record_car.py`.

| folder | contents |
|---|---|
| `00_compare_all_methods/<car>_<plan>.mp4` | every method side by side on the beta-25 plan of one track, 5 laps without reset, with the sideslip against the plan's |
| `01_ours_stopping_rule/` | the headline: our learner + DR (task objective), snapshot chosen by the sim-only stopping rule: the hardware stage (10 laps), the 8 evaluation plans zero-shot, after 10 (low budget) and after 48 real laps (asymptotic budget) |
| `02_ours_plus_dr_scratch/`, `03_ours_plus_dr_finetune/`, `04_ours_nominal_sim/` | our learner + DR (B: from scratch, A: DR fine-tune) and on the nominal sim only: hardware stage, zero-shot / after 10 laps |
| `05_ppo_dr/` | PPO+DR, then our hardware stage: zero-shot and after 10 real laps |
| `06_rma_best_nonoracle/` ... `09_fada_best_oracle/` | the RMA / FADA baselines: per method the best non-oracle and the best oracle variant (`src/paper/baselines.json`), at its own best real-data budget |
| `10_plan_lqr/` | the plans' LQR |

Files: `hardware_stage_<car>.mp4` (one panel per training plan, iteration by iteration, earlier iterations faded) and
`<when>_eval_<car>.mp4` (the 8 evaluation plans at once: beta 25 / 21 / 14 / 0 on both tracks, 21 and 14 never trained
on; 5 laps without reset). Cars: real_nom, real_mass (+25 % mass), real_mu (friction x0.8), real_act (actuators), real_lag
(tire stiffness / steering lag).
''')
