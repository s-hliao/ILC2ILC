#!/bin/bash
# car_redo_test3701.sh (2026-10-18): a second single frozen TEST pass of the car comparison on a fresh, never-used reeval seed
# (3701), now including the row that was missing: RMA (task) + our 10-lap ladder (car_redo_rma.sh, rmaL_task_s<k>_ladR_lap10).
# Spent seeds: 1701 (every car number before 2026-10-17), 301 (validation), 2701 (the first test pass, car_redo_test.sh).
# What is FROZEN before this runs (written before the first launch):
#   - the arms: every network, per-car ILC update and ladder rung was produced before this script, by car_redo.sh and
#     car_redo_rma.sh; nothing is trained, tuned or re-run here. RMA + ladder uses the identical LAD recipe as ours and
#     PPO+DR + ladder, run once (one configuration, 3 seeds), chosen on nothing;
#   - the arm list: every headline arm of the seed-2701 test table (ours zero-shot / + ladder, PPO+DR (task) zero-shot / on
#     our rungs / + ladder, RMA (task) zero-shot / on our rungs) plus RMA (task) + our ladder, all 3 seeds each, all reported;
#   - the protocol: reeval.py as before (12 runs x 5 chained laps per car and plan, the 8 cars, the b25 success metric,
#     baselines on our rungs = --rungs-from v3drBtL_s<k>_ladR_lap10), table_redo.py (incl. its paired cluster bootstrap);
#   - seed 3701 is used once; nothing is selected on its result.
# Checks first, stop on failure: (1) plan-translation invariance as in car_redo_test.sh -- re-score validation arms (ours +
# ladder, and the RMA zero-shot arm, which also exercises the RMA loader) on seed 301 with the current plan files and
# require the validation numbers exactly; (2) the RMA + ladder networks load as RMA (actor + the frozen adaptation module).
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu
P=${PROCS:-10}
for A in v3drBtL_s0_ladR_lap10 rmaL_task_s0_zs8; do
  F1T_PROCS=$P $PY -u reeval.py $A --seed 301 --tag _valcheck3701 >> logs/reeval_redo_valcheck3701.log 2>&1
done
$PY - <<PY || { echo "$(date '+%F %T') ABORT: a check failed"; exit 1; }
import json, sys
import reeval
ok = True
for A in ('v3drBtL_s0_ladR_lap10', 'rmaL_task_s0_zs8'):
    a = json.load(open(f'runs/hw/{A}/eval_big_val.json')); b = json.load(open(f'runs/hw/{A}/eval_big_valcheck3701.json'))
    d = max(abs(a[c][p]['success'] - b[c][p]['success']) for c in a for p in a[c] if p in b.get(c, {}))
    print(A, 'max |success diff| validation vs now:', d); ok &= d <= 1e-9
for s in (0, 1, 2):
    pol = reeval.load(f'runs/hw/rmaL_task_s{s}_ladR_lap10/ax:mu=0.6_policy.npz')
    print(f'rmaL_task_s{s}_ladR_lap10 loads as', sorted(pol) if isinstance(pol, dict) else 'a bare MLP')
    ok &= isinstance(pol, dict) and 'adapt' in pol
sys.exit(0 if ok else 1)
PY
ARMS=""; for s in 0 1 2; do ARMS+="v3drBtL_s${s}_zs8 v3drBtL_s${s}_ladR_lap10 ppoL_task_s${s}_zs8 ppoL_task_s${s}_ladR_lap10 rmaL_task_s${s}_zs8 rmaL_task_s${s}_ladR_lap10 "; done
echo "$(date '+%F %T') test pass start (seed 3701)"
F1T_PROCS=$P $PY -u reeval.py $ARMS --seed 3701 --tag _test3701 > logs/reeval_redo_test3701.log 2>&1
for s in 0 1 2; do
  for b in ppoL_task_s${s}_zs8 rmaL_task_s${s}_zs8; do
    F1T_PROCS=$P $PY -u reeval.py $b --seed 3701 --tag _test3701_rungs --rungs-from v3drBtL_s${s}_ladR_lap10 >> logs/reeval_redo_test3701.log 2>&1
  done
done
$PY table_redo.py --tag _test3701 > runs/hw/redo_test3701_table.txt
echo "$(date '+%F %T') CAR_REDO_TEST3701_DONE"
