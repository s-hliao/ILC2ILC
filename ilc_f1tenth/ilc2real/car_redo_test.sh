#!/bin/bash
# car_redo_test.sh (2026-10-17): the single frozen TEST pass of car_redo.sh's arms on a fresh, never-used reeval seed (2701).
# Nothing is selected on this result. The b25 plan files were translated (+0.25 m x; figfast also +0.3 m y) for the lab
# room by commit 7b5d9a6 after the validation pass; first re-score one validation arm with the translated plans and stop
# unless it reproduces the validation numbers (the car's errors are plan-relative, so a translation must change nothing).
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
A=v3drBtL_s0_ladR_lap10
F1T_PROCS=14 $PY -u reeval.py $A --seed 301 --tag _valcheck > logs/reeval_redo_valcheck.log 2>&1
$PY - <<PY || { echo "$(date '+%F %T') ABORT: the translated plans do not reproduce validation"; exit 1; }
import json, sys
a = json.load(open('runs/hw/$A/eval_big_adaptive_val.json')); b = json.load(open('runs/hw/$A/eval_big_adaptive_valcheck.json'))
d = max(abs(a[c][p]['success'] - b[c][p]['success']) for c in a for p in a[c] if p in b.get(c, {}))
print('max |success diff| validation vs translated plans:', d); sys.exit(0 if d <= 1e-9 else 1)
PY
ARMS=""; for s in 0 1 2; do ARMS+="v3drBtL_s${s}_zs8 v3drBtL_s${s}_ladR_lap10 ppoL_task_s${s}_zs8 ppoL_task_s${s}_ladR_lap10 rmaL_task_s${s}_zs8 "; done
echo "$(date '+%F %T') test pass start (seed 2701)"
F1T_PROCS=14 $PY -u reeval.py $ARMS --seed 2701 --tag _test > logs/reeval_redo_test.log 2>&1
for s in 0 1 2; do
  for b in ppoL_task_s${s}_zs8 rmaL_task_s${s}_zs8; do
    F1T_PROCS=14 $PY -u reeval.py $b --seed 2701 --tag _test_rungs --rungs-from v3drBtL_s${s}_ladR_lap10 >> logs/reeval_redo_test.log 2>&1
  done
done
$PY table_redo.py --tag _test > runs/hw/redo_test_table.txt
echo "$(date '+%F %T') CAR_REDO_TEST_DONE"
