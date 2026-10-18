#!/bin/bash
# final_test_3301.sh (2026-10-18): the third frozen pogo test pass on a FRESH seed (3301; 1301 and 2301 are spent).
# What changed, chosen on validation only (NOTES, "Perturbed training starts"): the hardware stage trains from perturbed
# drops (--train-pert 1.0) whose starts are drawn once per goal and kept across iterations (--pert-fixed 1). Validation,
# 36 trials: ours + DR (drNomInit) 99.5, PPO+DR + stage 98.8, ours nominal 95.3, RMA zero-shot 99.8.
# Pre-registered headline: the one protocol (PF: 3 s trials, fixed perturbed starts, 6 x 6 = 36 trials) for every arm that
# uses our hardware stage; the old-protocol arms (curve/it6) are re-evaluated as secondary rows.
# -> <arm>/summary_test3301.json, results_table_test3301.txt
cd "$(dirname "$0")"
export POGO_EVAL_SET=test POGO_TEST_SEED=3301 POGO_EVAL_TAG=3301 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=~/miniconda3/envs/pogo/bin/python
echo "$(date '+%F %T') test pass 3301 start" >> logs/final_test_3301.log
ARMS=(runs/v3/expert_zs)
for s in 0 1 2; do
  for n in drNomInit nom ppo_dr; do ARMS+=(runs/v3/${n}_s${s}_zs runs/v3/${n}_s${s}_PF runs/v3/${n}_s${s}_curve/it6); done
  ARMS+=(runs/v3/drLK20_s${s}_curve/it6 runs/v3/rmaR1_s${s}_zs runs/v3/fadaF2_s${s}_zs runs/v3/fadaF2_s${s}_hw24 runs/v3/ppo_s${s}_ep48 runs/v3/ppo_s${s}_ep384)
done
printf '%s\n' "${ARMS[@]}" | xargs -P 8 -n 1 $PY reeval.py >> logs/final_test_3301.log 2>&1
$PY table_v3.py --file summary_test3301.json > results_table_test3301.txt
echo "$(date '+%F %T') FINAL_TEST_3301_DONE" >> logs/final_test_3301.log
