#!/bin/bash
# final_test_2301.sh (2026-10-18): the second frozen pogo test pass on a FRESH seed (2301; 1301 is spent), for ours + DR
# started from the nominal sim network (drNomInit; chosen on validation: 94.3 vs drLK20's 91.1 % at 36 trials) together
# with the headline arms re-evaluated on the same seed -> <arm>/summary_test2301.json, results_table_test2301.txt.
cd "$(dirname "$0")"
export POGO_EVAL_SET=test POGO_TEST_SEED=2301 POGO_EVAL_TAG=2301 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=~/miniconda3/envs/pogo/bin/python
echo "$(date '+%F %T') test pass 2301 start" >> logs/final_test_2301.log
ARMS=(runs/v3/expert_zs)
for s in 0 1 2; do
  for n in drNomInit nom drLK20 ppo_dr; do ARMS+=(runs/v3/${n}_s${s}_zs runs/v3/${n}_s${s}_curve/it6); done
  ARMS+=(runs/v3/rmaR1_s${s}_zs runs/v3/fadaF2_s${s}_zs runs/v3/fadaF2_s${s}_hw24 runs/v3/ppo_s${s}_ep48 runs/v3/ppo_s${s}_ep384)
done
printf '%s\n' "${ARMS[@]}" | xargs -P 6 -n 1 $PY reeval.py >> logs/final_test_2301.log 2>&1
$PY table_v3.py --file summary_test2301.json > results_table_test2301.txt
echo "$(date '+%F %T') FINAL_TEST_2301_DONE" >> logs/final_test_2301.log
