#!/bin/bash
# final_test.sh: the single frozen test pass of the 2026-10-17 round (POGO_EVAL_SET=test, goals + drops from seed 1301).
# Every arm was chosen on the validation goals before this runs. Each arm's frozen networks are re-flown by reeval.py
# -> <arm>/summary_test.json; the two arms without saved networks (per-goal ILC: feedforwards; the RMA oracle: the
# true z) are rerun in test mode into runs/v3test/. Then table_v3.py --file summary_test.json.
cd "$(dirname "$0")"
export POGO_EVAL_SET=test OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=~/miniconda3/envs/pogo/bin/python
mkdir -p logs runs/v3test
echo "$(date '+%F %T') final test pass start" >> logs/final_test.log
ARMS=(runs/v3/expert_zs runs/v3/rma_s0_zs)
for s in 0 1 2; do
  ARMS+=(runs/v3/nom_s${s}_zs runs/v3/drLK20_s${s}_zs runs/v3/ppo_dr_s${s}_zs runs/v3/rmaR1_s${s}_zs
         runs/v3/fadaF2_s${s}_zs runs/v3/fadaF2_s${s}_hw24)
  for arm in nom drLK20 ppo_dr; do
    ARMS+=(runs/v3/${arm}_s${s}_curve/it2 runs/v3/${arm}_s${s}_curve/it4 runs/v3/${arm}_s${s}_curve/it6
           runs/v3/${arm}_s${s}_curve/it8)
  done
  for ep in 24 48 96 192 384; do ARMS+=(runs/v3/ppo_s${s}_ep${ep}); done
done
printf '%s\n' "${ARMS[@]}" | xargs -P 8 -n 1 $PY reeval.py >> logs/final_test.log 2>&1
$PY -u goal_ilc.py --out runs/v3test/goalilc --gpu 2 >> logs/final_test_goalilc.log 2>&1
$PY -u ppo_real.py --mode rma_oracle --seed 0 --checkpoints 24 --out runs/v3test --tag _s0 >> logs/final_test_oracle.log 2>&1
for d in runs/v3test/*/; do [ -f "$d/summary.json" ] && cp "$d/summary.json" "$d/summary_test.json"; done
$PY table_v3.py --file summary_test.json > results_table_test.txt
$PY table_v3.py --root runs/v3test --file summary_test.json >> results_table_test.txt
echo "$(date '+%F %T') FINAL_TEST_DONE" >> logs/final_test.log
