#!/bin/bash
# run_stage.sh (2026-10-22): the frozen PF hardware stage (scheduler_fix.py stages pfix / pfix2: --train-pert 1.0
# --pert-fixed 1 --iters 6, 3 s trials at the 6 TRAIN_GOALS) for ours nominal, ours + DR (drNomInit) and PPO+DR, seeds
# 0-2, re-run on the CPU with per-iteration checkpoints (--save-at 1..5; no intermediate evaluation). The final evaluation
# scores the VALIDATION set (POGO_EVAL_SET unset = val, seed 301). Outputs runs/<arm>_s<s>_PF/{it<k>/,}<robot>_policy.npz.
cd /home/henry/ilc_ws/src/ilc_pogo
OUT=/home/henry/ilc_ws/log/dilc/flows/pogo
PY=~/miniconda3/envs/pogo/bin/python
unset POGO_EVAL_SET POGO_TEST_SEED POGO_EVAL_TAG
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" XLA_PYTHON_CLIENT_PREALLOCATE=false
jobs=()
for s in 0 1 2; do
  jobs+=("nom_s$s runs/nom_s$s/policy_pick.npz" "drNomInit_s$s runs/drNomInit_s$s/policy_pick.npz" "ppo_dr_s$s runs/ppo_dr_s$s/policy_final.npz")
done
printf '%s\n' "${jobs[@]}" | xargs -P 8 -L 1 bash -c '[ -f '$OUT'/runs/$0_PF/summary.json ] || '$PY' -u hw_stage.py --policy $1 --out '$OUT'/runs/$0_PF --train-pert 1.0 --pert-fixed 1 --iters 6 --save-at 1 2 3 4 5 > '$OUT'/logs/$0_PF.log 2>&1'
echo "$(date '+%F %T') STAGE_DONE" >> $OUT/logs/run_stage.log
