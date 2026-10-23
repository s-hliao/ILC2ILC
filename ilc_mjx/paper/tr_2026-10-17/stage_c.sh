#!/bin/bash
# stage_c.sh (2026-10-22): the quad hardware stage, stage-side fix, on any start network. Same setup as
# landing_arms.sh: VALIDATION goals (planeval, seed 301), 6 goals x 4 iterations = 24 jumps per robot, robots r1 s1 r4
# r5, hardware seeds 9001-9003. Arms:
#   recipe      drB_hw24's step settings
#   pool        recipe + goal-pooled error (--pool-err 0.5: half the per-mode median error of the flown goals; the
#               robot's landing bias is shared by its goals, DIAG §2) + neighbour targets (nbr05, the one arm that beat
#               the random control): averages the jump-to-jump noise over goals and makes the update goal-smooth
#   ctl_random  the causal control: the recipe with a same-size random-direction step
# NET=<policy dir> MEMBER=<npz member> OUT=<dir> CLR=<m> ARMS="..." bash stage_c.sh
# GPU0 offsets the GPU rotation over NGPU GPUs (default 1-3).
P=/home/henry/ilc_ws/log/dilc/plane; S=/home/henry/ilc_ws/src/ilc_mjx; PY=~/miniconda3/envs/ilcmjx/bin/python
export ILC_QUAD_OVERLAY=/ilc_ws/log/dilc/plane/fixbox/overlay JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_ilc JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
GOALS="0.45 0.60 0.50,0.15 0.60,0.10 0.50,0.05 0.575,0.15"; ITERS=4; SEEDS=${SEEDS:-"9001 9002 9003"}
ROBOTS="real_r1 real_s1 real_r4 real_r5"
: ${NET:?NET=policy dir} ${MEMBER:?MEMBER=npz member} ${OUT:?OUT=output dir}
COMMON="--update gn --gn-delta 0.1 --est-window 0.025 --clear ${CLR:-0.03} --stall 3 --stall-sat 0.2 --sat-project --target-base own
   --eval planeval --perturbed --goals $GOALS --iters $ITERS --policy $NET --member $MEMBER"
RECIPE="--gn-beta 0.3 --step-rms 0.05 --bold 0.3 --gate 1 1.5 0.4"
declare -A ARM=(
  [recipe]="$RECIPE"
  [pool]="$RECIPE --pool-err 0.5 --nbr-radius 0.05 --nbr-weight 0.5"
  [ctl_random]="$RECIPE --control random"
)
ARMS=${ARMS:-"recipe pool ctl_random"}
mkdir -p $OUT; cd $S; n=0; g=0; NPAR=${NPAR:-3}
for seed in $SEEDS; do
  for arm in $ARMS; do
    for rb in $ROBOTS; do
      out=$OUT/${arm}_sd$seed
      [ -f $out/$rb/eval_planeval_final.json ] && continue
      START=--skip-start-eval; [ $arm = recipe ] && [ $seed = 9001 ] && START=""      # one zero-shot reference per robot
      $PY scripts/deploy.py $COMMON ${ARM[$arm]} --seed $seed $START --robots $rb --out $out --gpu $((g + ${GPU0:-1})) 2>&1 \
        | grep -v "warp\|bfc_alloc" > $OUT/${arm}_sd${seed}_$rb.log &
      g=$(( (g + 1) % ${NGPU:-3} )); n=$((n + 1))
      if (( n % NPAR == 0 )); then wait -n; n=$((n - 1)); fi
    done
  done
done
wait
echo "$(date +%T) STAGE_C_DONE $OUT"
