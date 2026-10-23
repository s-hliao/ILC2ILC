#!/bin/bash
# parts_arms.sh (matcha, 2026-10-19): which async / sensing group stops the quad hardware stage from converging? Partial
# twins real_{r1,s1,r4}_{a,s,m} (dilc_execute.py NAMED: clean dynamics + actuation / sensing / mocap only), recipe arm.
# (copied from clean_arms.sh:)
# Clean twins real_{r1,s1,r4,r5}c (dilc_execute.py NAMED: the validation robots' dynamics, no async / sensing effects).
# recipe vs ctl_random (same-size random direction) on the validation goals, recipe settings as landing_arms.sh, no
# --perturbed (its perturbation files exist only for the named robots). Outputs $P/clean/<arm>_sd<seed>.
#   SEEDS="9001" bash clean_arms.sh ; GPU0=1 (matcha GPUs 1-3)
P=/home/henry/ilc_ws/log/dilc/plane; S=/home/henry/ilc_ws/src/ilc_mjx; PY=~/miniconda3/envs/ilcmjx/bin/python
export ILC_QUAD_OVERLAY=/ilc_ws/log/dilc/plane/fixbox/overlay JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_ilc JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
GOALS="0.45 0.60 0.50,0.15 0.60,0.10 0.50,0.05 0.575,0.15"; ITERS=4; SEEDS=${SEEDS:-"9001 9002 9003"}
ROBOTS=${ROBOTS:-"real_r1_a real_s1_a real_r4_a real_r1_s real_s1_s real_r4_s real_r1_m real_s1_m real_r4_m"}; OUT=$P/parts
COMMON="--update gn --gn-delta 0.1 --est-window 0.025 --clear 0.03 --stall 3 --stall-sat 0.2 --sat-project --target-base own
   --eval planeval --goals $GOALS --iters $ITERS --policy $P/drB_s2/policy --member g0_plane_it30
   --gn-beta 0.3 --step-rms 0.05 --bold 0.3 --gate 1 1.5 0.4"
declare -A ARM=( [recipe]="" [ctl_random]="--control random" )
ARMS=${ARMS:-"recipe"}
mkdir -p $OUT; cd $S; n=0; g=0; NPAR=${NPAR:-3}
for seed in $SEEDS; do
  for arm in $ARMS; do
    for rb in $ROBOTS; do
      out=$OUT/${arm}_sd$seed
      [ -f $out/$rb/eval_planeval_final.json ] && continue
      START=--skip-start-eval; [ $arm = recipe ] && START=""      # zero-shot on the clean twins, per seed
      $PY scripts/deploy.py $COMMON ${ARM[$arm]} --seed $seed $START --robots $rb --out $out \
        --gpu $((g + ${GPU0:-0})) 2>&1 | grep -v "warp\|bfc_alloc" > $OUT/${arm}_sd${seed}_$rb.log &
      g=$(( (g + 1) % ${NGPU:-3} )); n=$((n + 1))
      if (( n % NPAR == 0 )); then wait -n; n=$((n - 1)); fi
    done
  done
done
wait
echo "$(date +%T) PARTS_ARMS_DONE"
