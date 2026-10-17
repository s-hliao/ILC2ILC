#!/bin/bash
# noise_arms.sh (ube, 2026-10-17): is the quad hardware stage limited by jump-to-jump noise? (TR_RESULTS.md: rho < 0 in
# half the steps). VALIDATION goals (planeval). 4 robots.
#   noise5     --tr 1, --reps 5 --iters 2 (60 jumps): the spread of J over 5 jumps of the SAME policy per goal (iteration
#              0) against the predicted improvement, and rho on 5-jump means (iteration 1). Seed 9001.
#   recipe_r2  drB_hw24's recipe, --reps 2 --iters 2 (24 jumps: the same budget as recipe_sd*, 2 jumps per step)
#   tr_r2      the trust region, --reps 2 --iters 2 (24 jumps)
P=/home/henry/ilc_ws/log/dilc/plane; S=/home/henry/ilc_ws/src/ilc_mjx; PY=~/miniconda3/envs/ilcmjx/bin/python
export ILC_QUAD_OVERLAY=/ilc_ws/log/dilc/plane/fixbox/overlay JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_ilc JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
COMMON="--update gn --gn-delta 0.1 --est-window 0.025 --clear 0.03 --stall 3 --stall-sat 0.2 --sat-project --target-base own
   --eval planeval --perturbed --skip-start-eval --goals 0.45 0.60 0.50,0.15 0.60,0.10 0.50,0.05 0.575,0.15
   --policy $P/drB_s2/policy --member g0_plane_it30"
RECIPE="--gn-beta 0.3 --step-rms 0.05 --bold 0.3 --gate 1 1.5 0.4"
TRA="--gn-beta 1.0 --step-rms 0.05 --tr 1 --anchor-w 0"
cd $S; n=0; g=0; NPAR=${NPAR:-4}
run() {  # name seed args robot
  [ -f $P/tr/$1_sd$2/$4/eval_planeval_final.json ] && return
  $PY scripts/deploy.py $COMMON $3 --seed $2 --robots $4 --out $P/tr/$1_sd$2 --gpu $g 2>&1 | grep -v "warp\|bfc_alloc" > $P/tr/$1_sd$2_$4.log &
  g=$(( (g + 1) % 3 )); n=$((n + 1)); if (( n % NPAR == 0 )); then wait -n; n=$((n - 1)); fi
}
for rb in real_r1 real_s1 real_r4 real_r5; do run noise5 9001 "$TRA --reps 5 --iters 2" $rb; done
for seed in 9001 9002 9003; do for rb in real_r1 real_s1 real_r4 real_r5; do
  run recipe_r2 $seed "$RECIPE --reps 2 --iters 2" $rb; run tr_r2 $seed "$TRA --reps 2 --iters 2" $rb
done; done
wait
echo "$(date +%T) NOISE_ARMS_DONE"
