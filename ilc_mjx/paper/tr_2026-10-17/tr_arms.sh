#!/bin/bash
# tr_arms.sh (ube, 2026-10-17): the quad hardware stage with ONE adaptive step rule (deploy.py --tr: Levenberg-Marquardt
# trust region per goal) vs drB_hw24's hand-set recipe. VALIDATION goals (planeval, seed 301), 24 jumps per robot
# (6 goals x 4 iterations), 4 robots, 3 hardware seeds (9001-9003).
#   recipe   drB_hw24's arguments: gn-beta 0.3, step-rms 0.05, anchor (default), bold 0.3, gate 1 1.5 0.4
#   tr       gn-beta 1.0, --tr 1 (radius starts at step-rms 0.05), anchor-w 0, no bold, no gate
#   tr_r20   the same, radius starting at 0.20 (sensitivity to the one remaining number)
P=/home/henry/ilc_ws/log/dilc/plane; S=/home/henry/ilc_ws/src/ilc_mjx; PY=~/miniconda3/envs/ilcmjx/bin/python
export ILC_QUAD_OVERLAY=/ilc_ws/log/dilc/plane/fixbox/overlay JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_ilc JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
COMMON="--update gn --gn-delta 0.1 --est-window 0.025 --clear 0.03 --stall 3 --stall-sat 0.2 --sat-project --target-base own
   --eval planeval --perturbed --goals 0.45 0.60 0.50,0.15 0.60,0.10 0.50,0.05 0.575,0.15 --iters 4
   --policy $P/drB_s2/policy --member g0_plane_it30"
declare -A ARM=(
  [recipe]="--gn-beta 0.3 --step-rms 0.05 --bold 0.3 --gate 1 1.5 0.4"
  [tr]="--gn-beta 1.0 --step-rms 0.05 --tr 1 --anchor-w 0"
  [tr_r20]="--gn-beta 1.0 --step-rms 0.20 --tr 1 --anchor-w 0"
)
cd $S; n=0; g=0; NPAR=${NPAR:-4}
for seed in 9001 9002 9003; do
  for arm in recipe tr tr_r20; do
    for rb in real_r1 real_s1 real_r4 real_r5; do
      out=$P/tr/${arm}_sd$seed
      [ -f $out/$rb/eval_planeval_final.json ] && continue
      START=--skip-start-eval; [ $arm = recipe ] && [ $seed = 9001 ] && START=""      # one zero-shot reference per robot
      $PY scripts/deploy.py $COMMON ${ARM[$arm]} --seed $seed $START --robots $rb --out $out --gpu $g 2>&1 \
        | grep -v "warp\|bfc_alloc" > $P/tr/${arm}_sd${seed}_$rb.log &
      g=$(( (g + 1) % 3 )); n=$((n + 1))
      if (( n % NPAR == 0 )); then wait -n; n=$((n - 1)); fi
    done
  done
done
wait
echo "$(date +%T) TR_ARMS_DONE"
