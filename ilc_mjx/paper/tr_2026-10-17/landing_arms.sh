#!/bin/bash
# landing_arms.sh (2026-10-18): the quad hardware stage is limited by its landing model (TR_RESULTS.md: it realizes ~25 %
# of the predicted improvement, wrong sign for ~1/3 of goals), not by the step rule. Model-side fixes, all on drB_hw24's
# recipe step settings, same setup as tr_arms.sh: VALIDATION goals (planeval, seed 301), 24 jumps per robot (6 goals x 4
# iterations), 4 robots, 3 hardware seeds (9001-9003). Outputs $P/tr/<arm>_sd<seed>, next to tr_arms.sh's recipe arm.
#   recipe       drB_hw24's arguments (the reference; skipped where tr_arms.sh already ran it)
#   jac_real     Jacobians with the sim restored to each trial's measured states (--jac-at real), not along the sim's own rollout
#   sens_goal    Broyden (secant) correction of the landing sensitivity from consecutive jumps, per goal (--sens-adapt goal)
#   sens_shared  the same, one correction per bank plan shared over goals (--sens-adapt shared)
#   sens_vres    value residual: ridge regression of measured - predicted landing on the state offset (--sens-adapt vres)
#   nbr05        neighbour targets (grid goals <= 5 cm get a weighted step, w 0.5): the interpolation test's best arm,
#                replicated here on validation goals and 3 seeds (INTERP_RESULTS.md was 1 seed on the spent test seed 701)
#   ctl_random   the causal control: a same-size step in a random direction (any gain left is the refit's)
# ARMS="jac_real sens_goal" bash landing_arms.sh   runs a subset (default: all). SMOKE=1: 2 goals x 2 iterations on real_r1,
# seed 9001, -> $P/smoke_landing (about 15 min; checks every arm starts, steps and evaluates).
P=/home/henry/ilc_ws/log/dilc/plane; S=/home/henry/ilc_ws/src/ilc_mjx; PY=~/miniconda3/envs/ilcmjx/bin/python
export ILC_QUAD_OVERLAY=/ilc_ws/log/dilc/plane/fixbox/overlay JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_ilc JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
GOALS="0.45 0.60 0.50,0.15 0.60,0.10 0.50,0.05 0.575,0.15"; ITERS=4; SEEDS="9001 9002 9003"
ROBOTS="real_r1 real_s1 real_r4 real_r5"; OUT=$P/tr
if [ -n "$SMOKE" ]; then GOALS="0.45 0.50,0.05"; ITERS=2; SEEDS=9001; ROBOTS=real_r1; OUT=$P/smoke_landing; fi
COMMON="--update gn --gn-delta 0.1 --est-window 0.025 --clear 0.03 --stall 3 --stall-sat 0.2 --sat-project --target-base own
   --eval planeval --perturbed --goals $GOALS --iters $ITERS --policy $P/drB_s2/policy --member g0_plane_it30"
RECIPE="--gn-beta 0.3 --step-rms 0.05 --bold 0.3 --gate 1 1.5 0.4"
declare -A ARM=(
  [recipe]="$RECIPE"
  [jac_real]="$RECIPE --jac-at real"
  [sens_goal]="$RECIPE --sens-adapt goal"
  [sens_shared]="$RECIPE --sens-adapt shared"
  [sens_vres]="$RECIPE --sens-adapt vres"
  [nbr05]="$RECIPE --nbr-radius 0.05 --nbr-weight 0.5"
  [ctl_random]="$RECIPE --control random"
)
ARMS=${ARMS:-"recipe jac_real sens_goal sens_shared sens_vres nbr05 ctl_random"}
mkdir -p $OUT; cd $S; n=0; g=0; NPAR=${NPAR:-4}
for seed in $SEEDS; do
  for arm in $ARMS; do
    for rb in $ROBOTS; do
      out=$OUT/${arm}_sd$seed
      [ -f $out/$rb/eval_planeval_final.json ] && continue
      START=--skip-start-eval; [ $arm = recipe ] && [ $seed = 9001 ] && START=""      # one zero-shot reference per robot
      $PY scripts/deploy.py $COMMON ${ARM[$arm]} --seed $seed $START --robots $rb --out $out --gpu $g 2>&1 \
        | grep -v "warp\|bfc_alloc" > $OUT/${arm}_sd${seed}_$rb.log &
      g=$(( (g + 1) % ${NGPU:-3} )); n=$((n + 1))
      if (( n % NPAR == 0 )); then wait -n; n=$((n - 1)); fi
    done
  done
done
wait
echo "$(date +%T) LANDING_ARMS_DONE"
