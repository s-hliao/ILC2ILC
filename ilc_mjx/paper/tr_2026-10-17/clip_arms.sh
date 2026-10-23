#!/bin/bash
# clip_arms.sh (matcha, 2026-10-19): why the quad hardware stage does not work (log/dilc/plane/tr/LANDING_RESULTS.md,
# log/dilc/plane/clip/DIAG.md): 54 % of the validation score is box jumps that land SHORT (> 10 cm, mostly r5) after
# clipping the box edge by ~1-1.5 cm. On such a trial the GN step stacks the measured landing (the clip's, another mode
# than the sim Jacobian's on-box landing; 15-30 cm) with the clearance rows (-4.5 cm residual) in raw units, so the
# wrong-mode landing dominates and the clearance is never fixed. Arms (same setup as landing_arms.sh, outputs $P/tr):
#   clipC      --clip-mode clear --clip-beta 1.0  (a short clip steps on its clearance rows alone, full gain)
#   clipW      --clip-mode weighted               (both, rows in tolerance units: landing / LW, clearance / 1 cm)
#   clipC_nbr  clipC + neighbour targets (nbr05: the only arm with an ILC effect over the random control)
#   ARMS="clipC" bash clip_arms.sh  runs a subset; GPU0=1 (matcha GPUs 1-3)
P=/home/henry/ilc_ws/log/dilc/plane; S=/home/henry/ilc_ws/src/ilc_mjx; PY=~/miniconda3/envs/ilcmjx/bin/python
export ILC_QUAD_OVERLAY=/ilc_ws/log/dilc/plane/fixbox/overlay JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_ilc JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
GOALS="0.45 0.60 0.50,0.15 0.60,0.10 0.50,0.05 0.575,0.15"; ITERS=4; SEEDS="9001 9002 9003"
ROBOTS="real_r1 real_s1 real_r4 real_r5"; OUT=$P/tr
COMMON="--update gn --gn-delta 0.1 --est-window 0.025 --clear 0.03 --stall 3 --stall-sat 0.2 --sat-project --target-base own
   --eval planeval --perturbed --goals $GOALS --iters $ITERS --policy $P/drB_s2/policy --member g0_plane_it30
   --gn-beta 0.3 --step-rms 0.05 --bold 0.3 --gate 1 1.5 0.4"
declare -A ARM=(
  [clipC]="--clip-mode clear --clip-beta 1.0"
  [clipW]="--clip-mode weighted"
  [clipC_nbr]="--clip-mode clear --clip-beta 1.0 --nbr-radius 0.05 --nbr-weight 0.5"
)
ARMS=${ARMS:-"clipC clipW clipC_nbr"}
mkdir -p $OUT; cd $S; n=0; g=0; NPAR=${NPAR:-4}
for seed in $SEEDS; do
  for arm in $ARMS; do
    for rb in $ROBOTS; do
      out=$OUT/${arm}_sd$seed
      [ -f $out/$rb/eval_planeval_final.json ] && continue
      $PY scripts/deploy.py $COMMON ${ARM[$arm]} --seed $seed --skip-start-eval --robots $rb --out $out \
        --gpu $((g + ${GPU0:-0})) 2>&1 | grep -v "warp\|bfc_alloc" > $OUT/${arm}_sd${seed}_$rb.log &
      g=$(( (g + 1) % ${NGPU:-3} )); n=$((n + 1))
      if (( n % NPAR == 0 )); then wait -n; n=$((n - 1)); fi
    done
  done
done
wait
echo "$(date +%T) CLIP_ARMS_DONE"
