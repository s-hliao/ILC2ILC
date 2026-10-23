#!/bin/bash
# jd_arms.sh (matcha, 2026-10-20): the delay-aware closed-loop sensitivity (deploy.py --jac-delay DS DA). log/dilc/plane/clip/DIAG.md
# §11-12: clean twins converge on the flown goals; the partial twins show the mocap and actuation groups break it, and the
# GPU sim's sensitivity assumes an instantaneous loop. 10 ms control samples: robots' actuation delay ~1 sample, mocap
# delay + 25 ms estimator fit ~1-2 samples. Nominal delays (deployment-wide, not per robot).
P=/home/henry/ilc_ws/log/dilc/plane; S=/home/henry/ilc_ws/src/ilc_mjx; PY=~/miniconda3/envs/ilcmjx/bin/python
export ILC_QUAD_OVERLAY=/ilc_ws/log/dilc/plane/fixbox/overlay JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_ilc JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
GOALS="0.45 0.60 0.50,0.15 0.60,0.10 0.50,0.05 0.575,0.15"; ITERS=4; SEEDS=${SEEDS:-"9001 9002 9003"}
ROBOTS="real_r1 real_s1 real_r4 real_r5"; OUT=$P/tr
COMMON="--update gn --gn-delta 0.1 --est-window 0.025 --clear 0.03 --stall 3 --stall-sat 0.2 --sat-project --target-base own
   --eval planeval --perturbed --goals $GOALS --iters $ITERS --policy $P/drB_s2/policy --member g0_plane_it30
   --gn-beta 0.3 --step-rms 0.05 --bold 0.3 --gate 1 1.5 0.4"
declare -A ARM=(
  [jd10]="--jac-delay 1 0"
  [jd01]="--jac-delay 0 1"
  [jd11]="--jac-delay 1 1"
)
ARMS=${ARMS:-"jd11 jd10 jd01"}
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
echo "$(date +%T) JD_ARMS_DONE"
