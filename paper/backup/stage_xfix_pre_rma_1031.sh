#!/bin/bash
# stage_xfix.sh NET CFG SEED ROBOT GPU (2026-10-30; user: "fix the car and quad to actually do better on extrapolation"): one quad stage
# run, stage_norm.sh's N1 / stage_interp.sh's I2 (neighbour targets, 12.5 cm) on NET = ours (ekfft_s2 it15) or ppo (base_ekf/ppo_dr it75).
# THE FIX: real_s1_m500 now practises ALL six goals. Its 15 cm boxes were dropped as "falls on every jump", but by the reachable rule
# (a pair is impossible only if NO jump of any run ever succeeded) they are reachable: (0.50, 0.15) 31 / 176 jumps succeeded, (0.575, 0.15)
# 13 / 176 (36 runs). Every far-extrapolated held-out failure (>= 6 cm from a practised goal) was on s1 or r5; r5's drops stay (0 successes).
# -> stage_xfix/<NET>_<CFG>_sd<SEED>/<ROBOT>/
P=/home/henry/ilc_ws/log/dilc/plane; S=/home/henry/ilc_ws/src/ilc_mjx; PY=${PY:-~/miniconda3/envs/ilcmjx/bin/python}
export ILC_QUAD_OVERLAY=/ilc_ws/log/dilc/plane/fixbox/overlay JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_ilc JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
export DILC_FLY_JOBS=${DILC_FLY_JOBS:-2}
net=$1 cfg=$2 seed=$3 rb=$4 g=$5
case $net in ours) NETD=$P/ekfft_s2; MEMBER=g0_plane_it15 ;; ppo) NETD=$P/base_ekf/ppo_dr/it75; MEMBER=policy ;;
  oursL) NETD=$P/ekfftL_s2; MEMBER=g0_plane_it10 ;; esac   # oursL (2026-10-30): the equal-budget run ekfftL_s2, it10 = best of its own sim DR eval (43 %)
ALL="0.45 0.60 0.50,0.15 0.60,0.10 0.50,0.05 0.575,0.15"
declare -A GOALS=([real_r1_m500]="$ALL" [real_r4_m500]="$ALL" [real_s1_m500]="$ALL" [real_r5_m500]="0.45 0.50,0.05")
# (2026-10-31, user: "figure out what is wrong with the trajectory ... if it doesn't work, remove it from the test set or make it easier"):
# every box plan lands the rear feet only 2.7-4.4 cm past the box face (plans_sbf8 info_feet_rel), the robots land 5-6.5 cm short,
# and on boxes >= 11 cm a rear foot off the edge falls (every arm, zero-shot 10.9 %). NET suffix "low": practise only goals <= 10 cm.
LOW="0.45 0.60 0.60,0.10 0.50,0.05"
[ "${LOWGOALS:-0}" = 1 ] && GOALS=([real_r1_m500]="$LOW" [real_r4_m500]="$LOW" [real_s1_m500]="$LOW" [real_r5_m500]="0.45 0.50,0.05")
BASE="--update gn --est-window 0.025 --clear 0.05 --target-base own --eval planeval --perturbed --skip-start-eval"
F1="--gn-beta 0.5 --gn-delta 0.02 --step-rms 0.1"; W="--clip-mode weighted"
declare -A CFG=([N1]="$F1 $W --iters 6" [I2]="$F1 $W --iters 6 --nbr-radius 0.125" [J1]="$F1 $W --iters 6 --reg-jac 1" [J10]="$F1 $W --iters 6 --reg-jac 10" [FS]="$F1 $W --iters 6 --nbr-radius 0.125 --stall 3" [FR]="$F1 $W --iters 6 --nbr-radius 0.125 --clear 0 --rollback 1.5 0.5" [FRG]="$F1 $W --iters 6 --nbr-radius 0.125 --clear 0 --rollback 1.5 0.5 --gate 1 1.5 0.4" [FRJ]="$F1 $W --iters 6 --nbr-radius 0.125 --clear 0 --rollback 1.5 0.5 --reg-jac 1")
# FS / FR (2026-10-30, user: "fix ppo+dr+stage ... only interested in cases with positivity"): I2 plus an existing safeguard. FS: the
# stall guard (--stall 3, the quad method's E3 guard: a goal that stops improving is held at its best so it cannot drag its
# neighbours through the shared network). FR: the theory's rollback rule (iv): a fall rolls the goal back (--clear 0: no clearance
# step on a fall) and a trial clearly worse than the goal's best (1.5x and +0.5) rolls back, its steps halved.
# J1 / J10 (2026-10-30, the slope anchor the user approved for all three domains: "that's ok"): N1 plus deploy.py --reg-jac, the policy's
# feedback d pi / d(error states) held at the anchor and trial states, weight 1 / 10 (the car's hw_stage.py --slope-w sweep runs alongside)
out=$P/stage_xfix/${net}_${cfg}${LOWGOALS:+L}_sd$seed; mkdir -p $out; cd $S
$PY scripts/deploy.py $BASE ${CFG[$cfg]} --goals ${GOALS[$rb]} --policy $NETD/policy --member $MEMBER --seed $seed \
  --robots $rb --out $out --gpu $g 2>&1 | grep -v "warp\|bfc_alloc" > $P/stage_xfix/${net}_${cfg}${LOWGOALS:+L}_sd${seed}_$rb.log
