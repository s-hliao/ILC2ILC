#!/bin/bash
# Quad fix, sim side (2026-10-22; log/dilc/plane/clip/DIAG.md, baseline_audit/AUDIT.md): our sim network is the bad
# start (zero-shot 21.3 vs fixed PPO+DR 12.9; 54 % of the score is box jumps clipping the edge by 1-1.5 cm), and the
# robots' timing (actuation delay / jitter / drops, mocap delay / noise) breaks the stage's convergence. Arms, all
# with drB's recipe (dr_variant_sweep.sh) and DR:
#   drC_s2    from scratch, --dr-async (timing DR, JumpEnv.sample_async) + --clear 0.05 (was 0.03), seed 2 (= drB_s2)
#   drCft_s2  fine-tune of drB_s2 it30, 15 its, --dr-async + --clear 0.05
#   drDft_s2  fine-tune of drB_s2 it30, 15 its, --clear 0.05 only (the control: clearance without the timing DR)
# Then the CPU val zero-shot (cpu_eval.sh, val goals) of each at its last snapshot.
cd /home/henry/ilc_ws/src/ilc_mjx
PY=~/miniconda3/envs/ilcmjx/bin/python
P=/home/henry/ilc_ws/log/dilc/plane
export ILC_QUAD_OVERLAY=/ilc_ws/log/dilc/plane/fixbox/overlay JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_ilc XLA_PYTHON_CLIENT_PREALLOCATE=false
R="--bank $P/bank_v3.json --beta 0.7 --cap 0.15 --replay 3 --steps 1000 --anchor-pool 20 --anchor-w 1.0 --lr-final 1e-4 --q-off 0.05 --a-off 0.03 --est-window 0.025 --dr --clear 0.05"
run() { n=$1; g=$2; shift 2
  OMP_NUM_THREADS=4 $PY scripts/plane_train.py --gpu $g $R --out $P/$n "$@" 2>&1 | grep --line-buffered -v "warp\|bfc_alloc" > $P/$n.out; }
run drC_s2 1 --seed 2 --iters 30 --dr-async &
run drCft_s2 2 --seed 2 --iters 15 --eval-every 5 --dr-async --init-npz $P/drB_s2/policy/g0_plane_it30.npz --lr 1e-4 &
run drDft_s2 3 --seed 2 --iters 15 --eval-every 5 --init-npz $P/drB_s2/policy/g0_plane_it30.npz --lr 1e-4 &
wait
echo "$(date +%T) training done"
for n in drC_s2:30 drCft_s2:15 drDft_s2:15; do
  d=${n%%:*}; it=${n##*:}
  [ -f $P/$d/policy/g0_plane_it$it.npz ] && JOBS=12 bash $P/cpu_eval.sh $P/$d/policy g0_plane_it$it val 2 >/dev/null 2>&1
done
echo "$(date +%T) RETRAIN_C_DONE"
