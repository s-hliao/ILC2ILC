#!/bin/bash
# run_fric45.sh (2026-10-23, ube): the quad with a conservative friction cone (plans_mu45, bank_mu45.json: mu 0.45, was
# 0.6). Diagnosis (log/dilc/plane/quad_r5_split.py): the low-friction robots (r5 0.59, r1 0.63) slip on the 0.6 plans,
# the nominal model's landing sensitivity loses its sign there (r5: median cos +0.25, 42 % wrong sign) -- positivity
# fails. Everything goes to log/dilc/plane/fric45/, apart from the 0.6 runs (the user keeps both).
#   1. the stage arms on drB_s2 it30 flying the 0.45 plans (no retraining): recipe / pool / ctl_random x 3 seeds x 4
#   2. (in parallel) sim fine-tunes of drB_s2 it30 on the 0.45 bank: f45B (drB's recipe, clear 0.03) and f45C (drCft's:
#      timing DR + clear 0.05), 15 its; then their CPU val zero-shot
#   3. the stage arms on the better fine-tune, if it beats drB_s2 on the 0.45 plans (18.96 on the 256-jump val)
set -u
P=/home/henry/ilc_ws/log/dilc/plane; F=$P/fric45; T=/home/henry/ilc_ws/src/ilc_mjx/paper/tr_2026-10-17
PY=~/miniconda3/envs/ilcmjx/bin/python
log() { echo "$(date +%T) $*"; }
export DILC_FLY_JOBS=4 GPU0=0 NGPU=3
( NET=$F/drB_s2/policy MEMBER=g0_plane_it30 CLR=0.03 OUT=$F/stage_drB NPAR=9 bash $T/stage_c.sh > $F/stage_drB.out 2>&1; log "stage on drB_s2 (0.45 plans) done" ) &
R="--bank $P/bank_mu45.json --beta 0.7 --cap 0.15 --replay 3 --steps 1000 --anchor-pool 20 --anchor-w 1.0 --lr-final 1e-4 --q-off 0.05 --a-off 0.03 --est-window 0.025 --dr"
FT="--seed 2 --iters 15 --eval-every 5 --init-npz $P/drB_s2/policy/g0_plane_it30.npz --lr 1e-4"
train() { n=$1; g=$2; shift 2
  ( cd /home/henry/ilc_ws/src/ilc_mjx
    ILC_QUAD_OVERLAY=/ilc_ws/log/dilc/plane/fixbox/overlay JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_ilc XLA_PYTHON_CLIENT_PREALLOCATE=false \
    OMP_NUM_THREADS=3 $PY scripts/plane_train.py --gpu $g $R --out $F/$n "$@" 2>&1 | grep --line-buffered -v "warp\|bfc_alloc" > $F/$n.out ) ; }
train f45B 0 --clear 0.03 $FT &
TB=$!
train f45C 1 --clear 0.05 --dr-async $FT &
TC=$!
wait $TB $TC
log "fine-tunes done"
for n in f45B f45C; do
  [ -f $F/$n/policy/g0_plane_it15.npz ] || continue
  JOBS=12 bash $P/cpu_eval.sh $F/$n/policy g0_plane_it15 val 8
done
BEST=$($PY - $F <<'PY'
import json, os, sys
import numpy as np
F = sys.argv[1]; best = ("drB_s2", 18.96)
for n in ("f45B", "f45C"):
    f = f"{F}/{n}/cpu_val_e8_g0_plane_it15.json"
    if os.path.exists(f):
        R = json.load(open(f)); s = np.mean([0.01 * r["landing_cost"] + 20 * r["fell"] for r in R])
        print(f"{n}: {s:.2f}", file=sys.stderr)
        if s < best[1]:
            best = (n, s)
print(best[0])
PY
)
log "best start on the 0.45 plans: $BEST"
if [ "$BEST" != drB_s2 ]; then
  c=0.03; [ $BEST = f45C ] && c=0.05
  NET=$F/$BEST/policy MEMBER=g0_plane_it15 CLR=$c OUT=$F/stage_$BEST NPAR=9 bash $T/stage_c.sh > $F/stage_$BEST.out 2>&1
  log "stage on $BEST done"
fi
wait
log "RUN_FRIC45_DONE"
