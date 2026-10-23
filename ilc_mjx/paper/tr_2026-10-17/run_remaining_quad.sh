#!/bin/bash
# run_remaining_quad.sh (2026-10-22): the quad fix's staged runs, for the machine with more compute. Read
# src/paper/HANDOFF_2026-10-22.md first. Steps (each skips what already exists):
#   1  the retrain (retrain_c.sh) if its networks are missing, then the CPU val zero-shot of each
#   2  pick the start network: the lowest val score of drC_s2 it30 / drCft_s2 it15 / drDft_s2 it15 (CPU val, the
#      four robots), printed next to drB_s2 it30 (the current one)
#   3  the stage arms (stage_c.sh: recipe / pool / ctl_random, 3 seeds x 4 robots, val goals) on the picked network,
#      --clear 0.05 (its training clearance) -> log/dilc/plane/stage_c_<net>
#   4  the pool arm on the OLD network drB_s2 it30 (--clear 0.03) -> log/dilc/plane/tr/pool_sd*, paired with tr/'s
#      recipe and ctl_random (the stage-side fix alone)
#   5  tables (quad_table.py)
# STEP=3 bash run_remaining_quad.sh starts at a step. NPAR / NGPU / GPU0 pass to stage_c.sh (default 3 / 3 / 1).
# More compute: raise NPAR (each deploy.py is one GPU process + DILC_FLY_JOBS CPU flights, default 3).
set -u
P=/home/henry/ilc_ws/log/dilc/plane; T=/home/henry/ilc_ws/src/ilc_mjx/paper/tr_2026-10-17
PY=~/miniconda3/envs/ilcmjx/bin/python
STEP=${STEP:-1}
if (( STEP <= 1 )); then
  # the scratch run drC_s2 was stopped on matcha at it0 (2026-10-22, too slow under load; its partial dir is
  # drC_s2_killed_it0): train it here, next to the fine-tunes if those did not finish on matcha either
  if [ ! -f $P/drC_s2/policy/g0_plane_it30.npz ]; then
    [ -d $P/drC_s2 ] && mv $P/drC_s2 $P/drC_s2_partial_$(date +%s)
    cd /home/henry/ilc_ws/src/ilc_mjx
    ILC_QUAD_OVERLAY=/ilc_ws/log/dilc/plane/fixbox/overlay JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_ilc XLA_PYTHON_CLIENT_PREALLOCATE=false \
    OMP_NUM_THREADS=4 $PY scripts/plane_train.py --gpu $(( ${GPU0:-1} + 2 )) --bank $P/bank_v3.json --beta 0.7 --cap 0.15 --replay 3 \
      --steps 1000 --anchor-pool 20 --anchor-w 1.0 --lr-final 1e-4 --q-off 0.05 --a-off 0.03 --est-window 0.025 --dr \
      --clear 0.05 --out $P/drC_s2 --seed 2 --iters 30 --dr-async 2>&1 | grep --line-buffered -v "warp\|bfc_alloc" > $P/drC_s2.out &
  fi
  # the fine-tunes (retrain_c.sh's lines): if matcha was packed before they finished, set the partial run aside and
  # train them here
  for n in drCft_s2 drDft_s2; do
    [ -f $P/$n/policy/g0_plane_it15.npz ] && continue
    [ -d $P/$n ] && mv $P/$n $P/${n}_partial_$(date +%s)
    X=""; GI=$(( ${GPU0:-1} + 1 )); [ $n = drCft_s2 ] && { X="--dr-async"; GI=${GPU0:-1}; }
    ( cd /home/henry/ilc_ws/src/ilc_mjx
      ILC_QUAD_OVERLAY=/ilc_ws/log/dilc/plane/fixbox/overlay JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_ilc XLA_PYTHON_CLIENT_PREALLOCATE=false \
      OMP_NUM_THREADS=4 $PY scripts/plane_train.py --gpu $GI --bank $P/bank_v3.json --beta 0.7 --cap 0.15 \
        --replay 3 --steps 1000 --anchor-pool 20 --anchor-w 1.0 --lr-final 1e-4 --q-off 0.05 --a-off 0.03 --est-window 0.025 \
        --dr --clear 0.05 --out $P/$n --seed 2 --iters 15 --eval-every 5 $X --init-npz $P/drB_s2/policy/g0_plane_it30.npz \
        --lr 1e-4 2>&1 | grep --line-buffered -v "warp\|bfc_alloc" > $P/$n.out ) &
  done
  wait
  for n in drC_s2:30 drCft_s2:15 drDft_s2:15 drB_s2:30; do
    d=${n%%:*}; it=${n##*:}
    [ -f $P/$d/cpu_val_g0_plane_it$it.json ] || JOBS=${JOBS:-12} bash $P/cpu_eval.sh $P/$d/policy g0_plane_it$it val 2
  done
fi
# the score as in quad_table.py: 0.01 x landing cost + 20 per fall, mean over the CPU val jumps (lower is better)
NETSEL=$($PY - $P <<'PY'
import json, os, sys
P = sys.argv[1]
best = None
for d, it in (("drB_s2", 30), ("drC_s2", 30), ("drCft_s2", 15), ("drDft_s2", 15)):
    f = f"{P}/{d}/cpu_val_g0_plane_it{it}.json"
    if not os.path.exists(f):
        print(f"{d} it{it}: missing", file=sys.stderr); continue
    r = json.load(open(f))
    sc = sum(0.01 * x["landing_cost"] + 20 * x["fell"] for x in r) / len(r)
    ok = sum((not x["fell"]) and abs(x["ex"]) <= 0.05 and abs(x["ez"]) <= 0.03 for x in r) / len(r)
    print(f"{d} it{it}: CPU val score {sc:.2f}, ok {ok:.0%} ({len(r)} jumps)", file=sys.stderr)
    if d != "drB_s2" and (best is None or sc < best[0]):
        best = (sc, d, it)
print(f"{best[1]}:{best[2]}" if best else "")
PY
)
echo "picked: $NETSEL"
if (( STEP <= 3 )) && [ -n "$NETSEL" ]; then
  d=${NETSEL%%:*}; it=${NETSEL##*:}
  NET=$P/$d/policy MEMBER=g0_plane_it$it CLR=0.05 OUT=$P/stage_c_$d bash $T/stage_c.sh
fi
if (( STEP <= 4 )); then
  NET=$P/drB_s2/policy MEMBER=g0_plane_it30 CLR=0.03 OUT=$P/tr ARMS=pool bash $T/stage_c.sh
fi
if (( STEP <= 5 )); then
  cd $T
  for d in $P/stage_c_*; do [ -d $d ] && $PY quad_table.py --dir $d | tee $d/table.txt; done
  $PY quad_table.py --dir $P/tr recipe pool ctl_random nbr05 | tee $P/tr/table_pool.txt
fi
echo "$(date +%T) RUN_REMAINING_QUAD_DONE"
