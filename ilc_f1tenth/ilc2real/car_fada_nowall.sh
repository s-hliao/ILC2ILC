#!/bin/bash
# User 2026-10-12: "what if you disable crashes?" -- FADA exact (per-car exact IDMs, oracle planner, rungs, the fixed lookahead)
# ADAPTED WITHOUT THE ROOM'S WALLS (F1T_SAFETY_EY=10: a wall contact no longer ends the episode; spin-outs still do), so its
# adaptation data are whole laps instead of crash-truncated ones; EVALUATED WITH the walls (0.3 m, reeval under env_tight.sh).
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
OURS=v3drBtL_s0_lad1_lap10
CK="24 96 384 1536 2016"
[ -f runs/hw/fada_nowall_ep2016/summary.json ] || F1T_SAFETY_EY=10 F1T_TRAIN_NO_WALLS=1 F1T_PROCS=8 $PY -u fada_real_car.py \
  --idm "runs/fada_exact_idm/idm_{car}.npz" --arm fada_nowall --horizon 4 --privileged 1 --oracle-arm $OURS --rungs-from $OURS \
  --checkpoints $CK --gpu 1 > logs/fada_nowall.log 2>&1
F1T_PROCS=8 $PY -u reeval.py fada_nowall_ep0 $(for k in $CK; do echo -n "fada_nowall_ep$k "; done) > logs/reeval_fada_nowall.log 2>&1
echo "$(date +%T) FADA_NOWALL_DONE"
