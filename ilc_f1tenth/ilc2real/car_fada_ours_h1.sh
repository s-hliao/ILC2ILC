#!/bin/bash
# Car FADA oracle, one-step IDM (H 1, trunk dims 0-4) on exact data flown by OUR per-car ladder networks (car_fada_ours.sh had H 4:
# 20 % at 0 laps despite a 0.006 IDM loss -- compounding over the 4-period target is the suspect).
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
OURS=v3drBtL_s0_lad1_lap10
[ -f runs/fada_ours_h1_idm/collect.json ] || F1T_PROCS=4 $PY -u fada_collect_car.py --out runs/fada_ours_h1_idm --init "runs/hw/$OURS/{car}_policy.npz" \
  --episodes 3000 --batch 400 --a-off 0.1 --dims 0 1 2 3 4 --horizon 1 --rungs-from $OURS --gpu 1 > logs/fada_ours_h1_collect.log 2>&1
CK="24 96"
[ -f runs/hw/fada_ours_h1_ep96/summary.json ] || F1T_PROCS=4 $PY -u fada_real_car.py --idm "runs/fada_ours_h1_idm/idm_{car}.npz" --arm fada_ours_h1 \
  --horizon 1 --privileged 1 --oracle-arm $OURS --rungs-from $OURS --checkpoints $CK --gpu 1 > logs/fada_ours_h1.log 2>&1
F1T_PROCS=4 $PY -u reeval.py fada_ours_h1_ep0 $(for k in $CK; do echo -n "fada_ours_h1_ep$k "; done) > logs/reeval_fada_ours_h1.log 2>&1
echo "$(date +%T) CAR_FADA_OURS_H1_DONE"
