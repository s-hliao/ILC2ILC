#!/bin/bash
# The fully idealized car FADA oracle: the IDM's data flown on each EXACT multi-body car by OUR per-car ladder network
# (v3drBtL_s0_lad1_lap10/<car>_policy.npz, 100 %) with small smooth action noise, on its rungs; the oracle planner from the same
# networks; the v2 IDM target (trunk dims 0-4, H 4). Its best budget is expected at 0 laps (adaptation degrades every FADA run).
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
OURS=v3drBtL_s0_lad1_lap10
[ -f runs/fada_ours_idm/collect.json ] || F1T_PROCS=4 $PY -u fada_collect_car.py --out runs/fada_ours_idm --init "runs/hw/$OURS/{car}_policy.npz" \
  --episodes 3000 --batch 400 --a-off 0.1 --dims 0 1 2 3 4 --horizon 4 --rungs-from $OURS --gpu 1 > logs/fada_ours_collect.log 2>&1
CK="24 96 384"
[ -f runs/hw/fada_ours_ep384/summary.json ] || F1T_PROCS=4 $PY -u fada_real_car.py --idm "runs/fada_ours_idm/idm_{car}.npz" --arm fada_ours \
  --horizon 4 --privileged 1 --oracle-arm $OURS --rungs-from $OURS --checkpoints $CK --gpu 1 > logs/fada_ours.log 2>&1
F1T_PROCS=4 $PY -u reeval.py fada_ours_ep0 $(for k in $CK; do echo -n "fada_ours_ep$k "; done) > logs/reeval_fada_ours.log 2>&1
echo "$(date +%T) CAR_FADA_OURS_DONE"
