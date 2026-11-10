#!/bin/bash
# FADA with the ORACLE planner and a RICHER goal set on the cars (2026-10-11, "allow it to fly at more goals than us, this is for a baseline"):
# the oracle planner = our headline at the 10-lap budget (v3drBt_s0_lap10: the stopping rule's network + 10 real laps). All 14 bank plans (the 8 evaluation plans and the 6 mu80 variants) instead of our 2 b25 plans, up to 2,016 laps per car.
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
PL14="$(for t in square2fast figfast; do for b in 0 10 14 18 21 25; do echo -n "mocap_${t}_b$b "; done; done)"
F1T_PROCS=8 $PY -u fada_real_car.py --idm runs/hw/fada_hw24/idm.npz --arm fada_real_oracle14 --oracle-arm v3drBt_s0_lap10 \
  --goals $PL14 --gpu 3 > logs/real_fada_oracle14.log 2>&1
ARMS=$(ls -d runs/hw/fada_real_oracle14_ep* | sed 's|runs/hw/||' | tr '\n' ' ')
F1T_PROCS=12 $PY -u reeval.py $ARMS --runs 6 > logs/reeval_fada_oracle14.log 2>&1
echo "$(date +%T) FADA_ORACLE14_DONE"
