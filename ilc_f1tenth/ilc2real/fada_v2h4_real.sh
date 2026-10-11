#!/bin/bash
# The car's best NON-ORACLE FADA inverse model (fada_v2_h4: no actuator states in the target, 4 periods ahead; 50 % zero-shot,
# vs 8 % for fada_hw24's IDM the first real-data runs started from) in the real-data study (2026-10-11, "the best oracle and
# non-oracle versions"): (1) its own planner (lam 0.8), (2) the oracle planner of our stopping-rule network's 10-lap arm, 14 plans;
# up to 2,016 laps per car each, then every checkpoint scored (reeval, 6 runs, as the other FADA arms).
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
PL14="$(for t in square2fast figfast; do for b in 0 10 14 18 21 25; do echo -n "mocap_${t}_b$b "; done; done)"
F1T_PROCS=6 $PY -u fada_real_car.py --idm runs/hw/fada_v2_h4_hw24/idm.npz --horizon 4 --arm fada_real_v2h4 --gpu 1 > logs/real_fada_v2h4.log 2>&1
F1T_PROCS=6 $PY -u fada_real_car.py --idm runs/hw/fada_v2_h4_hw24/idm.npz --horizon 4 --arm fada_real_v2h4_oracle14 \
  --oracle-arm v3drBt_stop_lap10 --goals $PL14 --gpu 1 > logs/real_fada_v2h4_oracle14.log 2>&1
ARMS=$(ls -d runs/hw/fada_real_v2h4_ep* runs/hw/fada_real_v2h4_oracle14_ep* | sed 's|runs/hw/||' | tr '\n' ' ')
F1T_PROCS=8 $PY -u reeval.py $ARMS --runs 6 > logs/reeval_fada_v2h4.log 2>&1
echo "$(date +%T) FADA_V2H4_DONE"
