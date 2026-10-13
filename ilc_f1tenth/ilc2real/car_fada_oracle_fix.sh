#!/bin/bash
# The paper's car FADA-oracle (fada_real_v2h4_oracle14: the v2 IDM, H 4, oracle planner from our stopping-rule network) rerun with
# the FIXED oracle lookahead (real_car.py 2026-10-12: H control periods ahead, not one 5 mm plan index):
#  (a) fada_v2h4_oracle14fix: the identical configuration (the corrected paper row)
#  (b) fada_v2h4_oracleL: the oracle planner AND the reference rungs from our ladder model (v3drBtL_s0_lad1_lap10) -- the
#      oracle measured against our 100 %
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
PL14="$(for t in square2fast figfast; do for b in 0 10 14 18 21 25; do echo -n "mocap_${t}_b$b "; done; done)"
CK="24 48 96 192 384 768 1536 2016"
[ -f runs/hw/fada_v2h4_oracle14fix_ep2016/summary.json ] || F1T_PROCS=3 $PY -u fada_real_car.py --idm runs/hw/fada_v2_h4_hw24/idm.npz \
  --horizon 4 --arm fada_v2h4_oracle14fix --oracle-arm v3drBt_stop_lap10 --goals $PL14 --checkpoints $CK --gpu 1 \
  > logs/fada_v2h4_oracle14fix.log 2>&1 &
[ -f runs/hw/fada_v2h4_oracleL_ep2016/summary.json ] || F1T_PROCS=3 $PY -u fada_real_car.py --idm runs/hw/fada_v2_h4_hw24/idm.npz \
  --horizon 4 --arm fada_v2h4_oracleL --oracle-arm v3drBtL_s0_lad1_lap10 --rungs-from v3drBtL_s0_lad1_lap10 --goals $PL14 \
  --checkpoints $CK --gpu 2 > logs/fada_v2h4_oracleL.log 2>&1 &
wait
for arm in fada_v2h4_oracle14fix fada_v2h4_oracleL; do
  F1T_PROCS=6 $PY -u reeval.py ${arm}_ep0 $(for k in $CK; do echo -n "${arm}_ep$k "; done) > logs/reeval_$arm.log 2>&1
done
echo "$(date +%T) FADA_ORACLE_FIX_DONE"
