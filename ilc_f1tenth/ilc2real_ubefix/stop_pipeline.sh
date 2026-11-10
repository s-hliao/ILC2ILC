#!/bin/bash
# The car headline (ours + DR (B), task objective) with the SIM-STAGE STOPPING RULE instead of the fixed 30 iterations
# (2026-10-11, "match against the stopping method"): train the same network to 90 iterations, score every 5th snapshot in sim
# only (sim_validate.py), pick by the rule (stop_pick.py), then the 10-lap budget arms (zero-shot / 2 / 10 real laps, the same
# arguments as v3drBt_s0_*), and FADA's oracle planner on the picked network's 10-lap arm (the data-matching baseline).
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
PL8="$(for t in square2fast figfast; do for b in 0 10 18 25; do echo -n "mocap_${t}_b$b "; done; done)"
V3="--anchor-w 1.0 --beta 0.3 --cap 0.04 --max-ey 0.2 --a-off 0.15 --pert 1.5 --explore-frac 0.7 --eval-every 5"
TASK="--err-scale 0.05 1e9 1e9 1e9 1e9 0.3"
G2="--goals mocap_square2fast_b25 mocap_figfast_b25"
[ -f runs/v3drBt_s0_long/policy_final.npz ] || $PY -u sim_train.py --out runs/v3drBt_s0_long --plans $PL8 --dr 1.0 --bc-iters 10 $V3 \
  --iters 90 --seed 0 $TASK --gpu 1 > logs/v3drBt_s0_long.log 2>&1
echo "$(date +%T) trained"
[ -f runs/stopping/sim_val_bt.json ] || $PY -u sim_validate.py --runs runs/v3drBt_s0_long --out runs/stopping/sim_val_bt.json --gpu 1 \
  > logs/stop_simval_bt.log 2>&1
$PY stop_pick.py runs/v3drBt_s0_long runs/stopping/sim_val_bt.json runs/v3drBt_stop
P=runs/v3drBt_stop/policy_final.npz
for spec in "zs:--iters 0" "lap2:--iters 1 $G2 --goal-cycle 1" "lap10:--iters 5 $G2 --goal-cycle 1"; do
  n=v3drBt_stop_${spec%%:*}
  [ -f runs/hw/$n/summary.json ] || F1T_PROCS=8 $PY -u hw_stage.py --policy $P --out runs/hw/$n ${spec#*:} --gpu 1 > logs/hw_$n.log 2>&1
done
F1T_PROCS=12 $PY -u reeval.py v3drBt_stop_zs v3drBt_stop_lap2 v3drBt_stop_lap10 > logs/reeval_stop.log 2>&1
echo "$(date +%T) STOP_ARMS_DONE"
PL14="$(for t in square2fast figfast; do for b in 0 10 14 18 21 25; do echo -n "mocap_${t}_b$b "; done; done)"
F1T_PROCS=8 $PY -u fada_real_car.py --idm runs/hw/fada_hw24/idm.npz --arm fada_real_oracle14 --oracle-arm v3drBt_stop_lap10 \
  --goals $PL14 --gpu 3 > logs/real_fada_oracle14.log 2>&1
ARMS=$(ls -d runs/hw/fada_real_oracle14_ep* | sed 's|runs/hw/||' | tr '\n' ' ')
F1T_PROCS=12 $PY -u reeval.py $ARMS --runs 6 > logs/reeval_fada_oracle14.log 2>&1
echo "$(date +%T) STOP_PIPELINE_DONE"
