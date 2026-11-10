#!/bin/bash
# The car's budget curve on the STOPPING-RULE network (2026-10-11, "a low amount where most of the value is gotten, and an
# asymptotic amount where the method plateaus"): the same 2-lap chained trials alternating the two b25 plans as the 10-lap arms,
# at 4 / 6 / 8 laps (with stop_pipeline.sh's 0 / 2 / 10) and 24 / 48 / 96 laps -- the long ones ONLY to locate the plateau.
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
until grep -q STOP_ARMS_DONE logs/stop_pipeline.out 2>/dev/null; do sleep 60; done
P=runs/v3drBt_stop/policy_final.npz
G2="--goals mocap_square2fast_b25 mocap_figfast_b25"
for L in 4 6 8 24 48 96; do
  n=v3drBt_stop_lap$L
  [ -f runs/hw/$n/summary.json ] || F1T_PROCS=6 $PY -u hw_stage.py --policy $P --out runs/hw/$n --iters $((L / 2)) $G2 --goal-cycle 1 --gpu 2 > logs/hw_$n.log 2>&1
done
F1T_PROCS=8 $PY -u reeval.py v3drBt_stop_lap4 v3drBt_stop_lap6 v3drBt_stop_lap8 v3drBt_stop_lap24 v3drBt_stop_lap48 v3drBt_stop_lap96 > logs/reeval_car_budget.log 2>&1
echo "$(date +%T) CAR_BUDGET_DONE"
