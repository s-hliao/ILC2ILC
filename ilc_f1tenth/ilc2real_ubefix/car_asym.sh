#!/bin/bash
# The car's ASYMPTOTIC real budget (2026-10-11): the stopping-rule network's tracking error plateaus at 48 laps (48 vs 96:
# mean RMS e_y 3.53 vs 3.54 cm, success 84.2 vs 84.4 %; car_budget.sh), so the figure rows that stop at 10 laps get a
# 48-lap arm with their 10-lap arms' exact recipe (the two b25 plans alternated, 2-lap chained trials; 24 iterations),
# 2 at a time (6 workers each; the CPU cap, beside car_stop_axes.sh), then reeval'd.
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
run() {   # NAME POLICY
  [ -f runs/hw/$1/summary.json ] || F1T_PROCS=6 $PY -u hw_stage.py --policy $2 --out runs/hw/$1 --iters 24 \
    --goals mocap_square2fast_b25 mocap_figfast_b25 --goal-cycle 1 --gpu 1 > logs/hw_$1.log 2>&1
  echo "$(date +%T) $1"
}
export -f run; export PY
printf '%s\n' "v3drB_s0_lap48 runs/v3drB_s0/policy_final.npz" "v3drB_s1_lap48 runs/v3drB_s1/policy_final.npz" \
  "v3drA_s0_lap48 runs/v3drA_s0/policy_final.npz" "v3drA_s1_lap48 runs/v3drA_s1/policy_final.npz" \
  "v3nom_s0_lap48 runs/v3nom_s0/policy_final.npz" "v3nom_s1_lap48 runs/v3nom_s1/policy_final.npz" \
  "v3nom_s2_lap48 runs/v3nom_s2/policy_final.npz" "ppo_dr_lap48 runs/ppo_dr_s0/policy_final.npz" \
  | xargs -P 2 -L 1 bash -c 'run "$0" "$1"'
F1T_PROCS=12 $PY -u reeval.py v3drB_s0_lap48 v3drB_s1_lap48 v3drA_s0_lap48 v3drA_s1_lap48 v3nom_s0_lap48 v3nom_s1_lap48 \
  v3nom_s2_lap48 ppo_dr_lap48 > logs/reeval_asym.log 2>&1
echo "$(date +%T) CAR_ASYM_DONE"
