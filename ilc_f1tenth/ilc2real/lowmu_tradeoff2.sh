#!/bin/bash
# low-friction car (real_mu), 10-lap budget (5 chained 2-lap trials alternating the two b25 plans): trade the pace against the path.
# The task objective divides e_y by EY (0.05 m) and |V| - p|V_plan| by V (0.30 m/s); a smaller EY or a larger V weighs the path
# more against the pace; p = --pace-target. Base: v3drBt_s0 (the headline network). (2026-10-11, "do all of them")
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
G2="--goals mocap_square2fast_b25 mocap_figfast_b25"
declare -A ARM=(
  [lm_c10_p90]="--err-scale 0.025 1e9 1e9 1e9 1e9 0.60 --pace-target 0.90 --cap 0.10 --beta 0.5"
  [lm_c10b8_p90]="--err-scale 0.025 1e9 1e9 1e9 1e9 0.60 --pace-target 0.90 --cap 0.10 --beta 0.8"
  [lm_c15_p90]="--err-scale 0.025 1e9 1e9 1e9 1e9 0.60 --pace-target 0.90 --cap 0.15 --beta 0.8"
  [lm_c10_p86]="--err-scale 0.025 1e9 1e9 1e9 1e9 0.60 --pace-target 0.86 --cap 0.10 --beta 0.8"
)
g=0
for n in lm_c10_p90 lm_c10b8_p90 lm_c15_p90 lm_c10_p86; do
  ( [ -f runs/hw/${n}_lap10/summary.json ] || F1T_PROCS=4 $PY -u hw_stage.py --policy runs/v3drBt_s0/policy_final.npz --out runs/hw/${n}_lap10 \
      --cars real_mu --iters 5 $G2 --goal-cycle 1 ${ARM[$n]} --gpu $g > logs/hw_${n}_lap10.log 2>&1 ) &
  g=$(( (g + 1) % 4 ))
done
wait
F1T_PROCS=12 $PY -u reeval.py lm_c10_p90_lap10 lm_c10b8_p90_lap10 lm_c15_p90_lap10 lm_c10_p86_lap10 --runs 6 > logs/reeval_lm_tradeoff2.log 2>&1
echo "$(date +%T) LM_TRADEOFF_DONE"
