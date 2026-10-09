#!/bin/bash
# low-friction check: Fiala plans at mu, then the plans' LQR zero-shot on the five multi-body cars at the same mu
cd "$(dirname "$0")"; PY=~/miniconda3/envs/f1t/bin/python
mu=$1; tag=mu${mu/./}; mkdir -p plans_$tag runs/mucheck
export F1T_MU=$mu F1T_PLANS=$PWD/plans_$tag
for t in mocap_square2fast mocap_figfast; do
  $PY -u trajopt_drift.py $t --steps 0 10 18 --beta 25 --out plans_$tag > logs/to_$tag_$t.log 2>&1 &
done; wait
$PY -c "import bank; [bank.load_plan(f'mocap_{t}_b{b}') for t in ('square2fast','figfast') for b in (0,10,18,25)]" > logs/lqr_$tag.log 2>&1
F1T_PROCS=8 $PY -u hw_stage.py --policy lqr --out runs/mucheck/lqr_$tag --iters 0 --eval-runs 4 --eval-laps 3 \
  --eval-plans mocap_square2fast_b25 mocap_figfast_b25 mocap_square2fast_b18 mocap_figfast_b18 mocap_square2fast_b10 mocap_figfast_b10 mocap_square2fast_b0 mocap_figfast_b0 \
  --gpu 0 > logs/mucheck_$tag.log 2>&1
echo done $mu
