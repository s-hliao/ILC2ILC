#!/bin/bash
# Low-friction fix for the car (user 2026-10-11: "fix our method for the car on low-grip scenarios, keep going until it is
# fixed"; criterion: a stable trajectory, grip OR drift; slower is fine). The hardware stage picks the reference by OUTCOME:
# hw_stage --fallback-suffix _mu80,_mu60 --fallback-on band (a goal out of the success band --fallback-after trials in a row
# steps one rung down a ladder of conservative plans: the same track / sideslip re-planned for 0.8x / 0.6x friction; no
# parameter estimation). fbL_a1 on v3drBt_s0: all 5 cars 100 %, x0.7 car 88 % (only figfast b0 _mu60 crashes: the network never
# saw a _mu60 plan; the plan's LQR flies it). Fix here: the same recipe (DR (B) sim stage, task objective) with the ladder's
# plans in the bank (PL8 + their _mu80 / _mu60 variants; the sim stays nominal), then the ladder hardware stage.
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
PL8="$(for t in square2fast figfast; do for b in 0 10 18 25; do echo -n "mocap_${t}_b$b "; done; done)"
PLL="$PL8$(for s in _mu80 _mu60; do for t in square2fast figfast; do for b in 0 10 18 25; do echo -n "mocap_${t}_b$b$s "; done; done; done)"
V3="--anchor-w 1.0 --beta 0.3 --cap 0.04 --max-ey 0.2 --a-off 0.15 --pert 1.5 --explore-frac 0.7 --eval-every 5"
TASK="--err-scale 0.05 1e9 1e9 1e9 1e9 0.3"
G2="--goals mocap_square2fast_b25 mocap_figfast_b25"
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.7 ax:mu=0.9 ax:mu=0.6"
LAD="--fallback-suffix _mu80,_mu60 --fallback-on band"
N=v3drBtL_s0
[ -f runs/$N/policy_final.npz ] || $PY -u sim_train.py --out runs/$N --plans $PLL --dr 1.0 --bc-iters 10 $V3 --iters 30 --seed 0 $TASK \
  --gpu 3 > logs/$N.log 2>&1
echo "$(date +%T) sim done"
arm() { [ -f runs/hw/$1/summary.json ] || F1T_PROCS=8 $PY -u hw_stage.py --policy runs/$N/policy_final.npz --out runs/hw/$1 $2 --cars $CARS \
  --gpu $3 > logs/hw_$1.log 2>&1; }
arm ${N}_zs "--iters 0" 0 & arm ${N}_lad1_lap10 "--iters 5 $G2 --goal-cycle 1 $LAD --fallback-after 1" 1 & wait
arm ${N}_lad1_lap4 "--iters 2 $G2 --goal-cycle 1 $LAD --fallback-after 1" 0 & arm ${N}_lad2_lap10 "--iters 5 $G2 --goal-cycle 1 $LAD --fallback-after 2" 1 & wait
# the old headline network with the ladder on the x0.6 car too (fbL_a1 had no x0.6 car)
[ -f runs/hw/fbL_a1_mu06_lap10/summary.json ] || F1T_PROCS=8 $PY -u hw_stage.py --policy runs/v3drBt_s0/policy_final.npz --out runs/hw/fbL_a1_mu06_lap10 \
  --iters 5 $G2 --goal-cycle 1 $LAD --fallback-after 1 --cars ax:mu=0.6 --gpu 2 > logs/hw_fbL_a1_mu06_lap10.log 2>&1
F1T_PROCS=16 $PY -u reeval.py ${N}_zs ${N}_lad1_lap10 ${N}_lad1_lap4 ${N}_lad2_lap10 fbL_a1_mu06_lap10 > logs/reeval_ladder.log 2>&1
echo "$(date +%T) LADDER_DONE"
