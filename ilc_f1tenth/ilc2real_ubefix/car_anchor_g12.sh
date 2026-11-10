#!/bin/bash
# car_anchor.sh (2026-10-30): the car stage's failure is goal-to-goal carry-over (freeDR5V_s0 S0, real_nom: it2 drives
# square2fast at ~1 cm from every start; after the figfast update it3 is 3 cm and crashes under the start perturbation,
# it4 crashes) -- the anchored refit holds the network only weakly (--anchor-w 1 x N0 / (N0 + laps), on the 2 goal plans'
# nominal-sim states, anchored to the PREVIOUS iterate). VALIDATION sweep of the anchor (existing knobs; the update rule
# unchanged), 1.0 m envelope (env_tight.sh), _v08 plans, seed 301, the 12 validation cars + 2 rubber cars:
#   A0  S0 (the frozen recipe)            A5   S0 + --anchor-w 5          A20  S0 + --anchor-w 20
#   A5P S0 + --anchor-w 5 + --anchor-plans all 15 bank plans (5 tracks x _v08/_v09/full)
# nets freeDR5V (ours + DR) / freePPOV (PPO + DR), seeds 0-2 -> runs/hw/<net>_s<s>_val<cfg>_stage, eval_free *_val<cfg>1m+stage
cd "$(dirname "$0")"; source env_tight.sh; mkdir -p logs runs/hw
C=$PWD; PY=~/miniconda3/envs/f1t/bin/python; L=/home/henry/ilc_ws/log/gpu_lock.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=3
EV=/home/henry/ilc_ws/log/dilc/car_free/eval_free.py
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.9 ax:mu=0.7 ax:mu=0.6 harsh_combo harsh_steer harsh_sense harsh_drift ax:mu=4.5 ax:mu=5.5,mass=1.1"
G="mocap_square2fast_b0_free_v08 mocap_figfast_b0_free_v08"
ALLP=""; for t in mocap_square2fast mocap_figfast mocap_square2fastM mocap_figfastM mocap_stadT; do for r in _v08 _v09 ""; do ALLP="$ALLP ${t}_b0_free$r"; done; done
BASE="--iters 5 --objective task --goal-cycle 1 --goal-order worst --trial-laps 2 \
  --eval-runs 4 --eval-laps 5 --eval-seed 301 --seed 9001 --launch 1.0 8.0 --save-its 1 --beta 0.5 --delta 0.05 --cap 0.05"
declare -A CFG=([A0]="--anchor-w 1.0 --anchor-plans $G" [A5]="--anchor-w 5.0 --anchor-plans $G" [A20]="--anchor-w 20.0 --anchor-plans $G"
                [A5P]="--anchor-w 5.0 --anchor-plans $ALLP")
NPAR=${NPAR:-6}; g=1
for cfg in A0 A5 A20 A5P; do for s in 0 1 2; do for n in freeDR5V freePPOV; do
  out=runs/hw/${n}_s${s}_val${cfg}_stage
  [ -f $out/summary.json ] && continue
  while [ $(jobs -r | wc -l) -ge $NPAR ]; do sleep 10; done
  ( rm -rf $out; $L tiny $g $PY -u hw_stage.py --policy runs/${n}_s$s/policy_final.npz --out $out --cars $CARS --goals $G \
      $BASE ${CFG[$cfg]} --gpu $g > logs/hw_${n}_s${s}_val${cfg}_stage.log 2>&1 ) &
  g=$(( g % 2 + 1 )); sleep 2
done; done; done
wait
echo "$(date +%T) stages done"
ARGS=""
for cfg in A0 A5 A20 A5P; do for s in 0 1 2; do for n in freeDR5V freePPOV; do
  [ -f runs/hw/${n}_s${s}_val${cfg}_stage/summary.json ] && ARGS="$ARGS ${n}_s${s}_val${cfg}1m+stage $C/runs/hw/${n}_s${s}_val${cfg}_stage"
done; done; done
EVAL_FREE_CARS="$CARS" EVAL_FREE_PLAN_SUFFIX=_v08 EVAL_FREE_NO_RUNGS=1 F1T_PROCS=12 $PY -u $EV $ARGS > logs/car_anchor_eval.log 2>&1
echo "$(date +%T) CAR_ANCHOR_DONE"
