#!/bin/bash
# car_redo.sh (2026-10-17): the car comparison redone with the baselines on OUR objective (task reward: e_y + pace) and the
# same plan bank as the headline network (PL8 + _mu80 / _mu60 variants, 24 plans), on the 8 cars (5 + friction x0.9/0.7/0.6).
# Selection / reporting here on the VALIDATION reeval seed 301 (tag _val); 1701 (every earlier car number) is spent; a fresh
# seed is used once at the end (car_redo_test.sh).
#   baselines: ppoL_task_s{0,1,2} (PPO+DR), rmaL_task_s{0,1,2} (RMA, on-policy phase 2, H 40, action matching), init = the
#              LQR-cloned network as before (runs/v1nom_s0/policy_bc.npz)
#   arms:      <net>_zs8 (zero-shot, 8 cars); ppoL_task_s<k>_ladR_lap10 (PPO+DR + our 10-lap ladder stage);
#              v3drBtL_s<k>_ladR_lap10 (ours, identical recipe for all seeds); baselines also scored on our rungs
#              (reeval --rungs-from v3drBtL_s<k>_ladR_lap10: the idealized reference, no real laps)
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
PL8="$(for t in square2fast figfast; do for b in 0 10 18 25; do echo -n "mocap_${t}_b$b "; done; done)"
PLL="$PL8$(for s in _mu80 _mu60; do for t in square2fast figfast; do for b in 0 10 18 25; do echo -n "mocap_${t}_b$b$s "; done; done; done)"
G2="--goals mocap_square2fast_b25 mocap_figfast_b25"
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.7 ax:mu=0.9 ax:mu=0.6"
LAD="--iters 5 $G2 --goal-cycle 1 --fallback-suffix _mu80,_mu60 --fallback-on band --fallback-after 1 --fallback-revert 1 --goal-order worst"   # = the existing v3drBtL_s*_ladR_lap10 recipe
P2="--hist 40 --p2-onpolicy 1 --p2-am 10 --adapt-iters 150"
INIT=runs/v1nom_s0/policy_bc.npz
arm() {  # name policy args gpu
  [ -f runs/hw/$1/summary.json ] || F1T_PROCS=4 $PY -u hw_stage.py --policy $2 --out runs/hw/$1 $3 --cars $CARS --gpu $4 > logs/hw_$1.log 2>&1
}
running() { pgrep -f "^$PY -u ppo_dr.py .*--out runs/$1 " > /dev/null; }
wait_for() { while running $1 && [ ! -f $2 ]; do sleep 60; done; }   # a training already in flight (a relaunch): wait for it
train() {  # seed gpu
  local s=$1 g=$2
  wait_for ppoL_task_s$s runs/ppoL_task_s$s/policy_final.npz
  [ -f runs/ppoL_task_s$s/policy_final.npz ] || $PY -u ppo_dr.py --plans $PLL --out runs/ppoL_task_s$s --init $INIT --mode ppo \
     --reward task --seed $s --gpu $g > logs/ppoL_task_s$s.log 2>&1
  arm ppoL_task_s${s}_zs8 runs/ppoL_task_s$s/policy_final.npz "--iters 0" $g &
  arm ppoL_task_s${s}_ladR_lap10 runs/ppoL_task_s$s/policy_final.npz "$LAD" $g &
  wait_for rmaL_task_s$s runs/rmaL_task_s$s/rma_adapt.npz
  [ -f runs/rmaL_task_s$s/rma_adapt.npz ] || $PY -u ppo_dr.py --plans $PLL --out runs/rmaL_task_s$s --init $INIT --mode rma \
     --reward task $P2 --seed $s --gpu $g > logs/rmaL_task_s$s.log 2>&1
  arm rmaL_task_s${s}_zs8 runs/rmaL_task_s$s/policy_final.npz "--iters 0" $g
  wait
}
ours() {  # seed gpu
  arm v3drBtL_s$1_zs8 runs/v3drBtL_s$1/policy_final.npz "--iters 0" $2
  arm v3drBtL_s$1_ladR_lap10 runs/v3drBtL_s$1/policy_final.npz "$LAD" $2
}
train 0 1 & train 1 2 & train 2 3 &
ours 0 1 & ours 1 2 & ours 2 3 &
wait
echo "$(date +%T) arms done"
ARMS=""; for s in 0 1 2; do ARMS+="v3drBtL_s${s}_zs8 v3drBtL_s${s}_ladR_lap10 ppoL_task_s${s}_zs8 ppoL_task_s${s}_ladR_lap10 rmaL_task_s${s}_zs8 "; done
F1T_PROCS=16 $PY -u reeval.py $ARMS --seed 301 --tag _val > logs/reeval_redo_val.log 2>&1
for s in 0 1 2; do                                  # the baselines on our rungs (idealized reference)
  for b in ppoL_task_s${s}_zs8 rmaL_task_s${s}_zs8; do
    F1T_PROCS=16 $PY -u reeval.py $b --seed 301 --tag _val_rungs --rungs-from v3drBtL_s${s}_ladR_lap10 >> logs/reeval_redo_val.log 2>&1
  done
done
echo "$(date +%T) CAR_REDO_DONE"
