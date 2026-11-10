#!/bin/bash
# car_extra.sh (2026-10-29): after car_tiers.sh, on the same 36 tier cars and test seed (EVAL_FREE_SEED 4301):
#  1. PPO + DR + PPO on the car (ppo_real_car.py: the privileged critic / reward, the 0.5 m envelope; from freePPO_s<k>),
#     1-lap episodes on the two training _mu80 plans, 10 per PPO iteration, checkpoints 10 (= DistIL-C_hw's 10 laps) 20 40
#     80 160 320 -> runs/hw/freePPO_s<k>_tiers_real_ep<K>/
#  2. the sim-data curve: every saved snapshot of the sim runs zero-shot (freeNom5 / freeDR5: bc, it005..030, final;
#     freePPO / freePPO5: it100 / 200 / 300, final); their sim laps from each run's log.jsonl
#  3. evaluations (eval_free.py) of 1 and 2 -> eval_free/*_tiers_test.json
cd "$(dirname "$0")"; source env_tight.sh; mkdir -p logs runs/hw
C=$PWD; PY=~/miniconda3/envs/f1t/bin/python; L=/home/henry/ilc_ws/log/gpu_lock.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
EV=/home/henry/ilc_ws/log/dilc/car_free/eval_free.py
CARS=$($PY -c "import mb_car as m; print(' '.join(m.TIER_A_CARS + m.TIER_B_CARS + m.TIER_C_CARS))")
G="mocap_square2fast_b0_free_mu80 mocap_figfast_b0_free_mu80"
CK="10 20 40 80 160 320"
echo "$(date +%T) PPO on the car"
for s in 0 1 2; do
  [ -f runs/hw/freePPO_s${s}_tiers_real_ep320/summary.json ] || F1T_PROCS=8 $L shared $s $PY -u ppo_real_car.py \
      --init runs/freePPO_s$s/policy_final.npz --arm freePPO_s${s}_tiers_real --cars $CARS --goals $G \
      --per-iter 10 --checkpoints $CK --seed $((4242 + s)) --gpu $s > logs/ppo_real_tiers_s$s.log 2>&1 &
done
wait
echo "$(date +%T) evaluations"
ARGS=""
for s in 0 1 2; do
  for k in $CK; do ARGS="$ARGS freePPO_s${s}_tiers_real_ep${k}_test $C/runs/hw/freePPO_s${s}_tiers_real_ep$k"; done
  for n in freeNom5 freeDR5; do
    for f in policy_bc policy_it005 policy_it010 policy_it015 policy_it020 policy_it025 policy_it030; do
      ARGS="$ARGS simh_${n}_s${s}_${f#policy_}_tiers_test $C/runs/${n}_s$s/$f.npz"; done
  done
  for n in freePPO freePPO5; do
    for f in policy_it100 policy_it200 policy_it300; do ARGS="$ARGS simh_${n}_s${s}_${f#policy_}_tiers_test $C/runs/${n}_s$s/$f.npz"; done
  done
done
EVAL_FREE_SEED=4301 EVAL_FREE_CARS="$CARS" EVAL_FREE_PLAN_SUFFIX=_mu80 EVAL_FREE_NO_RUNGS=1 F1T_PROCS=24 $PY -u $EV $ARGS > logs/free_extra_test.log 2>&1
echo "$(date +%T) CAR_EXTRA_DONE"
