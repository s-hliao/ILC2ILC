#!/bin/bash
# car_free_eval.sh (2026-10-26): after car_free_train.sh -- (1) zero-shot of every free-bank network on the 8 current
# + 4 harsh cars (eval_free.py: the free goals + the held-out mocap_squareH); (2) our hardware stage (the ladR recipe:
# 5 iterations x one 2-lap trial, goal cycle worst-first, the _mu80/_mu60 ladder) from ours nominal / ours + DR / the
# pogo-fix variants; (3) PPO on the cars (privileged critic) from PPO+DR, 24..384 episodes per car; (4) every adapted
# arm re-evaluated on the ORIGINAL free goals (eval_free.py; a ladder rung earns nothing there).
cd "$(dirname "$0")"; source env_tight.sh; mkdir -p logs runs/hw
PY=~/miniconda3/envs/f1t/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
EV=/home/henry/ilc_ws/log/dilc/car_free/eval_free.py
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.9 ax:mu=0.7 ax:mu=0.6 harsh_combo harsh_steer harsh_sense harsh_drift"
G="mocap_square2fast_b0_free mocap_figfast_b0_free"
LADR="--iters 5 --beta 0.5 --delta 0.05 --cap 0.05 --objective task --goal-cycle 1 --goal-order worst --trial-laps 2 --anchor-w 1.0 \
  --fallback-suffix _mu80,_mu60 --fallback-after 1 --fallback-on band --fallback-ey 0.1 --fallback-revert 1 --eval-runs 4 --eval-laps 5 \
  --eval-seed 301 --seed 9001 --launch 1.0 8.0 --save-its 1"
NETS=()
for s in 0 1 2; do
  NETS+=("freeNom_s$s:runs/freeNom_s$s/policy_final.npz" "freeDR_s$s:runs/freeDR_s$s/policy_final.npz")
  for W in 3e-4 1e-3; do NETS+=("freeDRT${W}_s$s:runs/freeDRT${W}_s${s}_stop/policy_final.npz"); done
  NETS+=("freeDRP_s$s:runs/freeDRP_s$s/policy_final.npz")
done
# the simplified recipe (user: anchor on hardware only, no line search / gain penalty): wait for freeDRP (car_free_drp.sh)
for s in 0 1 2; do while [ ! -f runs/freeDRP_s$s/policy_final.npz ]; do sleep 30; done; done
# (1) zero-shot
ARGS="lqr lqr"
for nt in "${NETS[@]}"; do ARGS="$ARGS ${nt%%:*} ${nt#*:}"; done
for s in 0 1 2; do ARGS="$ARGS freePPO_s$s runs/freePPO_s$s/policy_final.npz"; done
F1T_PROCS=12 $PY -u $EV $ARGS > logs/free_zs.log 2>&1
echo "$(date +%T) zero-shot done"
# (2) + (3)
NPAR=${NPAR:-5}
for nt in "${NETS[@]}"; do
  n=${nt%%:*}; pol=${nt#*:}
  while [ $(jobs -r | wc -l) -ge $NPAR ]; do sleep 10; done
  [ -f runs/hw/${n}_stage/summary.json ] || $PY -u hw_stage.py --policy $pol --out runs/hw/${n}_stage --cars $CARS --goals $G \
      --anchor-plans $G $LADR > logs/hw_${n}_stage.log 2>&1 &
done
for s in 0 1 2; do
  while [ $(jobs -r | wc -l) -ge $NPAR ]; do sleep 10; done
  [ -f runs/hw/freePPOreal_s${s}_ep384/summary.json ] || $PY -u ppo_real_car.py --init runs/freePPO_s$s/policy_final.npz \
      --arm freePPOreal_s$s --cars $CARS --goals $G --checkpoints 24 48 96 192 384 --gpu $s > logs/ppo_real_free_s$s.log 2>&1 &
done
wait
echo "$(date +%T) real-data phase done"
# (4)
ARGS=""
for nt in "${NETS[@]}"; do n=${nt%%:*}; [ -d runs/hw/${n}_stage ] && ARGS="$ARGS ${n}+stage runs/hw/${n}_stage"; done
for s in 0 1 2; do for k in 24 48 96 192 384; do [ -d runs/hw/freePPOreal_s${s}_ep$k ] && ARGS="$ARGS freePPOreal_s${s}_ep$k runs/hw/freePPOreal_s${s}_ep$k"; done; done
F1T_PROCS=12 $PY -u $EV $ARGS > logs/free_real.log 2>&1
echo "$(date +%T) CAR_FREE_EVAL_DONE"
