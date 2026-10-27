#!/bin/bash
# car_free_tracks.sh (2026-10-27): the free-goal networks retrained on 5 tracks instead of 2 (make_extra_tracks.py:
# square2fast / figfast mirrored, a stadium) -- on 2 tracks ours overfits them (held-out mocap_squareH: ours + stage
# 72-75 % vs PPO+DR / LQR ~97 %). mocap_squareH stays held out. Both methods get the same bank (fair):
#   plans     gen_free_plans.sh's sideslip-free plan + _mu80 / _mu60 rungs for each new track, LQR gains cached once
#   freeBC5   the LQR-cloned start on the 5-track bank (seed 0)
#   freeDR5   ours + DR (freeDR's recipe: --dr 1, anchor 1, V3), 384 lanes for the 2.5x bank, 3 seeds
#   freePPO5  PPO+DR (freePPO's recipe) from freeBC5, 3 seeds
# then zero-shot (eval_free.py: the ORIGINAL free goals + held-out squareH, unchanged), the ladR stage on the two
# original training tracks (as every other arm), and eval_free on the settled rungs (+stageR). GPU: gpu_lock shared.
cd "$(dirname "$0")"; source env_tight.sh; mkdir -p logs runs/hw
C=$PWD; PY=~/miniconda3/envs/f1t/bin/python; L=/home/henry/ilc_ws/log/gpu_lock.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=2
EV=/home/henry/ilc_ws/log/dilc/car_free/eval_free.py
NEW="mocap_square2fastM mocap_figfastM mocap_stadT"
for t in $NEW; do
  [ -f $F1T_PLANS/${t}_b0_free.npz ] || $PY -u trajopt_drift.py $t --steps 0 --beta 0 --w-beta 0 --tag _free --out $F1T_PLANS > logs/to_free_$t.log 2>&1 &
  for m in 80 60; do
    [ -f $F1T_PLANS/${t}_b0_free_mu$m.npz ] || $PY -u trajopt_drift.py $t --steps 0 --beta 0 --w-beta 0 --mu-plan 0.$m --tag _free_mu$m \
        --out $F1T_PLANS > logs/to_free_mu${m}_$t.log 2>&1 &
  done
done; wait
P=""
for t in mocap_square2fast mocap_figfast $NEW; do for r in "" _mu80 _mu60; do P="$P ${t}_b0_free$r"; done; done
for n in $P; do [ -f $F1T_PLANS/$n.npz ] || { echo "missing plan $n"; exit 1; }; done
$PY -c "import bank; [bank.load_plan(n) for n in '$P'.split()]; print('lqr cached')"
echo "$(date +%T) plans done: $P"
TASK="--err-scale 0.05 1e9 1e9 1e9 1e9 0.3"
V3="--beta 0.3 --cap 0.04 --max-ey 0.2 --a-off 0.15 --pert 1.5 --explore-frac 0.7 --eval-every 5 --bc-iters 10 --iters 30 --lanes 384 --hidden 256"
st() { n=$1; g=$2; shift 2; [ -f runs/$n/policy_final.npz ] || $L shared $g $PY -u sim_train.py --plans $P --out runs/$n $V3 $TASK --gpu $g "$@" > logs/$n.log 2>&1; }
st freeBC5_s0 0 --iters 0 --dr 0 --anchor-w 1.0 --seed 0
for s in 0 1 2; do
  st freeDR5_s$s $s --dr 1.0 --anchor-w 1.0 --seed $s &
  ( [ -f runs/freePPO5_s$s/policy_final.npz ] || $L shared $s $PY -u ppo_dr.py --plans $P --out runs/freePPO5_s$s --init runs/freeBC5_s0/policy_final.npz \
        --mode ppo --dr 1.0 --iters 300 --lanes 1024 --reward task --seed $s --gpu $s > logs/freePPO5_s$s.log 2>&1 ) &
done; wait
echo "$(date +%T) training done"
ARGS=""
for s in 0 1 2; do ARGS="$ARGS freeDR5_s$s $C/runs/freeDR5_s$s/policy_final.npz freePPO5_s$s $C/runs/freePPO5_s$s/policy_final.npz"; done
F1T_PROCS=12 OMP_NUM_THREADS=1 $PY -u $EV $ARGS > logs/free_zs5.log 2>&1
echo "$(date +%T) zero-shot done"
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.9 ax:mu=0.7 ax:mu=0.6 harsh_combo harsh_steer harsh_sense harsh_drift"
G="mocap_square2fast_b0_free mocap_figfast_b0_free"
LADR="--iters 5 --beta 0.5 --delta 0.05 --cap 0.05 --objective task --goal-cycle 1 --goal-order worst --trial-laps 2 --anchor-w 1.0 \
  --fallback-suffix _mu80,_mu60 --fallback-after 1 --fallback-on band --fallback-ey 0.1 --fallback-revert 1 --eval-runs 4 --eval-laps 5 \
  --eval-seed 301 --seed 9001 --launch 1.0 8.0 --save-its 1"
g=0
for n in freeDR5_s0 freeDR5_s1 freeDR5_s2 freePPO5_s0 freePPO5_s1 freePPO5_s2; do
  while [ $(jobs -r | wc -l) -ge 3 ]; do sleep 10; done
  [ -f runs/hw/${n}_stage/summary.json ] || F1T_PROCS=4 OMP_NUM_THREADS=1 $L shared $g $PY -u hw_stage.py --policy runs/$n/policy_final.npz \
      --out runs/hw/${n}_stage --cars $CARS --goals $G --anchor-plans $G $LADR --gpu $g > logs/hw_${n}_stage.log 2>&1 &
  g=$(( (g + 1) % 3 )); sleep 5
done; wait
ARGS=""
for n in freeDR5_s0 freeDR5_s1 freeDR5_s2 freePPO5_s0 freePPO5_s1 freePPO5_s2; do
  [ -f runs/hw/${n}_stage/summary.json ] && ARGS="$ARGS ${n}+stageR $C/runs/hw/${n}_stage"
  ARGS="$ARGS ${n}@R $C/runs/$n/policy_final.npz@$C/runs/hw/${n}_stage"
done
F1T_PROCS=12 OMP_NUM_THREADS=1 $PY -u $EV $ARGS > logs/free_real5.log 2>&1
echo "$(date +%T) CAR_FREE_TRACKS_DONE"
