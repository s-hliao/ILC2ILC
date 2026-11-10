#!/bin/bash
# car_tracks_redo.sh (2026-10-27): car_free_tracks.sh's training after its OOM (freeBC5_s0, freeDR5_s1; freePPO5 had no
# BC start) and without its ladder stage (user: replace the ladder; car_noladder.sh picks the pace weight): the 5-track
# bank, ours + DR and PPO+DR, then the zero-shot eval_free (original free goals + held-out squareH). gpu_lock shared.
cd "$(dirname "$0")"; source env_tight.sh; mkdir -p logs
C=$PWD; PY=~/miniconda3/envs/f1t/bin/python; L=/home/henry/ilc_ws/log/gpu_lock.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=2
EV=/home/henry/ilc_ws/log/dilc/car_free/eval_free.py
P=""; for t in mocap_square2fast mocap_figfast mocap_square2fastM mocap_figfastM mocap_stadT; do for r in "" _mu80 _mu60; do P="$P ${t}_b0_free$r"; done; done
TASK="--err-scale 0.05 1e9 1e9 1e9 1e9 0.3"
V3="--beta 0.3 --cap 0.04 --max-ey 0.2 --a-off 0.15 --pert 1.5 --explore-frac 0.7 --eval-every 5 --bc-iters 10 --iters 30 --lanes 384 --hidden 256"
st() { n=$1; g=$2; shift 2; [ -f runs/$n/policy_final.npz ] || $L shared $g $PY -u sim_train.py --plans $P --out runs/$n $V3 $TASK --gpu $g "$@" > logs/$n.log 2>&1; }
st freeBC5_s0 1 --iters 0 --dr 0 --anchor-w 1.0 --seed 0 &
st freeDR5_s1 1 --dr 1.0 --anchor-w 1.0 --seed 1 &
wait
for s in 0 1 2; do
  [ -f runs/freePPO5_s$s/policy_final.npz ] || $L shared $s $PY -u ppo_dr.py --plans $P --out runs/freePPO5_s$s --init runs/freeBC5_s0/policy_final.npz \
      --mode ppo --dr 1.0 --iters 300 --lanes 1024 --reward task --seed $s --gpu $s > logs/freePPO5_s$s.log 2>&1 &
done
while [ ! -f runs/freeDR5_s0/policy_final.npz ] || [ ! -f runs/freeDR5_s2/policy_final.npz ]; do sleep 30; done
wait
ARGS=""
for s in 0 1 2; do ARGS="$ARGS freeDR5_s$s $C/runs/freeDR5_s$s/policy_final.npz freePPO5_s$s $C/runs/freePPO5_s$s/policy_final.npz"; done
F1T_PROCS=12 OMP_NUM_THREADS=1 $PY -u $EV $ARGS > logs/free_zs5.log 2>&1
echo "$(date +%T) CAR_TRACKS_REDO_DONE"
