#!/bin/bash
# car_free_train.sh (2026-10-26; user: no sideslip target -- the track reference is the only goal; "go with all four"):
# the car's networks retrained on the sideslip-free plan bank (gen_free_plans.sh), objective = path + pace only
# (err-scale = consts.TASK_SCALE; PPO --reward task). 3 seeds each:
#   freeNom    ours, nominal sim (the car's v3 recipe, --dr 0)
#   freeDR     ours + DR (the headline v3drBtL recipe, --dr 1)
#   freePPO    PPO + DR (ppoL_task's recipe) from freeBC (the free bank's LQR-cloned start)
#   freeNomNA -> freeDRT<W>: the pogo fixes -- nominal WITHOUT the anchor, then the DR transfer WITH it plus the
#              feedback-gain penalty (--gain-w W, W = 3e-4 / 1e-3), snapshots every 5 its, picked by the car's sim-only
#              stopping rule (sim_validate.py on the free plans, held-out mocap_squareH; stop_pick.py)
cd "$(dirname "$0")"; source env_tight.sh; mkdir -p logs runs/stopping
PY=~/miniconda3/envs/f1t/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=2
P="mocap_square2fast_b0_free mocap_figfast_b0_free mocap_square2fast_b0_free_mu80 mocap_figfast_b0_free_mu80 mocap_square2fast_b0_free_mu60 mocap_figfast_b0_free_mu60"
TASK="--err-scale 0.05 1e9 1e9 1e9 1e9 0.3"
V3="--beta 0.3 --cap 0.04 --max-ey 0.2 --a-off 0.15 --pert 1.5 --explore-frac 0.7 --eval-every 5 --bc-iters 10 --iters 30 --lanes 192 --hidden 256"
st() { n=$1; g=$2; shift 2; [ -f runs/$n/policy_final.npz ] || $PY -u sim_train.py --plans $P --out runs/$n $V3 $TASK --gpu $g "$@" > logs/$n.log 2>&1; }
# BC start for PPO
st freeBC_s0 0 --iters 0 --dr 0 --anchor-w 1.0 --seed 0
for s in 0 1 2; do
  ( st freeNom_s$s $s --dr 0 --anchor-w 1.0 --seed $s
    st freeDR_s$s $s --dr 1.0 --anchor-w 1.0 --seed $s
    st freeNomNA_s$s $s --dr 0 --anchor-w 0 --seed $s
    for W in 3e-4 1e-3; do st freeDRT${W}_s$s $s --init runs/freeNomNA_s$s/policy_final.npz --dr 1.0 --anchor-w 1.0 --gain-w $W --seed $s; done
    echo "$(date +%T) ours s$s trained" ) &
  ( [ -f runs/freePPO_s$s/policy_final.npz ] || $PY -u ppo_dr.py --plans $P --out runs/freePPO_s$s --init runs/freeBC_s0/policy_final.npz \
        --mode ppo --dr 1.0 --iters 300 --lanes 1024 --reward task --seed $s --gpu $s > logs/freePPO_s$s.log 2>&1
    echo "$(date +%T) PPO s$s trained" ) &
done
wait
# the sim-only stopping rule for the pogo-fix variant
RUNS=$(ls -d runs/freeDRT*_s* | tr '\n' ' ')
$PY -u sim_validate.py --runs $RUNS --out runs/stopping/sim_val_free.json --train-plans $P --held-plans mocap_squareH_b0_free --gpu 0 > logs/sim_val_free.log 2>&1
for r in $RUNS; do $PY -u stop_pick.py $r runs/stopping/sim_val_free.json ${r}_stop >> logs/stop_pick_free.log 2>&1; done
echo "$(date +%T) CAR_FREE_TRAIN_DONE"
