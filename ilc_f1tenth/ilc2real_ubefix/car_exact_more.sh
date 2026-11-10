#!/bin/bash
# After car_exact_oracles.sh (RMA / PPO exact): C2 FADA exact (per-car IDMs from each multi-body car's own transitions,
# the trunk-only target, the oracle planner from our best arm, idealized rungs), then the ladder's seed-1 robustness check.
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
NP=14
OURS=v3drBtL_s0_lad1_lap10
until grep -q CAR_EXACT_DONE logs/car_exact_oracles.out 2>/dev/null; do sleep 60; done
[ -f runs/fada_exact_idm/collect.json ] || F1T_PROCS=$NP $PY -u fada_collect_car.py --out runs/fada_exact_idm --episodes 6000 \
  --rungs-from $OURS --gpu 1 > logs/fada_exact_collect.log 2>&1
CK="24 96 384 1536 2016"
[ -f runs/hw/fada_exact_ep2016/summary.json ] || F1T_PROCS=$NP $PY -u fada_real_car.py --idm "runs/fada_exact_idm/idm_{car}.npz" \
  --arm fada_exact --horizon 4 --privileged 1 --oracle-arm $OURS --rungs-from $OURS --checkpoints $CK --gpu 1 > logs/fada_exact.log 2>&1
F1T_PROCS=$NP $PY -u reeval.py fada_exact_ep0 $(for k in $CK; do echo -n "fada_exact_ep$k "; done) > logs/reeval_fada_exact.log 2>&1
echo "$(date +%T) C2 done"
# the ladder recipe, seed 1 (the same sim stage as v3drBtL_s0 and the same 10-lap ladder hardware stage, 8 cars)
PL8="$(for t in square2fast figfast; do for b in 0 10 18 25; do echo -n "mocap_${t}_b$b "; done; done)"
PLL="$PL8$(for s in _mu80 _mu60; do for t in square2fast figfast; do for b in 0 10 18 25; do echo -n "mocap_${t}_b$b$s "; done; done; done)"
V3="--anchor-w 1.0 --beta 0.3 --cap 0.04 --max-ey 0.2 --a-off 0.15 --pert 1.5 --explore-frac 0.7 --eval-every 5"
TASK="--err-scale 0.05 1e9 1e9 1e9 1e9 0.3"
[ -f runs/v3drBtL_s1/policy_final.npz ] || $PY -u sim_train.py --out runs/v3drBtL_s1 --plans $PLL --dr 1.0 --bc-iters 10 $V3 --iters 30 \
  --seed 1 $TASK --gpu 3 > logs/v3drBtL_s1.log 2>&1
[ -f runs/hw/v3drBtL_s1_lad1_lap10/summary.json ] || F1T_PROCS=$NP $PY -u hw_stage.py --policy runs/v3drBtL_s1/policy_final.npz \
  --out runs/hw/v3drBtL_s1_lad1_lap10 --iters 5 --goals mocap_square2fast_b25 mocap_figfast_b25 --goal-cycle 1 \
  --fallback-suffix _mu80,_mu60 --fallback-on band --fallback-after 1 \
  --cars real_nom real_mass real_mu real_act real_lag ax:mu=0.7 ax:mu=0.9 ax:mu=0.6 --gpu 3 > logs/hw_v3drBtL_s1_lad1_lap10.log 2>&1
F1T_PROCS=$NP $PY -u reeval.py v3drBtL_s1_lad1_lap10 > logs/reeval_ladder_s1.log 2>&1
echo "$(date +%T) CAR_MORE_DONE"
