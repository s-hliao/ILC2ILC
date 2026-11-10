#!/bin/bash
# car_m374.sh (2026-10-30; user: "car weight should be 3.74 as an f1tenth"): the car pipeline rebuilt at 3.74 kg
# (car_model.NOMINAL['m']; the multi-body cars scale from it -> now exactly the gym's F1TENTH MB set). Every recipe
# unchanged from the 4.6 kg runs (archived in runs/archive_m46_20261030, eval_free/archive_m46_20261030):
#   1 plans    gen_free_plans.sh + car_free_tracks.sh: the sideslip-free plan and its _mu80 / _mu60 variants on the 5
#              training tracks and the held-out squareH -> plans_mu02_tight_m374 (env_tight.sh), LQR gains cached
#   2 networks freeBC_s0 (2-track bank) / freeBC5_s0 (5-track) LQR clones; freeNom5 / freeDR5 (ours nominal / + DR,
#              5 tracks, V3); freePPO (PPO+DR from freeBC, 2 tracks) / freePPO5 (from freeBC5, 5 tracks); seeds 0-2
#   3 VALIDATION (car_drivable80_e5.sh's 12 cars, eval seed 301, _mu80 plans): zero-shot of every network; the hardware
#              stage (10 laps) of freeNom5 / freeDR5 / freePPO with the frozen recipe S0 and the knob sweep S1-S3
#              (car_stage_sweep.sh: S1 refit to convergence, S2 + the pogo's beta 1 / delta 0.005 / cap 0.1, S3 delta 0.02)
# The test tiers (car_tiers.sh / car_extra.sh, seed 4301) run after a config is chosen on validation.
cd "$(dirname "$0")"; source env_tight.sh; mkdir -p logs runs/hw
C=$PWD; PY=~/miniconda3/envs/f1t/bin/python; L=/home/henry/ilc_ws/log/gpu_lock.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=2
EV=/home/henry/ilc_ws/log/dilc/car_free/eval_free.py
echo "$(date +%T) plans -> $F1T_PLANS"
for t in mocap_square2fast mocap_figfast mocap_square2fastM mocap_figfastM mocap_stadT mocap_squareH; do
  [ -f $F1T_PLANS/${t}_b0_free.npz ] || $PY -u trajopt_drift.py $t --steps 0 --beta 0 --w-beta 0 --tag _free --out $F1T_PLANS > logs/m374_to_free_$t.log 2>&1 &
  for m in 80 60; do
    [ -f $F1T_PLANS/${t}_b0_free_mu$m.npz ] || $PY -u trajopt_drift.py $t --steps 0 --beta 0 --w-beta 0 --mu-plan 0.$m --tag _free_mu$m \
        --out $F1T_PLANS > logs/m374_to_free_mu${m}_$t.log 2>&1 &
  done
done; wait
ALLP=""; for t in mocap_square2fast mocap_figfast mocap_square2fastM mocap_figfastM mocap_stadT mocap_squareH; do
  for r in "" _mu80 _mu60; do ALLP="$ALLP ${t}_b0_free$r"; done; done
for n in $ALLP; do [ -f $F1T_PLANS/$n.npz ] || { echo "missing plan $n"; exit 1; }; done
$PY -c "import bank; [bank.load_plan(n) for n in '$ALLP'.split()]; print('lqr cached')"
echo "$(date +%T) plans done"
P2=""; for t in mocap_square2fast mocap_figfast; do for r in "" _mu80 _mu60; do P2="$P2 ${t}_b0_free$r"; done; done
P5=""; for t in mocap_square2fast mocap_figfast mocap_square2fastM mocap_figfastM mocap_stadT; do for r in "" _mu80 _mu60; do P5="$P5 ${t}_b0_free$r"; done; done
TASK="--err-scale 0.05 1e9 1e9 1e9 1e9 0.3"
V3="--beta 0.3 --cap 0.04 --max-ey 0.2 --a-off 0.15 --pert 1.5 --explore-frac 0.7 --eval-every 5 --bc-iters 10 --iters 30 --hidden 256"
st() { n=$1; g=$2; pl=$3; shift 3; [ -f runs/$n/policy_final.npz ] || $L shared $g $PY -u sim_train.py --plans $pl --out runs/$n $V3 $TASK --gpu $g "$@" > logs/$n.log 2>&1; }
ppo() { n=$1; g=$2; pl=$3; init=$4; s=$5; [ -f runs/$n/policy_final.npz ] || $L shared $g $PY -u ppo_dr.py --plans $pl --out runs/$n --init $init \
    --mode ppo --dr 1.0 --iters 300 --lanes 1024 --reward task --seed $s --gpu $g > logs/$n.log 2>&1; }
st freeBC_s0 0 "$P2" --lanes 192 --iters 0 --dr 0 --anchor-w 1.0 --seed 0 &
st freeBC5_s0 1 "$P5" --lanes 384 --iters 0 --dr 0 --anchor-w 1.0 --seed 0 &
wait
echo "$(date +%T) BC starts done"
for s in 0 1 2; do
  ( st freeNom5_s$s $s "$P5" --lanes 384 --dr 0 --anchor-w 1.0 --seed $s; st freeDR5_s$s $s "$P5" --lanes 384 --dr 1.0 --anchor-w 1.0 --seed $s ) &
  ( ppo freePPO_s$s $s "$P2" runs/freeBC_s0/policy_final.npz $s; ppo freePPO5_s$s $s "$P5" runs/freeBC5_s0/policy_final.npz $s ) &
done; wait
for n in freeNom5 freeDR5 freePPO freePPO5; do for s in 0 1 2; do [ -f runs/${n}_s$s/policy_final.npz ] || echo "MISSING runs/${n}_s$s"; done; done
echo "$(date +%T) training done"
# 3 validation
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.9 ax:mu=0.7 ax:mu=0.6 harsh_combo harsh_steer harsh_sense harsh_drift"
G="mocap_square2fast_b0_free_mu80 mocap_figfast_b0_free_mu80"
BASE="--iters 5 --objective task --goal-cycle 1 --goal-order worst --trial-laps 2 --anchor-w 1.0 \
  --eval-runs 4 --eval-laps 5 --eval-seed 301 --seed 9001 --launch 1.0 8.0 --save-its 1"
declare -A CFG=([S0]="--beta 0.5 --delta 0.05 --cap 0.05"
                [S1]="--beta 0.5 --delta 0.05 --cap 0.05 --fit-realize 0.9"
                [S2]="--beta 1.0 --delta 0.005 --cap 0.1 --fit-realize 0.9"
                [S3]="--beta 1.0 --delta 0.02 --cap 0.1 --fit-realize 0.9")
export OMP_NUM_THREADS=1 F1T_PROCS=3
NPAR=9
for cfg in S0 S2 S1 S3; do for s in 0 1 2; do for n in freeDR5 freeNom5 freePPO; do
  out=runs/hw/${n}_s${s}_val${cfg}_stage
  [ -f $out/summary.json ] && continue
  while [ $(jobs -r | wc -l) -ge $NPAR ]; do sleep 10; done
  ( rm -rf $out; $L tiny $s $PY -u hw_stage.py --policy runs/${n}_s$s/policy_final.npz --out $out --cars $CARS --goals $G \
      --anchor-plans $G $BASE ${CFG[$cfg]} --gpu $s > logs/hw_${n}_s${s}_val${cfg}_stage.log 2>&1 ) &
  sleep 2
done; done; done
wait
echo "$(date +%T) validation stages done"
ARGS="lqr_val lqr"
for s in 0 1 2; do
  for n in freeNom5 freeDR5 freePPO freePPO5; do ARGS="$ARGS ${n}_s${s}_valzs $C/runs/${n}_s$s/policy_final.npz"; done
  for cfg in S0 S1 S2 S3; do for n in freeDR5 freeNom5 freePPO; do
    [ -f runs/hw/${n}_s${s}_val${cfg}_stage/summary.json ] && ARGS="$ARGS ${n}_s${s}_val${cfg}+stage $C/runs/hw/${n}_s${s}_val${cfg}_stage"
  done; done
done
EVAL_FREE_PLAN_SUFFIX=_mu80 EVAL_FREE_NO_RUNGS=1 F1T_PROCS=27 $PY -u $EV $ARGS > logs/m374_val.log 2>&1
echo "$(date +%T) CAR_M374_VAL_DONE"
