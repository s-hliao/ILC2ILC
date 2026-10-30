#!/bin/bash
# car_v08.sh (2026-10-30; user: 3.74 kg car; on hardware BOTH plastic (mu 0.2) and rubber tires, "I don't care how they
# complete the trajectory, but they both have to converge in a few trajectories"): the car on ONE plan per track that
# both tires can drive -- the sideslip-free plan at 0.8 x speed (_free_v08: trajopt_drift --vscale 0.8; at mu 0.2 a grip
# plan, |beta| 2-7 deg, wheel slip ~1 %; the full-speed _mu80 plans are power slides, beta 30 deg / 45 % wheel spin,
# which put every grippier car (mu >= x2.5) into the wall in 0.2 laps, even the plan LQR). The same plan for every car
# and method (no ladder). Recipes unchanged; only the plan bank changes, the same for ours and PPO:
#   bank   5 training tracks x {_free_v08, _free_v09, _free} (squareH held out)
#   nets   freeBCV_s0 (2 tracks) / freeBC5V_s0 (5) LQR clones; freeNom5V / freeDR5V (ours nominal / + DR); freePPOV (PPO + DR
#          from freeBCV, 2 tracks) / freePPO5V (from freeBC5V, 5 tracks); the tire-covering DR pair (F1T_DR_MU "0.8 5.0":
#          friction x0.8..x5) freeDRg5V (ours) / freePPOg5V (PPO); seeds 0-2
#   VALIDATION (seed 301; the 12 validation cars + 2 rubber cars ax:mu=4.5 / ax:mu=5.5,mass=1.1; _v08 plans): zero-shot,
#          and the 10-lap hardware stage (goals square2fast / figfast _v08) with S0 (frozen recipe) and S1-S3
cd "$(dirname "$0")"; source env_tight.sh; mkdir -p logs runs/hw
C=$PWD; PY=~/miniconda3/envs/f1t/bin/python; L=/home/henry/ilc_ws/log/gpu_lock.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=2
EV=/home/henry/ilc_ws/log/dilc/car_free/eval_free.py
T5="mocap_square2fast mocap_figfast mocap_square2fastM mocap_figfastM mocap_stadT"
for t in $T5 mocap_squareH; do for v in 0.8 0.9; do
  [ -f $F1T_PLANS/${t}_b0_free_v${v/./}.npz ] || $PY -u trajopt_drift.py $t --steps 0 --beta 0 --w-beta 0 --vscale $v --tag _free_v${v/./} \
      --out $F1T_PLANS > logs/m374_to_free_v${v/./}_$t.log 2>&1 &
done; done; wait
P2=""; for t in mocap_square2fast mocap_figfast; do for r in _v08 _v09 ""; do P2="$P2 ${t}_b0_free$r"; done; done
P5=""; for t in $T5; do for r in _v08 _v09 ""; do P5="$P5 ${t}_b0_free$r"; done; done
for n in $P5 mocap_squareH_b0_free_v08; do [ -f $F1T_PLANS/$n.npz ] || { echo "missing plan $n"; exit 1; }; done
$PY -c "import bank; [bank.load_plan(n) for n in '$P5 mocap_squareH_b0_free_v08'.split()]; print('lqr cached')"
echo "$(date +%T) plans: $P5"
TASK="--err-scale 0.05 1e9 1e9 1e9 1e9 0.3"
V3="--beta 0.3 --cap 0.04 --max-ey 0.2 --a-off 0.15 --pert 1.5 --explore-frac 0.7 --eval-every 5 --bc-iters 10 --iters 30 --hidden 256"
st() { n=$1; g=$2; pl=$3; shift 3; [ -f runs/$n/policy_final.npz ] || $L shared $g $PY -u sim_train.py --plans $pl --out runs/$n $V3 $TASK --gpu $g "$@" > logs/$n.log 2>&1; }
ppo() { n=$1; g=$2; pl=$3; init=$4; s=$5; [ -f runs/$n/policy_final.npz ] || $L shared $g $PY -u ppo_dr.py --plans $pl --out runs/$n --init $init \
    --mode ppo --dr 1.0 --iters 300 --lanes 1024 --reward task --seed $s --gpu $g > logs/$n.log 2>&1; }
st freeBCV_s0 0 "$P2" --lanes 192 --iters 0 --dr 0 --anchor-w 1.0 --seed 0 &
st freeBC5V_s0 1 "$P5" --lanes 384 --iters 0 --dr 0 --anchor-w 1.0 --seed 0 &
wait
echo "$(date +%T) BC starts done"
for s in 0 1 2; do
  ( st freeNom5V_s$s $s "$P5" --lanes 384 --dr 0 --anchor-w 1.0 --seed $s; st freeDR5V_s$s $s "$P5" --lanes 384 --dr 1.0 --anchor-w 1.0 --seed $s
    F1T_DR_MU="0.8 5.0" st freeDRg5V_s$s $s "$P5" --lanes 384 --dr 1.0 --anchor-w 1.0 --seed $s ) &
  ( ppo freePPOV_s$s $s "$P2" runs/freeBCV_s0/policy_final.npz $s; ppo freePPO5V_s$s $s "$P5" runs/freeBC5V_s0/policy_final.npz $s
    F1T_DR_MU="0.8 5.0" ppo freePPOg5V_s$s $s "$P5" runs/freeBC5V_s0/policy_final.npz $s ) &
done; wait
NETS="freeNom5V freeDR5V freeDRg5V freePPOV freePPO5V freePPOg5V"
for n in $NETS; do for s in 0 1 2; do [ -f runs/${n}_s$s/policy_final.npz ] || echo "MISSING runs/${n}_s$s"; done; done
echo "$(date +%T) training done"
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.9 ax:mu=0.7 ax:mu=0.6 harsh_combo harsh_steer harsh_sense harsh_drift ax:mu=4.5 ax:mu=5.5,mass=1.1"
ARGS="lqr_V_val lqr"
for s in 0 1 2; do for n in $NETS; do ARGS="$ARGS ${n}_s${s}_valzs $C/runs/${n}_s$s/policy_final.npz"; done; done
EVAL_FREE_CARS="$CARS" EVAL_FREE_PLAN_SUFFIX=_v08 EVAL_FREE_NO_RUNGS=1 F1T_PROCS=27 OMP_NUM_THREADS=1 $PY -u $EV $ARGS > logs/v08_valzs.log 2>&1
echo "$(date +%T) zero-shot validation done"
G="mocap_square2fast_b0_free_v08 mocap_figfast_b0_free_v08"
BASE="--iters 5 --objective task --goal-cycle 1 --goal-order worst --trial-laps 2 --anchor-w 1.0 \
  --eval-runs 4 --eval-laps 5 --eval-seed 301 --seed 9001 --launch 1.0 8.0 --save-its 1"
declare -A CFG=([S0]="--beta 0.5 --delta 0.05 --cap 0.05"
                [S1]="--beta 0.5 --delta 0.05 --cap 0.05 --fit-realize 0.9"
                [S2]="--beta 1.0 --delta 0.005 --cap 0.1 --fit-realize 0.9"
                [S3]="--beta 1.0 --delta 0.02 --cap 0.1 --fit-realize 0.9")
export OMP_NUM_THREADS=1 F1T_PROCS=3
NPAR=9
for cfg in S0 S2 S1 S3; do for s in 0 1 2; do for n in freeDR5V freeNom5V freePPOV; do
  out=runs/hw/${n}_s${s}_val${cfg}_stage
  [ -f $out/summary.json ] && continue
  while [ $(jobs -r | wc -l) -ge $NPAR ]; do sleep 10; done
  ( rm -rf $out; $L tiny $s $PY -u hw_stage.py --policy runs/${n}_s$s/policy_final.npz --out $out --cars $CARS --goals $G \
      --anchor-plans $G $BASE ${CFG[$cfg]} --gpu $s > logs/hw_${n}_s${s}_val${cfg}_stage.log 2>&1 ) &
  sleep 2
done; done; done
wait
echo "$(date +%T) validation stages done"
ARGS=""
for s in 0 1 2; do for cfg in S0 S1 S2 S3; do for n in freeDR5V freeNom5V freePPOV; do
  [ -f runs/hw/${n}_s${s}_val${cfg}_stage/summary.json ] && ARGS="$ARGS ${n}_s${s}_val${cfg}+stage $C/runs/hw/${n}_s${s}_val${cfg}_stage"
done; done; done
EVAL_FREE_CARS="$CARS" EVAL_FREE_PLAN_SUFFIX=_v08 EVAL_FREE_NO_RUNGS=1 F1T_PROCS=27 $PY -u $EV $ARGS > logs/v08_valstage.log 2>&1
echo "$(date +%T) CAR_V08_VAL_DONE"
