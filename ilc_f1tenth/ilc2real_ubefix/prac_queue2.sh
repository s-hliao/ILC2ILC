#!/bin/bash
# prac_queue2.sh LIST SLOTS CPUSET [WAITFILE] (2026-11-08 ~02:10; user "new mask is good": EVERY stage, sim and hardware, uses
# hw_stage.py --skip-below 0.5 (count a trial from the first sample whose measured speed > 0.5 m/s) instead of --skip-launch 0.5; otherwise
# exactly prac_queue.sh. Out runs/screen/PRACB_*. Scored twice: ladder_eval.py (at-speed starts -> sb_practised.json) and
# ladder_eval_launch.py (launched starts, the trials' / hardware's start -> ladder_launch.json). Plastic 8 workers per run, rubber 2.
# Was: prac_queue.sh LIST SLOTS CPUSET [WAITFILE] (2026-11-08 ~00:50, written from matcha; user: "make ABSOLUTELY CERTAIN the drift ends up being
# generated out of our method ... may need to rerun a bunch of figure 1 and 2 ... and possibly 3"): the PRACTICE-ONLY plastic figure-2 runs.
# Each LIST line "PROT NET SEED SS TRACK": one SB stage run, exactly the sb_nom.sh / sb_two_1108.sh flags (SB, 4 iterations, --sim-launch 1
# --skip-launch 0.5), practising TRACK and scored on it (ladder_eval its 0-4, 5 laps, 4 starts, seed 701 -> sb_practised.json):
#   plastic square  mocap_square2fast_b0_free  anchors square2fast + figfast (nominal)      (= SBNOM)
#   plastic fig8    mocap_fig8R100_b0_free     anchors fig8R100 + square2fast (nominal)     (= SBTWO plastic)
#   rubber  square  mocap_square2fast_b0_free_x120 anchors square2fast_x120 + fig8R100_x120 (= SBTWO rubber)
#   rubber  fig8    mocap_fig8R100_b0_free_x120    anchors square2fast_x120 + fig8R100_x120
# Out: runs/screen/PRAC_<PROT>_<TRACK>_<NET>_s<SEED>_ss<SS>. SLOTS runs at a time, 8 workers each, pinned to CPUSET.
cd "$(dirname "$(readlink -f "$0")")"; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu F1T_PROCS=8
PY=~/miniconda3/envs/f1t/bin/python; mkdir -p logs/prac
LIST=$1; SLOTS=$2; CPUS=$3; WAIT=$4
[ -n "$WAIT" ] && until [ -f "$WAIT" ]; do sleep 30; done
one() {
  read PROT NET SEED SS TRACK <<< "$1"
  local PC='n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000'
  local RC='ax:mu=1.667 ax:mu=1.833,mass=1.1' CARS G A
  case $PROT/$TRACK in
    plastic/square) CARS=$PC; G=mocap_square2fast_b0_free; A='mocap_square2fast_b0_free mocap_figfast_b0_free';;
    plastic/fig8)   CARS=$PC; G=mocap_fig8R100_b0_free;    A='mocap_fig8R100_b0_free mocap_square2fast_b0_free';;
    plastic/fig8a45) CARS=$PC; G=mocap_fig8R100_b0_free_a45; A='mocap_fig8R100_b0_free_a45 mocap_square2fast_b0_free';;
    rubber/square)  CARS=$RC; G=mocap_square2fast_b0_free_x120; A='mocap_square2fast_b0_free_x120 mocap_fig8R100_b0_free_x120';;
    rubber/fig8)    CARS=$RC; G=mocap_fig8R100_b0_free_x120;    A='mocap_square2fast_b0_free_x120 mocap_fig8R100_b0_free_x120';;
    *) echo "bad line $1"; return;;
  esac
  local P=8; [ $PROT = rubber ] && P=2
  local D=runs/screen/PRACB_${PROT}_${TRACK}_${NET}_s${SEED}_ss${SS} X=''; [ $NET = rmaN ] && X='--z-path 1'
  if [ ! -f $D/summary.json ]; then
    F1T_PROCS=$P taskset -c $CPUS $PY -u hw_stage.py --policy runs/${NET}_s${SEED}/policy_final.npz --out $D --cars $CARS --goals $G --anchor-plans $A \
      --rollback 0.7 --tr 1 --tr-lo 0.25 --tr-hi 0.75 --tr-max 0.4 --fit-realize 0.9 --fit-max-steps 8000 --crash-rollback 1 --slope-w 1.0 \
      --iters 4 --beta 0.5 --delta 0.05 --cap 0.4 --fit-stall 0 --jac closed --objective task --err-scale 0.1 1e9 1e9 1e9 1e9 0.24 \
      --goal-cycle 1 --trial-laps 2 --anchor-w 20 --anchor-n0 6 --age-decay 0.5 --steps 800 --lr 1e-4 --eval-runs 0 \
      --eval-seed 701 --seed $SS --gpu 0 --backend sim --launch 1.0 8.0 --sim-launch 1 --skip-below 0.5 --save-its 1 $X \
      > logs/prac/$(basename $D).log 2>&1
  fi
  [ -f $D/summary.json ] && [ ! -f $D/ladder_launch.json ] && LADDER_ITS='0 1 2 3 4' LADDER_OUT=ladder_launch.json LADDER_LAPS=5 LADDER_PLANS=$G \
    LADDER_CARS="$(echo $CARS | tr ' ' '|')" F1T_PROCS=$P taskset -c $CPUS nice -n 5 $PY -u ladder_eval_launch.py $D >> logs/prac/$(basename $D).eval.log 2>&1
  [ -f $D/summary.json ] && [ ! -f $D/sb_practised.json ] && F1T_PROCS=$P LADDER_ITS='0 1 2 3 4' LADDER_OUT=sb_practised.json LADDER_LAPS=5 LADDER_PLANS=$G \
    LADDER_CARS="$(echo $CARS | tr ' ' '|')" taskset -c $CPUS nice -n 5 $PY -u ladder_eval.py $D >> logs/prac/$(basename $D).eval.log 2>&1
  echo "$(date +%H:%M) done $1" >> logs/prac/progress.txt
}
export -f one; export CPUS PY
grep -v '^#' $LIST | grep . | xargs -P $SLOTS -I{} bash -c 'one "{}"'
touch logs/prac/DONE2_$(basename $LIST)
