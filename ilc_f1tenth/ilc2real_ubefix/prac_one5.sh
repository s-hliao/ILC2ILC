#!/bin/bash
# prac_one5.sh = prac_one4.sh + plastic split-friction oval: turn 1 (lap 0.056-0.556) grippier x1.67 (oval34grip) or slicker x0.6 (oval34slick), 2026-11-08 ~16:30.
# prac_one4.sh = prac_one3.sh + rubber squarenom (the plastic nominal square plan + anchors), 2026-11-08 ~16:15.
# prac_one3.sh = prac_one2.sh + rubber square45 and the split-friction oval (turn 1 = lap 0.056-0.556 at friction x0.36), 2026-11-08 ~16:10.
# prac_one2.sh = prac_one.sh + the oval cells (2026-11-08 ~08:25, OVAL_RULES_1108.md); a new file so running prac_one.sh jobs are untouched.
# prac_one.sh "PROT NET SEED SS TRACK" CPUSET (2026-11-08 ~03:25, from matcha; user: "do the best you can ... schedule as best you can"): ONE SB stage
# run of the practice-only protocol, exactly prac_queue5.sh's flags (SB, 4 iterations, --sim-launch 1 --skip-below 0.5), then both scorers
# (ladder_eval_launch.py -> ladder_launch.json, ladder_eval.py -> sb_practised.json; its 0-4, seed 701, 4 starts). Every cell of tonight's plan:
#   plastic square (nominal) | square45 (a45) | fig8 (R100 nominal) | fig8r115 (R115 x090, the sweep rule) | pnutdeep (pnut100w100 x085) |
#   ethz (ETHZMobil, speed PETHZ below) ; rubber square (x120) | square33 (a33) | fig8 (R100 x120) | fig8r115 (R115 a30 = hardware, user) |
#   pnutdeep (x120) | ethz (x110) ; tier square. Anchors for the new cells = the practised plan + the same track's nominal plan.
#   ETHZMobil: 1-lap trials and 1-lap scoring (ETHZ_RULES.md); others 2-lap trials, 5-lap scoring. Out runs/screen/PRACB_<PROT>_<TRACK>_<NET>_s<SEED>_ss<SS>.
cd "$(dirname "$(readlink -f "$0")")"; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
PY=~/miniconda3/envs/f1t/bin/python; mkdir -p logs/prac
read PROT NET SEED SS TRACK <<< "$1"; CPUS=$2
PETHZ=$(cat logs/prac/PETHZ 2>/dev/null)          # plastic ETHZ plan, set once its sweep picks a speed
PC='n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000'
RC='ax:mu=1.667 ax:mu=1.833,mass=1.1'; TL=2; EL=5
case $PROT/$TRACK in
  plastic/square)   CARS=$PC; G=mocap_square2fast_b0_free;      A='mocap_square2fast_b0_free mocap_figfast_b0_free';;
  plastic/square45) CARS=$PC; G=mocap_square2fast_b0_free_a45;  A='mocap_square2fast_b0_free_a45 mocap_square2fast_b0_free';;
  plastic/fig8)     CARS=$PC; G=mocap_fig8R100_b0_free;         A='mocap_fig8R100_b0_free mocap_square2fast_b0_free';;
  plastic/fig8r115) CARS=$PC; G=mocap_fig8R115_b0_free_x090;    A='mocap_fig8R115_b0_free_x090 mocap_fig8R115_b0_free';;
  plastic/pnutdeep) CARS=$PC; G=mocap_pnut100w100_b0_free_x085; A='mocap_pnut100w100_b0_free_x085 mocap_pnut100w100_b0_free';;
  plastic/oval30)   CARS=$PC; G=mocap_oval165_b0_free_a30;     A="mocap_oval165_b0_free_a30 mocap_oval165_b0_free";;
  plastic/oval34)   CARS=$PC; G=mocap_oval165_b0_free_a34;     A="mocap_oval165_b0_free_a34 mocap_oval165_b0_free";;
  plastic/oval26)   CARS=$PC; G=mocap_oval165_b0_free_a26;     A="mocap_oval165_b0_free_a26 mocap_oval165_b0_free";;
  plastic/oval28)   CARS=$PC; G=mocap_oval165_b0_free_a28;     A="mocap_oval165_b0_free_a28 mocap_oval165_b0_free";;
  plastic/oval40)   CARS=$PC; G=mocap_oval165_b0_free_a40;     A="mocap_oval165_b0_free_a40 mocap_oval165_b0_free";;
  plastic/oval34grip)  CARS=$PC; G=mocap_oval165_b0_free_a34; A="mocap_oval165_b0_free_a34 mocap_oval165_b0_free"; export F1T_MU_PATCH=0.056,0.556,1.67;;
  plastic/oval34slick) CARS=$PC; G=mocap_oval165_b0_free_a34; A="mocap_oval165_b0_free_a34 mocap_oval165_b0_free"; export F1T_MU_PATCH=0.056,0.556,0.6;;
  rubber/squarenom) CARS=$RC; G=mocap_square2fast_b0_free;      A="mocap_square2fast_b0_free mocap_figfast_b0_free";;
  rubber/square45)  CARS=$RC; G=mocap_square2fast_b0_free_a45;  A="mocap_square2fast_b0_free_a45 mocap_square2fast_b0_free";;
  rubber/oval34split) CARS=$RC; G=mocap_oval165_b0_free_a34; A="mocap_oval165_b0_free_a34 mocap_oval165_b0_free"; export F1T_MU_PATCH=0.056,0.556,0.36;;
  rubber/oval30split) CARS=$RC; G=mocap_oval165_b0_free_a30; A="mocap_oval165_b0_free_a30 mocap_oval165_b0_free"; export F1T_MU_PATCH=0.056,0.556,0.36;;
  rubber/oval30)    CARS=$RC; G=mocap_oval165_b0_free_a30;     A="mocap_oval165_b0_free_a30 mocap_oval165_b0_free";;
  rubber/oval34)    CARS=$RC; G=mocap_oval165_b0_free_a34;     A="mocap_oval165_b0_free_a34 mocap_oval165_b0_free";;
  plastic/ethz)     [ -n "$PETHZ" ] || { echo "PETHZ not set"; exit 3; }; CARS=$PC; G=$PETHZ; A="$PETHZ ethzmobil43_b0_free"; TL=1; EL=1;;
  rubber/square)    CARS=$RC; G=mocap_square2fast_b0_free_x120; A='mocap_square2fast_b0_free_x120 mocap_fig8R100_b0_free_x120';;
  rubber/square33)  CARS=$RC; G=mocap_square2fast_b0_free_a33;  A='mocap_square2fast_b0_free_a33 mocap_square2fast_b0_free';;
  rubber/fig8)      CARS=$RC; G=mocap_fig8R100_b0_free_x120;    A='mocap_square2fast_b0_free_x120 mocap_fig8R100_b0_free_x120';;
  rubber/fig8r115)  CARS=$RC; G=mocap_fig8R115_b0_free_a30;     A='mocap_fig8R115_b0_free_a30 mocap_fig8R115_b0_free';;
  rubber/pnutdeep)  CARS=$RC; G=mocap_pnut100w100_b0_free_x120; A='mocap_pnut100w100_b0_free_x120 mocap_pnut100w100_b0_free';;
  rubber/ethz)      CARS=$RC; G=ethzmobil43_b0_free_x110;       A='ethzmobil43_b0_free_x110 ethzmobil43_b0_free'; TL=1; EL=1;;
  tier/square)      CARS=$($PY -c "import mb_car as m; print(' '.join(sorted(k for k in m.REAL_CARS if k[:5] in ('tiera','tierb','tierc'))))" 2>/dev/null)
                    G=mocap_square2fast_b0_free; A='mocap_square2fast_b0_free mocap_figfast_b0_free';;
  *) echo "bad item $1"; exit 2;;
esac
P=8; [ $PROT = rubber ] && P=2
POL=runs/${NET}_s${SEED}/policy_final.npz; case $NET in ours|oursdr) POL=runs/fig1/${NET}_s${SEED}/policy_final.npz;; esac
D=runs/screen/PRACB_${PROT}_${TRACK}_${NET}_s${SEED}_ss${SS} X=''; [ $NET = rmaN ] && X='--z-path 1'
if [ ! -f $D/summary.json ]; then
  F1T_PROCS=$P taskset -c $CPUS $PY -u hw_stage.py --policy $POL --out $D --cars $CARS --goals $G --anchor-plans $A \
    --rollback 0.7 --tr 1 --tr-lo 0.25 --tr-hi 0.75 --tr-max 0.4 --fit-realize 0.9 --fit-max-steps 8000 --crash-rollback 1 --slope-w 1.0 \
    --iters 4 --beta 0.5 --delta 0.05 --cap 0.4 --fit-stall 0 --jac closed --objective task --err-scale 0.1 1e9 1e9 1e9 1e9 0.24 \
    --goal-cycle 1 --trial-laps $TL --anchor-w 20 --anchor-n0 6 --age-decay 0.5 --steps 800 --lr 1e-4 --eval-runs 0 \
    --eval-seed 701 --seed $SS --gpu 0 --backend sim --launch 1.0 8.0 --sim-launch 1 --skip-below 0.5 --save-its 1 $X > logs/prac/$(basename $D).log 2>&1
fi
CB="$(echo $CARS | tr ' ' '|')"
[ -f $D/summary.json ] && [ ! -f $D/ladder_launch.json ] && LADDER_ITS='0 1 2 3 4' LADDER_OUT=ladder_launch.json LADDER_LAPS=$EL LADDER_PLANS=$G \
  LADDER_CARS="$CB" F1T_PROCS=$P taskset -c $CPUS nice -n 5 $PY -u ladder_eval_launch.py $D >> logs/prac/$(basename $D).eval.log 2>&1
[ -f $D/summary.json ] && [ ! -f $D/sb_practised.json ] && LADDER_ITS='0 1 2 3 4' LADDER_OUT=sb_practised.json LADDER_LAPS=$EL LADDER_PLANS=$G \
  LADDER_CARS="$CB" F1T_PROCS=$P taskset -c $CPUS nice -n 5 $PY -u ladder_eval.py $D >> logs/prac/$(basename $D).eval.log 2>&1
echo "$(date +%H:%M) done $1" >> logs/prac/progress.txt
