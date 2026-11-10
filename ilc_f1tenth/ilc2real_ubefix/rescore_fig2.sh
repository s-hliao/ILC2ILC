#!/bin/bash
# rescore_fig2.sh MACHINE CPUS PROCS (2026-11-08 ~17:00, user: "fig2 is only sim2sim transfer", practice-only protocol): every figure-2
# method's saved checkpoints scored on its panel's PRACTISED plan with the scorer figure 2 always used (ladder_eval.py: seed 701, 4 starts,
# 5 laps), output fig2_practised.json (baselines) / sb_practised.json (ours). Panels = the plans every method practised:
#   plastic square2fast_a45 (12 cars) | rubber fig8R100_x120 (2 cars) | tier square2fast nominal (36 cars). Existing outputs are kept.
cd ~/ilc_ws/src/ilc_f1tenth/$( [ "$1" = ube ] && echo ilc2real_fix || echo ilc2real ); source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu FADA_JIT=1 F1T_PROCS=$3
CPUS=$2; PY=~/miniconda3/envs/f1t/bin/python; mkdir -p logs/rescore
PC='n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000'
RC='ax:mu=1.667|ax:mu=1.833,mass=1.1'
TC=$($PY -c "import mb_car as m; print('|'.join(sorted(k for k in m.REAL_CARS if k[:5] in ('tiera','tierb','tierc'))))" 2>/dev/null)
declare -A G=([plastic]=mocap_square2fast_b0_free_a45 [rubber]=mocap_fig8R100_b0_free_x120 [tier]=mocap_square2fast_b0_free)
declare -A C=([plastic]="$PC" [rubber]="$RC" [tier]="$TC")
run() { # OUTNAME ITS PANEL DIRS...
  out=$1; its=$2; p=$3; shift 3; L=""; for d in "$@"; do [ -d $d ] && [ ! -s $d/$out ] && L="$L $d"; done; [ -z "$L" ] && return
  echo "$(date +%H:%M) $out $p:$L" >> logs/rescore/progress.txt
  LADDER_ITS="$its" LADDER_OUT=$out LADDER_LAPS=5 LADDER_PLANS=${G[$p]} LADDER_CARS="${C[$p]}" taskset -c $CPUS nice -n 5 $PY -u ladder_eval.py $L >> logs/rescore/rescore.log 2>&1
}
for p in rubber plastic tier; do for n in freePPON rmaN ours oursdr; do
  run sb_practised.json "0 1 2 3 4" $p runs/screen/SB_${p}_${n}_s?_ss????
done; done
touch logs/rescore/DONE
