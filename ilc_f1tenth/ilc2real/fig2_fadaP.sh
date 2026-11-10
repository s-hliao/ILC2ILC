#!/bin/bash
# fig2_fadaP.sh (2026-11-07, runs/screen/FADA_AUDIT_1107.md RERUN RULES): figure 2's FADA + adaptation with FADA's learned planner
# (fada_real_car2.py --planner figure 1's fada_d60_s<s>/planner.npz, frozen); otherwise fig2_base_ube.sh's fada() exactly. matcha: LoRA
# fits on the GPUs, rollouts on <= 6 x 3 CPU processes. Then the held-out scoring (fig2_base_eval_ube.sh's ladder_eval lines).
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=3 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=$HOME/miniconda3/envs/f1t/bin/python
TIER=$($PY -c "import mb_car as m; print(' '.join(sorted(k for k in m.REAL_CARS if k[:5] in ('tiera','tierb','tierc'))))")
declare -A CARS=([tier]="$TIER" [plastic]="n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000" [rubber]="ax:mu=1.667 ax:mu=1.833,mass=1.1")
declare -A GOAL=([tier]=mocap_square2fast_b0_free [plastic]=mocap_square2fast_b0_free_a45 [rubber]=mocap_fig8R100_b0_free_x120)
mkdir -p logs/fig2P
fada() { until [ -f runs/fig1/fada_d60_s$2/planner.npz ]; do sleep 60; done
         CUDA_VISIBLE_DEVICES=$3 $PY -u fada_real_car2.py --idm runs/fig1/fada_d60_s$2/idm.npz --planner runs/fig1/fada_d60_s$2/planner.npz \
          --arm fig2fadaP_$1_s$2 --cars ${CARS[$1]} --goals ${GOAL[$1]} --eval-plans ${GOAL[$1]} --horizon 4 --lam 0.8 --per-iter 8 \
          --checkpoints 8 24 96 384 --seed $((5151 + $2)) > logs/fig2P/fada_$1_s$2.log 2>&1; }
for s in 0 1 2; do fada rubber $s $((s + 1)) & done; for s in 0 1 2; do fada plastic $s $(( (s + 1) % 4 )) & done; wait
for s in 0 1 2; do fada tier $s $((s + 1)) & done; wait
touch logs/fig2P/RUNS_DONE
for p in tier plastic rubber; do for s in 0 1 2; do              # fig2_base_eval_ube.sh's layout: runs/fig2base/<arm>_<panel>_s<s>/its
  D=runs/fig2base/fig2fadaP_${p}_s$s; mkdir -p $D/its
  for K in 0 8 24 96 384; do E=runs/hw/fig2fadaP_${p}_s${s}_ep$K; [ -d $E ] || continue
    for f in $E/*_policy.npz; do c=$(basename $f _policy.npz); ln -sf $PWD/$f $D/its/it${K}_${c}_policy.npz; done
  done
done; done
TC=$(echo $TIER | tr ' ' '|'); PC="n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000"
export F1T_SPIN_RHO=3 F1T_PROCS=16 LADDER_ITS="0 8 24 96 384" JAX_PLATFORMS=cpu
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free LADDER_CARS="$TC" $PY -u ladder_eval.py runs/fig2base/fig2fadaP_tier_s? > logs/fig2P/eval.log 2>&1
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_a33 LADDER_CARS="$PC" $PY -u ladder_eval.py runs/fig2base/fig2fadaP_plastic_s? >> logs/fig2P/eval.log 2>&1
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_x120 $PY -u ladder_eval.py runs/fig2base/fig2fadaP_rubber_s? >> logs/fig2P/eval.log 2>&1
touch logs/fig2P/EVAL_DONE
