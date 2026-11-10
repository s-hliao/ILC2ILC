#!/bin/bash
# fadaR1_fig2.sh (2026-11-07 ~14:45, Lambda): figure 2 for THE FADA in the results (fadaR1 = our regime: RMA's data, memoryless; user
# decisions 14:05 / daggerR test 14:40): fig2_fadaT.sh's protocol (384 FADA-zs episodes, LoRA per budget 8 24 96 384, seed 5151 + s) on
# plastic + tier (rubber: hist_adapt_rubber.sh's runs/hw/fig2fadaR1_rubber_s<s>), then held-out scoring of all three panels.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=4 FADA_JIT=1 XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=$HOME/miniconda3/envs/f1t/bin/python
TIER=$($PY -c "import mb_car as m; print(' '.join(sorted(k for k in m.REAL_CARS if k[:5] in ('tiera','tierb','tierc'))))")
declare -A CARS=([tier]="$TIER" [plastic]="n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000")
declare -A GOAL=([tier]=mocap_square2fast_b0_free [plastic]=mocap_square2fast_b0_free_a45)
mkdir -p logs/fig2R1
fada() { [ -f runs/hw/fig2fadaR1_$1_s$2_ep384/summary.json ] && return
         CUDA_VISIBLE_DEVICES=$3 $PY -u fada_real_car_tf.py --model runs/fig1/fadaR1_s$2/it150.pkl --arm fig2fadaR1_$1_s$2 --cars ${CARS[$1]} \
          --goals ${GOAL[$1]} --checkpoints 8 24 96 384 --seed $((5151 + $2)) > logs/fig2R1/fada_$1_s$2.log 2>&1; echo "exit $?" >> logs/fig2R1/fada_$1_s$2.log; }
for s in 0 1 2; do fada plastic $s $((s % 2)) & done; for s in 0 1 2; do fada tier $s $(( (s + 1) % 2 )) & done; wait
for p in tier plastic rubber; do for s in 0 1 2; do
  D=runs/fig2base/fig2fadaR1_${p}_s$s; mkdir -p $D/its
  for K in 0 8 24 96 384; do E=runs/hw/fig2fadaR1_${p}_s${s}_ep$K; [ -d $E ] || continue
    for f in $E/*_policy.npz; do c=$(basename $f _policy.npz); ln -sf $PWD/$f $D/its/it${K}_${c}_policy.npz; done
  done
done; done
TC=$(echo $TIER | tr ' ' '|'); PC="n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000"
export F1T_PROCS=40 LADDER_ITS="0 8 24 96 384" JAX_PLATFORMS=cpu
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_x120 $PY -u ladder_eval.py runs/fig2base/fig2fadaR1_rubber_s? > logs/fig2R1/eval.log 2>&1
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_a33 LADDER_CARS="$PC" $PY -u ladder_eval.py runs/fig2base/fig2fadaR1_plastic_s? >> logs/fig2R1/eval.log 2>&1
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free LADDER_CARS="$TC" $PY -u ladder_eval.py runs/fig2base/fig2fadaR1_tier_s? >> logs/fig2R1/eval.log 2>&1
touch logs/fig2R1/DONE
