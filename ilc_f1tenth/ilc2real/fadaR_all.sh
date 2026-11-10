#!/bin/bash
# fadaR_all.sh (2026-11-07 ~12:10, Lambda; user: "only this version in the results, it is fairer"): FADA trained on EXACTLY RMA's
# adaptation-phase distillation data (fada_car_tf.py --data rma: on-policy student rollouts only, it 0 the teacher, 150 iterations x 512
# lanes x 160 periods, every 4th period, buffer of the last 8 iterations, labels = the teacher's action at the visited states; no
# suboptimal rollouts, no oracle shadows; 200 Adam steps / iteration, batch 512) with FADA's planner + IDM (leak-fixed o_t, H 30, K 6),
# teacher rma_s<s>, seeds 0-2. Then figure 1 zero-shot (fig1_eval.py, saved iterates) and figure 2 adaptation (fig2_fadaT.sh's protocol:
# one round of 384 FADA-zs episodes, LoRA per budget 8 24 96 384, held-out scoring) on rubber / plastic / tier.
# Outputs: runs/fig1/fadaR_s<s>, runs/fig1/fig1_eval_fadaR.json, runs/hw/fig2fadaR_<panel>_s<s>_ep<K>, runs/fig2base/fig2fadaR_<panel>_s<s>.
# CPU stages wait for the DAgger diagnostic's scoring (logs/dagger/DONE) so Lambda stays <= 46 cores.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=$HOME/miniconda3/envs/f1t/bin/python
SAVE="1 2 5 10 20 40 75 150"
mkdir -p logs/fadaR
for s in 0 1 2; do
  [ -f runs/fig1/fadaR_s$s/it150.pkl ] || CUDA_VISIBLE_DEVICES=$((s % 2)) $PY -u fada_car_tf.py --data rma --iters 150 --steps 200 \
     --save-at $SAVE --oracle runs/fig1/rma_s$s --out runs/fig1/fadaR_s$s --seed $s > logs/fadaR/s$s.log 2>&1 &
done
wait
touch logs/fadaR/TRAIN_DONE
until [ -f logs/dagger/DONE ]; do sleep 60; done
# ---- figure 1 zero-shot
L=""
for s in 0 1 2; do for k in $SAVE; do
  f=runs/fig1/fadaR_s$s/it${k}_policy.npz
  $PY -c "import numpy as np, os; np.savez('$f', mode='fada_tf', pkl=os.path.abspath('runs/fig1/fadaR_s$s/it$k.pkl'))"; L="$L $f"
done; done
F1T_SPIN_RHO=3 F1T_PROCS=20 FADA_JIT=1 nice -n 5 $PY -u fig1_eval.py runs/fig1/fig1_eval_fadaR.json $L > logs/fadaR/fig1_eval.log 2>&1 &
# ---- figure 2 adaptation
export F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=3 FADA_JIT=1 XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_ALLOCATOR=platform
TIER=$($PY -c "import mb_car as m; print(' '.join(sorted(k for k in m.REAL_CARS if k[:5] in ('tiera','tierb','tierc'))))")
declare -A CARS=([tier]="$TIER" [plastic]="n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000" [rubber]="ax:mu=1.667 ax:mu=1.833,mass=1.1")
declare -A GOAL=([tier]=mocap_square2fast_b0_free [plastic]=mocap_square2fast_b0_free_a45 [rubber]=mocap_fig8R100_b0_free_x120)
mkdir -p logs/fig2R
fada() { [ -f runs/hw/fig2fadaR_$1_s$2_ep384/summary.json ] && return
         CUDA_VISIBLE_DEVICES=$3 $PY -u fada_real_car_tf.py --model runs/fig1/fadaR_s$2/it150.pkl --arm fig2fadaR_$1_s$2 --cars ${CARS[$1]} \
          --goals ${GOAL[$1]} --checkpoints 8 24 96 384 --seed $((5151 + $2)) > logs/fig2R/fada_$1_s$2.log 2>&1; echo "exit $?" >> logs/fig2R/fada_$1_s$2.log; }
for s in 0 1 2; do fada rubber $s $((s % 2)) & done; for s in 0 1 2; do fada plastic $s $(( (s + 1) % 2 )) & done; wait
for s in 0 1 2; do fada tier $s $((s % 2)) & done; wait
wait
for p in tier plastic rubber; do for s in 0 1 2; do
  D=runs/fig2base/fig2fadaR_${p}_s$s; mkdir -p $D/its
  for K in 0 8 24 96 384; do E=runs/hw/fig2fadaR_${p}_s${s}_ep$K; [ -d $E ] || continue
    for f in $E/*_policy.npz; do c=$(basename $f _policy.npz); ln -sf $PWD/$f $D/its/it${K}_${c}_policy.npz; done
  done
done; done
TC=$(echo $TIER | tr ' ' '|'); PC="n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000"
export F1T_PROCS=40 LADDER_ITS="0 8 24 96 384" JAX_PLATFORMS=cpu
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_x120 $PY -u ladder_eval.py runs/fig2base/fig2fadaR_rubber_s? > logs/fig2R/eval.log 2>&1
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_a33 LADDER_CARS="$PC" $PY -u ladder_eval.py runs/fig2base/fig2fadaR_plastic_s? >> logs/fig2R/eval.log 2>&1
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free LADDER_CARS="$TC" $PY -u ladder_eval.py runs/fig2base/fig2fadaR_tier_s? >> logs/fig2R/eval.log 2>&1
touch logs/fadaR/DONE
