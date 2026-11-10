#!/bin/bash
# fig2_fadaT.sh (2026-11-07, runs/screen/FADA_AUDIT_1107.md FULL FIDELITY): figure 2's FADA + adaptation, the paper's protocol: the source
# FADA = fada_car_tf.py's final iterate runs/fig1/fadaT_s<s>/it8.pkl (transformer planner + IDM, H 30, K 6); one round of FADA-zs target
# rollouts (384 episodes per car, fig2_fadaP.sh's episodes: 1.05 laps of the panel's practised plan) and ONE offline LoRA fit per budget
# (8 24 96 384 episodes; r 8, alpha 16, dropout 0.05, 800 steps, lr 3e-4, batch 4096) -- fada_real_car_tf.py. LoRA fits on matcha's GPUs,
# rollouts on F1T_PROCS CPU workers per job. Runs on Lambda (2 x RTX 4090: GPU index s % 2). Then the held-out scoring exactly as fig2_fadaP.sh (ladder_eval, 5 laps, seed 701).
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=${F1T_PROCS:-4} XLA_PYTHON_CLIENT_PREALLOCATE=false FADA_JIT=1 XLA_FLAGS=--xla_gpu_enable_command_buffer=
PY=$HOME/miniconda3/envs/f1t/bin/python
TIER=$($PY -c "import mb_car as m; print(' '.join(sorted(k for k in m.REAL_CARS if k[:5] in ('tiera','tierb','tierc'))))")
declare -A CARS=([tier]="$TIER" [plastic]="n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000" [rubber]="ax:mu=1.667 ax:mu=1.833,mass=1.1")
declare -A GOAL=([tier]=mocap_square2fast_b0_free [plastic]=mocap_square2fast_b0_free_a45 [rubber]=mocap_fig8R100_b0_free_x120)
mkdir -p logs/fig2T
fada() { until [ -f runs/fig1/fadaT_s$2/it8.pkl ]; do sleep 60; done
         [ -f runs/hw/fig2fadaT_$1_s$2_ep384/summary.json ] && return
         CUDA_VISIBLE_DEVICES=$3 $PY -u fada_real_car_tf.py --model runs/fig1/fadaT_s$2/it8.pkl --arm fig2fadaT_$1_s$2 --cars ${CARS[$1]} \
          --goals ${GOAL[$1]} --checkpoints 8 24 96 384 --seed $((5151 + $2)) > logs/fig2T/fada_$1_s$2.log 2>&1; echo "exit $?" >> logs/fig2T/fada_$1_s$2.log; }
for s in 0 1 2; do fada rubber $s $((s % 2)) & done; for s in 0 1 2; do fada plastic $s $(( (s + 1) % 2 )) & done; wait
for s in 0 1 2; do fada tier $s $((s % 2)) & done; wait
touch logs/fig2T/RUNS_DONE
for p in tier plastic rubber; do for s in 0 1 2; do
  D=runs/fig2base/fig2fadaT_${p}_s$s; mkdir -p $D/its
  for K in 0 8 24 96 384; do E=runs/hw/fig2fadaT_${p}_s${s}_ep$K; [ -d $E ] || continue
    for f in $E/*_policy.npz; do c=$(basename $f _policy.npz); ln -sf $PWD/$f $D/its/it${K}_${c}_policy.npz; done
  done
done; done
TC=$(echo $TIER | tr ' ' '|'); PC="n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000"
export F1T_SPIN_RHO=3 F1T_PROCS=12 LADDER_ITS="0 8 24 96 384" JAX_PLATFORMS=cpu
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_x120 $PY -u ladder_eval.py runs/fig2base/fig2fadaT_rubber_s? > logs/fig2T/eval.log 2>&1
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_a33 LADDER_CARS="$PC" $PY -u ladder_eval.py runs/fig2base/fig2fadaT_plastic_s? >> logs/fig2T/eval.log 2>&1
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free LADDER_CARS="$TC" $PY -u ladder_eval.py runs/fig2base/fig2fadaT_tier_s? >> logs/fig2T/eval.log 2>&1
touch logs/fig2T/EVAL_DONE
