#!/bin/bash
# fadaL_all.sh (2026-11-07 ~11:05, LEAK FIX, runs/screen/FADA_AUDIT_1107.md): the full-fidelity FADA (fadaT) had o_t = [obs[:13], a_{t-1}],
# so the IDM's future chunk Y = o_{t+1..t+K} carried a_t in o_{t+1}'s previous-action slot -- the IDM's own target. Measured on target cars
# (n04_harsh_steer / n04_real_nom, scratchpad fada_idm_diag.py): base-IDM action MSE 2.9e-5 / 3.0e-5 vs 7.6e-3 / 8.1e-3 for copying a_{t-1};
# the action moved 0.0025 when the whole action history was zeroed, 0.17 for +1 std in Y. The IDM copied a_t out of Y (no dynamics), so
# the LoRA adaptation had nothing to correct (fig2 gains ~0) and the planner, predicting that slot, was itself the policy (a DAgger student of
# the privileged oracle). Fix: o_t = the state obs[:13] only; the executed actions enter only as the IDM's action tokens A^H (the paper's
# I(O^H, A^H, Y)). Everything else identical to fadaT_train_all.sh / fig1 scoring / fig2_fadaT.sh: oracle rma_s<s>, seeds 0-2, DAgger 8,
# LoRA r 8 / alpha 16 / 800 steps, budgets 8 24 96 384, the same held-out evaluation. Outputs: runs/fig1/fadaL_s<s>,
# runs/fig1/fig1_eval_fadaL.json, runs/hw/fig2fadaL_<panel>_s<s>_ep<K>, runs/fig2base/fig2fadaL_<panel>_s<s>/fig2_eval.json. (Tier panel: separate.)
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=$HOME/miniconda3/envs/f1t/bin/python
mkdir -p logs/fadaL
for s in 0 1 2; do
  [ -f runs/fig1/fadaL_s$s/it8.pkl ] && continue
  CUDA_VISIBLE_DEVICES=$((s + 1)) taskset -c $((20 + 2 * s))-$((21 + 2 * s)) $PY -u fada_car_tf.py \
    --oracle runs/fig1/rma_s$s --out runs/fig1/fadaL_s$s --seed $s > logs/fadaL/s$s.log 2>&1 &
done
wait
touch logs/fadaL/TRAIN_DONE
# ---- figure 1 (zero-shot) scoring of iterates 1 2 4 8, exactly as fadaT's
L=""
for s in 0 1 2; do for k in 1 2 4 8; do
  f=runs/fig1/fadaL_s$s/it${k}_policy.npz
  $PY -c "import numpy as np, os; np.savez('$f', mode='fada_tf', pkl=os.path.abspath('runs/fig1/fadaL_s$s/it$k.pkl'))"; L="$L $f"
done; done
F1T_SPIN_RHO=3 F1T_PROCS=6 FADA_JIT=1 nice -n 5 $PY -u fig1_eval.py runs/fig1/fig1_eval_fadaL.json $L > logs/fadaL/fig1_eval.log 2>&1 &
# ---- figure 2 adaptation (fig2_fadaT.sh's protocol), plastic + rubber, 3 jobs at a time, 3 rollout workers each (matcha load <= 34)
export F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=3 FADA_JIT=1 XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_ALLOCATOR=platform
declare -A CARS=([plastic]="n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000" [rubber]="ax:mu=1.667 ax:mu=1.833,mass=1.1")
declare -A GOAL=([plastic]=mocap_square2fast_b0_free_a45 [rubber]=mocap_fig8R100_b0_free_x120)
mkdir -p logs/fig2L
fada() { [ -f runs/hw/fig2fadaL_$1_s$2_ep384/summary.json ] && return
         CUDA_VISIBLE_DEVICES=$3 $PY -u fada_real_car_tf.py --model runs/fig1/fadaL_s$2/it8.pkl --arm fig2fadaL_$1_s$2 --cars ${CARS[$1]} \
          --goals ${GOAL[$1]} --checkpoints 8 24 96 384 --seed $((5151 + $2)) > logs/fig2L/fada_$1_s$2.log 2>&1; echo "exit $?" >> logs/fig2L/fada_$1_s$2.log; }
for s in 0 1 2; do fada rubber $s $((s + 1)) & done; wait
for s in 0 1 2; do fada plastic $s $((s + 1)) & done; wait
wait
for p in plastic rubber; do for s in 0 1 2; do
  D=runs/fig2base/fig2fadaL_${p}_s$s; mkdir -p $D/its
  for K in 0 8 24 96 384; do E=runs/hw/fig2fadaL_${p}_s${s}_ep$K; [ -d $E ] || continue
    for f in $E/*_policy.npz; do c=$(basename $f _policy.npz); ln -sf $PWD/$f $D/its/it${K}_${c}_policy.npz; done
  done
done; done
PC="n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000"
export F1T_PROCS=12 LADDER_ITS="0 8 24 96 384" JAX_PLATFORMS=cpu
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_x120 $PY -u ladder_eval.py runs/fig2base/fig2fadaL_rubber_s? > logs/fig2L/eval.log 2>&1
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_a33 LADDER_CARS="$PC" $PY -u ladder_eval.py runs/fig2base/fig2fadaL_plastic_s? >> logs/fig2L/eval.log 2>&1
touch logs/fadaL/DONE
