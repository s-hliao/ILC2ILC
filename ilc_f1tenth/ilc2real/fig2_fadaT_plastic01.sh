#!/bin/bash
# fig2_fadaT_plastic01.sh (2026-11-07 02:50): fig2_fadaT.sh's plastic s0 / s1, rerun after both ran GPU 1 out of memory (3 concurrent
# batch-4096 fits); fada_real_car_tf.py now serializes fits per GPU (flock). Everything else as fig2_fadaT.sh's fada(); both on GPU 0.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=${F1T_PROCS:-6} XLA_PYTHON_CLIENT_PREALLOCATE=false FADA_JIT=1 XLA_FLAGS=--xla_gpu_enable_command_buffer=
PY=$HOME/miniconda3/envs/f1t/bin/python
PC="n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000"
for s in 0 1; do
  mv logs/fig2T/fada_plastic_s$s.log logs/fig2T/fada_plastic_s$s.oom.log 2>/dev/null
  CUDA_VISIBLE_DEVICES=0 $PY -u fada_real_car_tf.py --model runs/fig1/fadaT_s$s/it8.pkl --arm fig2fadaT_plastic_s$s --cars $PC \
    --goals mocap_square2fast_b0_free_a45 --checkpoints 8 24 96 384 --seed $((5151 + s)) > logs/fig2T/fada_plastic_s$s.log 2>&1 &
done
wait
echo "exit $?" >> logs/fig2T/fada_plastic_s1.log
