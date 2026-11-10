#!/bin/bash
# fig2_fadaT_tier02.sh (2026-11-07 04:25): fig2_fadaT.sh's tier s0 / s2, rerun after both ran GPU 0 out of memory (JAX keeps a finished
# fit's memory pool: XLA_PYTHON_CLIENT_ALLOCATOR=platform returns it). Waits for fig2_fadaT_plastic01.sh (GPU 0); then the held-out
# scoring of these two exactly as fig2_fadaT.sh's (ladder_eval, mocap_peanutM_b0_free, 5 laps, seed 701).
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
while pgrep -f "fig2_fadaT_plastic0[1].sh" >/dev/null; do sleep 60; done
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=${F1T_PROCS:-8} XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform FADA_JIT=1 XLA_FLAGS=--xla_gpu_enable_command_buffer=
PY=$HOME/miniconda3/envs/f1t/bin/python
TIER=$($PY -c "import mb_car as m; print(' '.join(sorted(k for k in m.REAL_CARS if k[:5] in ('tiera','tierb','tierc'))))")
for s in 0 2; do
  mv logs/fig2T/fada_tier_s$s.log logs/fig2T/fada_tier_s$s.oom.log 2>/dev/null
  CUDA_VISIBLE_DEVICES=0 $PY -u fada_real_car_tf.py --model runs/fig1/fadaT_s$s/it8.pkl --arm fig2fadaT_tier_s$s --cars $TIER \
    --goals mocap_square2fast_b0_free --checkpoints 8 24 96 384 --seed $((5151 + s)) > logs/fig2T/fada_tier_s$s.log 2>&1 &
done
wait
for s in 0 2; do D=runs/fig2base/fig2fadaT_tier_s$s; mkdir -p $D/its
  for K in 0 8 24 96 384; do E=runs/hw/fig2fadaT_tier_s${s}_ep$K; [ -d $E ] || continue
    for f in $E/*_policy.npz; do c=$(basename $f _policy.npz); ln -sf $PWD/$f $D/its/it${K}_${c}_policy.npz; done; done; done
export F1T_PROCS=16 LADDER_ITS="0 8 24 96 384" JAX_PLATFORMS=cpu
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free LADDER_CARS="$(echo $TIER | tr ' ' '|')" $PY -u ladder_eval.py runs/fig2base/fig2fadaT_tier_s0 runs/fig2base/fig2fadaT_tier_s2 > logs/fig2T/eval_tier02.log 2>&1
touch logs/fig2T/EVAL_TIER02_DONE
