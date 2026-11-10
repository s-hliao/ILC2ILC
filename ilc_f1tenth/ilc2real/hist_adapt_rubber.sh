#!/bin/bash
# hist_adapt_rubber.sh (2026-11-07 ~13:10, user: "does the adaptation architecture benefit from the history or no"): figure 2's rubber
# adaptation (fig2_fadaT.sh protocol: 384 FADA-zs episodes, LoRA per budget 8 24 96 384, seed 5151 + s) for fadaR1 (history 1), then the
# held-out scoring (ladder_eval, peanut x120, 5 laps, its 0 8 24 96 384) of fadaR (history 30; its runs from fadaR_all.sh) and fadaR1 into
# fig2_eval_hist.json (fadaR_all.sh's own fig2_eval.json is untouched). Gain = it 0 -> it K per seed.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=4 FADA_JIT=1 XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=$HOME/miniconda3/envs/f1t/bin/python
mkdir -p logs/histA
until [ -f runs/fig1/fadaR1_s0/it150.pkl ] && [ -f runs/fig1/fadaR1_s1/it150.pkl ] && [ -f runs/fig1/fadaR1_s2/it150.pkl ]; do sleep 60; done
for s in 0 1 2; do
  [ -f runs/hw/fig2fadaR1_rubber_s${s}_ep384/summary.json ] || CUDA_VISIBLE_DEVICES=$((s % 2)) $PY -u fada_real_car_tf.py --model runs/fig1/fadaR1_s$s/it150.pkl \
    --arm fig2fadaR1_rubber_s$s --cars ax:mu=1.667 ax:mu=1.833,mass=1.1 --goals mocap_fig8R100_b0_free_x120 --checkpoints 8 24 96 384 \
    --seed $((5151 + s)) > logs/histA/fada_rubber_s$s.log 2>&1 &
done
wait
until [ -f runs/hw/fig2fadaR_rubber_s0_ep384/summary.json ] && [ -f runs/hw/fig2fadaR_rubber_s1_ep384/summary.json ] && [ -f runs/hw/fig2fadaR_rubber_s2_ep384/summary.json ]; do sleep 60; done
for arm in fadaR fadaR1; do for s in 0 1 2; do
  D=runs/fig2base/hist_${arm}_rubber_s$s; mkdir -p $D/its
  for K in 0 8 24 96 384; do E=runs/hw/fig2${arm}_rubber_s${s}_ep$K; [ -d $E ] || continue
    for f in $E/*_policy.npz; do c=$(basename $f _policy.npz); ln -sf $PWD/$f $D/its/it${K}_${c}_policy.npz; done
  done
done; done
export F1T_PROCS=24 LADDER_ITS="0 8 24 96 384" JAX_PLATFORMS=cpu
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_x120 $PY -u ladder_eval.py runs/fig2base/hist_fadaR_rubber_s? runs/fig2base/hist_fadaR1_rubber_s? > logs/histA/eval.log 2>&1
touch logs/histA/DONE
