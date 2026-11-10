#!/bin/bash
# fadaN_all.sh (2026-11-07 ~18:00, Lambda; user: "yes, no-teacher fada is fine"): the FADA for the adaptation figure (fig 2) with NO
# privileged teacher, distilled from the SAME base networks our stage starts from, so FADA and ours start equal and only the adaptation differs.
#   fadaNP_s<s>: labels = our PPO+DR network runs/freePPON_s<s> (obs only)            -> compared against PPO+DR + our stage
#   fadaNR_s<s>: labels = our deployed RMA runs/rmaN_s<s> (actor + adaptation module on the trajectory's own 40-step [obs[:8], a_prev]
#                history, newest last, zero-padded: real_car.py's loop)               -> compared against RMA + our stage
# Everything else is fadaR1's (fadaR1_run.sh): fada_car_tf.py --data rma --hist 1 (RMA phase-2 data: 150 its x 512 lanes x 160 periods,
# on-policy, it 0 the teacher flies, buffer of 8, every 4th period, 200 Adam steps / it), planner + IDM leak-fixed. --oracle rma_s<s> only
# supplies the plans / args (same 6 plans as freePPON / rmaN). Then fig1 zero-shot check (expect ~ its base) and fig2 adaptation
# (fadaR_all.sh protocol: one round of 384 FADA-zs episodes, LoRA per budget 8 24 96 384, held-out scoring) on rubber / plastic / tier.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
PY=$HOME/miniconda3/envs/f1t/bin/python
mkdir -p logs/fadaN
declare -A TK=([NP]=ppodr [NR]=rma) TN=([NP]=freePPON [NR]=rmaN)
for v in NP NR; do for s in 0 1 2; do
  [ -f runs/fig1/fada${v}_s$s/it150.pkl ] && continue
  CUDA_VISIBLE_DEVICES=$((s % 2)) $PY -u fada_car_tf.py --data rma --hist 1 --iters 150 --steps 200 --save-at 40 75 150 \
     --oracle runs/fig1/rma_s$s --teacher-kind ${TK[$v]} --teacher-net runs/${TN[$v]}_s$s/policy_final.npz \
     --out runs/fig1/fada${v}_s$s --seed $s > logs/fadaN/${v}_s$s.log 2>&1 &
done; done
wait
touch logs/fadaN/TRAIN_DONE
# ---- figure 1 zero-shot check (it 150 first)
L=""; for v in NP NR; do for s in 0 1 2; do for k in 150 40 75; do f=runs/fig1/fada${v}_s$s/it${k}_policy.npz
  $PY -c "import numpy as np, os; np.savez('$f', mode='fada_tf', pkl=os.path.abspath('runs/fig1/fada${v}_s$s/it$k.pkl'))"; L="$L $f"; done; done; done
F1T_SPIN_RHO=3 F1T_PROCS=16 FADA_JIT=1 nice -n 5 $PY -u fig1_eval.py runs/fig1/fig1_eval_fadaN.json $L > logs/fadaN/fig1_eval.log 2>&1 &
# ---- figure 2 adaptation
export F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=4 FADA_JIT=1 XLA_FLAGS=--xla_gpu_enable_command_buffer=
TIER=$($PY -c "import mb_car as m; print(' '.join(sorted(k for k in m.REAL_CARS if k[:5] in ('tiera','tierb','tierc'))))")
declare -A CARS=([tier]="$TIER" [plastic]="n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000" [rubber]="ax:mu=1.667 ax:mu=1.833,mass=1.1")
declare -A GOAL=([tier]=mocap_square2fast_b0_free [plastic]=mocap_square2fast_b0_free_a45 [rubber]=mocap_fig8R100_b0_free_x120)
fada() { [ -f runs/hw/fig2fada$1_$2_s$3_ep384/summary.json ] && return
         CUDA_VISIBLE_DEVICES=$4 $PY -u fada_real_car_tf.py --model runs/fig1/fada$1_s$3/it150.pkl --arm fig2fada$1_$2_s$3 --cars ${CARS[$2]} \
          --goals ${GOAL[$2]} --checkpoints 8 24 96 384 --seed $((5151 + $3)) > logs/fadaN/fig2_$1_$2_s$3.log 2>&1; echo "exit $?" >> logs/fadaN/fig2_$1_$2_s$3.log; }
for p in rubber plastic tier; do
  for v in NP NR; do for s in 0 1 2; do fada $v $p $s $((s % 2)) & done; done; wait
done
wait
for v in NP NR; do for p in tier plastic rubber; do for s in 0 1 2; do
  D=runs/fig2base/fig2fada${v}_${p}_s$s; mkdir -p $D/its
  for K in 0 8 24 96 384; do E=runs/hw/fig2fada${v}_${p}_s${s}_ep$K; [ -d $E ] || continue
    for f in $E/*_policy.npz; do c=$(basename $f _policy.npz); ln -sf $PWD/$f $D/its/it${K}_${c}_policy.npz; done
  done
done; done; done
TC=$(echo $TIER | tr ' ' '|'); PC="n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000"
export F1T_PROCS=40 LADDER_ITS="0 8 24 96 384" JAX_PLATFORMS=cpu
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_x120 $PY -u ladder_eval.py runs/fig2base/fig2fadaN?_rubber_s? > logs/fadaN/eval.log 2>&1
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_a33 LADDER_CARS="$PC" $PY -u ladder_eval.py runs/fig2base/fig2fadaN?_plastic_s? >> logs/fadaN/eval.log 2>&1
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free LADDER_CARS="$TC" $PY -u ladder_eval.py runs/fig2base/fig2fadaN?_tier_s? >> logs/fadaN/eval.log 2>&1
touch logs/fadaN/DONE
