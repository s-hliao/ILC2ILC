#!/bin/bash
# fadaM_train_eval.sh (2026-11-07 ~10:55, user: "yes, run it"): the architecture-vs-recipe test. fada_car_tf.py --arch mlp = FADA's
# full-fidelity source recipe (oracle rma_s<s>, DAgger 8 iterations, oracle shadows, Eq. 4.2 / 4.3, same data and steps) with the two
# transformers replaced by MLPs (fada_tf MLP modules, ~0.8 M parameters each), seeds 0-2 -> runs/fig1/fadaM_s<s>; then its iterates 1 2 4 8
# scored by fig1_eval.py exactly as figure 1 (-> runs/fig1/fig1_eval_fadaM.json). Rule (written before results): MLP-FADA it 8 vs RMA zero-shot
# (rma_s<s>) on plastic / rubber: MLP also below RMA -> the recipe explains FADA's zero-shot edge; MLP at / above RMA -> the architecture.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=$HOME/miniconda3/envs/f1t/bin/python
mkdir -p logs/fadaM
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$((s + 1)) taskset -c $((14 + 2 * s))-$((15 + 2 * s)) $PY -u fada_car_tf.py --arch mlp \
    --oracle runs/fig1/rma_s$s --out runs/fig1/fadaM_s$s --seed $s > logs/fadaM/s$s.log 2>&1 &
done
wait
L=""
for s in 0 1 2; do for k in 1 2 4 8; do
  f=runs/fig1/fadaM_s$s/it${k}_policy.npz
  $PY -c "import numpy as np, os; np.savez('$f', mode='fada_tf', pkl=os.path.abspath('runs/fig1/fadaM_s$s/it$k.pkl'))"; L="$L $f"
done; done
F1T_SPIN_RHO=3 F1T_PROCS=12 FADA_JIT=1 nice -n 5 $PY -u fig1_eval.py runs/fig1/fig1_eval_fadaM.json $L > logs/fadaM/fig1_eval.log 2>&1
touch logs/fadaM/DONE
