#!/bin/bash
# fig1_fadaT_eval.sh (2026-11-07, FADA_AUDIT_1107.md FULL FIDELITY): figure 1's FADA points = the full-fidelity source FADA's DAgger iterates
# runs/fig1/fadaT_s<s>/it<k>.pkl (simulated laps in each pickle's meta), each scored zero-shot by fig1_eval.py exactly as every figure-1
# arm (4 starts, 5 laps, seed 701, all panels) -> runs/fig1/fig1_eval_fadaT.json. npz wrappers it<k>_policy.npz (mode fada_tf).
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real
export F1T_SPIN_RHO=3 F1T_PROCS=${F1T_PROCS:-6} FADA_JIT=1
PY=$HOME/miniconda3/envs/f1t/bin/python
until [ -f runs/fig1/fadaT_s0/it8.pkl ] && [ -f runs/fig1/fadaT_s1/it8.pkl ] && [ -f runs/fig1/fadaT_s2/it8.pkl ]; do sleep 60; done
L=""
for s in 0 1 2; do for k in 1 2 3 4 5 6 7 8; do
  f=runs/fig1/fadaT_s$s/it${k}_policy.npz
  $PY -c "import numpy as np, os; np.savez('$f', mode='fada_tf', pkl=os.path.abspath('runs/fig1/fadaT_s$s/it$k.pkl'))"
  L="$L $f"
done; done
mkdir -p logs/fadaT
nice -n 5 $PY -u fig1_eval.py runs/fig1/fig1_eval_fadaT.json $L > logs/fadaT/fig1_eval.log 2>&1
echo "exit $?" >> logs/fadaT/fig1_eval.log
