#!/bin/bash
# FADA exact (per-car IDMs from each car's own transitions) re-adapted with the FIXED oracle lookahead (real_car.py, 2026-10-12:
# the oracle deviation H control periods ahead at the measured speed, not one 5 mm plan index): fada_exact2.
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
OURS=v3drBtL_s0_lad1_lap10
CK="24 96 384 1536 2016"
[ -f runs/hw/fada_exact2_ep2016/summary.json ] || F1T_PROCS=4 $PY -u fada_real_car.py --idm "runs/fada_exact_idm/idm_{car}.npz" \
  --arm fada_exact2 --horizon 4 --privileged 1 --oracle-arm $OURS --rungs-from $OURS --checkpoints $CK --gpu 1 > logs/fada_exact2.log 2>&1
F1T_PROCS=4 $PY -u reeval.py fada_exact2_ep0 $(for k in $CK; do echo -n "fada_exact2_ep$k "; done) > logs/reeval_fada_exact2.log 2>&1
echo "$(date +%T) FADA2_DONE"
