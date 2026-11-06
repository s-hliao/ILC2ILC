#!/bin/bash
# rescore_fix_rubber.sh (2026-11-06): the rubber cells on the fixed simulator (as rescore_fix.sh; the two rubber cars), after rescore_fix.sh.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real
until [ -f logs/RESCORE_FIX_DONE ]; do sleep 30; done
PY=$HOME/miniconda3/envs/f1t/bin/python
export F1T_SPIN_RHO=3 F1T_PROCS=10
LADDER_OUT=rescore_peanut.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_x120 nice -n 5 $PY -u ladder_eval.py runs/screen/SL50x120* runs/screen/ETHZ_rubber_* > logs/rescore_fix_rubber.log 2>&1
LADDER_OUT=rescore_ethz.json LADDER_LAPS=1 LADDER_PLANS=ethzmobil43_b0_free nice -n 5 $PY -u ladder_eval.py runs/screen/ETHZ_rubber_* >> logs/rescore_fix_rubber.log 2>&1
touch logs/RESCORE_FIX_RUBBER_DONE
