#!/bin/bash
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_fix
J=(); for n in freePPON rmaN; do for s in 0 1 2; do for ss in 9001 9002 9003; do J+=("$n $s $ss"); done; done; done
i=0; while [ $i -lt ${#J[@]} ]; do
  for g in 0 1 2; do [ $i -lt ${#J[@]} ] || break; bash ethz_x110.sh ${J[$i]} $g & i=$((i + 1)); sleep 10; done
  wait
done
PY=~/miniconda3/envs/f1t/bin/python
export F1T_SPIN_RHO=3
LADDER_OUT=ethz_eval.json LADDER_LAPS=1 F1T_PROCS=12 LADDER_PLANS=ethzmobil43_b0_free_x110 taskset -c 0-29 $PY -u ladder_eval.py runs/screen/ETHZX110_rubber_* > logs/ethzx110_eval.log 2>&1
LADDER_OUT=peanut_eval.json LADDER_LAPS=5 F1T_PROCS=12 LADDER_PLANS=mocap_peanutM_b0_free_x120 taskset -c 0-29 $PY -u ladder_eval.py runs/screen/ETHZX110_rubber_* >> logs/ethzx110_eval.log 2>&1
echo ETHZX110_DONE >> logs/ethzx110_eval.log
