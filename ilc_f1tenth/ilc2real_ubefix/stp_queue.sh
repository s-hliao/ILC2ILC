#!/bin/bash
# stp_queue.sh (2026-11-06, STEP_SCREEN_1106.md): 36 runs. Rubber ETHZ on the GPUs (one job per GPU, after ETHZX110's training),
# plastic on the CPU (at most 6 CPU stage runs at once on ube, counting the z-path runs), then the evaluations.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_fix
npy() { ps -eo comm,args | awk -v p="$1" '$1 ~ /^python/ && $0 ~ p' | wc -l; }
gpu_q() {
  until [ $(ps -eo comm,args | awk '$1 ~ /^bash/ && /ethz_x110.sh/' | wc -l) -eq 0 ]; do sleep 30; done
  J=(); for arm in 0 A B; do for s in 0 1 2; do for ss in 9001 9002; do J+=("$arm ethz $s $ss"); done; done; done
  i=0; while [ $i -lt ${#J[@]} ]; do
    for g in 0 1 2; do [ $i -lt ${#J[@]} ] || break; bash stp_run.sh ${J[$i]} $g & i=$((i + 1)); sleep 10; done
    wait
  done
}
cpu_q() {
  J=(); for s in 0 1 2; do for ss in 9001 9002; do for arm in 0 A B; do J+=("$arm plastic $s $ss"); done; done; done
  for j in "${J[@]}"; do
    while [ $(npy "hw_stage.py.*(ZP[HE]_|STP._plastic)") -ge 6 ]; do sleep 30; done
    bash stp_run.sh $j 0 &
    sleep 20
  done
  wait
}
gpu_q &
cpu_q &
wait
PY=~/miniconda3/envs/f1t/bin/python
PC="n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000"
export F1T_SPIN_RHO=3
E() { taskset -c 0-29 env "$@"; }
E LADDER_OUT=rescore_ethz.json LADDER_LAPS=1 F1T_PROCS=24 LADDER_PLANS=ethzmobil43_b0_free $PY -u ladder_eval.py runs/screen/STP?_ethz_rubber_* > logs/stp_eval.log 2>&1
E LADDER_OUT=rescore_peanut.json LADDER_LAPS=5 F1T_PROCS=24 LADDER_PLANS=mocap_peanutM_b0_free_x120 $PY -u ladder_eval.py runs/screen/STP?_ethz_rubber_* >> logs/stp_eval.log 2>&1
E LADDER_OUT=rescore_peanut.json LADDER_LAPS=5 F1T_PROCS=24 LADDER_PLANS=mocap_peanutM_b0_free_a33 LADDER_CARS="$PC" $PY -u ladder_eval.py runs/screen/STP?_plastic_* >> logs/stp_eval.log 2>&1
echo STP_DONE >> logs/stp_eval.log
