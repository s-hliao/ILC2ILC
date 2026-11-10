#!/bin/bash
# fig1_eval_queue.sh (2026-11-06, figure 1): score every checkpoint of every FINISHED figure-1 training run as soon as it finishes
# (fig1_eval.py, resumable), plus the clone and the deployed networks; until logs/fig1/DONE and nothing left. CPU cores 0-29.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_fix
PY=~/miniconda3/envs/f1t/bin/python; OUT=runs/fig1/fig1_eval.json
export F1T_SPIN_RHO=3 F1T_PROCS=${F1T_PROCS:-16}
list() {
  echo runs/freeBCN_s0/policy_final.npz runs/freePPON_s{0,1,2}/policy_final.npz runs/rmaN_s{0,1,2}/policy_final.npz
  for d in runs/fig1/{ppodr,ppo,ours,oursdr}_s?; do [ -f $d/policy_final.npz ] && ls $d/policy_it*.npz $d/policy_final.npz; done
  for d in runs/fig1/rma_s? runs/fig1/rma_s?_ck*; do [ -f $d/rma_adapt.npz ] && echo $d/policy_final.npz; done
  for d in runs/fig1/fada_d*_s?; do [ -f $d/summary.json ] && echo $d/idm.npz; done
}
while true; do
  done_flag=0; [ -f logs/fig1/DONE ] && done_flag=1
  taskset -c 0-29 nice -n 5 $PY -u fig1_eval.py $OUT $(list) >> logs/fig1/eval.log 2>&1
  [ $done_flag = 1 ] && break
  sleep 300
done
echo FIG1_EVAL_DONE >> logs/fig1/eval.log
