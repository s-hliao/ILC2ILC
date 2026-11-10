#!/bin/bash
# after zs_better.sh: zero-shot of each screen's final network (+ the published oursdr_s0 / ppodr_s0 for reference) on figure 1's practised panels.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix
until [ -e logs/zs_better/TRAIN_DONE ]; do sleep 30; done
L=""; for r in oursdrL oursdrW oursdrWL ppodrW; do f=runs/zs_better/${r}_s0/policy_final.npz; [ -f $f ] && L="$L $f"; done
F1T_SPIN_RHO=3 F1T_PROCS=10 nice -n 5 taskset -c 0-11 ~/miniconda3/envs/f1t/bin/python -u fig1_eval_prac.py runs/zs_better/zs_prac.json $L runs/fig1/oursdr_s0/policy_final.npz runs/fig1/ours_s0/policy_final.npz runs/fig1/ppodr_s0/policy_final.npz > logs/zs_better/eval.log 2>&1
touch logs/zs_better/EVAL_DONE
