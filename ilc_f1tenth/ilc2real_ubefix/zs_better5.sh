#!/bin/bash
# zs_better5.sh (2026-11-08 20:42): screen H moved here (its zs_better4 waiter waited on C, which OOMed at 1024 lanes on an 11 GB GPU).
#   H  published oursdr + max_ey 0.6 + 90 its, GPU2
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=~/miniconda3/envs/f1t/bin/python
PL='["mocap_square2fast_b0_free","mocap_square2fast_b0_free_v85","mocap_square2fast_b0_free_v70","mocap_figfast_b0_free","mocap_figfast_b0_free_v85","mocap_figfast_b0_free_v70"]'
BASE="\"plans\":$PL,\"init\":\"runs/freeBCN_s0/policy_final.npz\",\"eval_every\":10,\"seed\":0,\"max_ey\":0.6"
nice -n 5 taskset -c 24-35 $PY -u fig1_cmd.py sim_train.py runs/fig1/oursdr_s0/args.json runs/zs_better/oursdrH_s0 2 "{$BASE,\"iters\":90}" > logs/zs_better/H.log 2>&1
F1T_SPIN_RHO=3 F1T_PROCS=10 nice -n 5 taskset -c 24-35 $PY -u fig1_eval_prac.py runs/zs_better/zs_prac5.json runs/zs_better/oursdrH_s0/policy_final.npz > logs/zs_better/eval5.log 2>&1
touch logs/zs_better/EVAL5_DONE
