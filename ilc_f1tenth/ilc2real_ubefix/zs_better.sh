#!/bin/bash
# zs_better.sh (2026-11-08 ~17:15, user: "get our model to be better zero-shot"): seed-0 screens of our network's training (sim_train.py, as
# fig1's oursdr), matcha GPUs 0-3 (one job per GPU), ube's exact code (this sandbox). Levers, each alone first:
#   L   longer training: iters 90 (PPO+DR used 300 x 1024 lanes; ours 30 x 384)
#   W   friction DR widened to cover the low-grip cars: F1T_DR_MU "0.5 1.2" (log-uniform x0.5-1.2; default +-20 %)
#   WL  both
#   PW  FAIRNESS CHECK: PPO+DR (ppo_dr.py as freePPON) with the SAME widened friction DR
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=~/miniconda3/envs/f1t/bin/python
PL='["mocap_square2fast_b0_free","mocap_square2fast_b0_free_v85","mocap_square2fast_b0_free_v70","mocap_figfast_b0_free","mocap_figfast_b0_free_v85","mocap_figfast_b0_free_v70"]'
INIT=runs/freeBCN_s0/policy_final.npz
C() { nice -n 5 taskset -c 0-11 $PY -u fig1_cmd.py "$@"; }
C sim_train.py runs/freeDR5N_s0/args.json runs/zs_better/oursdrL_s0 0 "{\"plans\":$PL,\"init\":\"$INIT\",\"eval_every\":10,\"seed\":0,\"iters\":90}" > logs/zs_better/L.log 2>&1 &
F1T_DR_MU="0.5 1.2" C sim_train.py runs/freeDR5N_s0/args.json runs/zs_better/oursdrW_s0 1 "{\"plans\":$PL,\"init\":\"$INIT\",\"eval_every\":10,\"seed\":0}" > logs/zs_better/W.log 2>&1 &
F1T_DR_MU="0.5 1.2" C sim_train.py runs/freeDR5N_s0/args.json runs/zs_better/oursdrWL_s0 2 "{\"plans\":$PL,\"init\":\"$INIT\",\"eval_every\":10,\"seed\":0,\"iters\":90}" > logs/zs_better/WL.log 2>&1 &
F1T_DR_MU="0.5 1.2" C ppo_dr.py runs/freePPON_s0/args.json runs/zs_better/ppodrW_s0 3 "{}" > logs/zs_better/PW.log 2>&1 &
wait; touch logs/zs_better/TRAIN_DONE
