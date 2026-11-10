#!/bin/bash
# zs_better2.sh (2026-11-08 ~17:45): + the speed error weighted less in training (err_scale speed 0.3 -> 0.6, i.e. half weight vs lateral
# error), with the widened friction DR (W): WS 30 its, WSL 90 its. Then zero-shot on figure 1's practised panels.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=~/miniconda3/envs/f1t/bin/python
PL='["mocap_square2fast_b0_free","mocap_square2fast_b0_free_v85","mocap_square2fast_b0_free_v70","mocap_figfast_b0_free","mocap_figfast_b0_free_v85","mocap_figfast_b0_free_v70"]'
ES='[0.05,1000000000.0,1000000000.0,1000000000.0,1000000000.0,0.6]'
C() { nice -n 5 taskset -c 0-11 $PY -u fig1_cmd.py "$@"; }
F1T_DR_MU="0.5 1.2" C sim_train.py runs/freeDR5N_s0/args.json runs/zs_better/oursdrWS_s0 0 "{\"plans\":$PL,\"init\":\"runs/freeBCN_s0/policy_final.npz\",\"eval_every\":10,\"seed\":0,\"err_scale\":$ES}" > logs/zs_better/WS.log 2>&1 &
F1T_DR_MU="0.5 1.2" C sim_train.py runs/freeDR5N_s0/args.json runs/zs_better/oursdrWSL_s0 1 "{\"plans\":$PL,\"init\":\"runs/freeBCN_s0/policy_final.npz\",\"eval_every\":10,\"seed\":0,\"iters\":90,\"err_scale\":$ES}" > logs/zs_better/WSL.log 2>&1 &
wait
F1T_SPIN_RHO=3 F1T_PROCS=10 nice -n 5 taskset -c 0-11 $PY -u fig1_eval_prac.py runs/zs_better/zs_prac2.json runs/zs_better/oursdrWS_s0/policy_final.npz runs/zs_better/oursdrWSL_s0/policy_final.npz > logs/zs_better/eval2.log 2>&1
touch logs/zs_better/EVAL2_DONE
