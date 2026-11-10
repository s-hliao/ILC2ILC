#!/bin/bash
# fig1_rerun_oursdr_s2.sh (2026-11-06 18:55): fig1_queue.sh's oursdr seed-2 line, exactly, on GPU 2 (the 17:57 run died in XLA's
# GPU compiler: "Check failed: common_utilization <= producer_output_utilization"; folder kept as oursdr_s2_xlacrash_1757).
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_fix; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=~/miniconda3/envs/f1t/bin/python
PL='["mocap_square2fast_b0_free","mocap_square2fast_b0_free_v85","mocap_square2fast_b0_free_v70","mocap_figfast_b0_free","mocap_figfast_b0_free_v85","mocap_figfast_b0_free_v70"]'
rm -rf runs/fig1/oursdr_s2
taskset -c 30-33 $PY -u fig1_cmd.py sim_train.py runs/freeDR5N_s0/args.json runs/fig1/oursdr_s2 2 "{\"plans\":$PL,\"init\":\"runs/freeBCN_s0/policy_final.npz\",\"eval_every\":3,\"seed\":2}" > logs/fig1/oursdr_s2.log 2>&1
