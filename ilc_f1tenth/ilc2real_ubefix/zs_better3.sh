#!/bin/bash
# zs_better3.sh (2026-11-08 ~20:45; user: "if ppo is able to do it with the nominal model then we must be able to as well"): push WSL's levers.
#   A  W + speed error quarter weight (err_scale speed 1.2) + 90 its
#   B  friction DR x0.35-1.2 (log-uniform) + speed half weight + 90 its
#   C  WSL + 1024 lanes (PPO's lane count)
#   D  WSL + 180 its
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=~/miniconda3/envs/f1t/bin/python
PL='["mocap_square2fast_b0_free","mocap_square2fast_b0_free_v85","mocap_square2fast_b0_free_v70","mocap_figfast_b0_free","mocap_figfast_b0_free_v85","mocap_figfast_b0_free_v70"]'
E6='[0.05,1000000000.0,1000000000.0,1000000000.0,1000000000.0,0.6]'; E12='[0.05,1000000000.0,1000000000.0,1000000000.0,1000000000.0,1.2]'
BASE="\"plans\":$PL,\"init\":\"runs/freeBCN_s0/policy_final.npz\",\"eval_every\":10,\"seed\":0"
C() { nice -n 5 taskset -c 0-11 $PY -u fig1_cmd.py "$@"; }
F1T_DR_MU="0.5 1.2"  C sim_train.py runs/freeDR5N_s0/args.json runs/zs_better/oursdrA_s0 0 "{$BASE,\"iters\":90,\"err_scale\":$E12}" > logs/zs_better/A.log 2>&1 &
F1T_DR_MU="0.35 1.2" C sim_train.py runs/freeDR5N_s0/args.json runs/zs_better/oursdrB_s0 1 "{$BASE,\"iters\":90,\"err_scale\":$E6}" > logs/zs_better/B.log 2>&1 &
F1T_DR_MU="0.5 1.2"  C sim_train.py runs/freeDR5N_s0/args.json runs/zs_better/oursdrC_s0 2 "{$BASE,\"iters\":90,\"err_scale\":$E6,\"lanes\":1024}" > logs/zs_better/C.log 2>&1 &
F1T_DR_MU="0.5 1.2"  C sim_train.py runs/freeDR5N_s0/args.json runs/zs_better/oursdrD_s0 3 "{$BASE,\"iters\":180,\"err_scale\":$E6}" > logs/zs_better/D.log 2>&1 &
wait
L=""; for r in A B C D; do f=runs/zs_better/oursdr${r}_s0/policy_final.npz; [ -f $f ] && L="$L $f"; done
F1T_SPIN_RHO=3 F1T_PROCS=10 nice -n 5 taskset -c 0-11 $PY -u fig1_eval_prac.py runs/zs_better/zs_prac3.json $L > logs/zs_better/eval3.log 2>&1
touch logs/zs_better/EVAL3_DONE
