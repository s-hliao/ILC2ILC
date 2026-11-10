#!/bin/bash
# zs_better6.sh (2026-11-09 00:45). MEASURED (scratchpad gn_diag.py, published oursdr, square2fast, friction 0.4 lanes): the per-lane GN step
# points the right way (slow down, -0.11 A) once rows past 0.2 m are kept, but the trust-region cap 0.04 shrinks it ~40x
# (J -3 %); cap 0.5: -1.32 A, J -26 % (= a blunt -4 A offset). Marquardt damping: no change. Screens = the three together:
#   I  WSL (DR mu 0.5-1.2, speed weight 0.6, 90 its) + max_ey 0.6 + cap 0.15      GPU0
#   J  WSL + max_ey 0.6 + cap 0.5                                                   GPU1
#   K  published oursdr (default DR, 30 its) + max_ey 0.6 + cap 0.15                GPU2
#   M  DR mu 0.35-1.2 + speed 0.6 + max_ey 0.6 + cap 0.15, 90 its                    GPU3
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=~/miniconda3/envs/f1t/bin/python
PL='["mocap_square2fast_b0_free","mocap_square2fast_b0_free_v85","mocap_square2fast_b0_free_v70","mocap_figfast_b0_free","mocap_figfast_b0_free_v85","mocap_figfast_b0_free_v70"]'
E6='[0.05,1000000000.0,1000000000.0,1000000000.0,1000000000.0,0.6]'
BASE="\"plans\":$PL,\"init\":\"runs/freeBCN_s0/policy_final.npz\",\"eval_every\":10,\"seed\":0,\"max_ey\":0.6"
C() { nice -n 5 taskset -c 0-23 $PY -u fig1_cmd.py "$@"; }
F1T_DR_MU="0.5 1.2"  C sim_train.py runs/freeDR5N_s0/args.json runs/zs_better/oursdrI_s0 0 "{$BASE,\"iters\":90,\"err_scale\":$E6,\"cap\":0.15}" > logs/zs_better/I.log 2>&1 &
F1T_DR_MU="0.5 1.2"  C sim_train.py runs/freeDR5N_s0/args.json runs/zs_better/oursdrJ_s0 1 "{$BASE,\"iters\":90,\"err_scale\":$E6,\"cap\":0.5}" > logs/zs_better/J.log 2>&1 &
C sim_train.py runs/fig1/oursdr_s0/args.json runs/zs_better/oursdrK_s0 2 "{$BASE,\"cap\":0.15}" > logs/zs_better/K.log 2>&1 &
F1T_DR_MU="0.35 1.2" C sim_train.py runs/freeDR5N_s0/args.json runs/zs_better/oursdrM_s0 3 "{$BASE,\"iters\":90,\"err_scale\":$E6,\"cap\":0.15}" > logs/zs_better/M.log 2>&1 &
wait
L=""; for r in I J K M; do f=runs/zs_better/oursdr${r}_s0/policy_final.npz; [ -f $f ] && L="$L $f"; done
F1T_SPIN_RHO=3 F1T_PROCS=16 nice -n 5 taskset -c 0-23 $PY -u fig1_eval_prac.py runs/zs_better/zs_prac6.json $L > logs/zs_better/eval6.log 2>&1
touch logs/zs_better/EVAL6_DONE
