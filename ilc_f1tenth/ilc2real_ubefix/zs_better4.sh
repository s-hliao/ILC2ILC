#!/bin/bash
# zs_better4.sh (2026-11-08 ~21:00): the measured cause of our plastic zero-shot gap vs PPO: sim_train drops every row past the
# first |e_y| > --max-ey 0.2 (the sim fails only at 0.6), so the GN step never sees the cost of sliding off; PPO's reward stops at
# failure (a survival incentive). One-knob test: --max-ey 0.6 (= FAIL_EY), all else the published recipe.
#   E  WSL recipe (DR mu 0.5-1.2, speed weight 0.6, 90 its) + max_ey 0.6        GPU0 after A
#   F  published oursdr exactly (default DR, 30 its) + max_ey 0.6                GPU1 after B
#   G  published ours exactly (no DR) + max_ey 0.6                               GPU3 after D
#   H  published oursdr + max_ey 0.6 + 90 its                                    GPU2 after C
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=~/miniconda3/envs/f1t/bin/python
PL='["mocap_square2fast_b0_free","mocap_square2fast_b0_free_v85","mocap_square2fast_b0_free_v70","mocap_figfast_b0_free","mocap_figfast_b0_free_v85","mocap_figfast_b0_free_v70"]'
E6='[0.05,1000000000.0,1000000000.0,1000000000.0,1000000000.0,0.6]'
BASE="\"plans\":$PL,\"init\":\"runs/freeBCN_s0/policy_final.npz\",\"eval_every\":10,\"seed\":0,\"max_ey\":0.6"
W() { until [ -f runs/zs_better/oursdr$1_s0/policy_final.npz ]; do sleep 30; done; }
C() { nice -n 5 taskset -c 12-23 $PY -u fig1_cmd.py "$@"; }
( W A; F1T_DR_MU="0.5 1.2" C sim_train.py runs/freeDR5N_s0/args.json runs/zs_better/oursdrE_s0 0 "{$BASE,\"iters\":90,\"err_scale\":$E6}" > logs/zs_better/E.log 2>&1 ) &
( W B; C sim_train.py runs/fig1/oursdr_s0/args.json runs/zs_better/oursdrF_s0 1 "{$BASE}" > logs/zs_better/F.log 2>&1 ) &
( W D; C sim_train.py runs/fig1/ours_s0/args.json runs/zs_better/oursG_s0 3 "{$BASE}" > logs/zs_better/G.log 2>&1 ) &
( W C; C sim_train.py runs/fig1/oursdr_s0/args.json runs/zs_better/oursdrH_s0 2 "{$BASE,\"iters\":90}" > logs/zs_better/H.log 2>&1 ) &
wait
L=""; for r in oursdrE oursdrF oursG oursdrH; do f=runs/zs_better/${r}_s0/policy_final.npz; [ -f $f ] && L="$L $f"; done
until [ -f logs/zs_better/EVAL3_DONE ]; do sleep 30; done
F1T_SPIN_RHO=3 F1T_PROCS=12 nice -n 5 taskset -c 12-23 $PY -u fig1_eval_prac.py runs/zs_better/zs_prac4.json $L > logs/zs_better/eval4.log 2>&1
touch logs/zs_better/EVAL4_DONE
