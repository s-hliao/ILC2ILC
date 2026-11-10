#!/bin/bash
# fig1J_train.sh (2026-11-09 ~01:20, user: "run the new network, regenerate the figures"): our two networks retrained with screen J's
# settings (zs_better6.sh J: trust-region cap 0.5, rows kept to the sim's failure line max_ey 0.6, speed-error scale 0.6, 90 iterations;
# DistILC+DR also friction DR x0.5-1.2 log-uniform) on the published fig1 base (runs/fig1_preJ/oursdr_s0/args.json), seeds 0-2.
# Published runs moved to runs/fig1_preJ/. Checkpoints every 9 iterations for figure 1.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=~/miniconda3/envs/f1t/bin/python; mkdir -p logs/fig1J
E6='[0.05,1000000000.0,1000000000.0,1000000000.0,1000000000.0,0.6]'
O() { echo "{\"init\":\"runs/freeBCN_s0/policy_final.npz\",\"iters\":90,\"eval_every\":9,\"max_ey\":0.6,\"cap\":0.5,\"err_scale\":$E6,\"seed\":$1,\"dr\":$2}"; }
T() { # NET SEED GPU
  if [ $1 = oursdr ]; then export F1T_DR_MU="0.5 1.2"; DR=1.0; else unset F1T_DR_MU; DR=0.0; fi
  nice -n 5 taskset -c 24-35 $PY -u fig1_cmd.py sim_train.py runs/fig1_preJ/oursdr_s0/args.json runs/fig1/$1_s$2 $3 "$(O $2 $DR)" > logs/fig1J/$1_s$2.log 2>&1; }
( T oursdr 0 0; T ours 2 0 ) &
( T oursdr 1 1; T oursdr 2 1 ) &
( T ours 0 2 ) &
( T ours 1 3 ) &
wait; touch logs/fig1J/TRAIN_DONE
