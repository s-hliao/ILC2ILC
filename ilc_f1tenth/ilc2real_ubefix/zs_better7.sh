#!/bin/bash
# zs_better7.sh (2026-11-09 ~03:20): the retrained DistILC+DR (screen J) runs at ~0.90 pace, so nerr = max(ey/0.1, (1-pace)/0.1) is
# pace-bound and the stage's gains (trial J -13 %, all in e_y) don't show (fig 2: -2 %). J used speed-error scale 0.6 (half weight).
#   N  = J with the PUBLISHED speed scale 0.3 (DR mu 0.5-1.2, max_ey 0.6, cap 0.5, 90 its), seeds 0-2 on GPUs 0-2
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_DR_MU="0.5 1.2"
PY=~/miniconda3/envs/f1t/bin/python
E3='[0.05,1000000000.0,1000000000.0,1000000000.0,1000000000.0,0.3]'
for s in 0 1 2; do
  nice -n 5 taskset -c 0-23 $PY -u fig1_cmd.py sim_train.py runs/fig1_preJ/oursdr_s0/args.json runs/zs_better/oursdrN_s$s $s \
    "{\"init\":\"runs/freeBCN_s0/policy_final.npz\",\"iters\":90,\"eval_every\":9,\"max_ey\":0.6,\"cap\":0.5,\"err_scale\":$E3,\"seed\":$s,\"dr\":1.0}" > logs/zs_better/N_s$s.log 2>&1 &
done; wait
F1T_SPIN_RHO=3 F1T_PROCS=28 nice -n 5 taskset -c 0-31 $PY -u fig1_eval_prac.py runs/zs_better/zs_prac7.json runs/zs_better/oursdrN_s{0,1,2}/policy_final.npz > logs/zs_better/eval7.log 2>&1
touch logs/zs_better/EVAL7_DONE
