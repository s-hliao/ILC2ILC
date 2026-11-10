#!/bin/bash
# fig1_queue.sh (2026-11-06, paper/FIGURE_PLAN_1106.md figure 1): the car's simulation-training curves. Every arm from the same BC
# clone (runs/freeBCN_s0), the deployed networks' 6 plans and their own saved arguments, seeds 0-2, dense checkpoints:
#   ppodr  ppo_dr.py as runs/freePPON_s<s> + --save-at        ppo   the same with --dr 0
#   rma    ppo_dr.py as runs/rmaN_s<s> + --save-at, then phase 2 (--adapt-only) on the teacher at its 10 / 30 / 70 / 150 / 300
#   ours   sim_train.py as runs/freeNom5N_s0 (plans -> the 6, --init the clone, --eval-every 3, seed s)
#   oursdr sim_train.py as runs/freeDR5N_s0 (same overrides)
#   fada   fada_car.py as runs/hw/fada_v2_h4_hw24 (plans / goals -> the 6, --init the clone, --iters 0, data 5 / 10 / 20 / 40 / 60)
# GPUs 0-2 of ube, ONE job per GPU, after the step screen's GPU runs (18 STP?_ethz_rubber summaries).
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_fix; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=~/miniconda3/envs/f1t/bin/python
until [ $(ls runs/screen/STP?_ethz_rubber_*/summary.json 2>/dev/null | wc -l) -ge 18 ] && \
      [ $(ps -eo comm,args | awk '$1 ~ /^python/ && /hw_stage.py/ && /ETHZX110|STP._ethz/' | wc -l) -eq 0 ]; do sleep 60; done
PL='["mocap_square2fast_b0_free","mocap_square2fast_b0_free_v85","mocap_square2fast_b0_free_v70","mocap_figfast_b0_free","mocap_figfast_b0_free_v85","mocap_figfast_b0_free_v70"]'
INIT=runs/freeBCN_s0/policy_final.npz
SAVE='[5,10,20,30,50,70,150,250]'
mkdir -p runs/fig1 logs/fig1
C() { taskset -c 30-33 $PY -u fig1_cmd.py "$@"; }
worker() {
  s=$1 g=$2
  C ppo_dr.py runs/rmaN_s$s/args.json runs/fig1/rma_s$s $g "{\"save_at\":$SAVE}" > logs/fig1/rma_s$s.log 2>&1
  for k in 010 030 070 150; do
    C ppo_dr.py runs/rmaN_s$s/args.json runs/fig1/rma_s${s}_ck$k $g "{\"adapt_only\":true,\"teacher\":\"runs/fig1/rma_s$s/policy_it$k.npz\",\"init\":null}" > logs/fig1/rma_s${s}_ck$k.log 2>&1
  done
  C ppo_dr.py runs/freePPON_s$s/args.json runs/fig1/ppodr_s$s $g "{\"save_at\":$SAVE}" > logs/fig1/ppodr_s$s.log 2>&1
  C ppo_dr.py runs/freePPON_s$s/args.json runs/fig1/ppo_s$s $g "{\"dr\":0.0,\"save_at\":$SAVE}" > logs/fig1/ppo_s$s.log 2>&1
  C sim_train.py runs/freeNom5N_s0/args.json runs/fig1/ours_s$s $g "{\"plans\":$PL,\"init\":\"$INIT\",\"eval_every\":3,\"seed\":$s}" > logs/fig1/ours_s$s.log 2>&1
  C sim_train.py runs/freeDR5N_s0/args.json runs/fig1/oursdr_s$s $g "{\"plans\":$PL,\"init\":\"$INIT\",\"eval_every\":3,\"seed\":$s}" > logs/fig1/oursdr_s$s.log 2>&1
  for n in 5 10 20 40 60; do
    C fada_car.py runs/hw/fada_v2_h4_hw24/args.json runs/fig1/fada_d${n}_s$s $g "{\"plans\":$PL,\"goals\":$PL,\"init\":\"$INIT\",\"iters\":0,\"eval_runs\":0,\"data_iters\":$n,\"seed\":$s,\"policy\":null}" > logs/fig1/fada_d${n}_s$s.log 2>&1
  done
}
worker 0 0 & sleep 30; worker 1 1 & sleep 30; worker 2 2 &
wait
echo FIG1_TRAIN_DONE > logs/fig1/DONE
