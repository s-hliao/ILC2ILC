#!/bin/bash
# regenerate the car figures, trajectories and videos from the tight-track results once the real-data study is done
cd "$(dirname "$0")"
source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
until [ -f logs/reeval_match.done ] && [ -f logs/final.done ] && [ -f logs/reeval_laps.done ]; do sleep 120; done
export F1T_PROCS=6
$PY record_car.py --root . --eval ours_zeroshot=v3nom_s0_zs ours_24laps=v3nom_s0_hw24 ours_dr_a_zeroshot=v3drA_s0_zs \
  ours_dr_a_24laps=v3drA_s0_hw24 ours_dr_b_zeroshot=v3drB_s0_zs ours_dr_b_24laps=v3drB_s0_hw24 ppo_dr_zeroshot=ppo_dr_zs \
  ppo_dr_2laps=ppo_dr_lap2 ppo_dr_24laps=ppo_dr_hw24 rma_zeroshot=rma_zs fada_zeroshot=fada_hw24_zs fada_24laps=fada_hw24 \
  plan_lqr=lqr_zs 2>&1 | grep -v Warn
$PY record_car.py --root . --hwstage ours=v3nom_s0_hw24 ours_dr_a=v3drA_s0_hw24 ours_dr_b=v3drB_s0_hw24 ppo_dr=ppo_dr_hw24 2>&1 | grep -v Warn
MPLBACKEND=Agg $PY car_figures.py --tag "tight tracks, 0.3 m envelope"
MPLBACKEND=Agg $PY car_traj_figs.py --tag "tight tracks, 0.3 m envelope"
$PY make_car_videos.py --jobs 4
echo "$(date +%T) REGEN_TIGHT_DONE"
