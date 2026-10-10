#!/bin/bash
# recorded trajectories of the original-track mu 0.2 results (archive_mu02_wide) for the car videos / trajectory
# figures: every method's evaluation (3 chained laps on the 8 plans per car) and the key hardware stages re-flown with
# trial recording. The original environment: mu 0.2, plans_mu02, the original tracks, the 0.6 m departure limit.
cd "$(dirname "$0")"
export F1T_MU=0.2 F1T_PLANS=$PWD/plans_mu02 F1T_PROCS=6
unset F1T_TRACKS F1T_SAFETY_EY
PY=~/miniconda3/envs/f1t/bin/python
$PY record_car.py --root archive_mu02_wide --eval ours_zeroshot=v3nom_s0_zs ours_24laps=v3nom_s0_hw24 \
  ours_dr_a_zeroshot=v3drA_s0_zs ours_dr_a_24laps=v3drA_s0_hw24 ours_dr_b_zeroshot=v3drB_s0_zs ours_dr_b_24laps=v3drB_s0_hw24 \
  ppo_dr_zeroshot=ppo_dr_zs ppo_dr_2laps=ppo_dr_lap2 ppo_dr_24laps=ppo_dr_hw24 rma_zeroshot=rma_zs \
  fada_zeroshot=fada_hw24_zs fada_24laps=fada_hw24 plan_lqr=lqr_zs 2>&1 | grep -v Warn
$PY record_car.py --root archive_mu02_wide --hwstage ours=v3nom_s0_hw24 ours_dr_a=v3drA_s0_hw24 ours_dr_b=v3drB_s0_hw24 \
  ppo_dr=ppo_dr_hw24 ppo_dr_2laps=ppo_dr_lap2 2>&1 | grep -v Warn
echo "$(date +%T) RECORD_WIDE_DONE"
