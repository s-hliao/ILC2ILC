#!/bin/bash
# re-record the car trajectories at the 10-lap budget (2026-10-11) and rebuild the videos: ours = nominal-sim learner /
# + DR (A) / + DR (B, the headline v3drBt), each zero-shot and after 10 real laps (5 chained 2-lap trials); PPO+DR with our stage
cd "$(dirname "$0")"
source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
export F1T_PROCS=12
$PY record_car.py --root . --laps 5 --eval ours_zeroshot=v3nom_s0_zs ours_10laps=v3nom_s0_lap10 ours_dr_a_zeroshot=v3drA_s0_zs \
  ours_dr_a_10laps=v3drA_s0_lap10 ours_dr_b_zeroshot=v3drBt_s0_zs ours_dr_b_10laps=v3drBt_s0_lap10 ppo_dr_zeroshot=ppo_dr_zs \
  ppo_dr_2laps=ppo_dr_lap2 ppo_dr_10laps=ppo_dr_lap10 rma_zeroshot=rma_zs fada_zeroshot=fada_hw24_zs plan_lqr=lqr_zs 2>&1 | grep -v Warn
$PY record_car.py --root . --hwstage ours=v3nom_s0_lap10 ours_dr_a=v3drA_s0_lap10 ours_dr_b=v3drBt_s0_lap10 ppo_dr=ppo_dr_lap10 2>&1 | grep -v Warn
$PY make_car_videos.py --jobs 8
echo "$(date +%T) RECORD_10LAP_DONE"
