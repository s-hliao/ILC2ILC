#!/bin/bash
# Re-record the car trajectories for the paper's final rows (2026-10-11) and rebuild the videos and trajectory figures:
# the headline = the stopping-rule network (v3drBt_stop: zero-shot, 10 and 48 real laps), our learner + DR (B) / (A) / nominal
# sim (zero-shot, 10 laps), PPO+DR (zero-shot, + our stage 10 laps), the RMA / FADA picks (src/paper/baselines.json, at their
# own best budget), plan LQR. The old video folders were moved to ~/ilc_ws/archive/car_videos_v3drBt_s0/ first.
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python; export F1T_PROCS=8
PICKS=$(for r in rma_nonoracle rma_oracle fada_nonoracle fada_oracle; do python3 -c "import json
b = json.load(open('../../paper/baselines.json'))['car']['$r']['best']
print('$r=' + (b['source'] if b['kind'] == 'point' else b['source'] + '_ep' + str(b['best_budget'])))"; done | tr '\n' ' ')
echo "picks: $PICKS"
$PY record_car.py --root . --laps 5 --eval ours_stop_zeroshot=v3drBt_stop_zs ours_stop_10laps=v3drBt_stop_lap10 \
  ours_stop_48laps=v3drBt_stop_lap48 ours_dr_b_zeroshot=v3drB_s0_zs ours_dr_b_10laps=v3drB_s0_lap10 \
  ours_dr_a_zeroshot=v3drA_s0_zs ours_dr_a_10laps=v3drA_s0_lap10 ours_zeroshot=v3nom_s0_zs ours_10laps=v3nom_s0_lap10 \
  ppo_dr_zeroshot=ppo_dr_zs ppo_dr_10laps=ppo_dr_lap10 $PICKS plan_lqr=lqr_zs 2>&1 | grep -v Warn
$PY record_car.py --root . --hwstage ours_stop=v3drBt_stop_lap10 ours_dr_b=v3drB_s0_lap10 ours_dr_a=v3drA_s0_lap10 \
  ours=v3nom_s0_lap10 ppo_dr=ppo_dr_lap10 2>&1 | grep -v Warn
$PY make_car_videos.py --jobs 6
$PY car_traj_figs.py
echo "$(date +%T) RECORD_FINAL_DONE"
