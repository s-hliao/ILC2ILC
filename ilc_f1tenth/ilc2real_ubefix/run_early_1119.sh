#!/bin/bash
# early-base stages: rubber peanut x1.5 from BC-only (1 min) and PPO it100 (~5 min); plastic fig8 x0.95 from PPO it200 (~9 min; BC / it100 spin out zero-shot)
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix
{ for ss in 9001 9002 9003; do
  echo "mocap_peanutM_b0_free_x150 ax:mu=1.667 runs/freeBCN_s0/policy_final.npz bc $ss"
  echo "mocap_peanutM_b0_free_x150 ax:mu=1.667 runs/freePPON_s0/policy_it100.npz ppo100 $ss"
  echo "mocap_fig8R115_b0_free_x095 ax:mu=0.4 runs/freePPON_s0/policy_it200.npz ppo200 $ss"
done; } | xargs -P 3 -L 1 bash -c 'i=$(( ${4: -1} - 1 )); GPU=$i taskset -c $((24+4*i))-$((27+4*i)) ./screen_early_1119.sh $0 $1 $2 $3 $4'
echo ALLDONE
