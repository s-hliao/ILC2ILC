#!/bin/bash
# 2026-11-20: RMA arm (runs/rmaN_s0) on the plastic hardware track = the rotated R1.0 figure-8 x0.95 (fits the mocap room); 3 seeds
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix
for ss in 9001 9002 9003; do echo "mocap_fig8R100_b0_free_x095_rot25 ax:mu=0.4 runs/rmaN_s0/policy_final.npz rma $ss"; done \
  | xargs -P 3 -L 1 bash -c 'i=$(( ${4: -1} - 1 )); GPU=$i taskset -c $((4*i))-$((4*i+3)) ./screen_rma_1119.sh $0 $1 $2 $3 $4'
echo ALLDONE
