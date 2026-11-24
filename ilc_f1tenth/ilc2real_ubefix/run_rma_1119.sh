#!/bin/bash
# RMA arm (runs/rmaN_s0) on the hardware tracks: rubber peanut x1.5, plastic fig8 x0.95; 3 seeds each (sim stand-ins)
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix
{ for ss in 9001 9002 9003; do
  echo "mocap_peanutM_b0_free_x150 ax:mu=1.667 runs/rmaN_s0/policy_final.npz rma $ss"
  echo "mocap_fig8R115_b0_free_x095 ax:mu=0.4 runs/rmaN_s0/policy_final.npz rma $ss"
done; } | xargs -P 3 -L 1 bash -c 'i=$(( ${4: -1} - 1 )); GPU=$i taskset -c $((24+4*i))-$((27+4*i)) ./screen_rma_1119.sh $0 $1 $2 $3 $4'
echo ALLDONE
