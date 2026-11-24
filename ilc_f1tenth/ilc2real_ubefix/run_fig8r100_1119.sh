#!/bin/bash
# plastic on the SMALLER figure-8 (R 1.0 m, user 2026-11-19 option 2: fit the mocap room), freePPON + stage, 3 seeds, x0.95 and x0.90
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix
while ! grep -q ALLDONE logs/rm1119.out; do sleep 30; done
for G in mocap_fig8R100_b0_free_x095 mocap_fig8R100_b0_free_x090; do for ss in 9001 9002 9003; do echo "$G $ss"; done; done | \
  xargs -P 3 -L 1 bash -c 'i=$(( ${1: -1} - 1 )); GPU=$i taskset -c $((24+4*i))-$((27+4*i)) ./screen_track_1119p.sh $0 freePPON $1'
echo ALLDONE
