#!/bin/bash
# after the zero-shot screen: full stage (4 x 2-lap iterations), 3 seeds, rubber stand-in, on the plans >= 30 cm zero-shot
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix
while ! grep -q ALLDONE logs/zs1119t.out; do sleep 30; done
for G in mocap_peanutM_b0_free_x150 mocap_figfast_b0_free_x150; do for ss in 9001 9002 9003; do echo "$G $ss"; done; done | \
  xargs -P 3 -L 1 bash -c 'i=$(( ${1: -1} - 1 )); GPU=$i taskset -c $((24+4*i))-$((27+4*i)) ./screen_track_1119.sh $0 freePPON $1'
echo ALLDONE
