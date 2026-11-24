#!/bin/bash
# plastic stand-in, gentle launch (hand over at 2.0 m/s, 4 A), full stage 3 seeds on fig8R115 x1.0 and x0.95
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix
for G in mocap_fig8R115_b0_free mocap_fig8R115_b0_free_x095; do for ss in 9001 9002 9003; do echo "$G $ss"; done; done | \
  xargs -P 3 -L 1 bash -c 'i=$(( ${1: -1} - 1 )); GPU=$i taskset -c $((24+4*i))-$((27+4*i)) ./screen_track_1119pg.sh $0 freePPON $1'
echo ALLDONE
