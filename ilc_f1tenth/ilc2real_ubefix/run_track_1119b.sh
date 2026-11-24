#!/bin/bash
# rerun peanut x150 (3 seeds) after the figfast x150 stages: a peer session killed the first peanut runs at 18:36 (2026-11-19)
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix
while ! grep -q ALLDONE logs/tr1119.out; do sleep 30; done
for ss in 9001 9002 9003; do echo "mocap_peanutM_b0_free_x150 $ss"; done | \
  xargs -P 3 -L 1 bash -c 'i=$(( ${1: -1} - 1 )); GPU=$i taskset -c $((24+4*i))-$((27+4*i)) ./screen_track_1119.sh $0 freePPON $1'
echo ALLDONE
