#!/bin/bash
# after peanut x150: full stage at x1.65 (peer's zero-shot screen: figfast 52 cm, peanut 46 cm RMS), 3 seeds each
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix
while ! grep -q ALLDONE logs/tr1119b.out; do sleep 30; done
for G in mocap_peanutM_b0_free_x165 mocap_figfast_b0_free_x165; do for ss in 9001 9002 9003; do echo "$G $ss"; done; done | \
  xargs -P 3 -L 1 bash -c 'i=$(( ${1: -1} - 1 )); GPU=$i taskset -c $((24+4*i))-$((27+4*i)) ./screen_track_1119.sh $0 freePPON $1'
echo ALLDONE
