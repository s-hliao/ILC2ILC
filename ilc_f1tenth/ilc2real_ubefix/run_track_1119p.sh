#!/bin/bash
# plastic stand-in, full stage 3 seeds, on the plastic zero-shot screen's large-but-sane plans: peanut x1.0 (43 cm), figfast x0.8 (36 cm)
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix
while ! grep -q ALLDONE logs/tr1119c.out; do sleep 30; done
for G in mocap_peanutM_b0_free mocap_figfast_b0_free_x080; do for ss in 9001 9002 9003; do echo "$G $ss"; done; done | \
  xargs -P 3 -L 1 bash -c 'i=$(( ${1: -1} - 1 )); GPU=$i taskset -c $((24+4*i))-$((27+4*i)) ./screen_track_1119p.sh $0 freePPON $1'
echo ALLDONE
