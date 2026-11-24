#!/bin/bash
# queue 2026-11-19: zero-shot screens of x165/x180 + 3-seed stages on figfast x150 / peanut x150 (rubber); 3 at a time (matcha load <= 34), 4 cores each
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix
{ for p in mocap_figfast_b0_free_x165 mocap_figfast_b0_free_x180 mocap_peanutM_b0_free_x165; do echo "zs $p"; done
  for p in mocap_figfast_b0_free_x150 mocap_peanutM_b0_free_x150; do for s in 9001 9002 9003; do echo "st $p $s"; done; done; } | \
xargs -P 3 -L 1 bash -c 'slot=$(( ($(echo "$1$2" | cksum | cut -d" " -f1)) % 3 )); C=$((20+4*slot))-$((23+4*slot)); \
  if [ $0 = zs ]; then GPU=$((slot%4)) taskset -c $C ./screen_zs_1119.sh r_${1#mocap_} ax:mu=1.667 $1; \
  else GPU=$((slot%4)) taskset -c $C ./stage_track_1119.sh $1 freePPON $2; fi'
echo ALLDONE
