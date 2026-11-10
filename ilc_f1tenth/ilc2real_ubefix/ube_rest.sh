#!/bin/bash
# ube_rest.sh (2026-11-08 ~02:00, from matcha): the remaining plastic SQUARE practice runs of prac_ube.txt (its xargs was stopped so the
# two fig8-nominal items -- fig8 nominal deprecated by the user -- never start). Each item starts when fewer than 4 hw_stage.py run here,
# via prac_queue.sh (same flags, pinned 0-29).
cd ~/ilc_ws/src/ilc_f1tenth/ilc2real_fix
for it in "plastic freePPON 2 9003 square" "plastic rmaN 2 9001 square" "plastic rmaN 2 9002 square" "plastic rmaN 2 9003 square"; do
  while [ "$(ps -eo args= | awk '/hw_stage[.]py/' | wc -l)" -ge 4 ]; do sleep 30; done
  f=logs/prac/item_$(echo $it | tr ' ' '_').txt; echo "$it" > $f
  setsid nohup bash prac_queue.sh $f 1 0-29 > /dev/null 2>&1 < /dev/null &
  sleep 90
done
