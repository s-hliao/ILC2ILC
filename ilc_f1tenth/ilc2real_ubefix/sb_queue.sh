#!/bin/bash
# sb_queue.sh (2026-11-06, runs/screen/SB_RERUN_1106.md): every SB car stage run on ube's CPU, at most 6 stage runs at once, the main
# tables first; figure 2's DistILC_sim (+ DR) runs wait for figure 1's networks.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_fix
npy() { ps -eo comm,args | awk '$1 ~ /^python/ && /hw_stage.py/' | wc -l; }
J=()
for n in freePPON rmaN; do for s in 0 1 2; do for ss in 9001 9002 9003; do J+=("plastic $n $s $ss"); done; done; done
for n in freePPON rmaN; do for s in 0 1 2; do for ss in 9001 9002 9003 9004 9005 9006; do J+=("rubber $n $s $ss"); done; done; done
for n in freePPON rmaN; do for s in 0 1 2; do for ss in 9001 9002; do J+=("tier $n $s $ss"); done; done; done
for p in ethzr ethzp ethzr110; do for n in freePPON rmaN; do for s in 0 1 2; do for ss in 9001 9002 9003; do J+=("$p $n $s $ss"); done; done; done; done
for n in freePPON rmaN; do for s in 0 1 2; do J+=("peax150 $n $s 9001"); done; done
for n in ours oursdr; do for p in rubber plastic tier; do for s in 0 1 2; do J+=("$p $n $s 9001"); done; done; done
for j in "${J[@]}"; do
  set -- $j
  if [ $2 = ours ] || [ $2 = oursdr ]; then until [ -f runs/fig1/$2_s$3/policy_final.npz ]; do sleep 60; done; fi
  while [ $(npy) -ge 6 ]; do sleep 30; done
  bash sb_stage.sh $j & sleep 15
done
wait
touch logs/sb/SB_STAGE_DONE
