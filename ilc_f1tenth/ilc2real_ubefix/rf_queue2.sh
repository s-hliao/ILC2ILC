#!/bin/bash
# rf_queue2.sh (2026-11-07 ~13:00: copy of rf_queue.sh; its wait counted the file SB_RERUN_1106.md and expected 146, the SB rerun has 144 runs -> wait on directories only, >= 144)
# rf_queue.sh (2026-11-06, runs/screen/REFIT_SCREEN_1106.md): the 36 refit-screen runs on ube's CPU, after sb_queue.sh has launched
# all 146 SB runs; at most 6 car stage runs (SB + RF) at once.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_fix; mkdir -p logs/rf
npy() { ps -eo comm,args | awk '$1 ~ /^python/ && /hw_stage.py/' | wc -l; }
until [ $(ls -d runs/screen/SB_*/ 2>/dev/null | wc -l) -ge 144 ]; do sleep 120; done
for arm in AW0 AD0 FIT AW80 AD1 AD25; do for s in 0 1 2; do for ss in 9001 9002; do
  while [ $(npy) -ge 6 ]; do sleep 30; done
  bash rf_stage.sh $arm $s $ss & sleep 15
done; done; done
wait
touch logs/rf/RF_STAGE_DONE
