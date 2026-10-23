#!/bin/bash
# run_more_seeds_c.sh (2026-10-23 05:55, ube): stage_c on drCft_s2 (the second start network, 0.6 plans) at three more
# hardware seeds 9004-9006, recipe vs ctl_random -- a second replication next to run_more_seeds.sh (drB_s2)
P=/home/henry/ilc_ws/log/dilc/plane; T=/home/henry/ilc_ws/src/ilc_mjx/paper/tr_2026-10-17
export DILC_FLY_JOBS=2 GPU0=0 NGPU=3
NET=$P/drCft_s2/policy MEMBER=${MEMBER:?} CLR=${CLR:?} OUT=$P/stage_c_drCft_s2 SEEDS="9004 9005 9006" ARMS="recipe ctl_random" NPAR=6 \
  bash $T/stage_c.sh > $P/stage_c_drCft_s2/more_seeds.out 2>&1
echo "$(date +%T) MORE_SEEDS_C_DONE"
