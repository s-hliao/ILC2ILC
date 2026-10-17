#!/bin/bash
cd ~/ilc_ws/src/ilc_pogo
until grep -q ALL_DONE logs/scheduler_tr1.log 2>/dev/null; do sleep 60; done
POGO_FIX_STAGE=tr2 POGO_STATE=scheduler_tr2 POGO_GPUS=0,1,2 POGO_CPU_JOBS=12 ~/miniconda3/envs/pogo/bin/python -u scheduler.py --jobs scheduler_tr > logs/scheduler_tr2.out 2>&1
echo CHAIN_DONE
