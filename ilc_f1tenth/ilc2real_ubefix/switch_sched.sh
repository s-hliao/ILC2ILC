#!/bin/bash
# switch_sched.sh LIST BUDGET CPUSET (from matcha): stop the old queue SCRIPTS (not their running stage runs), pre-mark every list item whose run
# folder already exists as started, then start sched6.sh. Process matching via ps + awk on bracketed patterns (never pgrep -f).
cd "$(dirname "$(readlink -f "$0")")"; LIST=$1; mkdir -p logs/prac/started
K=$(ps -eo pid=,args= | awk '/bash (sched_[b]udget|prac_queue[2-5])[.]sh/ {print $1} $2 ~ /^xargs$/ && /-I\{\} bash -c [o]ne/ {print $1}')
echo "stopping queue scripts: $K"; [ -n "$K" ] && kill $K
sleep 2
n=0; grep -v '^#' $LIST | grep . | while read -r p ne s ss t; do D=runs/screen/PRACB_${p}_${t}_${ne}_s${s}_ss${ss}; [ -d $D ] && touch logs/prac/started/${p}_${ne}_${s}_${ss}_${t}; done
echo "pre-marked: $(ls logs/prac/started | wc -l); running stages: $(ps -eo args= | awk '/hw_stage(_yaw)?[.]py/' | wc -l)"
setsid nohup bash sched6.sh $LIST $2 $3 > logs/prac/sched6_$(basename $LIST .txt).out 2>&1 < /dev/null &
echo "sched6 started"
