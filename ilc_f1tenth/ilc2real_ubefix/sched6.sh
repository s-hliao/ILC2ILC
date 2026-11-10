#!/bin/bash
# sched6.sh LIST BUDGET CPUSET (2026-11-08 ~03:25, from matcha): priority scheduler. Re-reads LIST (a DATA file: lines "PROT NET SEED SS TRACK", top =
# first; editing it is safe) every 30 s and starts the first item that is neither done (summary.json + both scores) nor running nor started, when the
# cores in use by ALL stage runs here (any hw_stage*.py: rubber 2, others 8, from their --out) plus the item's cost fit in BUDGET. Runs prac_one.sh.
cd "$(dirname "$(readlink -f "$0")")"; LIST=$1; BUDGET=$2; CPUS=$3; mkdir -p logs/prac/started
used() { ps -eo args= | awk '/hw_stage(_yaw)?[.]py/ {for (i=1;i<=NF;i++) if ($i=="--out") {u += ($(i+1) ~ /_rubber_/) ? 2 : 8}} END {print u+0}'; }
dir_of() { read p n s ss t <<< "$1"; echo runs/screen/PRACB_${p}_${t}_${n}_s${s}_ss${ss}; }
while true; do
  started=0; left=0
  while read -r it; do
    [ -z "$it" ] && continue; [ "${it:0:1}" = "#" ] && continue
    D=$(dir_of "$it"); key=$(echo $it | tr ' ' '_')
    [ -f $D/summary.json ] && [ -f $D/ladder_launch.json ] && [ -f $D/sb_practised.json ] && continue
    [ -e logs/prac/started/$key ] && continue
    ps -eo args= | awk -v d="$D" '/hw_stage[.]py/ && index($0, d" ") {f=1} END {exit !f}' && continue
    left=$((left+1))
    c=8; [ "${it%% *}" = rubber ] && c=2
    [ "${it%% *}" = plastic ] && [ "${it##* }" = ethz ] && [ ! -s logs/prac/PETHZ ] && continue
    if [ $(( $(used) + c )) -le $BUDGET ]; then
      touch logs/prac/started/$key; setsid nohup bash prac_one.sh "$it" $CPUS > /dev/null 2>&1 < /dev/null &
      started=1; sleep 45; break
    fi
  done < $LIST
  [ $started = 0 ] && [ $left = 0 ] && [ -f logs/prac/STOP_$(basename $LIST) ] && break
  [ $started = 0 ] && sleep 30
done
