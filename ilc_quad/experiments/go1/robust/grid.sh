#!/bin/bash
# grid.sh OUT "task task ..." [conditions file]: every task under every condition, JOBS runs
# at a time, then the report. Conditions are lines "name|args" (conds.txt: the full set).
# EXTRA="--param NAME:=VALUE ..." is passed to every run, e.g. the paper's Stage III law:
#   EXTRA="--param stage3_safeguard:=false" ./grid.sh $LOG/robust/paper "f60_m85 b50_10_m85"
source "$(dirname "$0")/../env.sh"
OUT=${1:?usage: grid.sh OUT "tasks" [conds]}
TASKS=${2:?usage: grid.sh OUT "tasks" [conds]}
C=${3:-$EXP/robust/conds.txt}
for t in $TASKS; do
  while IFS="|" read -r c a; do [ -n "$c" ] && echo "$t|$c|$a"; done < "$C"
done | xargs -P "$JOBS" -I{} bash -c \
  'IFS="|" read -r t c a <<< "{}"; "$EXP/robust/job.sh" "$0" "$t" "$c" "$a"' "$OUT"
python3 "$EXP/robust/robust_report.py" "$OUT"
