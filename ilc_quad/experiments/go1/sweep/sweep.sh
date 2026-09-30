#!/bin/bash
# sweep.sh OUT JOBSFILE: every line "run|task|start|args" of JOBSFILE, JOBS at a time, then
# score.py. task is a tasks.txt name; start is "scratch" (the task's own TO forces) or
# "transfer" (retarget from $SUITE/<its source>, as validate.sh). args: any lockstep
# options (--param NAME:=VALUE, reality gap). Runs whose summary exists are skipped, so an
# interrupted sweep resumes. MAXT (default 20) caps the trials. Tasks come from tasks.txt
# or sweep/tasks_extra.txt (plans solved on first use, by `plans.jobs`: run it alone first).
source "$(dirname "$0")/../env.sh"
OUT=${1:?usage: sweep.sh OUT JOBSFILE}; JF=${2:?usage: sweep.sh OUT JOBSFILE}
MAXT=${MAXT:-20}; export OUT MAXT
mkdir -p "$OUT"
one() {
  IFS="|" read -r run task start args <<< "$1"
  [ -f "$OUT/$run.json" ] && return 0
  # tasks.txt, then sweep/tasks_extra.txt (extra plans; trailing columns = the TO settings
  # the plan was solved with, passed to every run so the node accepts the plan file)
  read name m dx dz xf h plan src targs <<< "$(grep -h "^$task " "$EXP/tasks.txt" "$EXP/sweep/tasks_extra.txt" | head -1)"
  [ -n "$name" ] || { echo "no task $task"; return 1; }
  local box="" tr=""
  [ "$h" != "0" ] && box="--box $xf $h"
  [ "$start" = transfer ] && [ "$src" != "-" ] && tr="--transfer-from $SUITE/$src --transfer-mode retarget"
  rm -rf "${OUT:?}/${run:?}"
  cd "$WS" && timeout 1800 python3 install/ilc_quad/lib/ilc_quad/ilc_jump_lockstep.py \
    --robot go1 --max-trials $MAXT --jump $dx $dz $box --param "phases:=[30,30,30]" \
    --param margin:=$m --reference-file "$EXP/plans/ref_$plan.npz" $targs $tr $args $EXTRA \
    --log-dir "$OUT" --run-name "$run" --summary "$OUT/$run.json" > "$OUT/$run.log" 2>&1
  echo "$run done ($?)"
}
export -f one
grep -v '^\s*#' "$JF" | grep -v '^\s*$' | xargs -P "$JOBS" -d '\n' -I{} bash -c 'one "$1"' _ {}
python3 "$EXP/sweep/score.py" "$OUT" > "$OUT/score.txt"
tail -3 "$OUT/score.txt"
