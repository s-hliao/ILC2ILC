#!/bin/bash
# job.sh OUT TASK COND "reality args": one task of tasks.txt under one reality condition
# (the lockstep runner's Reality options, and any --param for the controller). Transfer
# tasks start from the nominal suite's run of their source in $SUITE, so run
# ../validate.sh $SUITE first. In the args, @TASK@ is replaced by the task name and @LOG@
# by $LOG (for per-task plan files that must not overwrite the shared ones).
source "$(dirname "$0")/../env.sh"
OUT=$1 TASK=$2 COND=$3 RARGS=$4
read name m dx dz xf h plan src <<< "$(grep "^$TASK " "$EXP/tasks.txt")"
[ -n "$name" ] || { echo "no task $TASK in $EXP/tasks.txt"; exit 1; }
box=""; [ "$h" != "0" ] && box="--box $xf $h"
tr=""; [ "$src" != "-" ] && tr="--transfer-from $SUITE/$src --transfer-mode retarget"
RARGS=${RARGS//@TASK@/$TASK}
RARGS=${RARGS//@LOG@/$LOG}
run=${TASK}__$COND
mkdir -p "$OUT" "$LOG/plans_mu045"
rm -rf "${OUT:?}/${run:?}"
cd "$WS" && ROS_DOMAIN_ID=$((RANDOM % 100 + 100)) python3 \
  install/ilc_quad/lib/ilc_quad/ilc_jump_lockstep.py --robot go1 --max-trials 20 \
  --jump $dx $dz $box --param "phases:=[30,30,30]" --param margin:=$m \
  --reference-file "$EXP/plans/ref_$plan.npz" $tr $RARGS $EXTRA \
  --log-dir "$OUT" --run-name $run --summary "$OUT/$run.json" > "$OUT/$run.log" 2>&1
echo "$run done ($?)"
