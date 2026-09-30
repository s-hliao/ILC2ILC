#!/bin/bash
# validate.sh OUT: the 8-task Go1 suite (tasks.txt) with the node's defaults, JOBS runs at a
# time, then the report. Easy tasks learn from scratch; the harder boxes start by transfer
# (retarget) from an easier run that converged, in OUT. EXTRA="--param NAME:=VALUE ..." is
# passed to every run.
#
# tasks.txt columns: name margin dx dz box_x_front box_height plan transfer_source
source "$(dirname "$0")/env.sh"
OUT=${1:?usage: validate.sh OUT}
mkdir -p "$OUT"
export OUT
job() {
  local name=$1 m=$2 dx=$3 dz=$4 xf=$5 h=$6 plan=$7 src=$8 box="" tr=""
  rm -rf "${OUT:?}/${name:?}"
  [ "$h" != "0" ] && box="--box $xf $h"
  [ "$src" != "-" ] && tr="--transfer-from $OUT/$src --transfer-mode retarget"
  cd "$WS" && ROS_DOMAIN_ID=$((RANDOM % 100 + 100)) python3 \
    install/ilc_quad/lib/ilc_quad/ilc_jump_lockstep.py --robot go1 --max-trials 20 \
    --jump $dx $dz $box --param "phases:=[30,30,30]" --param margin:=$m \
    --reference-file "$EXP/plans/ref_$plan.npz" $tr $EXTRA \
    --log-dir "$OUT" --run-name $name --summary "$OUT/$name.json" > "$OUT/$name.log" 2>&1
  echo "$name done ($?)"
}
export -f job
T=$EXP/tasks.txt
sed -n 1,4p "$T" | xargs -P "$JOBS" -L 1 bash -c 'job "$@"' _     # from scratch
sed -n 5,7p "$T" | xargs -P "$JOBS" -L 1 bash -c 'job "$@"' _     # from the easy runs
sed -n 8,8p "$T" | xargs -P "$JOBS" -L 1 bash -c 'job "$@"' _     # from b50_20_m100
python3 "$EXP/suite_report.py" "$OUT"
