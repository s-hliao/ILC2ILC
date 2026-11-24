#!/bin/bash
# check_offset_1122.sh <plan> (2026-11-22, RUNBOOK_1122.md "Track offset per track"), ON THE WORKSTATION.
# Refuses (exit 1) when the car's RUNNING net_runner (car_runner.sh) has a track offset other than the one this plan needs:
#   mocap_fig8R100_b0_free_x095_rot25 -> 0.0 0.0  (designed shifted; at 0.5 it leaves the tracked area)
#   mocap_sq41R170CW_b0_free          -> 0.5 0.0  (as on 11-08)
# Other plans: no rule, exit 0. car_runner.sh reads ~/car_runs/track_offset.txt only when it starts: change the file, then restart it.
# Called by car_eval.sh and stage_track_1122.sh; SKIP_OFFSET_CHECK=1 bypasses it.
PLAN=$1
case "$PLAN" in
  mocap_fig8R100_b0_free_x095_rot25) WANT="0.0 0.0";;
  mocap_sq41R170CW_b0_free) WANT="0.5 0.0";;
  *) exit 0;;
esac
[ -n "$SKIP_OFFSET_CHECK" ] && { echo "!!! track offset check skipped (SKIP_OFFSET_CHECK)"; exit 0; }
R=$(ssh -o ConnectTimeout=6 car 'for p in $(pgrep -f "[n]et_runner.py"); do tr "\0" " " < /proc/$p/cmdline; echo; done; echo "FILE $(head -1 ~/car_runs/track_offset.txt)"' 2>/dev/null)
[ -z "$R" ] && { echo "!!! cannot reach the car to check the track offset (SKIP_OFFSET_CHECK=1 to go anyway)"; exit 1; }
FILE=$(echo "$R" | sed -n 's/^FILE //p')
RUN=$(echo "$R" | grep -o 'track_offset:=\[[^]]*\]' | head -1 | sed 's/track_offset:=\[//; s/\]//; s/,/ /')
same() { awk -v a="$1" -v b="$2" 'BEGIN{split(a,x," "); split(b,y," "); exit !((x[1]-y[1])^2 + (x[2]-y[2])^2 < 1e-6)}'; }
if ! same "$FILE" "$WANT"; then
  echo "!!! $PLAN needs track offset $WANT; the car's track_offset.txt says '$FILE'. Fix it on the car (first line), restart car_runner.sh."; exit 1
fi
if [ -z "$RUN" ]; then
  echo ">>> track offset: file '$FILE' OK for $PLAN; car_runner.sh is not running yet (it will read it at start)"; exit 0
fi
if ! same "$RUN" "$WANT"; then
  echo "!!! the RUNNING car_runner.sh uses track offset '$RUN', $PLAN needs $WANT (file says '$FILE'): restart car_runner.sh"; exit 1
fi
echo ">>> track offset $RUN OK for $PLAN (running net_runner and track_offset.txt agree)"
