#!/bin/bash
# stage_track.sh ppodr|rma rubber|plastic fig8|square [DAY] (2026-11-08, written from matcha; RUNBOOK_PRACTICE_1108.md): stage_run.sh for ONE
# network on ONE practice track, the runbook's SB command unchanged (3 iterations x 2-lap trials, --cap 0.4 --fit-stall 0, RMA --z-path 1,
# --skip-launch 0.5), with the track's plan as G, its own anchors A and its own output folder runs/hw/real_<NET>_<TIRE>_<TRACK>_1108.
#   rubber  fig8   mocap_fig8R100_b0_free_x120    A = square2fast_x120 + fig8R100_x120      (grip)
#   rubber  square mocap_square2fast_b0_free_x120 A = square2fast_x120 + fig8R100_x120      (grip, peak 9.4 m/s^2 < rubber 9.81)
#   plastic square mocap_square2fast_b0_free      A = square2fast + figfast (nominal)       (sliding needed; WAIT for matcha's go)
#   plastic fig8   mocap_fig8R100_b0_free         A = fig8R100 + square2fast (nominal)      (sliding needed; WAIT for matcha's go)
# Anchors = exactly the simulation checks' (SBNOM / SBTWO). Restarting a started stage: stage_resume.sh logic is not wired here -- ask matcha.
NETA=$1; TIRE=$2; TRACK=$3; DAY=${4:-$(date +%m%d)}
case "$NETA" in ppodr|freePPON) NET=freePPON;; rma|rmaN) NET=rmaN;; *) echo "usage: stage_track.sh ppodr|rma rubber|plastic fig8|square [DAY]"; exit 1;; esac
source ~/car_runs/car_env.sh $TIRE $DAY >/dev/null || exit 1
case "$TIRE/$TRACK" in
  rubber/fig8)    G=mocap_fig8R100_b0_free_x120;    A="mocap_square2fast_b0_free_x120 mocap_fig8R100_b0_free_x120";;
  rubber/square)  G=mocap_square2fast_b0_free_x120; A="mocap_square2fast_b0_free_x120 mocap_fig8R100_b0_free_x120";;
  plastic/square) G=mocap_square2fast_b0_free;      A="mocap_square2fast_b0_free mocap_figfast_b0_free";;
  plastic/fig8)   G=mocap_fig8R100_b0_free;         A="mocap_fig8R100_b0_free mocap_square2fast_b0_free";;
  *) echo "usage: stage_track.sh ppodr|rma rubber|plastic fig8|square [DAY]"; exit 1;;
esac
S=${SESSION:-$S}; OUT=${OUTROOT:-runs/hw}/real_${NET}_${TIRE}_${TRACK}_1108
[ -d "$S" ] || { echo "!!! session folder $S does not exist (start car_runner.sh $TIRE $DAY on the car first): nothing started"; exit 1; }
if pgrep -f 'hw_stage\.py' >/dev/null; then echo "!!! a hw_stage.py already runs (one stage at a time)"; exit 1; fi
[ -e "$OUT" ] && { echo "!!! $OUT exists: this stage was started before (ask matcha to resume it)"; exit 1; }
echo ">>> stage: $NETA, $TIRE, practice $G, session $S, out $OUT"
exec python -u hw_stage.py --backend manual --session-dir $S --cars real --policy runs/${NET}_s0/policy_final.npz \
  --out $OUT --goals $G --iters ${ITERS:-3} --objective task --goal-cycle 1 \
  --trial-laps 2 --anchor-w 20 --anchor-plans $A --fit-realize 0.9 \
  --fit-max-steps 8000 --seed 9001 --launch 1.0 8.0 --save-its 1 --beta 0.5 --delta 0.05 --cap 0.4 --fit-stall 0 --slope-w 1 --tr 1 \
  --err-scale 0.10 1e9 1e9 1e9 1e9 0.24 --eval-runs 0 --skip-launch 0.5 $([ $NET = rmaN ] && echo --z-path 1)
