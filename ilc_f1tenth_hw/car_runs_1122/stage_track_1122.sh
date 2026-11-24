#!/bin/bash
# stage_track_1122.sh ppodr|rma rubber|plastic fig8r100|sq41free [DAY]   (2026-11-22, written from matcha; RUNBOOK_1122.md)
# = stage_track_1116.sh (unchanged, kept) with the 11-22 arms and two settings:
#   PPO + rubber  + fig8r100 = mocap_fig8R100_b0_free_x095_rot25   (the new rotated R1.0 figure-8; TRACK OFFSET 0 0 for this track)
#   RMA + plastic + sq41free = mocap_sq41R170CW_b0_free             (track offset 0.5 0, as on 11-08)
#   PACE=<p>   -> hw_stage --pace-target p (user 2026-11-22: "pace target change is fine"): the zero-shot's own pace from
#                 offset_score_1122.py, REQUIRED for both arms (one rule; RMA diverged with the plan's pace in every sim screen, PPO's
#                 figure-8 improved more with it: 27 -> 11-12 cm repeatable vs 13-17 with the plan's pace, x0.45, 3 seeds)
#   HSCALE=<x> -> hw_stage --horizon-scale x, from: python ~/ros2_ws/src/ilc_f1tenth/ilc2real/horizon_scale.py $S/res_<zero-shot>.npz
#   ITERS defaults to 3 (= 6 real laps, the lap budget); every other argument is stage_track_1116.sh's (= the sim screens').
# Sim screens behind this (Lambda / matcha ~/car_screen_1122, 3 stage seeds, 5-lap evals, friction calibrated on the real zero-shots):
#   see RUNBOOK_1122.md.
NETA=$1; TIRE=$2; TRACK=$3; DAY=${4:-$(date +%m%d)}
case "$NETA" in ppodr|freePPON) NET=freePPON;; rma|rmaN) NET=rmaN;; *) echo "usage: stage_track_1122.sh ppodr rubber fig8r100 | rma plastic sq41free [DAY]"; exit 1;; esac
source ~/car_runs/car_env.sh $TIRE $DAY >/dev/null || exit 1
case "$TIRE/$TRACK" in
  rubber/fig8r100|plastic/fig8r100) G=mocap_fig8R100_b0_free_x095_rot25; A="mocap_fig8R100_b0_free_x095_rot25 mocap_fig8R115_b0_free"
    echo "!!! fig8r100: the car's track offset must be 0 0 (it was designed shifted; at +0.5 the sim zero-shot leaves the tracked area by 0.16 m)";;
  plastic/sq41free|rubber/sq41free) G=mocap_sq41R170CW_b0_free; A="mocap_sq41R170CW_b0_free mocap_fig8R115_b0_free";;
  *) echo "usage: stage_track_1122.sh ppodr rubber fig8r100 | rma plastic sq41free [DAY]"; exit 1;;
esac
[ -n "$PRINT_PLAN" ] && { echo $G; exit 0; }
S=${SESSION:-$S}; OUT=${OUTROOT:-runs/hw}/real_${NET}_${TIRE}_${TRACK}_1122
HSCALE=${HSCALE:-1}; [ "$HSCALE" != 1 ] && OUT=${OUT}_h$HSCALE
PACE=${PACE:-1.0}; [ "$PACE" != 1.0 ] && [ "$PACE" != 1 ] && OUT=${OUT}_p$PACE
[ "$HSCALE" = 1 ] && echo "!!! HSCALE not set: size it from the zero-shot: python horizon_scale.py \$S/res_<zero-shot>.npz"
{ [ "$PACE" = 1.0 ] || [ "$PACE" = 1 ]; } && { echo "!!! both arms need PACE=<zero-shot pace> (offset_score_1122.py; one rule for both, 11-22 screens); nothing started"; exit 1; }
[ -d "$S" ] || { echo "!!! session folder $S does not exist (start car_runner.sh $TIRE $DAY on the car first): nothing started"; exit 1; }
if pgrep -f 'hw_stage\.py' >/dev/null; then echo "!!! a hw_stage.py already runs (one stage at a time)"; exit 1; fi
[ -e "$OUT" ] && [ -z "$REPLAY" ] && { echo "!!! $OUT exists: this stage was started before"; exit 1; }
bash $HOME/car_runs/place_view/ensure_view.sh
echo ">>> stage: $NETA, $TIRE, practice $G, session $S, out $OUT, horizon scale $HSCALE, pace target $PACE, iters ${ITERS:-3}${REPLAY:+, replaying $REPLAY}"
exec python -u hw_stage.py --backend manual --session-dir $S --cars real --policy runs/${NET}_s0/policy_final.npz \
  --out $OUT --goals $G --iters ${ITERS:-3} --objective task --goal-cycle 1 \
  --trial-laps 2 --horizon-scale $HSCALE --pace-target $PACE --anchor-w 10 --anchor-plans $A --fit-realize 0.9 \
  --fit-max-steps 8000 --seed 9002 --launch 1.0 8.0 --save-its 1 --beta 0.5 --delta 0.05 --cap 0.8 --fit-stall 0 --slope-w 1 \
  --rollback 0.7 --tr 1 --tr-lo 0.25 --tr-hi 0.75 --tr-max 0.8 --crash-rollback 1 \
  --err-scale 0.10 1e9 1e9 1e9 1e9 1.0 --eval-runs 0 --skip-below 0.5 $([ $NET = rmaN ] && echo --z-path 1) ${REPLAY:+--replay $REPLAY}
