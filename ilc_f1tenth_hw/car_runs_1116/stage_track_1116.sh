#!/bin/bash
# stage_track_1116.sh ppodr|rma rubber|plastic peanut125|fig8x095 (or the older peanut120|fig8x090) [DAY] (2026-11-16, written from matcha; user: "see this one happen on
# rubber in hardware and then have a continuous drift one on plastic"): stage_track_v2.sh's hardware command with the settings of the
# sim screens that converged uniformly (ilc2real_ubefix/screen_hwplan_1116{,b}.sh = the BIGSTEP peanut run's args):
#   --iters 4 --cap 0.8 --tr-max 0.8 --anchor-w 10 --err-scale 0.10 1e9 1e9 1e9 1e9 1.0   (v2: 3 iters, cap 0.4, anchor 20, pace 0.24)
# Masking as on hardware since 11-08 (--skip-below 0.5). Sim results (stand-in cars, 3 stage seeds, lap rms e_y per iteration):
#   rubber  peanut120  mocap_peanutM_b0_free_x120   mu x1.667: 8.9/9.3/8.9 -> 5.9/5.0/4.4 cm, monotone; clean replay 8.1 -> 3.5 cm
#   plastic fig8x090   mocap_fig8R115_b0_free_x090  mu x0.4  : 32/31/32 -> 19/14/14 cm, monotone, no failures
# 2026-11-19 (later): rubber peanut150 = mocap_peanutM_b0_free_x150 (user: much larger initial deviation): sim, final network,
#   3 seeds: 34 -> 14-20 cm lap rms, peak 52 -> 22-40 cm, pace kept (0.79 -> 0.75-0.79); ilc2real_ubefix/screen_track_1119.sh
# 2026-11-19 speed ladder (ilc2real_ubefix/screen_ladder_1116.sh, same args), now the runbook's tracks:
#   rubber  peanut125  mocap_peanutM_b0_free_x125   mu x1.667: 12.1/11.9/11.8 -> 6.1/10.2/5.3 cm (peak accel 9.3 m/s^2; x1.3 needs 10.1, spun out at 10.6)
#   plastic fig8x095   mocap_fig8R115_b0_free_x095  mu x0.4  : 46.1/46.0/46.8 -> 30.5/12.7/18.3 cm
# 2026-11-20: plastic fig8r100 = mocap_fig8R100_b0_free_x095_rot25 (R 1.0 m figure-8 at x0.95, rotated 25 deg + shifted to fit the
#   mocap room; fig8R115 leaves the room). Sim, ax:mu=0.4, replay with the 8 A launch: lap1 45 -> 17/19/24 cm, peak 87 -> 27/52/35 cm;
#   every replayed path (zero-shot included) stays >= 0.34 m inside the tracked area x [-2.69, 2.22], y [-2.86, 2.80]
# (peanut x1.5 is infeasible on the real car: 5.55 m/s, 13.5 m/s^2; fig8R115 full speed: plastic leaves the 1.0 m envelope zero-shot)
NETA=$1; TIRE=$2; TRACK=$3; DAY=${4:-$(date +%m%d)}
case "$NETA" in ppodr|freePPON) NET=freePPON;; rma|rmaN) NET=rmaN;; *) echo "usage: stage_track_1116.sh ppodr|rma rubber|plastic peanut150|peanut125|fig8r100|fig8x095 [DAY]"; exit 1;; esac
source ~/car_runs/car_env.sh $TIRE $DAY >/dev/null || exit 1
case "$TIRE/$TRACK" in
  rubber/peanut150)  G=mocap_peanutM_b0_free_x150;  A="mocap_square2fast_b0_free_x120 mocap_fig8R100_b0_free_x120";;
  rubber/peanut125)  G=mocap_peanutM_b0_free_x125;  A="mocap_square2fast_b0_free_x120 mocap_fig8R100_b0_free_x120";;
  plastic/fig8r100)  G=mocap_fig8R100_b0_free_x095_rot25; A="mocap_fig8R100_b0_free_x095_rot25 mocap_fig8R115_b0_free";;
  plastic/fig8x095)  G=mocap_fig8R115_b0_free_x095; A="mocap_fig8R115_b0_free_x095 mocap_fig8R115_b0_free";;
  rubber/peanut120)  G=mocap_peanutM_b0_free_x120;  A="mocap_square2fast_b0_free_x120 mocap_fig8R100_b0_free_x120";;
  plastic/fig8x090)  G=mocap_fig8R115_b0_free_x090; A="mocap_fig8R115_b0_free_x090 mocap_fig8R115_b0_free";;
  *) echo "usage: stage_track_1116.sh ppodr|rma rubber peanut150|peanut125 | plastic fig8r100|fig8x095 [DAY]"; exit 1;;
esac
[ -n "$PRINT_PLAN" ] && { echo $G; exit 0; }
S=${SESSION:-$S}; OUT=${OUTROOT:-runs/hw}/real_${NET}_${TIRE}_${TRACK}_1116
[ -d "$S" ] || { echo "!!! session folder $S does not exist (start car_runner.sh $TIRE $DAY on the car first): nothing started"; exit 1; }
if pgrep -f 'hw_stage\.py' >/dev/null; then echo "!!! a hw_stage.py already runs (one stage at a time)"; exit 1; fi
[ -e "$OUT" ] && [ -z "$REPLAY" ] && { echo "!!! $OUT exists: this stage was started before"; exit 1; }
echo ">>> stage: $NETA, $TIRE, practice $G, session $S, out $OUT${REPLAY:+, replaying $REPLAY}"
exec python -u hw_stage.py --backend manual --session-dir $S --cars real --policy runs/${NET}_s0/policy_final.npz \
  --out $OUT --goals $G --iters ${ITERS:-4} --objective task --goal-cycle 1 \
  --trial-laps 2 --anchor-w 10 --anchor-plans $A --fit-realize 0.9 \
  --fit-max-steps 8000 --seed 9002 --launch 1.0 8.0 --save-its 1 --beta 0.5 --delta 0.05 --cap 0.8 --fit-stall 0 --slope-w 1 \
  --rollback 0.7 --tr 1 --tr-lo 0.25 --tr-hi 0.75 --tr-max 0.8 --crash-rollback 1 \
  --err-scale 0.10 1e9 1e9 1e9 1e9 1.0 --eval-runs 0 --skip-below 0.5 $([ $NET = rmaN ] && echo --z-path 1) ${REPLAY:+--replay $REPLAY}
