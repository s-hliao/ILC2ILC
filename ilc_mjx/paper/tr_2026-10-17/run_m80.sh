#!/bin/bash
# run_m80.sh (2026-10-23 06:00, ube): fix D -- the quad's plans with BOTH margins the diagnosis asks for: friction cone 0.45
# (positivity: r5 / r1 no longer slip) and torque margin 0.8 (authority: the heavy r5, mass x1.14, saturated the 0.9-margin
# box plans). One deployment-wide plan bank (bank_m80.json), sized for the DR family's edge (friction >= 0.5, mass <= 1.15),
# no per-robot identification; drB_s2 it30 unchanged. Stage arms recipe / ctl_random x 3 seeds x 4 robots.
P=/home/henry/ilc_ws/log/dilc/plane; T=/home/henry/ilc_ws/src/ilc_mjx/paper/tr_2026-10-17
export DILC_FLY_JOBS=${DILC_FLY_JOBS:-2} GPU0=0 NGPU=3
NET=$P/m80/drB_s2/policy MEMBER=g0_plane_it30 CLR=0.03 OUT=$P/m80/stage_drB ARMS="${ARMS:-recipe ctl_random}" NPAR=${NPAR:-12} \
  bash $T/stage_c.sh > $P/m80/stage_drB.out 2>&1
echo "$(date +%T) RUN_M80_DONE"
