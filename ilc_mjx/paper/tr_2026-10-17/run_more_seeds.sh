#!/bin/bash
# run_more_seeds.sh (2026-10-23 05:55, ube): the quad's original setup (drB_s2 it30, 0.6 plans, stage_c recipe -- the same
# protocol as tr/recipe and tr/ctl_random) at six more hardware seeds 9004-9009, recipe vs ctl_random only: the per-cell
# effect on the pre-registered success metric (no fall, |ex| <= 5 cm, |ez| <= 3 cm) was +4.7 pts [+2.1, +7.0] over 12
# cells on all four robots; the mean score's CI is wide because 5 % of the jumps (tumbles) carry ~30 % of it.
P=/home/henry/ilc_ws/log/dilc/plane; T=/home/henry/ilc_ws/src/ilc_mjx/paper/tr_2026-10-17
export DILC_FLY_JOBS=2 GPU0=0 NGPU=3
NET=$P/drB_s2/policy MEMBER=g0_plane_it30 CLR=0.03 OUT=$P/tr SEEDS="9004 9005 9006 9007 9008 9009" ARMS="recipe ctl_random" NPAR=12 \
  bash $T/stage_c.sh > $P/tr/more_seeds.out 2>&1
echo "$(date +%T) MORE_SEEDS_DONE"
