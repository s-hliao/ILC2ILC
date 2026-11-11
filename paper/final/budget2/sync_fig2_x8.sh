#!/bin/bash
# sync_fig2_x8.sh: the 8-iteration tier SB runs (ube runs/screen/SB_tier_<net>_s<s>_ss9001_x8: summary.json + sb_practised.json) -> fig2_src_x8/ube
cd "$(dirname "$0")"
for net in freePPON rmaN ours oursdr; do for s in 0 1 2; do d=SB_tier_${net}_s${s}_ss9001_x8; mkdir -p fig2_src_x8/ube/$d
  rsync -a henry@nsh1609server5.lan.local.cmu.edu:/home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_fix/runs/screen/$d/{summary.json,sb_practised.json} fig2_src_x8/ube/$d/ 2>/dev/null; done; done
