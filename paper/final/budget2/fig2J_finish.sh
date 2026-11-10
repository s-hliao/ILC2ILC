#!/bin/bash
# fig2J_finish.sh (2026-11-09 ~01:40): when ube's 18 sbJ runs (figure 2, retrained DistILC / DistILC+DR) are scored, swap them into
# fig2_src_prac/ube (published ones -> fig2_src_prac/ube_preJ) and rebuild figure 2 (stage_fig2_prac.sh).
cd "$(dirname "$0")"; U=henry@nsh1609server5.lan.local.cmu.edu; R=ilc_ws/src/ilc_f1tenth/ilc2real_fix
until [ "$(ssh -o BatchMode=yes $U "grep -c 'done' $R/logs/sbJ/progress.txt 2>/dev/null")" = 18 ]; do sleep 120; done
mkdir -p fig2_src_prac/ube_preJ
for p in plastic rubber tier; do for n in ours oursdr; do for s in 0 1 2; do b=SB_${p}_${n}_s${s}_ss9001
  [ -d fig2_src_prac/ube/$b ] && [ ! -d fig2_src_prac/ube_preJ/$b ] && mv fig2_src_prac/ube/$b fig2_src_prac/ube_preJ/
  mkdir -p fig2_src_prac/ube/$b; scp -q $U:$R/runs/screen/$b/summary.json $U:$R/runs/screen/$b/sb_practised.json fig2_src_prac/ube/$b/
done; done; done
bash stage_fig2_prac.sh > fig2J_build.log 2>&1; echo "$(date +%H:%M) figure 2 rebuilt" >> fig2J_build.log
