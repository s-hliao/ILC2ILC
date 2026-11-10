#!/bin/bash
# fig2J_finish3.sh (= finish2 with the re-pull skipping BOTH our arms: ube now holds the no-DR J runs under the published names)
# fig2J_finish2.sh (2026-11-09 ~02:20; replaces fig2J_finish.sh): figure 2 swaps in ONLY the retrained DistILC+DR (screen-J settings, ube
# SB_<panel>_oursdr_s<s>_ss9001, 9 runs); DistILC (no DR) stays the published network (J's settings made its tier-C zero-shot unstable,
# 0.59 -> 1.13; it never trains on low grip). Published DistILC+DR SB dirs -> fig2_src_prac/ube_preJ. Then rebuild figure 2.
cd "$(dirname "$0")"; U=henry@nsh1609server5.lan.local.cmu.edu; R=ilc_ws/src/ilc_f1tenth/ilc2real_fix
until [ "$(ssh -o BatchMode=yes $U "grep -c 'done [a-z]* oursdr ' $R/logs/sbJ/progress.txt 2>/dev/null")" = 9 ]; do sleep 120; done
mkdir -p fig2_src_prac/ube_preJ
for p in plastic rubber tier; do for s in 0 1 2; do b=SB_${p}_oursdr_s${s}_ss9001
  [ -d fig2_src_prac/ube/$b ] && [ ! -d fig2_src_prac/ube_preJ/$b ] && mv fig2_src_prac/ube/$b fig2_src_prac/ube_preJ/
  mkdir -p fig2_src_prac/ube/$b; scp -q $U:$R/runs/screen/$b/summary.json $U:$R/runs/screen/$b/sb_practised.json $U:$R/runs/screen/$b/args.json fig2_src_prac/ube/$b/
done; done
sed 's#^for d in fig2_src_prac/ube/SB_\*; do#for d in fig2_src_prac/ube/SB_*; do case $d in *_ours_*|*_oursdr_*) continue;; esac;#' stage_fig2_prac.sh > stage_fig2_prac_J.sh
bash stage_fig2_prac_J.sh > fig2J_build.log 2>&1; echo "$(date +%H:%M) figure 2 rebuilt" >> fig2J_build.log
