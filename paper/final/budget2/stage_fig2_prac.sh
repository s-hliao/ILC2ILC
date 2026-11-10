#!/bin/bash
# stage_fig2_prac.sh (2026-11-08): pull the practised-plan scores into fig2_src_prac (ube: SB sb_practised.json; Lambda: fig2_practised /
# gps_practised.json), then build car_fig2_prac(.json, rel) and the figures.
cd "$(dirname "$0")"
U=henry@nsh1609server5.lan.local.cmu.edu; L=henry@172.26.86.229
for d in fig2_src_prac/ube/SB_*; do b=$(basename $d); scp -q $U:ilc_ws/src/ilc_f1tenth/ilc2real_fix/runs/screen/$b/sb_practised.json $d/ 2>/dev/null; done
for d in fig2_src_prac/lam/fig2fadaN?_* fig2_src_prac/lam/fig2ppo_*; do b=$(basename $d); scp -q $L:ilc_ws/src/ilc_f1tenth/ilc2real/runs/fig2base/$b/fig2_practised.json $d/ 2>/dev/null; done
for d in fig2_src_prac/lam/GPS*; do b=$(basename $d); scp -q $L:ilc_ws/src/ilc_f1tenth/ilc2real/runs/screen/$b/gps_practised.json $d/ 2>/dev/null; done
python3 car_fig2_prac_data.py fig2_src_prac car_fig2_prac.json && python3 budget_figs2.py car_fig2_prac.json car_fig2prac
FIG2_REL=1 python3 car_fig2_prac_data.py fig2_src_prac car_fig2_prac_rel.json > /dev/null && python3 budget_figs2.py car_fig2_prac_rel.json car_fig2pracrel
