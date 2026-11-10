#!/bin/bash
# make_figs.sh (2026-11-08): rebuild figures 1, 2, 4, 5, 6 from their data builders (budget2/*_data.py, fig6/fig6_retarget.py) into
# paper/final/. Figure 1 / 2 read staged copies (budget2/fig1_src, fig2_src: re-stage from ube / Lambda if their runs change); figures 4 / 5 / 6
# read matcha's log/dilc/plane directly. Figure 3 (car flow) and 7 (hardware) have their own generators.
cd "$(dirname "$0")/budget2" || exit 1
python3 car_fig1_data.py fig1_src/runs car_fig1.json && python3 budget_figs2.py car_fig1.json car_fig1
python3 car_fig2_data.py fig2_src car_fig2.json > /dev/null && python3 budget_figs2.py car_fig2.json car_fig2
FIG2_REL=1 python3 car_fig2_data.py fig2_src car_fig2_rel.json > /dev/null && python3 budget_figs2.py car_fig2_rel.json car_fig2rel
python3 quad_fig4_data.py quad_fig4.json && python3 budget_figs2.py quad_fig4.json quad_fig4
python3 quad_fig5_data.py quad_fig5.json && python3 budget_figs2.py quad_fig5.json quad_fig5
python3 ../fig6/fig6_retarget.py
# (2026-11-08 evening) PRACTICE-ONLY versions (user: practice-only car protocol; fig 2 = sim-to-sim only): figure 1 zero-shot on the practised
# plans (fig1_eval_prac.py), figure 2 on the practised plans (stage_fig2_prac.sh pulls sb_/fig2_/gps_practised.json), behaviour panel.
cd "$(dirname "$0")/budget2" || exit 1
python3 car_fig1_prac_data.py fig1_src/runs car_fig1_prac.json && python3 budget_figs2.py car_fig1_prac.json car_fig1prac
bash stage_fig2_prac.sh
# behaviour (sideslip) panel, sim, practice-only cells: collect_beh.py over every machine's runs/screen/PRACB_*/ladder_launch.json ->
# behaviour/car_behaviour_runs.json (staged copy), then:
(cd "$(dirname "$0")/behaviour" && python3 fig_behaviour.py car_behaviour_runs.json car_behaviour_sim)
