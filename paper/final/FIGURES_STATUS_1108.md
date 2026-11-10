# Figures, state at 2026-11-09 00:26 (behaviour refreshed: 414 runs) (practice-only car protocol; hardware figures excluded)

| figure | file (paper/final/) | data | state |
|---|---|---|---|
| 1 car, zero-shot vs sim hours | car_fig1prac_sim | fig1_eval_prac + fadaR1; DistILC+DR = RETRAINED 2026-11-09 (screen-J settings: cap 0.5, max_ey 0.6, speed scale 0.6, friction DR x0.5-1.2, 90 its; old -> fig1_src/runs/fig1_preJ); DistILC = published | REBUILT 02:25 |
| 2 car, sim-to-sim adaptation | car_fig2prac_real, car_fig2pracrel_real | SB / FADA NP,NR / PPO-on-car / GPS re-scored on the practised plans; DistILC+DR SB rerun on the retrained network (ube sbJ, 9 runs; fig2J_finish3.sh rebuilds) | REBUILDING (~04:30) |
| car behaviour (sideslip), sim | behaviour/car_behaviour_sim (+ _table.txt) | 362 PRACB runs, 16 cells | DONE; refresh_behaviour.sh re-renders as cells finish |
| 3 car flow | car_flow*.pdf | 11-05/06 | unchanged (methods illustration) |
| 4 quad, zero-shot | quad_fig4_sim | log/dilc/plane | built 08:11 (peer session) |
| 5 quad, adaptation | quad_fig5_real | log/dilc/plane | built 08:11 (peer session) |
| 6 quad retargeting heatmap | fig6_retarget | stage_fl | built 08:12 (peer session) |
| pogo | pogo_*.pdf, method_pogo_ilc.pdf | 10-29/30 | unchanged |
| 7 car hardware | car_hw_*.pdf | 10-31 flights | EXCLUDED tonight (hardware) |
Superseded held-out versions of car figs 1/2: superseded_heldout/. Build: make_figs.sh (bottom: practice-only + behaviour).
Still filling (behaviour figure only; numbers move only slightly): last plastic oval foam-turn / deep-peanut seeds, rubber seed-1/2 deep peanut + ETHZ (matcha / Lambda, ~2-3 h). Plastic big fig8 COMPLETE.
