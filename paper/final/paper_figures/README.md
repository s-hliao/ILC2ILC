# Paper figures 1-6, final versions (moved here 2026-11-10 ~00:30 from paper/final/)

| fig | files | what | built |
|---|---|---|---|
| 1 | fig1/car_fig1prac_sim | car, zero-shot error vs simulated hours (practice-only protocol; DistILC+DR retrained with the screen-J settings) | 11-09 02:11 |
| 2 | fig2/car_fig2prac_real, car_fig2pracrel_real | car, sim-to-sim adaptation vs laps (absolute / relative to zero-shot) | 11-09 02:44 |
| 3 | fig3/car_flow_ethz_orbit_1s_w, car_flow_ethz_resid (+ caption_ethz.tex, caption_ethz_resid.tex, copied) | car flow on the ETHZ track: what the stage changes | 11-06 00:10 |
| 4 | fig4/quad_fig4_sim | quad, zero-shot error vs simulated hours | 11-08 08:11 |
| 5 | fig5/quad_fig5_real | quad, sim-to-sim adaptation vs jumps | 11-08 08:11 |
| 6 | fig6/fig6_retarget | quad, retargeting onto perturbed stances: PPO+DR / RMA / DistILC_sim (+DR) with our stage (RS), FADA with its own adaptation | 11-08 08:12 |

- Figure 7 (car hardware) has no current version. paper/final/car_hw_* are the 10-31 flights (old protocol) and stay in paper/final/.
- Not moved:
  - the car behaviour (sideslip) figure, behaviour/car_behaviour_sim;
  - the other car_flow variants;
  - the old Jacobian heatmap quad_pert/ (supplement: the stage-variant ablation).
- REBUILDS write to paper/final/, not here: make_figs.sh, budget2/budget_figs2.py, fig6/fig6_retarget.py and car_flow/flow_car.py. After a rebuild, move or copy the new files into this folder.
- Open for fig 6: the RMA-oracle reference (user 11-07) is not in it, and its stage rows are RS (the hardware now flies the SRB update).

## 2026-11-10 changes (user)
- Fig 6: the nominal-stance block was removed; only "score from the practised stance" remains. fig6/fig6_retarget.py now draws one
  panel; the two-panel script is fig6_retarget.py.bak_1110_twopanel.
- Figs 1 / 2: the plastic and rubber tire panels were removed (user). Fig 1 = A / B / C (budget2/car_fig1_prac_notire.json).
  Fig 2 = the coverage-tier runs split into A / B / C panels like fig 1 (user: "why are A-C all in the same figure"):
  `python3 car_fig2_prac_tiers_data.py fig2_src_prac car_fig2_prac_tiers.json && python3 budget_figs2.py car_fig2_prac_tiers.json car_fig2prac`
  (FIG2_REL=1 -> car_fig2_prac_tiers_rel.json -> car_fig2pracrel). budget_figs2.py gained optional "figw" / "legend_right" keys.
  FIG 2 IS A PLACEHOLDER: its "DistILC" curve is the pre-retrain (11-07, _preJ) network; the 8-iteration reruns on ube
  (ilc2real_fix/sbX_one.sh, sbX_fix.sh; best-iteration rule) replace every DistILC-stage curve. make_figs.sh still builds the old versions.
- Fig 4: our two arms are being rerun from the method's own start (log/dilc/plane/fig4s_job.sh, arms oursS / oursSdr); the figure
  is unchanged until they finish.
