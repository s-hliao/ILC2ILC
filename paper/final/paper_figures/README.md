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
- Fig 7 (2026-11-10): fig7/fig7_car_hw = the car hardware stage, the six NO-braking 5-lap evaluations of 11-08 / 11-09 (re-scored with
  car_score.py; equal to car_results_stage/README.md). Built by car_hw7/fig7_car_hw.py. Braking-on flights (1109_plastic after ~08:16) are
  excluded (user). One evaluation per condition; plastic flight-to-flight spread ~+-0.2-0.5 (README caveat).
- Fig 4 (2026-11-10, PLACEHOLDER): DistILC arms = the from-scratch runs (budget2/quad_fig4s_data.py -> quad_fig4s.json; oursS / oursSdr).
  Only the checkpoints scored on all 12 units are drawn (it30 both arms, it10 DR so far); it 0 / 5 / 10 / 15 are being scored. Rebuild:
  `python3 quad_fig4s_data.py quad_fig4s.json && python3 budget_figs2.py quad_fig4s.json quad_fig4`, copy quad_fig4_sim.* here.
- Fig 5 (2026-11-10): best-iteration rule applied (budget2/fig5_best_iter.py quad_fig5.json quad_fig5_best.json, then budget_figs2.py
  quad_fig5_best.json quad_fig5): PPO + DR + DistIL-C_hw stops at 12 jumps (nominal 2.17 -> 1.26; at 36 jumps it was 1.64, state it in the
  caption). The DistIL-C_sim (+ DR) + DistIL-C_hw rows (stage_xfix31.sh, 8 iterations, Lambda) are added when they finish.
- Fig 4 (2026-11-10 18:45, COMPLETE data): every DistILC checkpoint scored (it 0 / 5 / 10 / 15 / 30, 3 seeds x 4 robots). MEASURED: the
  from-scratch start (it 0, the plan's feedforward; 0 h, not drawable on the log axis) scores 3.05 nominal, and every sim-trained
  checkpoint is WORSE on the deployment robots (5.27 / 5.73 at 1.5 h) while the training-sim success rises to 81-86 %. The rise is mostly
  falls: 0-4 % at it 0 -> 9-31 % at it 30 per robot (r1 / r4 most); landing distance changes little. Cause not yet inspected (trajectories).
- Fig 5 (2026-11-10 19:30, SRB REBUILD, supersedes the at-real entry above): every DistIL-C_hw arm uses the hardware's update (stage_xfix32.sh:
  deploy_fm.py RS flags + --jac-at srbm --fall-mask 0.3, 8 iterations x 6 jumps, 3 stage seeds x 4 robots). Built by budget2/quad_fig5s_data.py
  -> quad_fig5s.json -> fig5_best_iter.py -> quad_fig5s_best.json -> budget_figs2.py quad_fig5s_best.json quad_fig5. Best-iteration rule:
  PPO + DR + DistIL-C_hw 2.17 -> 1.37 at 24 jumps (last, 48 jumps: 1.50); DistIL-C_sim + DistIL-C_hw best = its start 5.27 (48 jumps: 5.44);
  + DR 5.73 -> 5.36 at 48. y caps raised to 7 / 7.5 so the DistIL-C_sim arms are drawn, not clipped. MEASURED cause of the DistIL-C_sim arms:
  sim training raises landing pitch error (9.5 -> 28 deg on surviving landings) and falls (1 -> 22 %); the stage's GN steps are ~0 there.
- Pogo (2026-11-10 21:00): pogo/pogo_budget_{sim,real} = pogo_budget/budget_figs.py on pogo_budget/budget_data.json (made by
  ~/ilc_ws/log/pogo_servo/budget_data.py, copy in pogo_budget/). Frozen test pass (seed 4301), 3 seeds. ALIGNED with car / quad:
  sim FADA = the RMA-data-matched FADA (fada_pogo_tf.py, oracle teacher, it 25/50/100/200; hours = RMA teacher 428.5 h + FADA's own data,
  it200 = RMA's 513.8 h). Real: FADA (from PPO + DR) / (from RMA) + adaptation (no-teacher fadaNPp / fadaNRp, LoRA T1-T3, K = 0..384),
  RMA + DistIL-C_hw (hw_stage_rma.py), PPO + DR + GPS (gps_pogo.py, 30 rollouts per robot per iteration, it 0/1/2/4/8/12).
  MEASURED: GPS gets worse on test (A 0.64 -> 1.25 at 360); its log shows the C-step failing on many conditions (eta at the 1e16 cap,
  KL up to 100-2000x the bound, predicted costs 1e9-1e16: the time-varying linear fit through the impacts diverges). FADA adaptation
  flat. RMA + DistIL-C_hw: A 0.13 -> 0.30 and B 0.20 -> 0.37 (worse), C 1.11 -> 0.53 at 12. No best-iteration truncation on pogo.
- Fig 6 (2026-11-10 21:05, SRB REBUILD): fig6/fig6_retarget_srb.py (supersedes fig6_retarget.py's at-real rt_rs_* rows): every
  DistIL-C_hw row = the hardware's update (retarget_job_srb.sh, 6 iterations = 36 practice jumps; stage_fl/rt_srb_<net>_<P>), DistIL-C_sim
  rows = fig 4's from-scratch oursS_s0 / oursSdr_s0. MEASURED: PPO + DR + hw improves 4 / 5 stances (-0.17 .. -0.85, CI excl. 0), nose up
  +0.02; RMA + hw worse on front block (+0.84*) and crouched (+0.75*), better on tall (-0.89*); DistIL-C_sim + DR + hw large drops from
  a poor start (15.4 -> 6.2 front block). Final iteration shown (no mid-iteration evals for fig 6, so no best-iteration truncation).
- Fig 2 (2026-11-10 21:05, 8-ITERATION REBUILD): budget2/car_fig2_x8_data.py (-> car_fig2_x8.json / FIG2_REL=1 car_fig2_x8_rel.json ->
  budget_figs2.py ... car_fig2prac / car_fig2pracrel). DistILC-stage arms = ube SB_tier_<net>_s<s>_ss9001_x8 (8 iterations, 3 seeds; "ours"
  = the RETRAINED network, not _preJ; synced by budget2/sync_fig2_x8.sh to budget2/fig2_src_x8). Selection rule written before the results:
  stop at the lowest mean over all 36 tier cars. MEASURED (pooled means, it 0 -> 8): PPO + DR + hw 0.835 -> 0.612 and RMA + hw 0.568 ->
  0.372, both still falling at 8 (no stop); DistIL-C_sim + hw flat 0.661 -> 0.664 (stops at 0); + DR flat 0.446 -> 0.444 (stops at 3).
