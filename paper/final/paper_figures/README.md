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
- Pogo real (2026-11-11, user: "yes, best iteration rule on pogo as well"): pogo_budget/pogo_best_iter.py budget_data.json ->
  budget_data_best.json (budget_figs.py draws the real panel from it). Rule PER TIER PANEL (as stated to the user before the decision):
  each DistIL-C_hw curve stops at its lowest point. Stops: RMA + hw at 0 on A / B (0.13 / 0.20; last 0.30 / 0.37), at 12 on C (0.53; last
  0.66); PPO + DR + hw A 24 (0.30; last 0.32), B / C 36; DistIL-C_sim + hw A / B 24 (B last 0.89), C 36; + DR A 12, B 24, C 36. NOTE the
  car fig 2 rule pools the three tiers (one stop per arm); the pogo rule is per tier.
- Fig 3 (2026-11-11, user: "rebuild flow models"): fig3/car_flow_ethzx110_orbit_1s_w + car_flow_ethzx110_resid (+ caption_ethzx110*.tex)
  replace the 11-06 figure (moved to fig3/old_1106/: the pre-fix simulator and the x1.0 plan). Computed on ube with the FIXED simulator
  (FLOW_D=ilc2real_fix, F1T_SPIN_RHO=3, F1T_DEV_TIME=1; ~/flow_fig3_1111/run.sh) for the x1.1 plan of the ETHZX110 stage runs; example =
  the median of the 9 PPO+DR runs by score change (ETHZX110_rubber_freePPON_s2_ss9002, -11.1 %). Drawn locally: car_flow/flow_car.py
  ethzx110_orbit_1s / ethzx110_resid (data npz in car_flow/). MEASURED: line 9.4 -> 8.6 cm, perturbed laps 23.3 -> 20.5 cm rms, every
  perturbed lap still passes 0.5 m once in every row; nine runs (car ax:mu=1.667): PPO+DR -11.8 %, RMA -19.9 % (old figure: -24.0 / -29.0 %).
- Fig 5 (2026-11-11, user): nominal-stance panel only (perturbed stances are fig 6's subject); legend at the right (quad_fig5s_data.py
  "figw" 7.0, "legend_right" 0.64). Same data and best-iteration rule as before.
- Fig 6 (2026-11-11, user: "remove our stage+rma on the quadruped from the graphs"): the RMA + DistIL-C_hw row is removed; RMA stays as a
  zero-shot row (fig6_retarget_srb.py ZS_ONLY). Fig 5 never had a quad RMA + stage curve (RMA zero-shot point only).
- Quad HARDWARE evaluation (2026-11-10, deployment PC ~/quad_hw/srb_1110/eval_nominal; for the paper's hardware table): zero-shot PPO + DR
  (mem/ppo_m0/it300) vs the SRB stage's it4 network, 4 held-out goals (0.40 / 0.50 flat, 0.45 / 5 cm, 0.575 / 15 cm), 1 jump per goal per
  network, nominal stance: mean normalized error 1.31 -> 0.80 (-0.51; 4 of 4 goals better; 0 falls either network). Per goal 0.53 -> 0.51,
  0.99 -> 0.96, 1.22 -> 0.63, 2.50 -> 1.10. The 8 scored jumps are VALIDATED free of mocap errors (user 2026-11-11); three earlier attempts at
  0.45 / 5 cm with stale or stalled mocap frames were voided at the time with their reasons (it1/real/void/README.txt) and re-flown.
- Fig 3 candidates (2026-11-11, user: "search all tracks for significant effect"): fig3/candidates_1111/. Search: every PPO + DR stage run
  on the fixed simulator (2,105 run x car cells, car_flow/fig3_candidates_1111.json) pre-ranked by the stage's own first-to-last trial
  rms e_y drop; flows computed for the top 36 post-fix cells (SB_* / PRACB_* / ETHZX110; no friction-patch or ablation runs; ube
  ~/flow_fig3_1111/screen2). Ranked by the perturbed laps' rms change pi_0 -> pi_3. MEASURED:
  * flow_ethz_plastic_mu04_median: SB_ethzp_freePPON_s0_ss9001, plastic car mu x0.4, practised ETHZMobil (a33 plan): perturbed laps
    101.2 -> 56.1 cm rms, laps leaving the track 4/4 -> 2/4. All 7 screened runs of this cell drop 40-50 cm; this is their MEDIAN (-45.1 cm).
  * flow_deeppeanut_plastic_harshsteer: PRACB_plastic_pnutdeep_freePPON_s1_ss9002, plastic car n04_harsh_steer, practised deep peanut
    (mocap room, 0.5 m tube): 40.7 -> 25.3 cm rms, laps leaving the tube 4/4 -> 0/4. Best of its cell's 9 runs (-12 to -15 cm); disclose.
  The flow's zero-push line moves < 1 cm in both: the stage changes how the car recovers from offsets, not where it settles.
  Policy panels (2026-11-11): candidates_1111/policy_*_{steer,current} = pi_k - pi_0 on the flow grid (policy_resid.py on ube's fixed
  simulator; flow_car.py ethzx110_resid, FLOW_RESID_CH=current for the motor current). ETHZ plastic mu x0.4: steering barely changes
  (0.3 deg mean, max 1.6) -- the stage cuts the drive current before the corners (2.0 A mean, max 8 A): it slows the low-grip car into the
  turns. Deep peanut harsh-steer: both channels (steering 0.9 deg mean, max 6.8; current 3.6 A mean).
  (2026-11-11, user: "track deviation is fine, keep the run going, we can just report time outside the track"): flow_car.py now draws each
  perturbed lap in full (points outside the track / tube as x at the frame edge) and each row reports the time outside, summed over the 4
  perturbed laps: ETHZ plastic 14.8 -> 10.2 -> 8.8 -> 5.5 s; deep peanut see the row titles. Videos: videos/*_1111.mp4.
