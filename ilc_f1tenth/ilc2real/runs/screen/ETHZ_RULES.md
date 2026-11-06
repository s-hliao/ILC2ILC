# ETHZMobil practise study (2026-11-05), written before any result
User: "i need more extreme example ... run training on harder track", "don't stop the trial at 0.5 m deviation, just mark it as deviation
time", "ETHZMobil Track", "use shorter trials (i.e. halve it)". Replaces the ORCA study (ORCA at 4.3x needs more steering than the
F1TENTH has: planned radius 0.54 m vs its 0.93 m minimum turn; all ORCA results void, see the session notes).
- Track: BayesRace ETHZMobil centreline x4.3 (ethzmobil43): 12.9 x 7.7 m, 55.3 m, centreline radius >= 1.28 m; planned radius >= 1.11 m,
  steering at its limit 2.4 % (plastic plan) / 5.2 % (rubber plan) of the lap. Sim only.
- Zero-shot (PPO+DR seed 0, 4 starts): plastic score 1.26 (rms e_y 12.6 cm; peanut 0.87, 2.8 cm); rubber 2.19 (21.9 cm; peanut x1.2
  0.77, 7.7 cm). The plans reach 5.55 m/s, the training plans 3.88 m/s.
- Stage: ethz_stage.sh = the plastic / rubber stage's exact args, practised plan ethzmobil43 (plastic _free_a33, rubber _free), ONE-lap
  trials (the stage's two halved); PPO+DR and RMA, network seeds 0-2, stage seed 9001: 12 runs. F1T_DEV_TIME=1.
- Evaluation: ladder_eval.py on the practised ethzmobil43 plan (1 lap) and the held-out peanut (5 laps; plastic _a33, rubber _x120),
  it 0 vs 3. Reported for every run: score change (bootstrap CI over runs), rms e_y, pace, time outside the tube, spin-outs. Practised-track
  result, labelled as such; a figure example is the median run unless stated otherwise.

## Faster rubber ETHZMobil (2026-11-06 ~12:05; user: LLA-MPC runs this track ~25-30 % faster), written before any result
LLA-MPC (arXiv 2505.19512, ORCA 1:43, friction ~0.9) laps ETHZMobil in 5.7-6.0 s = 11.8-12.4 s / 4.4-4.7 m/s mean at x4.3 (Froude); our
plan (solved at mu 0.6) is 15.5 s / 3.57 m/s, the rubber cars have mu 1.0 / 1.1. Zero-shot check (fixed simulator: mb_fiala slip floor;
spin = the yaw rate above 3 x mu g / |v| for 0.1 s, F1T_SPIN_RHO=3; off-track / sliding is fine): rubber cars, PPO+DR and RMA s0-s2,
ETHZMobil x1.0 / x1.1 / x1.2 / x1.3, 4 starts (ladder_eval's), 1 lap. Rung rule: the fastest rung at which neither network spins on any
lap; that rung's stage runs (ETHZ settings, 9 runs per network) follow, scored as the ETHZ table.
- Zero-shot rung check (12:05, ethz_rungs.py, 24 laps per cell: 3 network seeds x 2 cars x 4 starts): spins PPO+DR / RMA: x1.0 0 / 0,
  x1.1 0 / 0, x1.2 13 / 0, x1.3 21 / 8. rms e_y (non-spin laps): x1.0 22.6 / 25.1, x1.1 18.6 / 17.5, x1.2 21.0 / 16.4, x1.3 25.6 / 19.6 cm;
  pace x1.0 1.07, x1.1 1.02, x1.2 0.96, x1.3 0.89. Rule -> x1.1 (lap 14.1 s, 3.93 m/s mean; LLA-MPC's pace ~ x1.25-1.3 is not reachable
  zero-shot without spins). Stage runs ETHZX110_*: ETHZ settings, 9 per network (s0-2 x ss9001-9003), RMA with --z-path 1, the fixed
  simulator and the spin rule (ube, ilc2real_fix), scored as the ETHZ table (x1.1 plan 1 lap; held-out peanut x1.2 5 laps).
