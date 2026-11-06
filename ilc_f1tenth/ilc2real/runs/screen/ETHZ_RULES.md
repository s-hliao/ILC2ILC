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
