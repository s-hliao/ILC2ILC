# Practise the peanut fast (2026-11-05), written before any result
User: "peanut magnitude is not large at all ... make it larger, might have to run faster". The speed ladder showed held-out gains fade
above the practised speed; ETHZMobil showed large gains where the car PRACTISES the hard track. So: practise the peanut itself at the fastest
ladder rung with a large zero-shot error and no spin-outs: plastic mocap_peanutM_b0_free_a33_x110 (zero-shot e_y 10.1 cm, score 1.63),
rubber mocap_peanutM_b0_free_x130 (13.9 cm, 1.39). pfast_stage.sh = the plastic / rubber stage's exact args, practised plan changed,
2-lap trials; PPO+DR and RMA, network seeds 0-2, stage seed 9001 (12 runs). F1T_DEV_TIME=1. A PRACTISED-track result for figures and
videos; the paper's held-out claim stays on the existing runs (the peanut is held-out there). Evaluation: ladder_eval.py on the practised
plan, 5 laps, it 0 vs 3; every run reported; the video uses the median run.

## Results, first peanut rungs (plastic x1.10, rubber x1.30; 3 runs per cell, ss9001; eval_score.py pfast_eval.json; 2026-11-06 00:40)
| cell | score | change [95 % CI over runs] | runs better | e_y | pace |
|---|---|---|---|---|---|
| plastic PPO+DR | 1.63 -> 1.48 | -10.5 % [-14.1, -6.3] | 3/3 | 10.1 -> 10.8 cm | 0.839 -> 0.856 |
| plastic RMA | 1.38 -> 1.30 | -7.1 % [-11.0, -3.6] | 3/3 | 8.3 -> 8.7 cm | 0.862 -> 0.871 |
| rubber PPO+DR | 1.39 -> 1.20 | -13.9 % [-18.7, -9.9] | 3/3 | 13.9 -> 12.0 cm | 0.910 -> 0.941 |
| rubber RMA | 0.95 -> 0.85 | -11.4 % [-17.5, -0.5] | 3/3 | 9.5 -> 7.9 cm | 0.926 -> 0.937 |
Practising the peanut itself at these speeds gives a smaller change than ETHZMobil practice (-24 / -29 %); plastic gain is pace again.

## Results, rule rungs (PFAST2; 3 runs per cell, ss9001; eval_score.py pfast2_eval.json; 2026-11-06 ~01:45)
| cell | score | change [95 % CI over runs] | runs better | e_y | pace |
|---|---|---|---|---|---|
| plastic square2fast x1.2, PPO+DR | 2.93 -> 2.93 | -0.3 % [-0.6, +0.0] | 2/3 | 16.0 -> 16.0 cm | 0.715 -> 0.715 |
| plastic square2fast x1.2, RMA | 2.84 -> 2.83 | -0.4 % [-0.8, -0.1] | 3/3 | 11.3 -> 11.3 cm | 0.758 -> 0.759 |
| plastic peanut x1.2, PPO+DR | 2.24 -> 2.25 | -0.1 % [-4.6, +3.0] | 1/3 | 18.8 -> 20.5 cm | 0.798 -> 0.814 |
| plastic peanut x1.2, RMA | 1.99 -> 1.96 | -2.1 % [-2.2, -2.0] | 3/3 | 14.2 -> 15.2 cm | 0.808 -> 0.815 |
| rubber fig8 x1.4, PPO+DR | 1.62 -> 1.61 | -0.1 % [-6.0, +4.7] | 1/3 | 16.2 -> 16.1 cm | 0.868 -> 0.884 |
| rubber fig8 x1.4, RMA | 1.37 -> 1.36 | -1.4 % [-13.3, +12.7] | 2/3 | 11.5 -> 13.4 cm | 0.865 -> 0.890 |
| rubber peanut x1.5, PPO+DR | 3.15 -> 2.50 | **-21.7 % [-30.8, -11.0]** | 3/3 | 31.5 -> 25.0 cm | 0.816 -> 0.850 |
| rubber peanut x1.5, RMA | 2.19 -> 2.11 | -3.8 % [-4.8, -3.2] | 3/3 | 21.9 -> 21.1 cm | 0.819 -> 0.840 |
Plastic square2fast x1.2 is VOID as a stage test: on 11/12 cars per run the network after the stage is bit-identical to zero-shot. Cause,
measured (hw_stage F1T_GN_DUMP reproduction, n04_real_nom): every trial has 0 valid samples -- the car launched from rest tops out at
2.42 m/s and the plan never goes below 2.84 m/s, so --skip-launch 0.5 (the approved hardware rule) never starts the trial's data; the
fit then runs on an empty set (loss NaN, J 0). The rung rule (fastest without spin-outs) picked a speed the plastic car cannot reach.

## Corrected rung rule, plastic square2fast (2026-11-06 ~02:05), written before checking any rung
The rung must also be REACHABLE: the fastest rung (x1.05, x1.10, x1.15) at which every zero-shot trial (PPO+DR s0 and RMA s0, all 12
plastic cars, launched from rest as the stage does: v0 0.6 m/s, launch controller to 1.0 m/s at 8 A, 2 laps, sensing noise) keeps at least
50 % of its samples after --skip-launch 0.5's cut, and has no spin-out. That rung is practised (6 runs: PPO+DR and RMA, network seeds 0-2,
ss9001; pfast2_stage.sh) and scored like the others. If none qualifies, x1.0 (the existing PLSL result) stands and no rerun is made.
- Corrected-rule result (logs/rung_reach.py / rung_reach100.py, 2026-11-06 ~02:20): NO rung qualifies. x1.05: ax:mu=0.4667 / 0.4000
  keep 0 % (PPO+DR; RMA 0.4000 too); x1.10 and x1.15 worse. Even x1.0 fails the 50 % bar for one cell (PPO+DR ax:mu=0.4000: 0 % kept).
  Per the rule x1.0 stands, no rerun. Consequence checked in the main plastic stage runs (PLSL_*): PPO+DR's ax:mu=0.4000 network is
  bit-identical after the stage in 2 of 3 runs (every other car / network changed): that car contributes a zero change there (dilutes,
  does not inflate, the plastic gain). Disclose. Rubber fig8 x1.4: the stage lowered its own cost J (e.g. 1293 -> 849) through the speed
  row (pace 0.868 -> 0.884), lateral flat (practice laps 20.9 -> 21.8 cm), so the e_y-bound score does not move: pace, not tracking.
