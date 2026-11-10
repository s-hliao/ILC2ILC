# Car: making the results larger (2026-11-06 ~02:35; user: "make the car results much more significant ... check rma if you have to"),
# written before any result
The paper's headline protocol is unchanged (3 iterations = 6 practice laps, the hardware lap budget). Two additions, both reported in full:

## A. Longer practice in sim (the stage's convergence)
Measured motivation: the stage's own cost was still falling at its last iteration (e.g. rubber peanut x1.5: J 2011 -> 1740 -> 1631).
Exactly the main runs' settings (PLSL plastic: practise square2fast_a45, 12 cars; SL50x120 rubber: practise fig8R100_x120, 2 cars;
pfast2_stage.sh's flags) with --iters 8 (16 practice laps), networks PPO+DR and RMA, seeds 0-2, stage seed 9001 (6 runs per tire).
Held-out peanut evaluation exactly as the main tables (ladder_eval.py: plastic a33 / rubber x120, 5 laps, 4 starts, seed 701), at
iterations 0, 3, 5, 8. Reported: change vs zero-shot at every iteration, 95 % CI over runs, e_y and pace. No selection: all reported.

## B. RMA: the stage's sensitivity through the adaptation module (runs/screen/RMA_SENS_1106.md)
hw_stage --z-path 1: the chain's policy reaction includes z_hat = phi(last 40 x [obs[:8], previous action]) -- the history buffer is
carried through the closed-loop recursion (ilc_core). Gate before any run: (i) with the phi path switched off it reproduces the
current chain exactly; (ii) its policy-reaction model matches the true reaction (car_rma_impulse.py's measure) far better than the
current one. Then RMA runs as A (iters 8, evaluated at 0, 3, 5, 8) with --z-path 1, paired with A's RMA runs (same seeds).
Adopted as the RMA stage if its it-3 held-out change is lower than A's RMA it-3 on the same seeds on both tires; otherwise reported only.

## A. RESULT (2026-11-06 ~04:30, long_score.py -> LONG_SCORE_1106.txt; held-out peanut, 3 runs per cell, every run reported)
| cell | it3 | it5 | it8 |
|---|---|---|---|
| plastic PPO+DR | -25.2 % [-28.7, -21.1] | -32.0 % [-36.2, -27.8] | -33.6 % [-39.6, -28.6] |
| plastic RMA | -9.5 % [-10.3, -8.3] | -14.7 % [-17.4, -9.8] | -15.3 % [-19.4, -9.8] |
| rubber PPO+DR | -7.7 % [-12.8, -2.6] | -7.4 % [-11.5, -2.8] | -8.2 % [-11.7, -4.2] |
| rubber RMA | -4.9 % [-7.7, -2.8] | -3.9 % [-5.5, -2.9] | -4.7 % [-7.2, -2.2] |
All runs better at every iteration. Plastic keeps improving with practice (pace 0.885 -> 0.944 for PPO+DR; lateral 4.7 -> 6.5 cm, inside the
10 cm tolerance); rubber plateaus by it3: its zero-shot is already inside tolerance on the held-out peanut (score 0.76; e_y 7.6 cm, pace
0.97), so little is left to gain there -- rubber's large gains are on the harder practised tasks (ETHZMobil -24 / -29 %, peanut x1.5 -21.7 %).

## B. RESULT (2026-11-06 ~05:10, long_score.py; 3 runs per cell paired by seed with A's RMA runs)
| cell | it3 | it5 | it8 |
|---|---|---|---|
| plastic RMA (A) | -9.5 % [-10.3, -8.3] | -14.7 % | -15.3 % [-19.4, -9.8] |
| plastic RMA + z-path | -11.4 % [-13.2, -10.5] | -17.5 % [-19.5, -14.0] | -19.2 % [-23.8, -12.7] |
| rubber RMA (A) | -4.9 % [-7.7, -2.8] | -3.9 % | -4.7 % [-7.2, -2.2] |
| rubber RMA + z-path | -5.6 % [-10.8, -0.0] | -6.0 % [-9.9, -0.1] | -6.1 % [-9.0, -2.5] |
Rule B (it3, paired): plastic -2.0 pts, rubber -0.6 pts -> lower on both tires -> PASSES: --z-path 1 is the RMA stage. The effect is
small (CIs overlap, 3 seeds); same direction at every iteration on both tires. Hardware: the RMA runbook (HANDOFF_2026-11-04.md) still
flies the old chain; switching it is flagged to the user, not done silently (hw_stage --z-path 1 needs each lap's obs and a, which the
manual backend records -- to verify on a recorded session before flying).

## C. RMA main cells with --z-path 1 (user 2026-11-06 ~10:40: "ok, run it for rma"), written before any result
Exactly the main RMA runs' settings and seeds, only --z-path 1 added (JAX on ube's CPUs):
  held-out peanut, plastic: PLSL settings, rmaN s0-2 x stage seeds 9001-9003 (9 runs)        -> ZPH_plastic_*
  held-out peanut, rubber : SL50x120 settings, rmaN s0-2 x stage seeds 9001-9006 (18 runs)   -> ZPH_rubber_*
  ETHZMobil practised      : ethz_stage.sh settings, rmaN s0-2 x 9001-9003, both tires (18) -> ZPE_*
Evaluation identical to the main tables (ladder_eval.py: held-out peanut 5 laps -- plastic a33 / rubber x120; ETHZMobil 1 lap + the
held-out peanut), it 0 vs 3. Reported next to the existing RMA cells (paired by network seed and stage seed). These become the paper's
RMA rows (the method with z-path, as the hardware will fly if the user adopts it); the old rows are reported as the fixed-z_hat version.

## D. Fixed-simulator re-score (2026-11-06 ~13:00; runs/screen_final/rescore_table.py; rescore_*.json next to the old files)
The multi-body car (mb_fiala) now floors its slip denominators at 0.1 m/s (bit-identical while every wheel rolls) and a run ends
only on a spin (yaw rate above 3 mu g / |v| for 0.1 s, F1T_SPIN_RHO=3). Every table cell re-evaluated (same policies, same seeds
and starts); practice itself ran on the old simulator (disclosed).
| cell | old simulator | fixed simulator | spin-outs (fixed) |
|---|---|---|---|
| plastic held-out peanut, PPO+DR (9) | -27.3 % [-29.4, -25.0] | -27.3 % [-29.4, -25.0] | 0 -> 0 % |
| plastic held-out peanut, RMA (9) | -13.4 % [-15.5, -11.6] | -13.4 % [-15.4, -11.6] | 0 -> 0 % |
| rubber held-out peanut, PPO+DR (18) | -5.3 % [-6.8, -3.9] | -5.3 % [-6.9, -4.0] | 0 -> 0 % |
| rubber held-out peanut, RMA (18) | -4.2 % [-5.9, -2.5] | -4.2 % [-6.0, -2.6] | 0 -> 0 % |
| rubber peanut x1.5 practised, PPO+DR (3) | -21.7 % [-30.8, -11.0] | -21.7 % [-30.8, -11.0] | 0 -> 0 % |
| rubber peanut x1.5 practised, RMA (3) | -3.8 % [-4.8, -3.2] | -3.8 % [-4.8, -3.2] | 0 -> 0 % |
| rubber fig8 x1.4, PPO+DR / RMA (3 each) | -0.1 % / -1.4 % | -0.1 % / -1.4 % | 0 -> 0 % |
| plastic peanut x1.2, PPO+DR / RMA (3 each) | -0.1 % / -2.1 % | -0.1 % / -2.1 % | 0 -> 0 % |
| plastic square2fast x1.2 (VOID, PFAST_RULES) | -0.3 % / -0.4 % | -0.3 % / -0.4 % | RMA 2.1 % both its |
ETHZMobil (36 runs): only 13 runs' policies are on matcha -- on those (4 / 3 / 3 / 3 runs per cell) plastic PPO+DR -16.9 % (old
-14.9 % on all 9; the old simulator failed 4.2 % of its laps, the fixed one 0.0 -> 0.5 %), plastic RMA +1.7 %, rubber PPO+DR -25.4 %,
rubber RMA -30.3 %. All 36 re-scored on ube (ilc2real_fix/ethz_rescore_fix.sh, after zp_rescore_fix.sh): RESULT pending.
Also fixed before use: ilc2real_fix/ladder_eval.py imported ../ilc2real's (unfixed) simulator through a hard-coded path; now its own.
D (cont., 14:30): ALL 36 ETHZMobil runs re-scored on ube (fixed simulator): plastic PPO+DR -17.7 % [-20.2, -15.3] 9/9 (old -14.9 %; the
old simulator failed 4.2 % of its laps, the fixed 0.0 -> 0.5 %), plastic RMA +1.0 % (n.s., unchanged), rubber PPO+DR -24.0 % [-26.1, -21.9],
rubber RMA -29.0 % [-31.5, -26.0] (unchanged). Held-out peanut after ETHZ practice: unchanged (plastic +2.7 / +5.1 %, rubber -1.7 / +6.0 %).

## C. RESULT (14:30; fixed simulator, ladder_eval as the main tables; paired by seeds with the old RMA rows)
| RMA + stage | old chain (fixed z_hat) | + z-path (C) |
|---|---|---|
| plastic, held-out peanut (9 runs) | -13.4 % [-15.4, -11.6] | -17.2 % [-19.9, -14.6], 9/9, 108/108 cells |
| rubber, held-out peanut (18 runs) | -4.2 % [-6.0, -2.6] | -5.7 % [-7.5, -4.1], 18/18 |
| plastic, ETHZMobil practised (9) | +1.0 % [-3.5, +6.0] | -0.3 % [-2.7, +2.2] (n.s.) |
| rubber, ETHZMobil practised (9) | -29.0 % [-31.5, -26.0] | -24.3 % [-28.5, -20.4] |
Per C these are the paper's RMA rows (z-path); the fixed-z_hat rows are reported as the variant. The z-path is better on the held-out
peanut (both tires) and on plastic ETHZ, worse on rubber ETHZ (-24.3 vs -29.0 %).

## Faster rubber ETHZMobil x1.1 RESULT (ETHZ_RULES.md; fixed simulator, spin rule; 9 runs per network; RMA with --z-path 1)
practised x1.1: PPO+DR -12.1 % [-15.5, -8.8] 9/9 (e_y 18.6 -> 16.3 cm), RMA -19.8 % [-24.2, -15.4] 9/9 (17.5 -> 14.0 cm), no spin-outs;
held-out peanut x1.2 after it: PPO+DR -4.4 % [-5.8, -2.7], RMA -8.7 % [-11.5, -5.7]. (x1.0 above: -24.0 / -29.0 %.)
