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
