# Car step-size screen (2026-11-06 ~13:30; user: "why do the first steps of the car show so little improvement? maybe the steps are
# too small" -> "yes"), written before any result

## Measured motivation (stage logs, summary.json of the main runs)
- The trust radius starts at --cap 0.1 (rms of the normalized action change) and only doubles after an iteration whose measured /
  predicted J reduction rho > 0.75, up to --tr-max 0.4; 3 iterations never reach 0.4. Rubber ETHZMobil: median rho 1.26 at it 1
  (97 % of cars > 0.75): the model was right, the radius conservative. Plastic / rubber fig8: median rho 0.59-0.83, 11-24 % < 0.25.
- The refit realizes only part of the (capped) commanded step: median 49-59 % at it 0 on rubber, 18-23 % on the plastic headline
  (--fit-realize 0.9 is never reached; the fit stops when the realized fraction gains < 0.003 per 1000 steps).
- Predicted J change of the capped step at it 0: -9 % (rubber ETHZ) to -28 % (plastic ETHZ). That the 0.1 cap binds at it 0 is
  INFERRED (an uncapped beta = 0.5 GN step would predict roughly -50 to -75 %), not measured.

## Arms (everything else exactly the main runs' flags; the fixed simulator ilc2real_fix, F1T_SPIN_RHO=3, for all three)
- S0: current (--cap 0.1)
- SA: --cap 0.4 (the trust region starts at its maximum; rho < 0.25 still halves it, rho < 0 still rejects the step)
- SB: SA + --fit-stall 0 (the refit runs to --fit-max-steps 8000 unless it reaches --fit-realize 0.9)
## Cells (PPO+DR, network seeds 0-2 x stage seeds 9001-9002 = 6 runs per arm and cell, 36 runs)
- rubber ETHZMobil x1.0 (ethz_stage.sh settings, 1-lap trials, 2 rubber cars) -> STP{0,A,B}_ethz_rubber_freePPON_s*_ss*
- plastic held-out protocol (car_long.sh / PLSL settings: practise square2fast_a45, 12 plastic cars) -> STP{0,A,B}_plastic_freePPON_s*_ss*
## Evaluation (as the main tables, the fixed simulator, it 0 vs 3)
- rubber: ladder_eval.py on ethzmobil43_b0_free (1 lap) and the held-out peanut x120 (5 laps)
- plastic: ladder_eval.py on the held-out peanut a33 (5 laps), 12 cars
Statistic: per run the mean over its cars of the % score change it 3 vs it 0 (car_final_score.py's); paired by seeds against S0.
## Rule
An arm (SA or SB) qualifies if its mean change is lower than S0's on BOTH cells (rubber: the practised ETHZMobil score; plastic: the
held-out peanut) and its spin-out share at it 3 is not higher than S0's. If both qualify: the lower mean over the two cells. If one
is adopted, the user is asked before every car cell is rerun with it (it changes the method's step settings). If neither: the
current settings stay and this screen is reported as is.

## RESULT (2026-11-06 16:20; fixed simulator; 6 runs per arm and cell, paired by seeds; scored on ube)
| cell | S0 (cap 0.1) | SA (cap 0.4) | SB (cap 0.4 + full refit) |
|---|---|---|---|
| rubber ETHZMobil practised | -21.9 % [-23.4, -20.4] | -35.8 % [-38.1, -32.6] | -34.2 % [-38.2, -29.4] |
| plastic held-out peanut | -28.4 % [-30.6, -25.7] | -36.3 % [-42.0, -30.7] | -31.9 % [-34.8, -29.1] |
| spin-outs at it 3 (rubber / plastic) | 0.0 / 0.0 % | 0.0 / 0.3 % | 0.0 / 0.0 % |
| paired vs S0 (rubber / plastic) | | -13.9 [-16.5, -10.9] / -7.9 [-15.6, -0.8] pts | -12.3 [-16.6, -7.6] / -3.5 [-8.5, +0.4] pts |
(rubber ETHZ runs on the held-out peanut after practice: S0 -2.1 %, SA -0.0 %, SB +0.2 %; n.s. differences, reported.)
RULE: SA is lower than S0 on both cells but its spin-out share at it 3 is higher on plastic (0.3 % vs 0.0 %: one spin in ~288
car-runs) -> does NOT qualify. SB is lower on both cells with no more spin-outs -> QUALIFIES -> SB is the rule's pick. Adopting it means
rerunning every car stage cell with --cap 0.4 --fit-stall 0: asked to the user before anything is rerun (rule).
