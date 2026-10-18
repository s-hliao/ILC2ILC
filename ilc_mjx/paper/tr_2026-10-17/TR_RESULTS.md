# LM trust region on the quadruped (2026-10-17, machine ube)

## Question
Does ONE adaptive step rule replace `drB_hw24`'s hand-set step knobs?
- The knobs: gn-beta 0.3, step-rms 0.05, the anchor, the bold ×1.25 / ×0.5 heuristic, the sim gate.
- The rule (`deploy.py --tr 1`): a per-goal Levenberg–Marquardt trust region on the full GN step (gain 1).
  - rho = (J_prev − J) / (J_prev − J_pred), with J_pred from the landing sensitivity times the commanded step.
  - rho < 0.25 halves the radius; rho < 0 rolls the goal back to its best trial; rho > 0.75 doubles it (max 0.4).
  - No anchor, no bold, no gate.

## Setup
- `tr_arms.sh`; validation goals (planeval, seed 301).
- 6 training goals × 4 iterations = 24 jumps per robot; 4 robots (real_r1, s1, r4, r5); 3 hardware seeds (9001–9003).
- Starting network: drB_s2 it30, fixed robots (overlay).
- ube reproduces matcha: the first iteration matches `fixbox/drB_hw24`'s log number for number.

## Results
Score = 0.01 × landing cost + 20 per fall; **lower is better**. The 95 % CI is a bootstrap over the 12 (seed, robot)
cells.

| arm | final | 95 % CI | per seed | perturbed | falls (12 runs) |
|---|---|---|---|---|---|
| zero-shot (start) | 21.30 | | | 49.89 | |
| `recipe` (drB_hw24's knobs) | **19.74** | [13.8, 25.7] | 18.0 / 22.3 / 18.9 | 48.22 | 21 |
| `tr` (radius from 0.05) | 20.30 | [14.0, 26.2] | 22.4 / 19.1 / 19.3 | 48.96 | 25 |
| `tr_r20` (radius from 0.20) | 21.83 | [14.5, 29.3] | 22.8 / 22.3 / 20.4 | 49.34 | 19 |

Paired differences against the recipe (same seed and robot):
- tr: **+0.56 [−1.9, +3.6]**;
- tr_r20: **+2.09 [−1.8, +6.4]**.

Per robot, mean over seeds (zero-shot → recipe / tr / tr_r20):

| robot | zero-shot | recipe | tr | tr_r20 |
|---|---|---|---|---|
| r1 | 30.2 | 29.0 | 27.2 | 32.2 |
| s1 | 7.0 | 6.2 | 5.2 | 6.0 |
| r4 | 13.8 | 13.2 | 17.0 | 15.2 |
| r5 | 34.3 | 30.5 | 31.9 | 33.9 |

**rho, the trust region's own diagnostic:**
- median −0.16 (tr) and −0.04 (tr_r20);
- rho < 0 in 52–53 % of the 133–141 per-goal steps;
- rho > 0.75 in 20–22 %.

## Reading
- **No step rule matters on the quad, because the stage itself barely moves the score.**
  - The recipe improves the zero-shot score by 1.6 (21.3 → 19.7) with a CI that contains the zero-shot value.
  - The trust region is level with the recipe within CI, from either radius.
- **The cause is visible in rho.** In more than half of the steps, a jump scored WORSE than the previous one at that
  goal, although the model predicted an improvement.
  - Either the landing model's predicted change is wrong in direction on these robots, or the jump-to-jump noise is as
    large as the predicted improvement.
  - The earlier diagnosis supports the second: most robot-goals are 2–4 cm near-misses inside trial noise, and a few
    are 17–58 cm misses the model cannot fix (r5's far / box goals).
  - Positivity, the theory's key assumption, does not hold reliably at the cost level here. That is why the trust
    region keeps rejecting or shrinking, and why the gate on matcha rejected most full steps.
- **What would help** is not a step rule but:
  - noise handling: more than 1 jump per goal per iteration (`--reps 2`), or ρ averaged over reps before
    accepting;
  - a better landing sensitivity: the "jac-at real" / corrected-model options.

  The quad hardware stage needs a measurement-noise fix before its step size is even identifiable.
- **The honest statement for the paper:**
  - on the quad, the LM rule with textbook constants is as good as the hand-tuned recipe;
  - neither is significantly better than zero-shot on the validation goals with 24 jumps;
  - the per-step rho shows why.

## Noise floor and averaging (`noise_arms.sh`; validation goals, 4 robots)

**Noise floor** (`noise5_sd9001`: 5 jumps per goal of the SAME policy at iteration 0, then one trust-region step on
their means):
- jump-to-jump spread of J: median sd **3.0**;
- predicted improvement of the step: median **2.7**;
- sd / predicted improvement: median **0.9**. One jump cannot tell whether a step helped.
- 5 of the 24 robot-goals have no rho: falls, or the goal held. Two are dominated by falls: r1 g(0.575, 0.15) J 135 ± 83;
  r4 / r5 g(0.5, 0.15) J 16-21 ± 17-19.
- rho on 5-jump means: median **+0.24**; rho < 0 for **37 %** (single jumps: −0.16 and 53 %). Averaging helps but does
  not rescue it.
  - Even on means, the measured improvement is about a quarter of the predicted one.
  - Over a third of the steps go the wrong way, while the sd of a 5-jump mean (≈ 1.35) is half the predicted gain.
  - **The landing model over-predicts systematically: model error, not only noise.**

**Matched budget, 24 jumps** (final score, lower is better; zero-shot 21.30; 3 seeds × 4 robots):

| arm | final | 95 % CI | perturbed | falls |
|---|---|---|---|---|
| recipe, 1 jump × 4 iterations | 19.74 | [13.8, 25.7] | 48.22 | 21 |
| recipe, 2 jumps × 2 iterations | 20.51 | [13.9, 27.3] | 47.71 | 24 |
| trust region, 1 × 4 | 20.30 | [14.0, 26.2] | 48.96 | 25 |
| trust region, 2 × 2 | 19.27 | [12.9, 25.8] | 47.63 | 30 |
| (trust region, 5 × 2 = 60 jumps, 1 seed) | 19.56 | | | |

Paired differences:
- recipe 2×2 − 1×4: +0.8 [−1.0, +2.7];
- trust region 2×2 − 1×4: −1.0 [−4.2, +1.7];
- trust region − recipe at 2×2: −1.2 [−4.0, +0.9].

**Conclusion for the quadruped.**
- Averaging jumps does not help at equal budget, and 60 jumps do no better than 24.
- The hardware stage moves the validation score by 1-2 points out of 21 with any step rule or batching. The limit is
  the landing sensitivity's accuracy on these robots: it realizes about 25 % of the predicted improvement and has the
  wrong sign for about 1/3 of the goals, even on 5-jump means. Trial noise is a second, comparable factor.
- The fix has to be on the model side:
  - the Jacobians at the robot's measured states (`--jac-at real`) or the corrected sensitivity (`--sens-adapt`);
  - or, for the long misses, plans the robots can reach.

  Not on the step size.
