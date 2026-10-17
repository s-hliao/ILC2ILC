# DistILC on a toy 2-D pogo stick (2026-10-13)

Plan and design: `PLAN.md`. Figures: `src/paper/pogo/figures/` (`figures.py`). Animations: `src/paper/pogo/videos/`
(`animate_pogo.py`). Full table: `results_table.txt` (`table.py`). The v1 arms (pre-fix defaults) and the tuning
sweeps are in `archive_v1/hw/`.

## Setup in one paragraph

**The pogo stick** is the simplest 2-D model: a SLIP.
- A 10 kg point mass rides on a massless spring leg (k = 4000 N/m, l0 = 0.5 m).
- The foot is pinned in stance (perfect friction).
- Controls: a thrust (the spring's rest length pushed out, 0–8 cm) and the leg angle in flight.
- One network is conditioned on the goal (forward speed v\* in [0, 1.5] m/s, apex height h\* in [0.55, 0.75] m).

**The nominal model** (JAX, GPU, 0.2 ms steps) has no leg damping.
- The touchdown's leg-angle gradient enters through straight-through saltation terms.

**The six "real" sticks** are a separate numpy simulator at 0.1 ms with leg damping 15 N s/m, plus each robot's own
perturbation:

| robot | perturbation |
|---|---|
| nom | none |
| mass | ×1.2 |
| spring | ×0.8 |
| act | thrust ×0.75 |
| offset | leg angle +0.05 rad |
| prog | progressive spring |

- Sensing is idealized: exact state, no delay.
- Inside the DR family: nom, mass, spring. Outside it: act, offset, prog.

**Error rows and evaluation**
- ILC error rows are time-local and in flight only: the ballistic apex predictor z + vz²/2g − h\*, and vx − v\*.
- Test: 8 reserved goals (seed 701) × 4 perturbed drops × 10 s (~20 hops), on 6 robots = 192 runs per arm.
- Success = no fall, and from hop 3 on, RMS apex error ≤ 2 cm and |mean speed error| ≤ 0.1 m/s.
- Real trials are 3 s drops at the 6 training goals.

## Main result (`fig1_methods`, `fig2_budget`)

Success over the 6 robots, test goals (± about 6 points at n = 192).

| method | zero-shot | +6 | +12 | +24 | +48 real trials / robot |
|---|---|---|---|---|---|
| Raibert controller (classical) | 4.2 | | | | |
| per-goal classical ILC (on its own 6 goals) | | 13.9 | 13.9 | 19.4 | 29.2 |
| **ours, nominal sim** (s0) | 0.0 | 58.9 | 72.4 | **79.7** | **87.0** |
| ours, nominal sim, seeds 1 / 2 | 0.0 / 2.1 | | | 77.1 / 88.0 | |
| ours + DR (A), s0 / s1 | 47.9 / 52.1 | | | 70.8 / 66.1 | |
| ours + DR (B), s0 / s1 | 52.1 / 43.8 | 56.2 | 60.4 | 63.0 / 66.7 | 64.6 |
| PPO+DR (+ our hardware stage) | 77.1 | 79.2 | 79.7 | **94.3** | 91.1 |
| our DR sim stage from the PPO net (ppoDR) | 74.0 | | | **95.8** | |
| RMA (estimated latent) | 31.8 | | | | |
| FADA (+ LoRA adaptation) | 1.6 | | | 0.0 | |

**What the table shows**
- **The nominal-sim network fails zero-shot and gets there with real trials.**
  - It starts at 0 % because the lossless nominal model knows nothing about leg damping (apex 4.5 cm low).
  - The hardware stage takes it to 80 % (3 seeds: 80 / 77 / 88) in 24 real 3-s trials, and 87 % in 48.
  - RMS apex error falls 4.45 → 0.73 cm and the speed error 0.10 → 0.044 m/s.
  - No falls anywhere.
- **The hardware stage is orthogonal to the warm start.**
  - On PPO+DR it gives 77 → 94 %.
  - Our DR sim stage started from the PPO net, then the hardware stage, gives 96 %, the best arm.
  - It fixes PPO+DR's out-of-family robots: offset 0 → 94 %, prog 75 → 100 %.
- **DR (PPO+DR, ours + DR) is strong zero-shot inside its family:** 92–96 % on nom / mass / spring. It loses outside
  it: PPO+DR's offset is 0 %.
- **Classical baselines stay low.**
  - Raibert: 4 %.
  - Per-goal ILC: 29 % at 48 trials. A time-indexed feedforward over 10 s drifts against the hop timing; J falls about
    7 % per trial against the 67 % it predicts.
- **RMA** identifies poorly: 32 %. With the oracle latent it reaches 94 % (data matching below).
- **FADA** fails here as on the car (≤ 2 %). Adaptation fits the transitions but does not move the task error.
- **Our DR (B) network has a weak spot: the heavy and soft sticks** (mass ×1.2 and spring ×0.8, both INSIDE the DR
  family), at 12 % before and after the stage.
  - Mechanism: a speed run-away. The heavy stick accelerates to 2.1 m/s at v\* = 1.2 (`videos/pogo_mass.gif`).
  - The hardware stage's positivity there is negative (fig4), so its steps are rolled back.
  - PPO+DR has no such mode (mass 100 %).

## Coverage and positivity (`fig3_coverage`, `fig4_positivity`)

- **Sweep:** 6 single-axis sweeps on real_nom, 7 values each:

| axis | range |
|---|---|
| spring | ×0.5–1.5 |
| mass | ×0.7–1.5 |
| thrust | ×0.5–1.3 |
| leg offset | ±0.1 rad |
| progressive spring | kp 0–3 |
| damping | 0–60 N s/m |

- **Mean success inside vs outside the DR range** (19 / 23 robots):

| arm | inside DR | outside DR |
|---|---|---|
| PPO+DR zero-shot | 81 % | 33 % |
| ours + DR (B) zero-shot | 79 % | 43 % |
| ours (nominal) + 24 trials | 90 % | 65 % |
| ours + DR (B) + 24 | 86 % | 63 % |
| PPO+DR + 24 (our stage) | 96 % | 72 % |
| RMA | 33 % | 20 % |
| Raibert | 8 % | 9 % |

  - **DR's zero-shot score drops by half outside its family** (the coverage assumption).
  - **The hardware stage keeps most of it**: thrust ×0.5 is 90 % for ours vs 12 % for PPO+DR. On leg offset
    −0.06 rad, ours hits 38 %, every DR arm 0 %.
  - Where ours fails (spring ×0.5, offset −0.1, mass ×1.5), so does everything else.
- **Positivity, (A1) of the thesis.** cos(G·step, measured change of the error vector) over consecutive trials:
  - positive in 97 % of the trial pairs for ours (median 0.88), 96 % for PPO+DR + stage (0.86), 90 % for DR (B)
    (0.68);
  - **it predicts what the stage gains.** Over 125 axis robots × 3 adapted arms, Spearman (median cos, success gain)
    = 0.63:

| median cos | runs | gain | success after |
|---|---|---|---|
| < 0.5 | 16 | +1 point | 12 % |
| 0.5–0.8 | 32 | +17 points | 79 % |
| ≥ 0.8 | 77 | +55 points | 92 % |

  - DR (B)'s cos goes negative exactly where it fails: mass ×1.2, offset −0.03, thrust ×0.5.
  - The flipped-G control (the 180° case) gives 0 %; open-loop G gives 5 % (median cos 0.18).

## Controls (`fig5_controls`; nominal s0, 24 trials)

| arm | success |
|---|---|
| default (full GN step, no trust region) | 79.7 |
| + trust region (anchor-w 1) | 77.1 |
| v1 defaults (beta 0.5, cap 0.05, delta 0.02, anchor-w 1) | 41.1 |
| hardware-stage open-loop G | 5.2 |
| hardware-stage flipped G | 0.0 |
| no rollback | 79.7 (identical; no rollback fired) |
| perturbed training drops / replicate seed | 80.7 / 80.7 |
| sim-stage final snapshot instead of the stopping-rule pick | 34.4 |
| sim stage without exploration | 86.5 |
| sim stage with open-loop G | 75.0 |

- **The no-exploration sim stage does better here** (86.5 vs 79.7), unlike the quadruped and car. The toy's starts
  are already perturbed drops.

## Data matching (`fig8_matching`; `ppo_real.py`, privileged critic, on the robots themselves)

| trials / robot | 0 | 24 | 96 | 384 | 768 | 1,536 | 2,000 |
|---|---|---|---|---|---|---|---|
| PPO on the robot (from PPO+DR) | 77.1 | 78.6 | 79.2 | **100** | 92.2 | 100 | 100 |
| RMA oracle latent + PPO on the robot | 93.8 | 87.5 | 99.0 | 86.5 | 89.6 | 95.8 | 95.8 |

- **Against ours:** ours reaches 80 % at 24 trials and 87 % at 48. PPO on the robot needs between 96 and 384 trials
  (4–16× ours) to pass that.
- **With exact real data and enough of it, RL reaches 100 %**, the same "RL can transfer under idealized conditions"
  check as on the car. It runs with a KL stop and lr 3e-5; without them, single updates wrecked good policies.
- **The best arm uses both:** PPO+DR + our stage reaches 94 % at 24 trials.

## Cost

| | sim hours | real trials |
|---|---|---|
| ours | ~11 (BC 1.7 + deep ILC 9.2) | 24 |
| PPO+DR | 427 | — |
| RMA | 452 | — |
| FADA | 12.8 (data) | — |

## How the method got here (v1 → v2, for the record)

1. **Touchdown gradient bug (fixed before any arm).** The foot placement had a zero derivative with respect to the leg
   angle. Fixed with straight-through saltation terms in `common.substep`; J_pred / J went from 0.8 to 0.3.
2. **The nominal sim stage degrades late.**
   - Iterations 30 → 40 weaken the speed loop: on the real sticks the speed runs away by 0.5 m/s.
   - The nominal sim score stays at 100 %.
   - A sim-only stopping rule (`sim_validate.py`: the best mean success under DR draws and over 20 s on the validation
     goals; the car's finding) picks iteration 15 and fixes it: the hardware stage then gets 41 % instead of 18 %.
3. **Hardware-stage defaults.**
   - The full GN step (beta 1, delta 0.005, cap 0.1, no trust region) beats the gentle v1 step (41 → 80 %).
   - **Caveat: this was first seen on the test goals.** It was then re-checked on the validation goals (seed 301)
     with the saved per-robot networks before adopting it. Validation success: v1 46 %, beta 1 79 %, + delta 0.005
     83 %, no trust region 90 %, both 94 % (adopted).
4. **Baselines.**
   - PPO's reward had to be defined in every phase. With the flight-only ILC rows it learned not to hop.
   - RMA needs a one-hop history (H = 50 periods). With H = 10 it got 9 %.

## Caveats

- **Sim-to-sim under idealized sensing only.** The "real" gap is damping, step size and parameters. The equations are
  the same SLIP, so (A1) is easy here by construction except where the network itself is fragile.
- **Small evaluation.**
  - About 6 points of noise at n = 192 per arm, and one PPO / RMA seed.
  - The sim-stage seed spread for ours is 77–88 % at 24 trials.
- **Ours + DR is weaker than PPO+DR as a warm start on this toy.** The DR (B) speed run-away is unexplained beyond the
  mechanism. A late-snapshot check with a long DR horizon in the rule did not flag it; it was tried and reverted.
- **Per-goal ILC is scored on its own 6 training goals**, since a feedforward exists only for those. All other arms
  are scored on the 8 reserved test goals.

## Files and commands

**Model and simulators**
- `common.py`: the model, task errors, Raibert controller, metrics.
- `pogo_jax.py`: the nominal model, Jacobians and the GN step.
- `real_pogo.py`: the real sticks, including `ax:` robots.
- `policies.py`: policy loading.

**Our method**
- `sim_train.py` (sim stage), `sim_validate.py` (stopping rule).
- `hw_stage.py` (hardware stage, including the positivity log).

**Baselines**
- `goal_ilc.py`.
- `ppo_dr.py` / `rma_np.py`.
- `fada.py` / `fada_np.py`.
- `ppo_real.py` (data matching).

**Schedulers, tables, figures**
- `scheduler.py` (main matrix) and `scheduler_axes.py` (coverage).
- `table.py`, `figures.py`, `animate_pogo.py`.

**Run commands**
- **Rerun everything** (it is re-runnable; it skips jobs whose outputs exist):
  `nohup ~/miniconda3/envs/pogo/bin/python -u scheduler.py > logs/scheduler.out &`
- **Coverage suite:**
  `POGO_STATE=scheduler_axes POGO_CPU_JOBS=8 nohup python scheduler.py --jobs scheduler_axes &`
- **Table and figures:** `python table.py > results_table.txt; python figures.py`

## Overnight 2026-10-13: fixing the sim stage, ours + DR vs PPO+DR (v3 arms in runs/hw/v3/)

**What was wrong with our DR sim stage, and the fixes (each confirmed in sim on in-family conditions first)**
1. **Drift without a check.** A regression onto model-predicted corrections can drift where the prediction is poor
   (the speed loop through the touchdown angle). Fix: the measured-cost check, `sim_train --accept`.
   - `--accept 1` (reject and halve) stalls on unlucky seeds.
   - `--accept 2` is a backtracking line search on the measured cost along the regression direction (Abbeel et al.'s
     line search, in sim). It fixed the stalls: all 3 seeds learn the heavy and soft sticks in sim.
2. **A late drift remained** at about iterations 40–50, whatever the learning-rate schedule (constant, low, no anchor:
   all drift).
   - Cause, confirmed: the sensitivity uses the network's OWN feedback, so the Jacobian co-evolves with the gains it
     shapes (the same loop that broke the VG critic on the quadruped).
   - A fixed (Raibert) feedback in the Jacobian removes the drift but learns less (DR success 22–28 %).
   - Fix: a lagged feedback, the snapshot refreshed every 20 iterations (`--fbk-refresh 20`). No drift through 80–120
     iterations; DR success 56–59 % in sim. Every 10 iterations still drifts.
3. **Stopping rule.** It now uses the mean normalized task error under DR draws (128 lanes) and over 20 s, instead of
   the thresholded success rate over 32 lanes (`sim_validate.py`; sim-only).

**Real sticks, test goals, 6 robots**

| | zero-shot | +24 trials | +48 trials |
|---|---|---|---|
| ours + DR, v2 (drB_s0, no fixes) | 52 % | 63 % | 65 % |
| ours + DR, line search, 3 seeds | 51 / 64 / 53 % | 84 / 90 / 85 % | |
| ours + DR, line search + lagged feedback, s0 / s1 / s2 / s0 120 it | 67 / 59 / 56 / 66 % | 89 / 91 / 69 / 87 % | 90 / 93 % (s0 / s1) |
| PPO+DR, 3 seeds | 77 / 79 / 65 % (mean 74) | with OUR stage: 94 / 99.5 / 100 % (mean 98) | 91 % (s0) |
| PPO on the robots (from PPO+DR s0, privileged critic) | 77 % | 79 % | 79 % at 96, 100 % at 384 |

**Reading.**
- Ours + DR (about 50 sim hours) beats PPO+DR zero-shot (74 %) with 24 real trials (84 % mean of 4 runs). It also
  beats PPO+DR fine-tuned by PPO on the robots at equal real budget (79 % at 24–96 trials).
- It does NOT beat PPO+DR's network followed by our hardware stage (98 %). PPO's 427 sim hours give a better warm start
  (74 vs 62 % zero-shot, and more consistent across seeds). Inside the DR family after the stage, ours is level or
  better (96–97 % vs 91–100 %); the gap is the leg-offset robot, where both sit at the 0.1 m/s speed threshold.

**Mode cells** (`real_pogo.signature`; `hw_stage --mode-accept`; distilc_theory.tex "Mode cells"):
- the pogo hardware stage never changes its mode cell (0 % topological changes, events compared on their common prefix);
- events cross 10 ms period boundaries in 96–98 % of trial pairs (kinks: the piecewise-differentiable regime).

**GPS-style fitted local dynamics** (`hw_stage --fit-samples`, runs/hw/gps/): 0–9 % at up to 192 trials per robot with
data-dominated fits (positivity cos 0.15–0.21), 56–60 % with a strong nominal prior, vs ours 80 / 87 % at 24 / 48.
- 4x lanes (1024 per iteration, 250 sim hours, `drBlsLK20x4_s{0,1}`): 61 / 68 % zero-shot, 82 / 88 % after 24 trials.
  No better than 256 lanes: more simulation alone does not close the gap to PPO+DR's warm start.

## 2026-10-17 round: baseline fixes, clean test set, seeds, budget (in progress; validation goals, seed 301)

Protocol: every choice below is made on the validation goals (POGO_EVAL_SET=val, the default). The test seed is now
1301; 701 is superseded (the hardware-stage defaults were seen on it). The test set is evaluated once, at the end
(reeval.py). Outputs: runs/v3/ (scheduler_fix.py), table_v3.py (seeds, cluster-bootstrap CIs, ci.py), budget.py.

**RMA, fixed.**
- Cause: phase 2 regressed z on the teacher's own rollouts (off-policy). At deployment, small z errors move the
  student into states the estimator never saw. With the oracle z the teacher reaches 94 %.
- Fix: phase 2 on the student's own rollouts (DAgger, `--p2-onpolicy 1`, 200 iterations).
- Val zero-shot: 35 % → 100 % on all six sticks (apex 0.3 cm). Action matching (`--p2-am`) and a wider phase-2 range
  (`--p2-dr 1.5 / 2.0`) also give 100 %.
- No leak (sanity check). The estimate of the leg offset is 1.68 vs the true 1.67, outside the DR range. On the mass /
  spring sticks it recovers the identifiable k/m ratio, not m and k separately.
- The old estimator was biased on every stick (e.g. z_hat k = −0.8 on the mass stick).

**FADA, fixed.**
- Cause: its source was the nominal-sim network (0 % zero-shot), and FADA reproduces its source. FADA's own recipe
  uses a DR source, so the source is now PPO+DR.
- Val zero-shot: 71–73 % vs 77 % for the source (inside the CI).
- After 24 trials of LoRA adaptation:
  - 4-step target: 92 %;
  - next-apex target (`--target apex`, an identifiable keyframe; the pogo's actions act only at touchdown and in
    stance): **96 %**.
- OU noise on the real trials hurts: 78 % at rms 0.05, 42 % at 0.1.
- Chosen: apex target, no noise. A sim check still shows FADA below its source in DR sim (22–27 % vs 42–45 %);
  `--idm-clean` variants are queued.

**Minimum hardware budget (budget.py, validation goals, 3 seeds, 6 goals × 1 trial (3 s) per iteration).**
B\* is the smallest budget within 2 points of the plateau (the best budget ≥ 48 trials) and not significantly worse
(paired cluster bootstrap).

| arm | 6 | 12 | 18 | 24 | 36 | 48 | 72 | 96 | B\* |
|---|---|---|---|---|---|---|---|---|---|
| ours nominal | 67 | 84 | 92 | 93 | 96 | **97** | 97 | 97 | 36 trials (108 s) |
| ours + DR | 80 | 87 | 86 | 90 | 91 | **92** | 91 | 92 | 24 trials (72 s) |
| PPO+DR + our stage | 83 | 91 | 96 | 96 | 99 | **99** | 97 | 95 | 36 trials (108 s) |

- 24 was not the right number: the curves are still rising for the nominal net and for PPO+DR.
- All three plateau by 36–48 trials. More does not help; PPO+DR + stage drifts down at 72–96 (87 % on one seed).
  36–48 is also an upper bound worth respecting.
- Batch B (2 or 3 goals per iteration; 1.5 s trials) is running, to see whether the same result is reachable with fewer
  real seconds.

**Test-pass freeze (2026-10-17 07:23:36).**
- Arms frozen as listed in `final_test.sh`, all chosen on the validation goals:
  - Raibert; old RMA; RMA (on-policy phase 2, 3 seeds); FADA (apex target, 3 seeds, zs and 24 trials);
  - ours nominal / ours + DR / PPO+DR + our stage, at 12 / 24 / 36 / 48 trials (3 seeds each);
  - PPO on the sticks at 24–384 trials (3 seeds); per-goal ILC; RMA oracle.
- Headline budget: 36 trials (B\* for nominal and PPO+DR; ours + DR B\* = 24). Batch B (val) is informative only and
  does not change the list.
- The test seed 1301 has not been evaluated by anything before this line.

**Batch B: the shape of a trial (validation goals, 3 seeds; success at 12 / 24 / 48 trials).**

| | 6 goals × 3 s (curve) | 3 goals per iteration | 2 goals per iteration | 6 goals × 1.5 s |
|---|---|---|---|---|
| ours nominal | 84 / 93 / **97** | 88 / 87 / 87 | 73 / 72 / 73 | 83 / 91 / 91 |
| ours + DR | 87 / 90 / **92** | 82 / 83 / 82 | 83 / 82 / 79 | 78 / 83 / 81 |
| PPO+DR + our stage | 91 / 96 / **99** | 93 / 85 / 83 | 88 / 85 / 81 | 87 / 92 / 90 |

- **The number of distinct goals sets the ceiling. Trials only set how fast it is reached.**
  - With 2–3 goals, the stage converges within 6–12 trials, but to a lower level (73–88 %).
  - PPO+DR + stage then degrades with more trials (93 → 83 % with 3 goals): it fits the few flown goals at the expense
    of the unflown test goals.
- **Half-length trials (1.5 s, about 2 hops)** reach about 90 % in 18–24 trials (27–36 s of hopping), but plateau 6–8
  points below the 3 s trials.
- **The cheapest route to the best result is 6 goals × 3 s trials, about 36 trials (108 s of hopping per stick).**

## Final test pass (seed 1301, frozen 2026-10-17 07:23; `results_table_test.txt`, `compare_test.py`)
Six sticks × 8 test goals × 4 drops × 10 s, 3 seeds per arm, 95 % cluster-bootstrap CI over (robot, goal).

| arm | zero-shot | 12 trials | 24 trials | **36 trials** | 48 trials |
|---|---|---|---|---|---|
| Raibert | 12.5 [4, 23] | | | | |
| ours, nominal sim | 7.8 ± 3.8 | 87.0 | 94.6 | **97.9 ± 1.3** [94.6, 100] | 97.0 |
| ours + DR | 71.2 ± 6.3 | 77.4 | 85.1 | **86.3 ± 9.4** [79.5, 92.5] | 86.8 |
| PPO+DR + our stage | 78.5 ± 2.0 | 89.2 | 96.2 | **97.2 ± 2.1** [93.6, 100] | 99.3 |
| PPO+DR | 78.5 ± 2.0 [67, 88] | | | | |
| RMA, fixed (on-policy phase 2) | **99.3 ± 0.6** [98.1, 100] | | | | |
| RMA, old (off-policy phase 2) | 34.9 | | | | |
| RMA teacher with the true z (oracle) | 97.9 | | | | |
| FADA, fixed (PPO+DR source, apex target) | 64.9 ± 4.1 | | 93.9 ± 3.6 [89.6, 97.4] | | |
| PPO on the sticks (privileged critic) | 78.5 | | 78.5 | | 82.6 |

PPO on the sticks continues at 96 / 192 / 384 trials: 87.2 / 93.2 / 95.8.

Paired differences, percentage points [95 % CI]; \* = excludes 0:

| comparison | difference |
|---|---|
| ours nominal @36 − PPO+DR zs | +19.4 [+10.2, +29.9] \* |
| ours nominal @36 − PPO on the sticks @48 | +15.3 [+8.2, +23.6] \* |
| ours nominal @36 − PPO on the sticks @384 | +2.1 [−0.7, +5.0] |
| ours nominal @36 − RMA (fixed) zs | −1.4 [−4.3, +0.3] |
| ours nominal @24 − FADA @24 | +0.7 [−7.3, +7.6] |
| ours + DR @24 − PPO+DR zs | +6.6 [−2.1, +15.8] |
| ours + DR @36 − ours nominal @36 | −11.6 [−18.6, −5.7] \* |
| PPO+DR + stage @36 − RMA (fixed) zs | −2.1 [−4.9, +0.2] |
| PPO+DR + stage @36 − ours nominal @36 | −0.7 [−4.2, +3.3] |
| FADA @24 − PPO+DR zs | +15.5 [+5.0, +26.9] \* |

**What the test pass supports (and does not).**
- **The test pass reproduces the validation numbers within about 2 points everywhere** (e.g. ours nominal @36:
  96.4 val / 97.9 test). The tuning did not overfit the validation goals.
- **Ours from the nominal sim reaches 98 % in 36 trials (108 s of hopping per stick) from 8 % zero-shot.**
  - That is +19 points over PPO+DR zero-shot and +15 over PPO fine-tuned on the sticks at a matched-or-larger budget
    (48 trials).
  - PPO on the sticks needs about 384 trials to come within CI (95.8 %).
- **With the baselines fixed, ours is not better than the best baselines on this toy.**
  - RMA's estimator, trained on its own rollouts, adapts zero-shot to every stick (99.3 %), including the
    out-of-family ones. It is level with ours @36 within CI, while ours needs 36 trials.
  - FADA with 24 trials is level with ours at 24.
  - The honest claim is "matches the best sim-trained adaptive baselines without a DR family, a parameterized latent,
    or an inverse model, from a nominal simulator, in ~100 s of real data". It is not "beats them".
  - RMA succeeds here because every real perturbation is either a parameter it randomized or one whose effect a 1-hop
    history reveals. The toy does not test unmodelled effects, where its latent has nothing to estimate.
- **Our stage is policy-agnostic:** PPO+DR 78.5 → 97.2 % @36 (99.3 % @48).
- **Ours + DR is the weakest of our variants** (86 % @36, 11.6 points below ours nominal). This is the known DR(B)
  weak spot: one seed sits at 79 %.
- **Per-goal ILC (test, its own 6 training goals, deterministic):** 12.5 / 12.5 / 17.4 / 22.9 % at 6 / 12 / 24 / 48
  trials.
- **Figures:** `src/paper/pogo/figures/fig1_methods_test.png` and `fig2_budget_test.png` (`figures_v3.py`).

## Ours + DR diagnosis (2026-10-17, after the test pass; validation goals only)

**Where it fails.**
- Mostly the heavy and soft sticks, both inside its own DR family, plus the offset stick.
- On the heavy / soft sticks the misses are at the high-apex goal (0.52, 0.74): the apex is about 4.5 cm short.
- The nominal net reaches 100 % on the same sticks after the stage.
- **Ruled out:**
  - Saturation: thrust and leg angle stay inside their limits.
  - A weak DR sim stage: our DR net scores 57 % on the sim-DR metric vs PPO+DR's 41 %.
  - The step direction: positivity median cos 0.8–0.85.

**The stage converges slower on the DR net.**
- On the heavy stick the training error shrinks about ×0.6 per iteration, against about ×0.2 for the nominal net.
- |measured change| / |predicted change| is 0.28–0.37 (nominal net 0.43–0.73).

**The refit realizes only part of each GN step.** hw_stage now logs `realize`: the fraction of the step the refit
applies on its own trials.
- Default refit (800 Adam steps, lr 1e-4, age decay 0.5): 0.8 at iteration 0, falling to 0.13–0.4 by iteration 3. Old
  targets dominate the loss.
- **A stronger refit raises it but does not fix ours + DR** (`drfix`, 36 trials, seeds 0 / 2; success at 12 / 24 / 36
  trials):

| refit | s0 | s2 | realized (median) |
|---|---|---|---|
| default | 88 / 90 / 90 | 82 / 84 / 87 | 0.36–0.40 |
| 3000 steps | 89 / 88 / 94 | 82 / 83 / 82 | 0.44–0.48 |
| lr 3e-4 | 87 / 90 / 90 | 82 / 88 / 88 | 0.39–0.44 |
| 3000 steps + lr 3e-4 | 90 / 87 / 84 | 90 / 90 / 89 | 0.50–0.51 |
| (nominal net s2, default) | | 83 / 92 / 95 | 0.48 |

- s2 stays at 78 % on the heavy and soft sticks with every refit.
- The training-goal error there does converge (0.004 at the nearby training goal (0.6, 0.74)). The correction just
  does not carry over to the validation goal (0.52, 0.74) for the DR net, while it does for the nominal net.
- **Remaining hypothesis: goal generalization of the DR net's correction, not the step or the refit.** Next tests:
  - a 7th training goal near the high-apex corner;
  - the adaptive trust region (proposed) that replaces the fixed step knobs.
- No new test pass: nothing was adopted.
