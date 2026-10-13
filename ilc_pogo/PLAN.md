# DistILC on a toy 2-D pogo stick: full plan

**Status (2026-10-13 01:45): executed. Results, the v1 -> v2 fixes and caveats: NOTES.md.**

## Context
The user wants a third, minimal domain for the DistILC paper (after the Go1 quadruped and the F1TENTH car): a toy pogo
stick, a simple classical dynamics program. It should run the same matrix as the car:
- ours, nominal and with DR;
- classical ILC;
- PPO+DR, RMA, FADA;
- the same DR benchmarks;
- sim-to-sim transfer under idealized conditions (perfect state, no delay), no hardware.

Constraints from the user during this session:
- no MuJoCo; everything in numpy / JAX + matplotlib;
- entirely 2-D;
- "even simpler" (no multi-body hopper);
- perfect friction.

Standing rules:
- one network, no residual policy, no system ID on the robots;
- the training / Jacobian sim stays nominal, and perturbations exist only on the "real" sticks;
- DR is an orthogonal variant;
- small fast batches (≤10 min per step);
- ≤32 CPUs and 4 GPUs;
- background waiters, no blocking sleeps.

Why a toy: it is small enough to sweep a perturbation axis continuously and show the thesis in one figure.
- ILC regresses on measured outcomes, so it converges whenever the nominal Jacobian is within 90 deg of the truth
  (positivity).
- DR only works inside its family's coverage.

**Decisions (user, 2026-10-13):**
- main objective: the goal-conditioned gait (v*, h*);
- the real-vs-nominal gap: damping plus a finer step, plus each robot's perturbation;
- included in this run: coverage + positivity, data matching, animations.

## Already written (in `src/ilc_pogo/`, env `~/miniconda3/envs/pogo` = clone of ilcmjx + optax 0.2.8)
| file | state |
|---|---|
| `common.py` | done, tested: SLIP model `substep` (numpy/jax-agnostic), straight-through touchdown saltation terms, obs `features` (8), task `errors` (2), `failed`, Raibert `expert_action`, `mlp`, policy io, `hop_metrics` |
| `pogo_jax.py` | done, tested: nominal batched sim (0.2 ms steps), `dr_params`, `rollout`, `lane_jacobians` (policy / expert / none feedback), `gn_lane` (closed-loop GN, from the car's `ilc_core.make_gn`) |
| `real_pogo.py` | done, tested: numpy vectorized "real" sticks (0.1 ms steps), `ROBOTS`, `ax:` single-axis robots, `simulate`, `evaluate`, `summarize` |
| `sim_train.py` | written (port of the car's `sim_train.py`), first debug run done before the gradient fix; needs one re-run |
| `PLAN.md` | outdated (3-body / MuJoCo wording): rewrite from this plan |

Measured so far:
- **Raibert controller**:
  - nominal sim: apex error ≤ 0.1 cm, speed error ≤ 0.03 m/s;
  - real sticks: apex error 2-7 cm (fails the 2 cm band); the offset robot is also 0.33 m/s slow.
- **Classical ILC with closed-loop G** (nominal Jacobians, a damped + heavy test system): J 607 → 175 in 6 trials.
  Open-loop G diverges.
- **Bug fixed**: before the straight-through terms, the touchdown had a zero gradient with respect to the leg angle.

## 1. The pogo stick (model)
- **2-D SLIP**: a point mass m on a massless spring leg of rest length l0, in the x-z plane. Perfect friction: the foot
  is pinned in stance.
  - State x = [x, z, vx, vz, xf, stance].
  - Flight is ballistic; the leg points along the commanded angle phi.
  - Touchdown when z ≤ l0 cos phi, descending; the foot is placed at x + l0 sin phi.
  - Stance: the leg force is max(0, k(1 + kp c/l0)(l0 + gth·delta − L) − b·L̇) along the leg.
  - Liftoff when L ≥ l0 + gth·delta, extending.
- **Actions** (normalized to [-1, 1]):
  - the thrust delta in [0, 0.08] m (the spring's rest length pushed out, Raibert-style);
  - the leg angle phi in ±0.6 rad, used in flight.
- **Nominal**: m 10 kg, k 4000 N/m, l0 0.5 m, b 0, gth 1, phi_off 0, kp 0; 100 Hz control.
- **Goal** g = (v*, h*) in the box [0, 1.5] m/s × [0.55, 0.75] m.
  - Hardware-stage goals: the 6 `TRAIN_GOALS`.
  - Reserved test goals: 8 from seed 701. Validation goals: 8 from seed 301.
- **Observation** (8): z − h*, vx − v*, vz, stance, (x − xf)·stance, (L − l0)·stance, v*, h*.
- **Task error rows** (in flight only; no time alignment to any reference):
  - the ballistic apex z + vz²/2g − h* (scale 2 cm);
  - vx − v* (scale 0.1 m/s).
- **Success** (`hop_metrics`), per 10 s run without a reset (~15 hops): no fall (z < 0.25), and from hop 3 on, RMS apex
  error ≤ 2 cm and |mean vx − v*| ≤ 0.1 m/s.

## 2. Candidate objectives (to list in PLAN.md; #1 is the main task)
1. **Goal-conditioned gait (v*, h*)** — chosen. Continuous and stationary, like the car's drift; a goal plane, like
   the quadruped.
2. Hop in place to a target apex (1-D subset of 1, v* = 0): the cleanest axis sweeps.
3. Hop to a target spot and stop: a terminal landing error, the quadruped's task.
4. Stepping stones: a footstep sequence with variable spacing, which needs a preview.
5. Speed / height transitions inside one trial: a goal step at t = 5 s, scored on the settling.
6. Hop onto a box / down a step: a terrain change, a mode change at a different height.

Options 3-6 reuse the same model with a new error row. They are follow-ups, not in this run.

## 3. "Real" pogo sticks and the DR family
- **Real simulator**: an independent numpy run at 0.1 ms steps (vs the nominal's 0.2 ms in JAX float32).
  - Idealized sensing: exact state, no delay.
  - All robots carry leg damping b = 15 N s/m, which the nominal model does not have.

| robot | perturbation (on top of b = 15) | in DR family? |
|---|---|---|
| real_nom | none | yes |
| real_mass | m ×1.2 | yes |
| real_spring | k ×0.8 | yes (edge) |
| real_act | thrust gain ×0.75 | no (DR ±20 %) |
| real_offset | leg-angle offset +0.05 rad | no (DR ±0.03) |
| real_prog | progressive spring, kp = 1.5 | no (different spring law) |

- **DR family** (`pogo_jax.dr_params`):
  - m ±25 %, k ±20 %, b 0-30;
  - thrust gain ±20 %, phi offset ±0.03;
  - l0 ±4 %.
- **Axis sweeps** (`ax:` robots, for the coverage figure):

| axis | values |
|---|---|
| k | ×0.5-1.5 |
| m | ×0.7-1.5 |
| gth | ×0.5-1.3 |
| phi_off | −0.1 to +0.1 |
| kp | 0-3 |
| b | 0-40 |

  Each axis has 7 points; the DR range is shaded in the figure.

## 4. Methods (each mirrors a car / quadruped file)
| arm | file (new) | port of |
|---|---|---|
| Raibert zero-shot (classical controller) | `hw_stage.py --policy expert --iters 0` | car `hw_stage.py --policy lqr` |
| Per-goal classical ILC: time-indexed feedforward on Raibert, nominal G with the expert's feedback, fixed drop start; evaluated on its own goals | `goal_ilc.py` | car `track_ilc.py` |
| **Ours, sim stage**: BC (DAgger onto Raibert), then deep ILC (closed-loop GN, exploration, chaining with new goals, replay, anchor) | `sim_train.py` | car `sim_train.py` |
| **Ours, hardware stage**: per robot, GN-ILC from nominal Jacobians at the measured states, the network's own K, regression on all trials (age decay), trust region on nominal-sim states, rollback, crash rollback | `hw_stage.py` | car `hw_stage.py` (no plan bank, no ladder) |
| Ours + DR (A: DR fine-tune of the nominal net; B: DR from BC) | `sim_train.py --dr 1` | car v3drA / v3drB |
| PPO+DR (warm start = BC net; reward 1 − 0.1 min(|e/scale|², 10), fall ends the lane, plus an alive bonus) | `ppo_dr.py --mode ppo` | car `ppo_dr.py` |
| RMA (teacher sees z; adaptation module from a 10-step history of (obs, a) → z) | `ppo_dr.py --mode rma` | car `ppo_dr.py` |
| FADA: IDM(o_t, target next-state deviation) → a over DR; planner d* = λ^H d toward the goal (apex / speed deviations); LoRA adaptation on real transitions | `fada.py` | car `fada_car.py` |
| PPO+DR + our hardware stage | `hw_stage.py --policy runs/ppo_dr_s0/...` | car `ppo_dr_hw*` |

Hardware-stage defaults:
- 6 goals × 1 trial (3 s, ~6 hops) per iteration;
- budgets 0 / 6 / 12 / 24 / 48 trials per robot (`--iters 0/1/2/4/8`);
- beta 0.5, cap 0.05, anchor-w 1 with n0 6, age decay 0.5, 800 fit steps, lr 1e-4.

## 5. Ablation matrix (`scheduler.py`, the car's dependency scheduler; GPU and CPU slots)
- **Sim stages** (GPU, ~2-5 min each):
  - `bc` (seed 0);
  - `nom_s{0,1,2}`;
  - `olG_s0` (`--no-feedback-jac`);
  - `noexp_s0` (`--explore-frac 0 --chain 0`);
  - `drB_s{0,1}`, `drA_s{0,1}`;
  - `ppo_dr_s0`, `rma_s0`;
  - `fada` (CPU arm, IDM trained on the GPU);
  - `ppoNom_s0` / `ppoDR_s0` (our sim stage from the PPO net; optional).
- **Hardware / evaluation arms** (CPU), on all 6 robots:
  - `expert_zs`;
  - `goalilc_b{1,2,4,8}` (trials per goal);
  - for each of nom_s0-2, drA, drB, olG, noexp, ppo_dr: `_zs` and `_hw24`;
  - nom_s0 and drB_s0: `_hw6`, `_hw12`, `_hw48` (budget curve);
  - nom_s0 controls: `_hw24_openG`, `_hw24_flip`, `_hw24_noTR` (anchor-w 0), `_hw24_noRB` (rollback 0);
  - `rma_zs`;
  - `fada_zs` / `fada_hw24`.
- **Coverage suite**: for each axis robot (6 axes × 7 values = 42 robots):
  - Raibert;
  - ours nominal zs and hw24;
  - ours + DR (B) zs and hw24;
  - PPO+DR zs;
  - RMA zs.

  Plus the **positivity** per robot: the cosine between the predicted (G·step) and the measured change of the error
  vector, from consecutive trial pairs in the hw24 runs, logged by `hw_stage.py`.
- **Data matching** (phase 2, after the main matrix):
  - `ppo_real.py`: PPO with a privileged critic on each real stick, from the PPO+DR / RMA-oracle nets, up to 2,000
    trials per robot, checkpoints at 24 / 96 / 384 / 1,536 / 2,000;
  - the number of trials needed to match ours + hw24.
- **Final evaluation**: `reeval.py`, each arm × 6 robots × 8 test goals × 4 perturbed drops × 10 s (192 runs per arm),
  reserved seed 701, 3 replicate seeds for headline arms.
  - `table.py` → `results_table.txt`.
  - `figures.py` → `src/paper/pogo/figures/`:
    1. methods bar (zs vs hw24);
    2. budget curve;
    3. coverage (success vs axis value, DR band shaded);
    4. positivity along the axes;
    5. controls;
    6. sim cost (sim hours) vs real trials;
    7. goal-plane maps.
  - `animate_pogo.py`: matplotlib 2-D animations (stick, spring, foot, apex target line) → MP4 / GIF in
    `src/paper/pogo/videos/`.
  - `NOTES.md`: results and claims, car-style.

## 6. Order of work
1. Rewrite PLAN.md. Re-run the `sim_train.py` debug (20 iterations).
   - Target: nominal val success clearly above BC (BC 69 %), J falling.
   - If BC is weak: hidden 256, more DAgger iterations, an observed L̇.
2. Write `hw_stage.py`. Smoke test: 1 robot, 2 iterations.
3. Write `goal_ilc.py`, `ppo_dr.py`, `fada.py`, `reeval.py`, `table.py`. Smoke test each with tiny sizes.
4. Write `scheduler.py` and launch the main matrix (background, log `logs/scheduler.log`), with a background waiter
   that wakes me.
5. Coverage suite and positivity (`scheduler_axes.py`).
6. Figures, animations, NOTES.md. Then data matching.

## 7. Compute estimate
- Nominal GPU rollout: 256 lanes × 3 s takes ~0.2 s; the GN step ~0.5 s. A sim stage of 40 iterations is about 3 min.
- PPO: 300 iterations × 1,024 lanes is ~15 min.
- Real sim (numpy, vectorized over runs): ~6 ms per control step for any batch size.
  - A 10 s eval of 192 runs takes ~6 s per robot.
  - A hw24 arm (4 iterations × 36 trials batched) takes ~1 min plus its eval.
- Whole main matrix: ~30-40 jobs at ≤8 parallel, under 1 h. Coverage suite: ~30 min.

## 8. Verification
- **Unit checks** (in the smoke scripts):
  - numpy vs JAX substep agreement at an equal step (positions within mm over 3 s);
  - FD vs autodiff check of the step Jacobian on a lane with a touchdown;
  - the Raibert controller holds every goal in the nominal sim.
- **Classical-ILC sanity**: J decreases with closed G and the flipped G fails (done for J; the flip check comes with
  `hw_stage`).
- **Pipeline**:
  - each new script runs on tiny sizes before the scheduler;
  - the scheduler status JSON shows every job done;
  - `reeval.py` reproduces `hw_stage`'s own eval for one arm.
- **Results sanity**:
  - ours hw24 beats Raibert and ours zs on the in-family robots;
  - DR (B) zs wins in-family but drops on real_act / offset / prog and on the axes outside its band;
  - positivity stays > 0 where ours converges.
