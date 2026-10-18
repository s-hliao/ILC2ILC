# LM trust region across the three domains (ube, 2026-10-17)

Branch `ube-2026-10-17`. The full per-domain write-ups are in:
- pogo: `ilc_pogo/NOTES.md`, section "LM trust region";
- car: `ilc_f1tenth/ilc2real/NOTES.md`, section "LM trust region";
- quadruped: `ilc_mjx/paper/tr_2026-10-17/TR_RESULTS.md`.

The theory side is `paper/theory/distilc_theory.tex`, remark `rem:posdomains`.

## Question
Can ONE adaptive step rule replace each domain's hand-set step knobs (gain beta, cap, rollback, anchor, bold/gate)?

## The rule (same in all three domains)
A Levenberg–Marquardt trust region (Nocedal–Wright 4.1) on the Gauss–Newton step:

- rho = (measured reduction of J) / (predicted reduction of J).
- rho < 0.25 halves the radius r. rho < 0 (or a new fall or crash) rejects the step: roll back to the last kept
  network. rho > 0.75 doubles r, up to a maximum of 0.4.
- The prediction J_pred, per domain:
  - pogo: J(f) = ½|e + f·de|², for the change the refit actually realized (fraction f);
  - car: J(f) = J − (2f − f²)(J − J_pred), per goal, against that goal's own last kept trial (the goals cycle);
  - quadruped: J_pred = J_prev · (e_predᵀQe_pred) / (eᵀQe) per goal, with e_pred = e − S·step.

Flags:

| domain | file | flags |
|---|---|---|
| pogo | `ilc_pogo/hw_stage.py` | `--tr 1 --tr-lo 0.25 --tr-hi 0.75 --tr-max 0.4`; `--cap` = r0; optional `--fit-realize 0.95` (refit to convergence, plateau rule) |
| car | `ilc_f1tenth/ilc2real/hw_stage.py` | `--tr 1`; `--cap` = r0; `--fit-realize` |
| quadruped | `ilc_mjx/scripts/deploy.py` | `--tr 1 --tr-lo/--tr-hi/--tr-max`; `--step-rms` = r0. With `--tr`, the bold heuristic is off and the scale is 1 |

## Headline
- **Pogo:** the step size is not a knob.
  - With the full GN step (beta 1), the trust region from any r0 ≥ 0.05 matches the fixed default within CI.
  - Caps from 0.05 to 0.20 give identical runs.
- **Car:** the full step is harmful.
  - beta 1 without the anchor crashes, and the trust region cannot save it on a 10-lap budget.
  - With the recipe's damping kept (beta 0.5 and the anchor), the trust region **replaces the fixed cap and the
    progress rollback**. It is level with the recipe from either r0.
- **Quadruped:** no step rule matters. The hardware stage moves the score by 1–2 / 21 under any rule or batching.
  - The limit is the landing model. It realizes about 25 % of the predicted improvement and has the wrong sign for
    about 1/3 of goals, even on 5-jump means.
  - Trial noise is a second, comparable factor.
- **Theory link:** rho is the measured version of the positivity assumption, and beta ≤ μ/Γ² is the step bound.
  - pogo: positivity margin large (rho 0.5–0.7);
  - car: margin small (full-step rho −3 to −15), so beta 0.5 is the theorem's step bound, not tuning;
  - quadruped: margin about 0 (rho median −0.16 on single jumps, +0.24 on 5-jump means).

## Pogo (validation goals, seed 301; 36 trials = 6 goals × 6 iterations; 3 networks × 3 seeds; `runs/v4/`)
ube reproduces matcha: fix10 gives 96.0 / 91.5 / 99.0 % against matcha's 96.4 / 91.1 / 98.8 %.

Success % at 12 / 24 / 36 trials:

| arm | nominal | ours + DR (drLK20) | PPO+DR |
|---|---|---|---|
| fixed cap 0.025 | **44** / 90 / 94.1 | 84 / 89 / 89.9 | 91 / 96 / 99.3 |
| fixed cap 0.05 | 84 / 92 / 96.4 | 86 / 87 / 87.8 | 91 / 96 / 99.0 |
| fixed cap 0.10 (default) | 84 / 93 / **96.0** [91, 100] | 87 / 89 / **91.5** [86, 96] | 91 / 96 / **99.0** [98, 100] |
| fixed cap 0.20 | 84 / 93 / 96.0 | 87 / 90 / 91.3 | 91 / 96 / 99.1 |
| trust region, r0 0.025 | **55** / 90 / 94.1 | 84 / 89 / 88.9 | 91 / 96 / 99.3 |
| trust region, r0 0.10 | 84 / 91 / 94.6 [89, 99] | 87 / 90 / 89.9 [84, 95] | 91 / 96 / 99.1 [98, 100] |
| trust region, r0 0.40 | 84 / 91 / 94.4 | 87 / 89 / 89.8 | 91 / 96 / 99.0 |
| fix10 + refit to convergence | 91 / 91 / 87.5 | 86 / 83 / 82.1 | 94 / 90 / 87.5 |
| tr10 + refit to convergence | 91 / 91 / 88.4 | 86 / 81 / 79.7 | 93 / 93 / 89.2 |

- **Median rho is 0.5–0.7:** the model over-predicts the gain by about 1.5–2×. That is about 1 rejected step per run.
- **Only a too-small cap binds, and only early** (0.025 → 44 % at 12 trials; the trust region recovers to 55 %).
- **Refit-to-convergence (age decay 0) hurts** by 8–12 points at 36 trials. It forgets what the older trials fixed.
- **Side result, ours + DR from the nominal sim network (`drNomInit`):**
  - after the hardware stage, 94.3 % [88, 99] at 36 trials, against 91.5 % for drLK20; the heavy / soft sticks go from
    91 to 97 %;
  - it is worse zero-shot: 54 % against 74 %.
- Reproduce: `cd ilc_pogo; POGO_FIX_STAGE=tr1 POGO_STATE=scheduler_tr1 POGO_GPUS=0,1,2 POGO_CPU_JOBS=12 nohup
  python scheduler.py --jobs scheduler_tr &`. Then repeat with stages `tr2` and `drnom`.

## Car (10-lap ladder, 8 cars, 3 sim seeds v3drBtL_s0..2; rung-aware scoring; `runs/tr/`)
ube reproduces matcha: the recipe gives b25 100 / 67 / 100 %, against matcha's 100 / 66–74 / 99.5 %.

| arm | b25 success s0 / s1 / s2 | all plans | crashes % | rollbacks / run | rho median |
|---|---|---|---|---|---|
| `lad` recipe (beta 0.5, anchor 1, cap 0.05, progress rollback) | **100 / 67 / 100** | 100 / 88 / 100 | 0 / 6 / 0 | 5 / 10 / 4 | – |
| `lad_tr` full step (beta 1, no anchor), r0 0.05 | 66 / 61 / 56 | 52 / 68 / 63 | 33 / 18 / 21 | 12 / 15 / 15 | −3 / −15 / −11 |
| `lad_tr_r20` full step, r0 0.20 (s0 / –– / s2) | 50 / – / 61 | 63 / – / 54 | 19 / – / 24 | 16 / – / 16 | – |
| `lad_trd` trust region + recipe damping, r0 0.05 | 98 / 64 / 98 | 99.5 / 84 / 99.5 | 0 / 6.5 / 0 | 9 / 13 / 11 | 0.61 / 0.02 / −0.31 |
| `lad_trd_r20` trust region + recipe damping, r0 0.20 | 100 / 75 / 94 | 100 / 89 / 94 | 0 / 2 / 0 | 9 / 13 / 13 | 0.55 / −1.2 / −1.0 |

- **The full step hurts.** Steps that the nominal Jacobians predict to help crash the car. Each rejection costs a crash
  and drops the car a ladder rung; even the unperturbed car ends on `_mu60` plans.
- **With the recipe's damping, the trust region replaces the fixed cap.** It is insensitive to r0. It pays 9–13
  rejected steps per run, against 4–10 for the recipe.
- **What this leaves:**
  - beta 0.5 and the anchor are justified by the small positivity margin;
  - the cap is not a tuned number.
- **`lad_tr_r20_s1` is not available.** It hung twice in the hw_stage worker pool at iteration 3 (log silent, about 4 %
  CPU) and was killed. The conclusion does not depend on it. The pool hang itself is not debugged; fix it before more
  car runs on ube.
- Reproduce:
  - `cd ilc_f1tenth/ilc2real; bash gen_plans.sh` (regenerates `plans_mu02_tight`, including `_mu80` / `_mu60`);
  - then `NPAR=3 bash car_tr.sh` and `NPAR=2 bash car_trd.sh`;
  - tables: `tr_table.py` in the same folder (rung-aware, as `reeval.py`; pogo has its own `ilc_pogo/tr_table.py`).

## Quadruped (validation goals planeval; 6 goals × 4 iterations = 24 jumps per robot; robots r1 / s1 / r4 / r5; seeds 9001–9003)
Starting network: drB_s2 it30 on the fixbox overlay. Score = 0.01 × landing cost + 20 per fall, **lower is better**.
The CI is a bootstrap over the 12 (seed, robot) cells.

| arm | final | 95 % CI | falls / 12 runs |
|---|---|---|---|
| zero-shot | 21.30 | | |
| `recipe` (drB_hw24: gn-beta 0.3, step-rms 0.05, anchor, bold, gate), 1 jump × 4 iterations | **19.74** | [13.8, 25.7] | 21 |
| `tr` (beta 1, r0 0.05, no anchor / bold / gate), 1 × 4 | 20.30 | [14.0, 26.2] | 25 |
| `tr_r20` (r0 0.20), 1 × 4 | 21.83 | [14.5, 29.3] | 19 |
| recipe, 2 jumps × 2 iterations | 20.51 | [13.9, 27.3] | 24 |
| trust region, 2 × 2 | 19.27 | [12.9, 25.8] | 30 |
| (trust region, 5 × 2 = 60 jumps, 1 seed) | 19.56 | | |

Paired differences: none is significant. All CIs contain 0, and the largest is tr_r20 − recipe = +2.1 [−1.8, +6.4].

**rho:**
- on single jumps: median −0.16, < 0 in 53 % of steps;
- on 5-jump means: median +0.24, < 0 in 37 %.

**Noise floor:** the jump-to-jump sd of J (median 3.0) is about equal to the predicted improvement per step (2.7).
- Reading: the stage is limited by the landing sensitivity (it over-predicts about 4×), with noise as a second factor.
  The step rule is not the limit.
- Next, model-side: `deploy.py --jac-at real` (Jacobians at the measured states) and `--sens-adapt goal|shared|vres`
  (Broyden / residual correction). Both flags already exist and have not been tested in this setting.
- Reproduce: in `ilc_mjx/paper/tr_2026-10-17/`, `NPAR=4 bash tr_arms.sh` and `bash noise_arms.sh`. They need the
  `ilcmjx` env, `log/dilc/plane/{drB_s2,fixbox/overlay,plans_v3}` and `~/mujoco_menagerie`. Outputs go to
  `log/dilc/plane/tr/`.

## Open items (not started; ube is idle)
1. **Car: our hardware stage vs the task-reward baselines.** This decides the car claim.
   - Train PPO+DR and RMA with `--reward task` (RMA also `--hist 40 --p2-onpolicy 1 --p2-am 10 --adapt-iters 150`):
     seeds 1 and 2, and ideally on the ladder bank PLL (PL8 + `_mu80` / `_mu60`), as v3drBtL was.
   - Evaluate them on the 8 ladder cars with the same plan-rung fallback (no network update) at the 10-lap budget.
     Compare against `lad` / `lad_trd`.
   - Then a held-out test set and seeds.
   - **Blocker on ube:** the BC init `runs/v1nom_s0/policy_bc.npz` that `car_baseline_fix.sh` uses is missing (it lives
     on matcha). Either copy it over or retrain it.
2. **Quadruped:** the `--jac-at real` and `--sens-adapt` arms with the `tr_arms.sh` setup (recipe step settings,
   24 jumps, 3 seeds × 4 robots).
3. **Pogo:** the interpolation test (error at goals never flown on hardware). This was on matcha with the other session.

## Environment notes (ube)
- JAX 0.4.35 with CUDA 12.3 libraries, pinned for driver 535. Envs: `pogo`, `f1t` (+ numba, casadi), `ilcmjx`
  (+ mujoco / mjx 3.12.0).
- The quadruped container is `ilc_quad` (image `ilc_quad:humble`). Run docker through `sg docker -c`.
- Never `pkill -f <pattern>` from a command whose own text contains the pattern. Kill by PID or by the anchored binary
  path.
