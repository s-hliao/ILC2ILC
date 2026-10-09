# ILC2Real on the F1TENTH: continuous drift (started 2026-10-09, overnight)

The quadruped pipeline ported to the car. The target is a continuous, multi-lap (potentially infinite) drift on
`mocap_square2fast` and `mocap_figfast`. Everything is in `src/ilc_f1tenth/ilc2real/`. Run with the `f1t` conda env
(`~/miniconda3/envs/f1t`: JAX 0.9 CUDA, numba 0.68, CasADi 3.8).

## Pieces (quadruped analog in brackets)

| file | role |
|---|---|
| `car_model.py` | **Nominal model** [GPU MJX sim]. A two-axle single-track car with Pacejka combined-slip tires and longitudinal load transfer as a state. The driveline is current-driven, with a fixed torque split and open diffs. One model is written for numpy, JAX and CasADi. |
| `mb_car.py` | **"Real" cars** [CPU MuJoCo robots]. The f1tenth_gym fork's multi-body model, from `vehicle_dynamics_mb`: sprung body with roll and pitch, 4 wheels with their own spin, suspension, and per-corner MF tires including the full-scale offsets. RK4 at 0.25 ms in numba. It sits behind the same actuation as the nominal model, with five variants: `real_nom`, `real_mass` (+25 % mass), `real_mu` (friction ×0.8), `real_act` (motor ×0.8, steering offset 0.03 rad, slow servo, 1-period current delay) and `real_lag` (lateral tire stiffness ×0.8, 1-period steering delay, slower servo). All five add sensing noise. |
| `tracks.py` | Track geometry from the llampc mocap waypoints: a periodic spline, resampled by arc length. |
| `trajopt_drift.py` | **Trial-1 plans** [TO bank]. IPOPT (CasADi), Radau-3 collocation in **arc length**, with periodicity z(L) = z(0), so a plan can be flown lap after lap. The cost is lateral offset, sideslip against the target β*(s) = −β_d·tanh(κ/0.25), speed against the track's design speed, and input rates. A homotopy runs β_d = 0 → 10 → 18 → 25°. |
| `plan_lqr.py` | **Periodic LQR in arc length** around each plan, used as the warm-start structure. |
| `bank.py` | The plan bank: plans plus cached LQR gains. |
| `sim_jax.py` | **Nominal GPU sim**: batched rollouts. It provides the path projection (local search, since the figure-eight crosses itself), the observation (deviation from the plan at the car's own s, raw state, and a 1.5 m preview of the plan), the ILC error, exact Jacobians, and the DR family. |
| `sim_train.py` | **Sim stage** [plane_train.py]. First, DAgger onto the LQR. Then per iteration: rollouts with exploration and lap chaining, a closed-loop GN ILC step per lane, and regression onto μ + step. |
| `ilc_core.py` | The closed-loop GN ILC step on any batch of runs, from nominal Jacobians at the given (measured) states. |
| `real_car.py`, `sim_jax_np.py` | Fly a network, the LQR, or LQR + feedforward on a real car, and compute metrics. |
| `hw_stage.py` | **Hardware stage** [deploy.py], then the multi-lap evaluation. |
| `track_ilc.py` | **Per-track ILC baseline** [JumpILC]: LQR plus a periodic feedforward ff(s), learned per car and per plan from scratch. |
| `ppo_dr.py` | **PPO+DR and RMA baselines** in the same sim, with the same observation and the same warm start. |
| `table.py` | The results table. |

## Design decisions (and why)

- **RWD drift car (front torque share 0).** With the 50/50 AWD split, the nominal model has no drift equilibria at
  all (a multi-start search found none for β = −10…−40°). RWD gives β = −10…−30° equilibria at V ≈ 3.0 m/s
  (R = 1 m) and 3.7 m/s (R = 1.5 m). These match the tracks' design speeds: square2fast 3.75 m/s at a median
  R of 1.56 m, figfast 2.5 m/s. On hardware this is the usual RC drift conversion (front driveshaft out).
- **Current (torque) actuation**, as a VESC in current mode, plus a steering servo set point. Power-over drift
  needs rear wheel spin, which a speed command through the gym's `pid` cannot produce.
- **Tire.** The full-scale MF tire's dimensionless coefficients, so the lateral force peaks at about 9° of slip.
  The Froude-scaled set in `lifted_linear_tire_20261009/mb_sim` peaks at about 35°, too soft for drift (DRIFT.md
  item 6). Friction stays at the gym's F1TENTH μ = 1.0489.
- **Sim/real gap.** The nominal single-track model and the multi-body car differ in structure as well as in
  parameters: roll, left/right load transfer, per-wheel open-diff spin, camber, ply-steer offsets and compliance.
  This is the analog of MJX vs CPU MuJoCo plus perturbations, and the sim never sees the perturbations.
- **Arc-length indexing everywhere.** The plan, the LQR and the ILC error (against the plan at the car's own s)
  are all indexed by arc length, so there's no phase drift. Laps chain without resets, and the policy is
  stationary over laps (the "infinite drift" requirement).
- **Open-loop drift is violently unstable.** The per-lap growth of the plan's linearization is 7e10 (square) and
  4e17 (figure-eight), against 1e2–1e4 for grip. The periodic LQR brings it to ≤ 2e-4. So the ILC must sit
  around feedback: the network carries the feedback, and the ILC's G is the closed-loop sensitivity, as in the
  quadruped.
- **LQR weights** (R: 0.1 rad, 3 A) were swept. The more aggressive current gains chatter with the wheel-spin mode
  and lose the figfast drift at the left/right transition.

## Status log

### 2026-10-09 03:40–05:00: first full pass

**Sim stage.**
- The v1 recipe was LQR DAgger with narrow exploration (action noise 0.05, start perturbation 1), then 60 deep-ILC
  iterations with no trust region. Its failure rate rose after about 20 iterations, and one seed collapsed to NaN.
  - The non-finite guard was added afterwards.
  - The cause is the quadruped's v6/v7 problem: no trust region.
- More importantly, the v1 network TRANSFERS FAR WORSE THAN ITS LQR TEACHER: 7.5 % against 37.5 % full-drift success
  zero-shot. On the multi-body cars, the plan deviations the network sees are 5–10x the clean-sim ones.
- Broad exploration in the warm start fixes it (bcB: DAgger with action noise 0.3, start perturbation 2): 30 %.
  This is the quadruped's "exploration is essential" finding again.
- **v3 recipe:**
  1. bcB warm start;
  2. 30 deep-ILC iterations with the trust region (`--anchor-w 1`) and a gentle step (β 0.3, cap 0.04);
  3. rows only within |e_y| < 0.2 m;
  4. exploration: action noise 0.15, start perturbation 1.5, 70 % of the lanes exploring, 25 % chained lanes.

  It is stable, at about 6,000 sim laps and 10 min on one GPU.

**Hardware stage.**
- 6 goals (both tracks, β* = 10/18/25°) × 4 iterations = 24 laps per car, the quadruped's 6 × 4.
- A NaN from a lap that spun out early (a non-finite predicted error) poisoned a network. Two fixes:
  - a non-finite guard;
  - rollback: a car whose laps lose ground goes back to its previous network, with its step cap halved, as the
    quadruped's fall rollback.
- The trust region now covers the whole bank. With only the drift goals in it, grip degraded.
- Settings probed: step size 0.04 → 0.08, anchor weight 1 → 0.3, age decay 0.5 → 0.8. All within noise; defaults kept.

**Positivity (A1) on the car.** Consecutive laps of the same goal on the same car. The question is whether the
nominal model's predicted change in the lap's error has the same sign as the measured change:
- ours: 64–69 % (nominal sim) and 80 % (DR sims);
- 180° flip control: 25 %;
- open-loop G: 53 % (chance).

The quadruped's figure was 79 %.

**Controls.** Open-loop G is useless for an unstable drift: zero-shot 35 % → 35 % after 24 laps. The flipped
Jacobian destroys it: 7.5 %. The closed-loop sensitivity through the network's own feedback is what makes the car's
hardware stage work.

**Baselines.**
- **PPO+DR and RMA:** same sim, same observation and network, warm-started from the v1 LQR clone. Budget: 300
  iterations × 1024 lanes ≈ 280k sim laps, 54 min.
- **FADA-style:** IDM(o, λ·d) over the DR family. 2.4M transitions ≈ 14k sim laps, then LoRA hindsight adaptation
  on the same 24 laps.
  - It **cannot drift** at any planner gain (λ = 0–0.95: spins out within 0.5 laps); it can grip-drive at λ ≥ 0.8.
  - A one-step inverse model doesn't stabilize an open-loop-unstable drift.
  - Caveat: the planner is a decaying deviation target, not FADA's learned planner.
- **Per-track ILC:** plan LQR + periodic feedforward. It trains directly on every evaluation plan, held-out ones
  included.

**real_mu (friction ×0.8) is where the methods split.**
- PPO+DR drifts it zero-shot (success 1.0 on both tracks): about 9 % slower than the plan, with less throttle.
- Our networks keep the plan's speed and push more current. Like the LQR they were cloned from, they over-rotate
  (|β| 34–49° against 22°) and spin out within half a lap. The DR-B sim stage doesn't fix it: the low-friction lanes
  drop out of the ILC rows, the targets track a plan that sits at the stability edge, and DAgger labels come from the
  nominal LQR.
- Our hardware stage makes slow progress there: after 48+ laps the square2fast laps stop spinning out. 24 laps
  aren't enough, even with the speed rows loosened (`--err-scale ... 1.0` / `3.0` on vx).
- So for continuous drift, DR's robustness and ILC's convergence are complementary. PPO+DR plus our hardware stage
  is the strongest all-round arm, the car version of the quadruped's "initialized from a DR policy, the same
  hardware stage gives our best result".

## RESULTS (2026-10-09 ~05:40; final evaluation `reeval.py`, `results_table.txt`)

**Protocol.**
- 5 multi-body cars × 8 plans × 12 runs, each run 5 chained laps with no reset.
- Starts spread over the lap with a perturbation; identical seeds for every arm (1701).
- Success: no spin-out or departure, RMS e_y ≤ 10 cm, and RMS sideslip error ≤ 8° over all laps.
- Groups: **drift** = β* 25° on both tracks (the end product); **held-out** = β* 14 and 21° (never trained on,
  neither in sim nor on the cars); **grip** = β* 0.
- Each drift cell is 120 runs (±4.5 points binomial; more in practice, since runs cluster by car).
- The per-track ILC rows train on every evaluation plan, held-out ones included.

```
arm                          | drift ok% | fail% | e_y cm | dbeta deg | held-out ok% | fail% | grip ok% | fail% | real laps/car
-----------------------------+-----------+-------+--------+-----------+--------------+-------+----------+-------+--------------
lqr_zs [x12]                 |      34.2 |  29.2 |    7.4 |       9.6 |         40.0 |  20.4 |     89.2 |   0.8 |             0
v3nom_s0_zs [x12]            |      34.2 |  35.0 |    6.6 |       8.6 |         35.0 |  25.8 |     89.2 |  10.0 |             0
v3nom_s1_zs [x12]            |      38.3 |  29.2 |    7.0 |       9.2 |         33.3 |  28.7 |     90.0 |   7.5 |             0
v3nom_s2_zs [x12]            |      32.5 |  24.2 |    7.0 |       9.3 |         30.0 |  27.1 |     90.0 |   8.3 |             0
v3nom_s0_hw24r [x12]         |      49.2 |  37.5 |    4.8 |       6.0 |         43.8 |  25.0 |     85.0 |  10.0 |            22
v3nom_s1_hw24r [x12]         |      60.0 |  24.2 |    5.2 |       7.0 |         40.0 |  29.6 |     80.8 |   5.0 |            22
v3nom_s2_hw24 [x12]          |      59.2 |  22.5 |    4.9 |       6.7 |         40.0 |  35.8 |     80.0 |  20.0 |            22
v3nom_s0_hw24_s2 [x12]       |      55.0 |  31.7 |    4.5 |       5.8 |         39.6 |  32.9 |     90.0 |  10.0 |            22
v3nom_s0_hw24vs [x12]        |      53.3 |  31.7 |    4.8 |       6.1 |         40.0 |  25.8 |     90.0 |  10.0 |            22
v3nom_s0_hw48 [x12]          |      50.8 |  37.5 |    5.2 |       6.3 |         37.1 |  33.3 |     88.3 |  10.0 |            43
v3nom_s0_hw96 [x12]          |      48.3 |  41.7 |    4.1 |       5.4 |         30.0 |  51.7 |     90.0 |  10.0 |            87
v3drA_s0_zs [x12]            |      58.3 |  31.7 |    5.6 |       7.8 |         34.6 |  27.9 |     81.7 |  14.2 |             0
v3drA_s0_hw24 [x12]          |      56.7 |  33.3 |    3.9 |       5.4 |         39.6 |  33.3 |     86.7 |  10.8 |            22
v3drA_s1_zs [x12]            |      52.5 |  27.5 |    6.4 |       8.5 |         30.4 |  36.2 |     87.5 |  10.0 |             0
v3drA_s1_hw24 [x12]          |      60.0 |  27.5 |    4.8 |       6.5 |         41.2 |  38.3 |     90.0 |  10.0 |            22
v3drB_s0_zs [x12]            |      50.0 |  27.5 |    6.0 |       8.3 |         32.5 |  25.0 |     90.0 |  10.0 |             0
v3drB_s0_hw24 [x12]          |      56.7 |  24.2 |    3.9 |       5.9 |         44.6 |  27.9 |     83.3 |   7.5 |            22
v3drB_s1_zs [x12]            |      50.0 |  37.5 |    5.9 |       8.3 |         38.3 |  18.3 |     90.0 |   2.5 |             0
v3drB_s1_hw24 [x12]          |      50.0 |  31.7 |    4.4 |       6.5 |         35.8 |  25.8 |     90.0 |   4.2 |            22
ppo_dr_zs [x12]              |      27.5 |   5.8 |    7.5 |      10.9 |         31.2 |   1.7 |     90.0 |   0.0 |             0
ppo_dr_hw24 [x12]            |      54.2 |  19.2 |    5.8 |       8.0 |         51.7 |   0.8 |     90.0 |   0.0 |            25
ppo_dr_hw24_s2 [x12]         |      56.7 |  20.8 |    5.8 |       7.6 |         57.5 |   0.4 |     90.0 |   0.0 |            25
ppo_dr_hw48 [x12]            |      65.0 |  12.5 |    3.5 |       6.3 |         58.3 |   3.3 |     90.0 |   0.0 |            49
rma_zs [x12]                 |      19.2 |  20.0 |    8.2 |      10.7 |         39.2 |   4.2 |     90.0 |   0.0 |             0
fada_hw24_zs [x12]           |       0.0 |  93.3 |   35.1 |      23.0 |          0.0 |  82.5 |     43.3 |  20.0 |             0
fada_hw24 [x12]              |       0.0 | 100.0 |    nan |       nan |          0.0 | 100.0 |      0.0 | 100.0 |             7
v3nom_s0_hw24_openG [x12]    |      35.0 |  35.0 |    6.6 |       8.6 |         35.0 |  25.8 |     88.3 |  10.0 |            22
v3nom_s0_hw24_flip [x12]     |       8.3 |  54.2 |    8.2 |      11.3 |         22.1 |  27.1 |     67.5 |  20.0 |            21
v3nom_s0_hw24_noTR [x12]     |      50.8 |  30.0 |    5.7 |       7.2 |         33.8 |  32.9 |     80.0 |  19.2 |            22
v3nom_s0_hw24_noRB [x12]     |      49.2 |  35.8 |    4.8 |       6.1 |         41.7 |  25.0 |     83.3 |  10.0 |            22
v3nom_s0_hw24_b25only [x12]  |      39.2 |  50.8 |    3.5 |       4.5 |         18.7 |  52.1 |     74.2 |  20.0 |            22
bcB_zs [x12]                 |      32.5 |  38.3 |    6.7 |       9.0 |         38.3 |  24.6 |     90.0 |   0.0 |             0
bcB_hw24b [x12]              |      49.2 |  36.7 |    5.2 |       6.9 |         40.0 |  25.0 |     80.0 |   9.2 |            20
v1narrow_hw24 [x12]          |       9.2 |  35.8 |    8.8 |      13.4 |         18.7 |  29.6 |     90.0 |   0.8 |            22
v3noexp_s0_zs [x12]          |      23.3 |  40.8 |    6.7 |       9.4 |         34.6 |  30.8 |     80.0 |  10.0 |             0
v3noexp_s0_hw24 [x12]        |      43.3 |  40.8 |    5.2 |       7.0 |         39.6 |  22.5 |     89.2 |  10.8 |            22
v3olG_s0_zs [x12]            |      34.2 |  33.3 |    6.7 |       8.9 |         34.2 |  21.2 |     90.0 |   5.0 |             0
v3olG_s0_hw24 [x12]          |      49.2 |  25.0 |    5.5 |       7.4 |         36.2 |  26.7 |     90.0 |  10.0 |            22
v4drA_s0_zs [x12]            |      58.3 |  30.8 |    5.6 |       7.7 |         33.3 |  44.2 |     75.0 |  25.0 |             0
v4drB_s0_zs [x12]            |      47.5 |  20.8 |    6.6 |       8.0 |         39.6 |  25.8 |     90.0 |  10.0 |             0
track_ilc (3 laps per plan)  |      50.0 |  37.5 |    5.0 |       6.5 |         50.0 |  20.4 |     90.0 |   0.8 |            24
track_ilc (4 laps per plan)  |      50.0 |  38.3 |    4.6 |       5.9 |         50.0 |  20.4 |     90.0 |  10.0 |            32
track_ilc (12 laps per plan) |      60.0 |  30.0 |    3.2 |       4.4 |         64.6 |  22.1 |     90.0 |   6.7 |            96
```

**Main numbers.**

| Arm | Drift success | Notes |
|---|---|---|
| Ours, nominal sim, zero-shot | 35.0 % (32.5 / 38.3 / 34.2 over 3 sim seeds) | the plans' LQR: 34.2 % |
| Ours + 24 real laps per car | **56.1 %** (49.2 / 60.0 / 59.2; a second hardware-stage seed on sim seed 0: 55.0) | e_y 6.9 → 5.0 cm, sideslip error 9.0 → 6.6°; held-out 33 → 41 % |
| Per-track ILC, equal budget (3 laps per plan, 24 per car) | 50.0 % | 60.0 % at 96 laps per car, trained on the evaluation plans themselves |
| DR-A (DR fine-tune) | 55 → 58 % with the hardware stage | |
| DR-B (DR from scratch) | 50 → 53 % | |
| PPO+DR | zero-shot 27.5 % with only 5.8 % spin-outs → **54–57 %** with our hardware stage (24 laps), **65.0 %** with 48 laps | held-out 31 → 58 %, grip 90 %: the best result |
| RMA, zero-shot | 19.2 % | |
| FADA-style | 0 % | cannot drift |

The pattern matches the quadruped's:
- DR raises zero-shot (35 → 50–58 %).
- The hardware stage lifts the nominal network to the DR level.
- Initialized from a DR policy (PPO+DR), the same hardware stage gives the best result: the two are compatible.

**Sample and time efficiency** (sim stage, one GPU, shared):

| Method | Sim laps | Wall time | vs ours |
|---|---|---|---|
| Ours (bcB warm start + 30 deep-ILC iterations) | ≈ 6,000 (916 + 5,068) | ≈ 10 min | — |
| DR-B | ≈ 4,900 | 6.4 min | — |
| PPO+DR / RMA | ≈ 280,000 | 54 min | **47× more samples, 5.4× more time** |
| FADA data | 14,400 | — | — |

**Ablations** (ours, sim seed 0, 24 laps unless noted):

| Ablation | Drift success | Note |
|---|---|---|
| Open-loop G | 35 % (no gain) | |
| Flipped G | 8 % | |
| No trust region | 51 % | held-out 34 %, grip 80 % |
| No rollback | 49 % | |
| All 24 laps on the two β 25 plans | 39 % | lowest e_y (3.5 cm), but held-out collapses to 19 % |
| Varied starts | 53 % | |
| 48 laps | 51 % | |
| 96 laps | 48 % | e_y keeps falling, 4.1 cm; held-out degrades to 30 %: specialization |

Sim-stage ablations:

| Variant | Zero-shot → +24 laps |
|---|---|
| Narrow exploration (v1) | — → 9 % |
| No exploration | 23 → 43 % |
| Open-loop G in the sim stage | 34 → 49 % (≈ the closed-loop sim stage: in sim the network's own feedback barely matters) |
| bcB warm start only (no sim ILC) | 32.5 → 49 % |

**Infinite drift (`long_eval.py`, 20 laps without reset, from the plan's start).**
- PPO+DR + 48 laps: 9 of 10 car × track chains complete 17–21 laps, e_y 1.6–3.8 cm (8 cm on real_mass figfast).
  The per-lap error slope is within ±0.2 cm per 10 laps on 8 of them; the real_mass figfast chain drifts by
  0.44 cm per 10 laps.
- Ours (nominal sim) + 24 laps: 6–8 of 10.
- square2fast is held for all 20 laps on every car except real_mu. The figfast transitions on the perturbed cars
  are the hard part.

**GIFs** (`gifs/`):
- `endproduct_figfast_b25.gif`, `endproduct_square2fast_b25.gif`: PPO+DR plus our hardware stage on real_nom,
  real_mu and real_lag; 3 continuous laps, heading against velocity, β against the plan's β*.
- `drift_square2fast_real_mass.gif`: zero-shot against +24 laps on the heavy car.
- `drift_figfast_hw24.gif`: ours after the hardware stage on real_nom and real_lag.
- 920×620 to 1380×620, under 1 MB.

**Figures:** `figs/methods.png` (zero-shot against +24 laps per method), `figs/budget.png` (success and e_y against
real laps: ours, PPO+DR + ours, per-track ILC).

## Limitations and next steps
1. **real_mu (friction ×0.8).** Our LQR-cloned networks over-rotate and spin out within half a lap, so the ILC gets
   almost no rows, even from DR sims and with loosened speed rows (v4). PPO+DR survives it.
   - Candidates: a sim-stage DAgger expert re-solved for the lane's friction (the LQR around a friction-specific
     plan), or a recovery row (penalize |β| past the plan's).
2. **Success saturates at about 55–65 %** under the strict figfast sideslip bound on the perturbed cars. Tracking
   error keeps falling with more laps, but held-out plans degrade: the trust region fades as N0 / (N0 + n). A
   constant anchor for long budgets is untested.
3. **The FADA baseline is simplified** (no learned planner), so treat its 0 % with care.
4. **No ROS in the loop.** The cars are the gym fork's MB dynamics run in-process at a fixed 40 Hz. `f1tenth_gym_ros`
   (dev-humble) builds in the `f1t` container, and `lifted_linear_tire_20261009/mb_sim/mb_bridge.py` is the ROS
   front end, but no runs went through ROS tonight.
5. **Plans** are a homotopy in β_d (0, 10, 14, 18, 21, 25°), saved in `plans/`. Re-solve with
   `python trajopt_drift.py <track> --beta 25`.
6. **Drift entry.** Every run starts in the drift, from the plan's state. Initiating the drift from rest isn't
   planned or learned yet: a one-off entry segment in the TO is the next step.

## Reproduce
```bash
cd src/ilc_f1tenth/ilc2real; PY=~/miniconda3/envs/f1t/bin/python
$PY trajopt_drift.py mocap_square2fast --out plans; $PY trajopt_drift.py mocap_figfast --out plans
PL="mocap_square2fast_b0 mocap_square2fast_b10 mocap_square2fast_b18 mocap_square2fast_b25 mocap_figfast_b0 mocap_figfast_b10 mocap_figfast_b18 mocap_figfast_b25"
$PY sim_train.py --plans $PL --out runs/bcB --bc-iters 10 --iters 0 --a-off 0.3 --pert 2.0 --explore-frac 0.8
$PY sim_train.py --plans $PL --out runs/v3nom_s0 --init runs/bcB/policy_final.npz --anchor-w 1.0 --beta 0.3 --cap 0.04 \
    --max-ey 0.2 --a-off 0.15 --pert 1.5 --explore-frac 0.7 --iters 30
$PY hw_stage.py --policy runs/v3nom_s0/policy_final.npz --out runs/hw/<arm> --iters 4     # 24 real laps per car
$PY reeval.py <arm>; $PY table.py --big; $PY long_eval.py <arm> --laps 20; $PY figures.py
```

## Follow-ups (05:20–05:40)

**The best arm, PPO+DR + our hardware stage at 48 laps.**
- Second hardware-stage seed: **76.7 %** drift success.
- Mean over the two seeds: **70.8 %** (65.0 / 76.7). Spin-outs 12.5 %, e_y 3.4 cm, held-out 54–58 %, grip 90 %.
- At 24 laps: 54.2 / 56.7 %.

**Our sim stage started from the PPO+DR network** (`v5ppoNom`: nominal sim; `v5ppoDR`: DR sim).
- Zero-shot drift success: 27 / 28 %, but spin-outs rise from 6 % to 22–25 %.
- +24 laps: 49 / 47 %.
- So our plan-tracking sim ILC erodes the low-friction survival that DR-RL learned. The robustness comes from DR's
  survival reward, the convergence from the ILC hardware stage, and the order matters: DR first, then ILC on the car.

**GIFs** `gifs/endproduct_*.gif` now show the best arm (`ppo_dr_hw48_s2`). In this single sample from the plan's
start, real_mu holds figfast for 3 laps but spins out on square2fast at lap 1.6, the arm's remaining failure mode.

**Suggested sentence for the abstract's [CAR DRIFTING] slot.** Not inserted into `abstract.txt`; your call:
> On a 1/10-scale car, the same pipeline learns a continuous multi-lap drift on two motion-capture tracks with 47
> times fewer simulated laps than PPO with domain randomization; 24 laps on each of five multi-body test cars raise
> full-drift success from 35 % to 56 %, and initialized from the domain-randomized policy, 48 laps reach 71 %.

**Check:** `check_real_vs_sim.py` compares the numpy twin used on the real cars with the JAX sim. Max differences:
observation 7e-6, ILC error 8e-7, expert action 1e-4 relative, projection index 0. The real-car runs see exactly
the observations the network was trained on.

**Replicates and a constant trust region (06:00).**
- PPO+DR + 48 laps over three hardware-stage seeds: 65.0 / 76.7 / 77.5 %, **mean 73.1 %**, spin-outs 10–12.5 %,
  held-out 54–58 %.
- Constant anchor weight (`--anchor-n0 1e9`) at 96 laps:
  - ours: 48.3 → 52.5 % drift with fewer spin-outs, but held-out stays at 30 %, so the fading trust region was not
    the cause of the held-out decline;
  - PPO+DR: 67.5 % drift, held-out **65.4 %**, the best held-out of any arm.

**real_mu diagnosis.**
- The plan's LQR re-designed for μ ×0.8 also spins out at μ ×0.8 in the nominal sim (0.12–0.26 laps), the same as
  the nominal LQR. So it is the plan, not the gains: the nominal plan's speed is infeasible at that friction.
- Plans re-solved for μ ×0.8 (`plans/*_mu80.npz`, `trajopt_drift.py --mu-plan 0.8 --tag _mu80`) drift the same path
  and sideslip about 8 % slower. Flown by their LQR they give real_mu 100 % success on both tracks, but the
  nominal-friction cars 0 %: the goal must match the car.
- Hence the **goal fallback**, `hw_stage.py --fallback-suffix _mu80`, the quadruped's stall guard for unreachable
  goals:
  - a goal that spins out twice in a row switches, for that car, to its conservative variant;
  - it is triggered by outcomes only, with no system ID;
  - the network's bank (v6: 8 + 6 conservative plans, nominal sim) contains both variants.

## Goal fallback result (06:15): closes most of the real_mu gap without DR or system ID

**v6** is the v3 recipe on a 14-plan bank (the 8 nominal plans plus the 6 conservative `_mu80` drift plans), in the
nominal sim, 2 seeds. The hardware stage runs with `--fallback-suffix _mu80`.
- **Scoring:** a car that switched a goal is scored on the variant it settled on for the nearest trained goal of the
  same track (`eval_big_adaptive.json`). The conservative plan has the same path and sideslip, about 8 % slower.
- **Without the fallback** (24 laps): 49.2 / 60.0 % drift success, spin-outs 40.8 / 30.8 %.
- **With the fallback** (24 laps; triggered after 2 or 3 consecutive spin-outs, 4 runs): 60.0 / 69.2 / 60.0 / 60.0 %,
  **mean 62.3 %**, spin-outs **7.9 %** mean (0.8–10.8 %).
- **With the fallback, 48 laps:** 60.0 %, 6.7 % spin-outs, held-out 53 %.
- real_mu switches its drift goals after 1–3 iterations, then drifts figfast at 100 % success and square2fast at
  92 % (one seed still spins out on square2fast).
- At 24 laps our pipeline with the fallback (62 %, 8 % spin-outs) now **beats PPO+DR + our hardware stage** (55 %,
  20 %), from 47× fewer sim samples. PPO+DR reaches 73 % at 48 laps.

Interpretation: the ILC converges to whatever goal it is given. When a goal is unreachable on a car, the fix is
choosing the goal from outcomes, not changing the update.

**20-lap chains of the goal-fallback arm** (`v6nom_s1_hw24fb`, `long_eval_{nominal,mu80}.json`):
- **real_mu, on the conservative plans it switched to:** 18.7 laps on square2fast (e_y 5.6 cm, sideslip error 4.4°)
  and 19.5 laps on figfast (e_y 2.4 cm, 2.9°). Per-lap error is flat. Continuous drift on the low-friction car.
- **real_nom and real_mass:** all 20 laps on the nominal plans, e_y 2–3 cm on square2fast. real_mass figfast:
  6 cm, with shallow sideslip.
- **real_act and real_lag:** they switched figfast to the conservative plan, where they complete 23–24 laps without
  a spin-out but drift shallowly (|β| 10–11° against about 20°). On square2fast they stay on the nominal plan for 19
  laps.

**GIFs** (`gifs/`):
- `ours_fallback_{figfast,square2fast}_mu80.gif`: our pipeline (v6 + goal fallback, 24 laps) on real_mu, 3
  continuous laps.
- `endproduct_*.gif`: PPO+DR + our hardware stage (48 laps) on real_nom, real_mu and real_lag.
- `drift_square2fast_real_mass.gif`: zero-shot against +24 laps.
- `drift_figfast_hw24.gif`: ours, nominal sim + 24 laps.

**Suggested abstract sentence, updated:**
> On a 1/10-scale car, the same pipeline learns a continuous multi-lap drift on two motion-capture tracks with 47
> times fewer simulated laps than PPO with domain randomization; 24 laps on each of five multi-body test cars raise
> full-drift success from 45 % to 62 % and cut spin-outs from 40 % to 8 %, and initialized from the
> domain-randomized policy, 48 laps reach 73 %.

All three numbers are on the same footing: the v6 network, 14-plan bank, zero-shot → + 24 laps with the goal
fallback, mean over 2 sim seeds × 2 trigger settings. The PPO+DR figure is the 48-lap mean over 3 hardware-stage
seeds.
