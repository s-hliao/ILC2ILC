# ILC2Real on the F1TENTH: continuous drift on the LLA-MPC car

## μ 0.2 rerun (plastic tires), 2026-10-09 11:46–13:46

This is the current result. The μ 0.6 rerun from this morning is kept below; its files are in `archive_mu06/`.
The overnight RWD / Pacejka run is in `archive_rwd_mf/`.

### What changed from the μ 0.6 rerun
- **μ 0.2 everywhere** (`F1T_MU=0.2`), for plastic tires. This covers the Fiala TO, the Fiala nominal sim and the
  multi-body cars, whose real_mu variant is ×0.8, i.e. 0.16. The plans were re-solved in `plans_mu02/`; the speed
  target scales with μ (A_LAT = 5.5·μ/0.6), and the `_mu80` fallback plans are re-solved too.
- **Objective (per the user: "the network should be able to converge for EITHER drifting or gripping").** The sim
  stage still tracks the whole plan (ERR_SCALE). The hardware stage and the per-track ILC use the **task objective**
  (TASK_SCALE): path e_y plus pace |V| against the plan, with sideslip free. Success = on track, RMS e_y ≤ 10 cm and
  pace ≥ 90 % of the plan's. |β| is reported, not scored. The μ 0.6 table scored drift success on sideslip, so its
  0 % column is not directly comparable. Its grip column, and its e_y of 20–30 cm, are.
- **Lap-budget ablation** (`scheduler_laps.py`): 2–10 laps per car on the two β-25 plans, for ours (sim seeds 0 and
  1) and PPO+DR, plus per-track ILC at 1–5 laps per plan.
- Same pipeline otherwise: lifted-time G, no state lifting, nominal Fiala sim, and perturbations only on the
  multi-body cars. Evaluation: `reeval.py`, 5 cars × 8 plans × 12 runs × 5 chained laps.
  - `results_table.txt`, `figs/lap_budget.png` (`fig_laps.py`), `figs/budget.png`, `figs/methods.png`.
  - 20-lap chains: `logs/long_*.log`.

### Result: the drift plans now transfer, at partial sideslip
Full-drift plans (β 25°, both tracks). 5 cars × 2 plans × 12 runs per row.

| Arm | b25 success | fail | e_y (cm) | pace | \|β\| flown | held-out ok | grip ok |
|---|---|---|---|---|---|---|---|
| Plan LQR, zero-shot | 62 % | 22 % | 7.4 | 98 % | 13.2° | 74 % | 80 % |
| Ours (seed 0), zero-shot | 78 % | 4 % | 7.0 | 96 % | 13.3° | 79 % | 90 % |
| Ours (seed 0) + 2 laps | 79 % | 3 % | 6.2 | 96 % | 13.4° | 79 % | 90 % |
| Ours (seed 0) + 10 laps | 79 % | 8 % | 4.5 | 96 % | 13.6° | 79 % | 88 % |
| Ours (seed 0) + 24 laps | 79 % | 11 % | 4.1 | 97 % | 13.5° | 79 % | 79 % |
| Ours (seed 0) + 48 / 96 laps | 79 / 45 % | 11 / 28 % | 3.5 / 9.9 | 98 / 94 % | 13.6 / 12.8° | 78 / 43 % | 32 / 18 % |
| Ours (seed 1), zero-shot → + 24 laps | 70 → 70 % | 17 → 18 % | 8.3 → 5.6 | 96 % | 13.4–13.9° | 69 → 75 % | 81 → 88 % |
| Our learner + DR, A / B, zero-shot (seed 0) | 79 / 80 % | 3 / 0 % | 6.5 / 5.7 | 96 % | 13.0–13.3° | 79 / 80 % | 100 % |
| Our learner + DR, A / B, + 24 laps (seed 0) | 76 / 70 % | 13 / 10 % | 4.0 / 4.6 | 97 % | 13.8–13.9° | 79 / 73 % | 90 / 80 % |
| PPO+DR, zero-shot | 73 % | 2 % | 7.3 | 97 % | 14.4° | 92 % | 100 % |
| PPO+DR + 2 laps of our hardware stage | **96 %** | 0 % | 5.7 | 97 % | 14.8° | 91 % | 90 % |
| PPO+DR + 4 / 6 / 8 / 10 laps | 86 / 78 / 87 / 79 % | 0–4 % | 5.4–5.9 | 97 % | 14.5–14.9° | 84–91 % | 80–90 % |
| PPO+DR + 24 / 48 laps (2 / 3 seeds) | 83–84 / 56–84 % | 4–28 % | 4.5–7.8 | 95–98 % | 14.5–17.2° | 78–95 % | 80–100 % |
| RMA, zero-shot | 60 % | 2 % | 9.5 | 98 % | 14.5° | 85 % | 90 % |
| FADA-style, + 24 laps | 0 % | 100 % | — | — | — | 0 % | 0 % |
| Per-track ILC, 1 → 5 laps per plan | 69 → 77 % | 21 → 4 % | 5.8 → 5.6 | 97–98 % | 13.5–14.0° | 80 % | 99–100 % |
| Per-track ILC, 12 laps per plan | 80 % | 0 % | 5.5 | 97 % | 14.7° | 80 % | 97 % |

**Pattern.**
- **Every working method converges to a partial drift.** Mean |β| is 13–15° against the plan's 25°, at 96–98 % pace,
  on every arm. The task objective leaves sideslip free, and the cars settle between grip and the planned slide. The
  20-lap chains hold it with no drift-off:
  - ours + 24 laps covers about 19–20 laps per chain at e_y 0.9–3.4 cm and |β| 10–16° on real_nom, mass, act and lag;
  - the e_y slope is about 0 cm per 10 laps.
- **real_mu (μ 0.16) is the ceiling.** It caps the ILC-trained networks at 80 %: the other 4 cars are at 92–100 %,
  and real_mu at **0 %** for every budget. That holds for ours, for our learner + DR, and for the per-track ILC.
  - On real_mu the networks run wide (e_y 13–20 cm) or slide out (|β| 18–31°).
  - PPO+DR is the only family that handles real_mu: 83 % zero-shot and after 2 laps, at e_y 9.3 cm. Its DR covers μ.
  - Our hardware stage then trades real_mu away with more laps: 46 % at 24 laps, 38 % at 48.
  - RMA reaches 46 % on real_mu.
- **The hardware stage fixes PPO+DR's nominal-car gap in 2 laps.** PPO+DR zero-shot gets 50 % on real_nom and 54 % on
  real_mass; after 2 laps those reach 96 % and 100 %. That is the best arm here, 96 % b25 success, against 73 %
  zero-shot. With more laps it falls back toward 80–87 %, and 48 laps is noisy across seeds (56–84 %).
- **Ours (nominal sim) is already at the 4-car ceiling zero-shot.** Laps buy accuracy rather than success: e_y goes
  7.0 → 6.2 → 4.5 → 4.1 → 3.5 cm at 0 / 2 / 10 / 24 / 48 laps.

### Lap budget (`figs/lap_budget.png`)
Success, e_y and |β| against real laps per car on the β-25 plans: 0 = `_zs`, 2–10 = `_lapN`, 24 / 48 / 96 = `_hw*`.
Per-track ILC at b laps per plan is plotted at 2b; it spends the same again on the 6 other plans.
- **4–10 laps is enough.** Ours reaches its best success by 2 laps and half its e_y gain by 10. The rest of the gain
  (to 3.5 cm) costs 48 laps.
- **Past about 48 laps the stage over-fits the drift plans.** Grip-plan success drops 88 → 32 → 18 % at
  10 → 48 → 96 laps, with failures rising to 27–36 %. At 96 laps b25 success falls to 45 %, and a constant trust
  region (`_cTR`) only partly helps (59 %). Those cars are being pushed to drift a gripping plan.
- **Seed 1** starts lower (70 %) and stays at 68–76 %; e_y drops 8.3 → 4.7 cm by 10 laps.
- **Variants at the same budget** (seed 0):
  - 6 goals × 1 iteration (`lap6g6`) instead of 2 goals × 3: same success, worse e_y (6.5 against 5.7 cm).
  - Larger steps (`lap6big` / `lap10big`): same b25 success. Grip-plan success drops to 80 / 50 %, and held-out
    failures rise to 9–12 %.
  - Big steps don't help.
- **Per-track ILC** needs 10 laps on the β-25 plans to reach 77 %, and 96 laps to reach 80 %. It also fails real_mu
  at every budget, and gets 50 % on real_lag up to 4 laps per plan.

### Controls and ablations (seed 0, + 24 laps unless noted)
- **Sim stage:**
  - The sim-stage exploration matters: without it (`v3noexp`) zero-shot success is 21 %, against 78 %.
  - The open-loop-G sim stage (`v3olG`) is as good as ours: 78 % zero-shot, 80 % after 24 laps.
  - The lifted closed-loop G is not what carries the sim stage on this car.
- **Hardware-stage Jacobian:**
  - flipped G: 51 % and e_y 12 cm, so the direction matters;
  - open G: 79 % at e_y 6.2 cm;
  - no trust region / no rollback: 78 % / 79 %, so neither is needed at 24 laps.
- **Goal fallback to the `_mu80` plans** (`v6*fb`): 48–52 %, worse than without it (58 %). On the 20-lap chain the
  conservative plans do run real_mu clean (e_y 2.4–5.6 cm, |β| 20–25°), but against the nominal plan's pace that
  isn't a success.
- **RL warm-started from our sim stage** (`v5ppoNom` / `v5ppoDR`, zero-shot): 42 %.
- **FADA:** 0 %. The LoRA adaptation spins every car out.

### Compared with the earlier runs
- **μ 0.6 (`archive_mu06/`):**
  - No arm drifted the multi-body cars (0 % drift success). Grip transferred at 80–100 %, but the drift runs gripped
    up at e_y 20–30 cm.
  - At μ 0.2 the same pipeline tracks the drift plans at e_y 4–7 cm and holds 13–15° of sideslip.
  - The open-diff load-transfer problem behind μ 0.6 shrinks with μ: lateral load transfer scales with the lateral
    acceleration, about μ·g.
- **RWD / Pacejka overnight run (`archive_rwd_mf/`):** a different car (made-up driveline, μ 1.05); not comparable.

### Open
- **real_mu:** the nominal-sim ILC networks never handle a 20 % lower μ on the drift plans, while DR over μ does.
  Our learner + DR used the same DR family and still got 0 %, so its DR range or weighting needs checking.
- **96-lap over-fitting:** stop the hardware stage by budget (≤ 10–24 laps), or anchor the grip plans.


## μ 0.6 rerun, 2026-10-09 09:05–10:17 (superseded by the μ 0.2 rerun above)

This supersedes the overnight run. That run used a car I had made up: RWD, Pacejka tires, the gym's μ 1.05, and
double the real motor constant. Its notes, results, GIFs and code snapshot are in `archive_rwd_mf/`.

### Setup (per the user, 2026-10-09 morning)

- **The car:** LLA-MPC's model, `LLA-MPC-online` `origin/current`, `llampc/llampc/nmpc_gen_fiala_fixed.py`
  (`export_model`, exact=False), with the NMPC's mean tire model from `llampc_fiala_fixed.fiala_setup`:
  - single track, Fiala/brush tires with combined slip, Cf 250 and Cr 225 N/rad, μ 0.6, static axle loads;
  - **one wheel speed driving both axles** (front axle driven);
  - 4.6 kg; torque constant 11.82 · 1.5 · 2 · 0.000726 N·m/A;
  - current −25…50 A at 300 A/s, steering ±0.34 rad at 3.2 rad/s.

  `car_model.py` matches LLA's equations to 3e-16, verified against `ilc_f1tenth/dynamics.py`, which is the same
  code. The network acts through set points for steering and current; the servo and current loop are first
  order, rate limited, and clamped by a softplus (identity inside the range).
- **Fiala for both** (the user's instruction): the initial TO (IPOPT, `trajopt_drift.py`) and the nominal sim and
  linearization (`sim_jax.py`), with the **lifted-time** ILC G and **no state lifting**. The Fiala state is the
  multi-body car's planar projection [X, Y, ψ, vx, vy, r, ω_w, I, δ], so the Fiala Jacobians are taken directly at
  measured states.
- **"Real" cars:** the f1tenth_gym fork's multi-body model (`mb_fiala.py`, generated by `make_mb_fiala.py` from the
  fork's `multi_body.py` with only the tire block replaced by LLA's Fiala tire on each corner). Calibrated to the
  LLA car: mass, inertias, steering, per-tire stiffness Cf/2 and Cr/2, μ 0.6 on each wheel's own load.
  - Driveline: a locked centre shaft (motor inertia) into **open front and rear diffs**, the stock Traxxas 4x4
    setup (`mb_car.py`; viscous LSD coupling `LSD = 0`).
  - Five variants, as last night: real_nom, real_mass (+25 %), real_mu (×0.8), real_act, real_lag.
- **Drift plans:** IPOPT, Radau in arc length, periodic, homotopy in β_d. Speed target is
  min(design, √(5.5 m/s² / |κ|)). At β 25°:
  - square2fast 3.15 m/s, 3.55 s lap, β 16–25°, e_y ≤ 6.6 cm;
  - figfast 2.61 m/s, 5.84 s lap, β ±26–29°.

  On this car a drift is a **power slide**: all four wheels spin (slip 0.2–0.3), the current runs up to 50 A,
  and the steering points *into* the turn. The equilibria (β −10…−35°) are at V ≈ 2.1 / 2.4 / 2.95 m/s for
  R 0.75 / 1 / 1.5 m. At R 0.75 m only drift works; grip is steering-limited there. Conservative `_mu80` plans
  (solved for μ ×0.8) are about 10 % slower.
- **Scheduler:** `scheduler.py`, 80 jobs, the whole of last night's matrix. All done, none failed. Final
  evaluation: `reeval.py`, 5 cars × 8 plans × 12 runs × 5 chained laps (`results_table.txt`).

### Result: no method drifts the multi-body cars

| Arm | Drift success | Spin-outs | e_y (cm) | Sideslip error (°) | Grip success |
|---|---|---|---|---|---|
| Plan LQR, zero-shot | 0 % | 21 % | 30 | 17 | 80 % |
| Ours (nominal Fiala sim), zero-shot, 3 seeds | 0 % | 4–11 % | 21–24 | 17 | 80 % |
| Ours + 24 laps (3 seeds) | 0 % | 6–13 % | 22–23 | 17 | 80–87 % |
| Ours + 48 / 96 laps (96 also with a constant trust region) | 0 % | 10–16 % | 22–23 | 16–17 | 80 % |
| DR-A / DR-B, + 24 laps (2 seeds each) | 0 % | 5–53 % | 21–26 | 15–17 | 90 % |
| PPO+DR, zero-shot | 0 % | 56 % | 15 | 15 | 100 % |
| PPO+DR + our hardware stage, 24 / 48 / 96 laps | 0 % | 58–69 % | 14–19 | 14–15 | 100 % |
| RMA, zero-shot | 0 % | 63 % | 11 | 15 | 100 % |
| FADA-style | 0 % | 100 % | — | — | 0–60 % |
| Per-track ILC, 3 / 4 / 12 laps per plan | 0 % | 21–34 % | 29–30 | 16–17 | 85–90 % |
| Goal fallback (`_mu80`), 24 / 48 laps | 0 % | 4–11 % | 24 | 17 | 80–90 % |

**Pattern.**
- The ILC-trained networks stay on track by **gripping**. |β| is about 3–11° against the planned 22°, the car runs
  wide (e_y 20–30 cm), and it rarely spins out.
- The RL policies keep more sideslip but **spin out in 55–70 %** of the drift runs.
- Grip transfers for every method (80–100 %).
- The 20-lap chains (`long_eval.py`) agree. Ours + 24 laps covers 17–18 laps on real_nom without spinning out, but
  at |β| 3–10° and e_y 19–22 cm. PPO+DR + 48 laps completes square2fast on real_nom at |β| 11°, but spins out on
  figfast and on real_mu within half a lap.
- The flipped and open-loop controls show nothing: there is no drift progress to destroy.

**Why the hardware-stage ILC does not converge to the drift.**
- Lap costs stay flat over iterations. On square2fast 25°, real_nom: 2224 → 2264 → 2293 → 2255, sideslip error
  steady at 14°, while each Gauss–Newton step predicts about 5 % improvement.
- An aggressive diagnostic (`runs/diag/aggr_nom`: 6× the step cap, full GN step, no trust region, 8 laps per plan)
  lowered the square2fast cost noisily (2249 → 1583) without changing the sideslip error (13–15°). Figfast did not
  move.
- The Fiala linearization does not point toward a drift the multi-body car can hold. In the multi-body car the
  planned power slide breaks down in about 0.3 s:
  - **open diffs:** the unloaded wheels spin up (128 against 78 rad/s), the drive force is capped by them, the yaw
    rate collapses, and the car grips up;
  - **viscous LSD (0.01–1 N·m·s/rad) or locked axles:** left/right scrub gives an understeer yaw moment and the car
    departs (`t_lsd` sweep this morning).
- LLA's single-track model has one wheel speed for all four wheels and no left/right load or speed difference.
  The drift it plans depends on exactly that.

**Unsettled: can the multi-body car drift at all?** If it cannot, these drift numbers measure a plant with no
reachable drift rather than the methods. The next step, not done: optimize a steady drift directly on the multi-body
car. The JAX twin is built and checked: `mb_car_jax.py` and `mb_jax.py` match numba to 1e-15. Do this across diff
setups (open, LSD, locked) and decide which setup represents the real car, or plan with a model that has the
track-width effects.

Grip-only conclusions are not affected. `figs/` holds the summary plots of this rerun; no GIFs were made for it.

### Files added this morning
- `car_model.py`: rewritten to LLA's Fiala car.
- `mb_fiala.py` / `make_mb_fiala.py`: the multi-body model with per-corner Fiala tires.
- `mb_car.py`: rewritten. LLA calibration, driveline, `LSD` coupling.
- `mb_jax.py` / `make_mb_jax.py`, `mb_car_jax.py`, `check_mb_jax.py`: the JAX multi-body twin. Not used in this
  rerun.
- `scheduler.py`, `cache_lqr.py`.
- The rest of the pipeline was ported to the 9-state layout and offset-centred actions. `check_real_vs_sim.py`
  still matches (observation 6e-6).
