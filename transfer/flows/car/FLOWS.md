# Car (F1TENTH drift) flows: status 2026-10-22 17:30 (work moving to another machine)

## Setup (flows_car.py)
- **Map:** the closed loop over H control periods (default H = 20, 0.5 s; `--H`), in the plan-relative error
  coordinates (e_y, e_psi, vx, vy, r minus the plan at the car's own arc length, + the speed error).
  - V^s_pi(x) = e(F(x)) - e(x).
  - **sim** = the nominal Fiala single-track (sim_jax.step), with friction set to the plan's design value: 1 for the
    plans themselves, 0.8 / 0.6 for the ladder rungs `_mu80` / `_mu60`.
    - Why: in the nominal-friction sim, pi_0 crashes on the `_mu60` rungs (RMS e_y 0.29 m), so that flow is not the
      intended one.
    - In the design-friction sim, pi_0 flies them: ours e_y 0.02–0.05 m, PPO+DR 0.03–0.10 m. This is an analysis
      reference only; the method is unchanged.
  - **car** = mb_car started from the recorded multi-body state on the cars' own trajectories (perturbations added to
    it), or from mb_from_planar for the sim's states.
    - Delays and offsets are applied, and the command history is warm-started at pi's action. No sensing noise.
  - **RMA:** z_hat is held at its own trajectory's value at the nearest plan index.
- **Field error:** RMS over (e_y, e_psi, vx, vy, r) / (0.1 m, 0.2 rad, 0.5, 0.5 m/s, 1 rad/s) of
  V^car_{pi_i} - V^sim_{pi_0}, averaged over the tube.
  - Also reported: against the plan's LQR flow, the task coordinates only (e_y, speed), e_y alone, and the policy-induced
    change |V^car_{pi_i} - V^car_{pi_0}|.
- **Tube:** every 40th period of sim pi_0 and of every pi_i on the car (3 runs × 2 laps from the validation starts,
  seed 301), plus one copy perturbed at relative size 0.25. Off-tube: relative size 1.0 (IN only).
- **Goal sets**, each plan on the car's final ladder rung:
  - IN = the b25 plans, i.e. the stage's goals;
  - HULL = the b21 plans of the same tracks;
  - OUT = the b0 grip plans of the same tracks + `mocap_squareH_b25` (held-out track).
- **Task error:** RMS e_y over the 3 × 2-lap validation runs, with periods after a crash counted at 0.3 m; plus the
  stage's success test.

## Update 2026-10-22 19:50
**Seed-0 stage re-runs: finished** (18:33). They took 78–82 min each here (1 + 1 processes, single-threaded XLA).
- Dirs: `runs/{v3drBtL,ppoL_task,rmaL_task}_s0_ladR_val_its`, with `its/it0..5_<car>_policy.npz` + `its/its.json`.
- Vs the original GPU runs:
  - rungs identical on all 3 arms; same number of trials;
  - ours: per-trial J within 2.7 %;
  - PPO+DR / RMA: later trials drift (CPU float noise amplified through the iterations; max rel. ΔJ 0.53 / 0.84).
- The stages' own validation eval (`--eval-seed 301`), b25 success over the 8 cars: ours 1.000, PPO+DR 0.859, RMA 0.641.
  - The originals (eval seed 701): 1.000 / 0.875 / 0.734.

**Seed-0 flows: no numbers.**
- `run_s0_flows.sh` (H 20, tube stride 120, 2 + 2 + 1 workers) ran 18:33 – 19:45. At 19:45:15 every one of its
  processes was killed at once by something external, not by this worker. The runner, the RMA compute and the
  workers all died with no traceback in the main processes.
- That was mid-way through the car-field step (ours / PPO+DR) and the RMA sim-field step, so no `fields/*.json` were
  written. Not relaunched (past the ~1 h budget; the external kill may be the machine handoff).
- `run_remaining.sh` skips the three finished stages and redoes their flows.
- Runtime of one stage's compute at H 20, stride 120, 2 workers: sim trajectories ~20 min, car runs ~12 min, car fields
  > 40 min. So ~1.2 h per stage with 2 workers. RMA is slower: serial here.
  - Default stride 40 is ~3x the car-field cost: use FPROCS >= 16 on the big machine, or `--stride 120`.

**H tests on the fake stage** (it0 = start network, it1 = the ORIGINAL final network; real_mu and ax:mu=0.6; new code:
multi-body starts, design-friction reference; IN, field error / task-coord field error / policy change / task error):

| H | car | field err it0 → it1 | task-coord field err | policy change |V_1 - V_0| | task err (m) |
|---|---|---|---|---|---|
| 4 (0.1 s) | real_mu | 0.040 → 0.043 | 0.005 → 0.015 | 0.007 | 0.056 → 0.034 |
| 4 | ax:mu=0.6 | 0.030 → 0.030 | 0.004 → 0.005 | 0.001 | 0.045 → 0.044 |
| 20 (0.5 s) | real_mu | 0.176 → 0.184 | 0.031 → 0.064 | 0.032 | 0.056 → 0.034 |
| 20 | ax:mu=0.6 | 0.118 → 0.118 | 0.022 → 0.024 | 0.003 | 0.045 → 0.044 |

- Starting the car from its recorded multi-body state roughly halves the 0.1 s corruption: 0.072 → 0.040 on real_mu.
- **Reading so far** (2 cars, 1 seed, an old final network; not a result):
  - The stage changes the car's field by much less than the sim-to-car gap: 0.032 vs 0.18 at 0.5 s on real_mu.
  - It does NOT move the car's field back towards sim pi_0's. The field error stays flat, and in the task coordinates it
    grows (0.031 → 0.064), while the task error falls 40 %.
  - This is consistent with the task objective: the ILC asks only for e_y and pace, and leaves the drift state free.
  - So on the car the picture may be "re-correction of the task, not of the sim field". The per-iteration seed-0 run is
    needed to say this properly.
- ax:mu=0.6's final network barely differs from the start (change 0.003). Its improvement comes from the ladder rung
  (`_mu60` plan), not the ILC updates, as NOTES.md says for the low-friction cars.

## Status / remaining (final, 2026-10-22 19:55; all jobs on matcha killed by the user, work moves machines)
**Finished**
- `hw_stage.py --save-its` (append-only; default 0 reproduces old runs). Not committed.
- The seed-0 stage re-runs: `runs/{v3drBtL,ppoL_task,rmaL_task}_s0_ladR_val_its`.
  - Complete: summary.json with the validation eval, plus `its/` (networks after iterations 0–5 per car).
- flows_car.py, tested end to end on `runs/_test_fake` at H 4 and H 20 (table above).

**Not done:** any per-iteration flows on real stages, i.e. no `fields/`, `summary.json`, figures or portraits.
- The seed-0 computes were killed mid-run at 19:45.

**`run_remaining.sh` will** (idempotent):
1. Skip the three finished seed-0 stages if their dirs are copied. Otherwise it re-runs them.
2. Run the stages for seeds 1 and 2 of ours / PPO+DR / RMA, plus the extra `v3nom_s0`: 7 stages, ~80 min each at 1 + 1
   processes.
3. Run `flows_car.py compute` on all 10 stages (H 20; the default tube stride is 40).
   - That is ~3x the cost of the stride-120 estimate of ~1.2 h per stage at 2 workers. Use FPROCS >= 16, or add
     `--stride 120` in the script.
4. Make the portraits (s0; real_mu and ax:mu=0.6) and run `report`: `summary.json`, `summary_extra.json`, `table.md`,
   `figs/field_err.png`, `figs/task_err.png`.

**Inputs to copy:** listed in the header of `run_remaining.sh`.
- The ilc2real code with the uncommitted hw_stage.py / ilc_core.py edits, plans_mu02_tight, the start networks +
  rma_adapt.npz, tracks_tight, mb_sim.
- This directory, incl. `runs/*_s0_ladR_val_its` to skip their stages.
