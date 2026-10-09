# Progress and restart notes: lifted ILC for F1TENTH (as of 2026-10-09)

The goal is to make the F1TENTH ILC work like the quadruped's `JumpILC`:
- a nonlinear trajectory optimization (TO) produces the trial-1 controls;
- every trial, a lifted-form QP on a simple model's sensitivities (the linear tire) updates the controls;
- a safeguard based on the measured cost rolls back bad trials.

The method, files and results are in `README.md`, and the drift design is in `DRIFT.md`. This file covers where things stand and how to pick up again.

## Environment

- **Container:** `docker_lla_new` (image `lla:snapshot`). It mounts `~/docker_workspaces` as `/workspaces`, so this folder is `/workspaces/ilc_ws/src/ilc_f1tenth/lifted_linear_tire_20261009`.
- **Shell setup** for every command inside the container:
  ```bash
  source /opt/ros/foxy/setup.bash; source /workspaces/lla_drive_ws/install/setup.bash
  export ROS_DOMAIN_ID=42 OPENBLAS_NUM_THREADS=2 ACADOS_SOURCE_DIR=/acados LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/acados/lib
  ```
- **Plant: multi-body (current).** `cd mb_sim && python3 mb_bridge.py` (add `--log x.npz` to record hidden states).
- **Plant: stock single-track (earlier runs).**
  ```bash
  ros2 run f1tenth_gym_ros gym_bridge --ros-args -r __node:=bridge --params-file sim_open40.yaml
  ```
  The two bridges publish on the same topics, so run only one at a time.
- **Container quirks:** the `ros2 topic` CLI shows no messages here, but rclpy works. Files written from the container are owned by root, so rename or delete them from inside the container.
- **Nothing is running.** All jobs were stopped on 2026-10-09, including the MB bridge.

## Done

- `lifted_ilc.py`: lifted linear-tire ILC.
  - Projected-Newton box QP, plus an exterior penalty for the steering and speed-command limits.
  - Sparse OSQP fallback.
  - Safeguard.
  - Validated: matches OSQP to 4e-7, and the figure-eight QP solves in about 7 s.
- `trajopt.py`: **the TO to use** (you asked for IPOPT instead of acados).
  - CasADi Opti and IPOPT, multiple shooting.
  - Linear-tire plan first, then the Fiala plan from it (Cf = Cr = 225), with 16 RK4 substeps.
  - The cost is the task cost from `weights.json`.
  - vx ≥ −0.5. Both tire models brake at standstill when the steering is at full lock.
- `initial_to.py`:
  - **References:** S-curve, the original figure-eight, and llampc mocap tracks via `mocap_reference`. Mocap tracks start from rest, with steering clipped to ±0.34.
  - **Old acados TO path:** the Fiala and blend NMPCs. Kept for the record and no longer the default.
- `mb_sim/`: the multi-body plant from the `s-hliao/f1tenth_gym` fork, with F1TENTH-scale parameters (Froude-scaled estimates), the stock front end, and the ROS bridge. Validated offline in `mb_sim/validate_mb.json`.
- `run_lifted.py`: the runner.
  - `--method lifted|ilqr|repeat`.
  - Takes `--history <TO npz>`; without it, it runs `trajopt.py` itself.
  - Re-flies a trial whose command misses a whole period.
  - Logs the achieved control rate.

### Results so far (position RMSE in m: trial 0, then best)

| Run (folder in `results/`) | Lifted | iLQR (alpha 0.5) |
|---|---|---|
| Stock ST, 35 Hz, S-curve (`lifted_linear`, `ilqr_alpha05`) | 0.345 → 0.132 (by trial 2) | 0.346 → 0.137 (trial 6) |
| Stock ST, 40 Hz, S-curve, blend TO (`40hz_s_curve_*`) | 0.349 → 0.128 | 0.355 → 0.132 |
| MB, 40 Hz, S-curve, acados Fiala TO (`mb_40hz_s_curve_*`) | 0.276 → 0.143 | 0.277 → 0.142 |
| MB, 40 Hz, figure-eight, acados Fiala TO (`mb_40hz_figure_eight_*`) | 3.997 → **0.048** (trial 6; run stopped at trial 10 on a missed deadline) | 3.973 → 0.249 (stalled, rejecting) |

Repeat noise: about ±0.003 m (`repeat_u0`, `mb_40hz_s_curve_repeat`).

### TO status at 40 Hz (`references/`)

| Reference | IPOPT (`*_ipopt_to.npz`) | acados Fiala (`*_fiala_to.npz`) |
|---|---|---|
| s_curve | converged, cost 6.33 | converged |
| figure_eight | **not done**: used 20 GB of RAM and was killed | converged (used by the MB runs above) |
| mocap_square | converged, 4177 (no longer wanted) | converged |
| mocap_fig8slow | converged, 1837 (no longer wanted) | failed (QP MINSTEP, even from the blend guess or with merit backtracking) |
| mocap_figslow | converged, 1377 (no longer wanted) | converged |
| **mocap_square2fast** | **not done** | n/a |
| **mocap_figfast** | **not done** | n/a |

## To do next, in order

1. **Fix the IPOPT TO memory on long horizons.** The figure-eight (N = 783) ran out of RAM. Try one of these in `trajopt.trajopt`:
   - fewer substeps (8);
   - `'hessian_approximation': 'limited-memory'`;
   - `expand: False`.

   Check it still converges on the S-curve.
2. **Build the TOs for the mocap tracks you picked:**
   ```bash
   python3 trajopt.py mocap_square2fast --rate 40
   python3 trajopt.py mocap_figfast --rate 40
   ```
   - **Open decision:** `mocap_square2fast` stores two constant speeds, 3.75 m/s (`mus` 0) and 5.0 m/s (`mus` 1). `mocap_reference` uses `vs[0]`, i.e. 3.75. At about 0.9 m corner radius either speed needs far more than μg of lateral acceleration, so the TO will slow down or cut corners.
   - Both tracks have corners tighter than the minimum turning radius (about 0.93 m at 0.34 rad), so some path error can't be removed. Report the TO plan's own path error as the floor.
3. **Start the MB bridge, then run `run_queue_mocap.sh`** (already pointed at square2fast and figfast). It runs:
   - **Ablations** on S-curve and figure-eight, from the acados Fiala TOs so they compare with the MB runs above: lifted ILC with blend-model G (`--model blend`), and iLQR with `--alpha 1.0`. These answer whether the lifted method wins because of the full constrained step or the model.
   - **Mocap tracks:** lifted ILC and iLQR from the IPOPT TOs.
4. **Optional:** rerun the MB S-curve and figure-eight from IPOPT TOs, so every trial-1 controls come from the same TO.

## Known issues

- **Bug in the `f1tenth_gym` fork:** `VehicleParameters.to_array(DynamicModel.MB)` is offset by two, because `collision_body_center_x/y` sit at indices 18–19. `mb_sim/fork_import.mb_vector` works around it here; the fork itself isn't fixed.
- **MB parameters are estimates.** The tire peaks at about 35° of slip, too soft for limit or drift studies (see `DRIFT.md`).
- **Odometry has vy = 0** (stock contract), so sideslip is never measured.
- **No `best_refresh` in the safeguard.** A lucky best trial causes repeated rejections.
- **Old acados NMPC:** the snapshot's 20-iteration SQP cap stops before acados' tolerance. Use ≥ 200 if acados is ever used again.
- **Unsuitable QP solvers:** IPOPT and qpOASES are impractical on the dense lifted QP (more than 25 min and 97 s respectively on the figure-eight). Keep the box QP.
