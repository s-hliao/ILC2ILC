# F1TENTH 9-state defect-aware iLQR + ILC

## Information boundary: old oracle-assisted version versus current version

Here “cheat” means using simulator-only information unavailable to the intended real-hardware controller. The earlier oracle-assisted execution-chain version crossed that boundary. This does not mean every historical baseline used those inputs.

| Earlier oracle-assisted implementation | Why it was privileged | Current successful implementation |
|---|---|---|
| `timing_gym_bridge.py` logged the simulator's true state, including actual steering angle and sideslip | These were read directly from simulator internals, without a hardware sensor/estimator interface | Uses public odometry for x/y/heading/velocities/yaw rate. Actual actuator angle and sideslip are not read. Steering state is the controller's command state. |
| Read `car.steer_buffer` | Exact internal actuator command buffer is not externally observable | No internal buffer state is used. |
| Read actual simulation tick counts and command-receipt ticks through `timing_alignment.align_trial()` | Supplied exact internal execution timing and delay | Runs at nominal 35 Hz with fixed deadlines and a public-odometry receiver. No simulator ticks or internal command-receipt timing enter ILC. Unknown delay remains part of the prediction mismatch. |
| Passed true augmented states/timing into `measured_context` for old execution-chain prediction | Backward/forward prediction received simulator ground truth | No oracle `measured_context`. Nine-state measured rollout combines public odometry with known controller command states and derived wheel speed. |
| `ExecutionChainModel` imported simulator `pid` and `vehicle_dynamics_st`, with simulator-specific parameter defaults | The predictor assumed knowledge of the plant's exact implementation and parameters | Uses our existing nine-state `blend_model` and its nominal Jacobians. No simulator dynamics/PID import, ExecutionChainModel, GP or neural network. Nominal vehicle parameters remain engineering assumptions, not an identified exact hardware plant. |

The modified bridge also logged environment parameters, timestep and collision information. Not every logged field was used by the optimizer; the old `measured_context` did use true states/buffers, ticks and command delay. In the later D ablation, public-only state/timing estimation removed those oracle inputs, **but the predictor still called simulator dynamics**. D therefore did not satisfy the full information boundary requested for hardware deployment either.

The modified bridge inherited the original bridge's dynamics and timer behavior; its added role was internal logging. The stock bridge was restored for the current experiments. Current controller code has no dependency on those telemetry logs. The simulator necessarily remains the plant used for testing; its hidden implementation is not used as the controller's prediction model.

### How we handle model mismatch now

We retain the nominal blend model and calculate a fixed correction from the previous publicly observed trial:

```text
d[k] = X_measured[k+1] - f_blend(X_measured[k], U_old[k])
f_defect,k(x,u) = f_blend(x,u) + d[k]
```

Heading residuals are wrapped. X_measured contains public odometry plus controller-owned/derived quantities; it is not the simulator's complete true state. The defect uses no next-trial or future-execution data. During each update it is held constant, and A/B remain the original nominal-model Jacobians. This makes the previous measured nominal consistent with that iteration's corrected forward dynamics and lets alpha=0 recover old controls.

Candidate acceptance compares old and candidate controls through the same defect-aware model. Actual measured performance is evaluated after execution and reported separately. Original nominal predictions are diagnostics only.

**Current successful version: public observations + known controls + nine-state nominal blend model + fixed previous-trial additive defects.** It excludes the oracle execution-chain files and Response V2. No simulator-only information is required by this learning update. This is an information-boundary statement, not a claim of demonstrated hardware robustness; real odometry quality, model parameters, timing and repeatability still need hardware validation.


Independent successful experiment snapshot. Baseline source files in the parent project remain unchanged. No ExecutionChainModel, simulator dynamics/PID, hidden steering states, telemetry, GP or neural network. Observations come from public ROS odometry; wheel speed derives from vx; current and steering are controller command states, not measured actuator states.

Fixed previous-trial defects: d[k]=X_measured[k+1]-f_blend(X_measured[k],U_old[k]), heading residual wrapped. Original nominal Jacobians unchanged. Formal acceptance compares defect-aware candidate cost against unchanged controls in the same model. Original nominal costs are diagnostic only. Measured actual costs/RMSE logged separately. Rate limits, reference and weights fixed within each run. Alpha-zero and reconstruction checks run on every update.

Weights match the tested effective values in weights.json, including terminal current weight zero (existing integer-truncation behavior preserved). This snapshot does not repair that earlier bug.

## Environment

Run in the existing F1TENTH ROS2/acados container. Source /opt/ros/foxy/setup.bash and /sim_ws/install/setup.bash; configure ACADOS_SOURCE_DIR, PYTHONPATH and LD_LIBRARY_PATH as in the existing project. Start the stock simulator bridge with the open 40m arena and public /ego_racecar/odom, /drive and /initialpose topics. No telemetry bridge is required. Runtime-generated acados solvers are ignored by Git. Commands below are run from this folder; output paths are relative to this folder.

## Offline preflight (no vehicle execution)

```bash
python3 ilc_f1tenth_ilqr_defect_aware.py --history references/s_curve_epoch0.npz --epoch 0 --alphas 0.5 --output results/preflight
```

## Explicit simulation runs

These commands execute an initial NMPC rollout and then the requested ILC trials. Each output directory must be new.

```bash
OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg python3 -u run_ilc.py --path s_curve --alpha 0.5 --epochs 10 --output results/s_curve_alpha05
OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg python3 -u run_ilc.py --path s_curve --alpha 0.2 --epochs 20 --output results/s_curve_alpha02
OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg python3 -u run_ilc.py --path figure_eight --alpha 0.5 --epochs 10 --output results/figure_eight_alpha05
```

The standalone runner solves fresh blend NMPC for the bundled reference. Previously tested S-curve runs reexecuted stored initial NMPC controls; a new solve need not reproduce them exactly. Reference geometry/time sampling and tested weights are preserved. Figure-eight runs used fresh NMPC; the recorded solve returned status2 (not converged).

## Recorded validation

S-curve alpha=.5, 10 updates: RMSE .3462 -> .1348 m. S-curve alpha=.2, 20 updates: .3415 -> .1378 m. Figure-eight alpha=.5, 10 updates:1.8255 -> .4758 m, best .4713 m; six updates accepted, later candidates rejected and old controls reexecuted. Best repeated rollout variability is not a new learned control. These are simulation results, not hardware robustness/convergence proofs. Lightweight summaries and plots are in validation/; large histories and logs stay in the original organized_results folders.

## Changes and findings on 2026-10-08

### Problem: measured nominal and forward dynamics were inconsistent

The previous backward pass used the last measured rollout as its nominal trajectory. That trajectory generally did not satisfy the prediction model:

```text
X_measured[k+1] != f_nominal(X_measured[k], U_old[k])
```

The forward pass started from the initial state and propagated the model continuously. Even with unchanged controls, its states drifted away from the measured nominal. The feedback term `K[k] @ (X_predicted[k] - X_measured[k])` then changed controls even when the feedforward step `alpha*k[k]` was zero. Reducing alpha alone therefore did not recover the previous controls.

Offline replay reproduced the saved controls to numerical precision. In the previous Response V2 run, all 10 candidate updates had higher predicted cost than an unchanged-control rollout through the same model. An apparent improvement relative to measured nominal cost was not evidence of model-predicted improvement.

### Implemented change: fixed previous-trial additive defects

The independent experiment uses the original nine-state blend model with state order:

```text
[x, y, heading, vx, vy, yaw_rate, wheel_speed, current, steering_command]
```

After each measured rollout it constructs:

```text
d[k] = X_measured[k+1] - f_nominal(X_measured[k], U_old[k])
f_defect,k(x,u) = f_nominal(x,u) + d[k]
```

The heading residual uses `atan2(sin(error), cos(error))`. Defects remain fixed during that update, so their derivatives with respect to state and control are zero. The original nominal A/B Jacobians and backward recursion are reused. Both feedback forward prediction and open-loop candidate evaluation include the same defects. These defects are computed from the previous trial, not future trial measurements.

Each update checks that the defect-aware model reconstructs the measured nominal and that alpha=0 reconstructs old controls. Typical zero-alpha control differences in S-curve experiments were about 1e-13; the longer figure-eight experiment reached about 2e-11, still within the diagnostic tolerance. The reported aggregate state-error magnitude mixes units; per-state diagnostics are also saved.

### Implemented change: fair candidate acceptance

Three quantities are kept separate:

1. `J_measured_old`: cost of the previous actual rollout.
2. `J_defect_old` and `J_defect_candidate`: predictions through the same defect-aware model.
3. `J_nominal_old` and `J_nominal_candidate`: predictions through the original uncompensated model, for diagnostics only.

Formal acceptance requires `J_defect_candidate < J_defect_old`. Old and candidate predictions share the initial state, time step, clipping rules, reference and cost weights. Candidate selection never compares a model-predicted cost against measured cost. If no candidate improves the unchanged-control prediction, old controls are retained. After execution, measured cost/RMSE changes are reported separately.

Predicted acceptance does not guarantee measured improvement. There is currently no automatic measured-performance rollback; best measured trajectories are saved for review.

### What remains unchanged and what this version excludes

The original baseline files remain untouched. The method remains time-indexed, model-based iLQR + trial-to-trial ILC. Control definitions, nine-state ordering, blend dynamics, RK4 discretization and NMPC initialization logic are preserved. The packaged runner performs fresh initial NMPC solves; earlier S-curve experiments reused stored initial NMPC controls.

The successful version uses public odometry and controller-owned command states with 35 Hz fixed-deadline execution. It does not use Response V2, ExecutionChainModel, simulator dynamics/PID functions, true actuator states, internal steering buffers, simulator ticks or command-receipt telemetry. Earlier oracle-assisted experiments did use privileged information and are excluded from this folder. No GP or neural network was added.

A terminal-current weight issue was discovered: integer `Q_f` storage had truncated configured `Qf_I=0.015` to zero. This snapshot preserves the tested effective zero weight; that issue has not been repaired. Other weights were not changed in the successful alpha experiments.

### Experiment findings

- Same-nine-state on/off comparison: without defect correction, all 10 candidates were rejected and the initial controls stayed unchanged; final position RMSE was about 0.3436 m. Defect-aware alpha=.03 accepted updates and reached about 0.3134 m.
- Small-alpha tracking improvement was mainly longitudinal. Lowering steering-reference Q in offline tests produced only a small additional predicted lateral improvement; it was not promoted into the working configuration.
- Offline alpha ablation used fixed k/K and defects for each source rollout. Increasing alpha from .03 to 1 reduced defect-aware predicted cost without control-rate clipping, but large-alpha original-model diagnostics did not always improve. This motivated actual trials rather than assuming predictions were correct.
- From the beginning, alpha=.5 over 10 updates reached 0.1348 m position RMSE; alpha=.2 over 20 updates reached 0.1378 m. Both runs later fluctuated around 0.14 m. This is not a proven lower bound or convergence guarantee.
- A separate notebook-derived figure-eight reference was generated with signed curvature, fixed 35 Hz time sampling, all nine reference states and smooth start/stop. Its length is about 36.15 m, duration19.6 s, maximum speed2 m/s and maximum reference steering11.6 degrees. Six updates were accepted, after which alpha=.5 candidates were rejected. Final RMSE was about0.4758 m. The initial NMPC solve returned status2, so this experiment starts from a nonconverged solution.

### Current loop structure and files

```text
run_ilc.py
  -> initial blend NMPC solution
  -> fixed_deadline_rollout.py: execute controls and collect public odometry
  -> ilc_f1tenth_ilqr_defect_aware.py: construct fixed defects
  -> iLQR_blend.py: backward pass and original Jacobians
  -> defect-aware forward pass and candidate acceptance
  -> execute next control sequence
  -> report actual measured performance
```

During a real/simulator trial, the control sequence is executed open-loop: realtime odometry is logged but is not used with K to correct that trial's controls. K feedback is used inside the model forward pass when generating the next sequence. This is **model-based ILC with open-loop trial execution and measured trial-to-trial feedback**, rather than realtime closed-loop trajectory tracking.

Supporting files are `blend_solver.py` (vehicle dynamics/NMPC), `params.py` (vehicle parameters), `weights.json` (tested costs), and `references/*.npz` (fixed references). `trajectory_generator.py` and `trajectory_analysis.py` are retained dependencies/helpers; loading the packaged references does not regenerate them. Simulation validation does not establish hardware robustness.
