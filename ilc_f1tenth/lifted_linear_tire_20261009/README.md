# Lifted-form ILC with a linear tire model (quadruped-style), 2026-10-09

F1TENTH counterpart of the quadruped's `JumpILC` (`ilc_quad/ilc_gen.py`):

| | quadruped | here |
|---|---|---|
| trial 1 | full-body TO (IPOPT) | Fiala/brush-tire NMPC over the whole reference (llampc `nmpc_gen_fiala_fixed`, acados SQP), required to converge (status 0) |
| learning model | planar SRB | 9-state single-track, **linear tires** (`lifted_ilc.linear_tire_model`) |
| update | one QP on the lifted `G` along the measured trial | same: `min ½‖e − GΔu‖²_W + ½‖Δu‖²_S + ½‖U+Δu‖²_R`, rate limits, steering/speed-command limits |
| safeguard | worse trial → back to the best, heavier step | same (`accept_tol` 2 %, `Qu` ×4 per rejection, ÷2 per improvement) |

`e = x_ref − x_measured`, so at Δu = 0 the prediction is the measured error: no defect
correction is needed, only the sensitivities come from the model. W, R are the snapshot's
`weights.json` task cost, so the QP minimizes the reported "measured cost".

The snapshot `../successful_defect_aware_20261008` is imported, never modified.

## Files

- `lifted_ilc.py`: model, lifted G, QP (projected-Newton box QP on the dense lifted H, plus an exterior penalty for any binding steering/speed-command rows; falls back to sparse OSQP), `LiftedILC.update`
- `initial_to.py`: references at any rate (S-curve and figure-eight generators with `dt` as a parameter), and the initial TO (`--model fiala|blend`, SQP cap raised to 1000; at 20 the NMPC stops short of the 1e-6 stationarity tolerance)
- `run_lifted.py`: trials on the ROS sim, `--method lifted|ilqr|repeat` (`ilqr` = the snapshot's defect-aware iLQR, for comparison from the same TO controls)
- `mb_sim/`: multi-body plant
  - `fork_import.py`: imports the `s-hliao/f1tenth_gym` fork's dynamics without gymnasium
  - `mb_params.py`: F1TENTH-scale MB parameters
  - `mb_plant.py`: stock f110_gym front end (2-sample steering delay, `pid`), with MB integrated by RK4 at 0.25 ms
  - `mb_bridge.py`: ROS bridge on the stock topics
  - `validate_mb.py`: offline checks
- `run_suite_mb.sh`: the MB experiments; `run_suite_40hz.sh`: the same on the stock ST bridge

## Running (inside `docker_lla_new`, `ROS_DOMAIN_ID=42`)

```bash
cd /workspaces/ilc_ws/src/ilc_f1tenth/lifted_linear_tire_20261009
(cd mb_sim && python3 mb_bridge.py &)                    # or the stock bridge with sim_open40.yaml
python3 initial_to.py figure_eight --rate 40 --model fiala
python3 -u run_lifted.py --method lifted --history references/figure_eight_40hz_fiala_to.npz \
    --trials 12 --output results/<new_dir>
```

## Results: position RMSE (m) by trial

| plant, rate, path | method | trial 0 | 1 | 2 | 4 | best |
|---|---|---|---|---|---|---|
| ST 35 Hz S-curve | lifted | 0.345 | 0.138 | 0.132 | 0.134 | 0.132 |
| ST 35 Hz S-curve | iLQR | 0.346 | 0.298 | 0.163 | 0.142 | 0.137 |
| ST 40 Hz S-curve | lifted | 0.349 | 0.145 | 0.135 | 0.134 | 0.128 |
| ST 40 Hz S-curve | iLQR | 0.355 | 0.304 | 0.164 | 0.145 | 0.132 |
| MB 40 Hz S-curve | lifted | 0.276 | 0.155 | 0.151 | 0.147 | 0.143 |
| MB 40 Hz S-curve | iLQR | 0.277 | 0.261 | 0.174 | 0.152 | 0.142 |
| MB 40 Hz figure-eight | lifted | 3.997 | 1.744 | 0.421 | 0.070 | **0.048** |
| MB 40 Hz figure-eight | iLQR | 3.973 | 1.572 | 1.014 | 0.342 | 0.249 |

The ST runs used the blend TO (stored 35 Hz controls, then fresh 40 Hz solves); the MB runs used the Fiala TO.
Repeating the same controls: about ±0.003 m (ST). Every run held 40.00 Hz (periods 23–27 ms).

## Caveats

- MB parameters are estimates. They are anchored on the gym's F1TENTH single-track values and `params.py`, with the rest Froude-scaled from CommonRoad vehicle 2; see the `mb_params.py` docstring.
- The fork has a bug: `VehicleParameters.to_array(DynamicModel.MB)` is offset by two (`collision_body_center_x/y` were added at indices 18–19, but the MB code doesn't skip them). `fork_import.mb_vector` works around it here.
- Odometry follows the stock bridge: `linear.y` is always 0, so vy is never measured.
- The MB figure-eight lifted run stopped at trial 10, when one command went out a full period late. `run_lifted.py` now re-flies the trial in that case.
- The safeguard has no `best_refresh` (averaging the best trial's cost over re-flights). A lucky best can therefore trigger repeated rejections, as in ST 40 Hz S-curve trials 6–10.
- IPOPT on the figure-eight lifted QP (N = 783, 1566 inputs) didn't finish within 25 min. qpOASES took 97 s; the penalized box QP takes about 7 s (0.08 % above the qpOASES optimum).
