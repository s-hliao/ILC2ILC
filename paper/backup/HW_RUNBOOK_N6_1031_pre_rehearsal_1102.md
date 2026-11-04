# Car hardware, nominal-mu-0.6 pipeline (2026-10-31; 2026-11-01: peanut CLOCKWISE held out for both tires; rubber practises ONE plan, the figure-eight _x120, from the start line)

Arms (user, 2026-10-31): **RMA zero-shot, PPO+DR zero-shot, then our stage on both** (RMA + stage, PPO+DR + stage).
Tires: plastic and rubber; one session folder per tire. Crash envelope **0.5 m** (|e_y| beyond it ends a run).

## Tracks

| track | role | notes |
|---|---|---|
| `mocap_square2fast` | practised (plastic) | min radius 1.14 m: steering never at its limit |
| **`mocap_fig8R100`** | practised (rubber); scored (plastic) | **the new figure-eight**: lobes R 1.00 m, 60 deg crossing, 2.01 x 4.31 m, lap 12.8 m. Replaces figfast (min radius 0.58 m, tighter than the car can steer: the stage had no authority there). Layout: `HW_TRACK_fig8R100_centerline.csv`, `HW_TRACK_fig8R100.png` (room coordinates, centred on the old area's centre) |
| `mocap_peanut` | the counter-clockwise direction: not flown (both tires' held-out is peanut CLOCKWISE, below; user 2026-11-01 "keep the peanut clockwise") | 2.87 x 3.88 m, lap 11.2 m, the old 2.9 x 3.9 m area. Layout: `HW_TRACK_peanut_centerline.csv`. Steering command beyond the limit at zero-shot 6-15 % (plastic) / 1-3 % (rubber) of the lap |
| **`mocap_peanutM`** | **held out, BOTH tires** (user 2026-11-01: "why not just make peanut right turns?", "keep the peanut clockwise") | **the same tape, driven clockwise**: peanut is symmetric about both axes (the mirrored track = the original point set), so its mirror is the same layout in the other direction. 85 % right turns, the direction square2fast (100 % right turns) practises. Start (-0.719, 1.640) heading +x; `HW_TRACK_peanutM_centerline.csv` (the CSV's mirror lies within 1.7 cm of the original tape) |
| `mocap_squareH` | not flown on hardware | replaced as the held-out track: 47 % of its lap needs > 88 % of the steering lock at any speed (corners tighter than the car's 0.93 m kinematic radius); steering command beyond the limit 43-46 % (plastic) / 71-75 % (rubber) of the lap |

The planner and the sim never see the tires: every plan is solved for the nominal car (mu 0.6, plans_nom06).
- **Plastic** (lower planned speeds only): `trajopt_drift.py --a-max A --vstar 5.5`, a hard bound A on the planned total
  acceleration (corners slower, straights unchanged): **4.5 m/s^2 on square2fast (`_a45`), 3.3 m/s^2 on the figure-eight and
  peanut (`_a33`)**: at 4.5 the figure-eight crashed 10-21 % and peanut 15-21 % of the validation runs (at 3.3: 2-4 %).
- **Rubber**: the nominal plans flown faster (scale_plan.py: uniform time scaling, no friction model). **Practice: ONE plan, the
  figure-eight `_x120`, every trial from the start line** (user 2026-11-01: "i don't want to have to run multiple plans for training";
  "no varied states, just fly the network at the plan"). Held-out: peanut clockwise at `_x120`, one speed for both networks (user 2026-11-02). Note: peanut `_x120` asks 9.94 m/s^2 in one corner, above the softer rubber
  car's grip (9.81); even the LQR slides there: disclose.
- **The launch (found 2026-11-01 06:00):** on the car every trial starts with the slow rolling launch and the network takes over at
  1 m/s; the earlier simulated stages started at the plan speed. With the launch simulated (`hw_stage.py --sim-launch 1`, the
  car_session_sim protocol) the first third of each trial is the acceleration (rms e_y 21 cm vs 8.5 cm after), which the GN step
  then fits. **Approved (user, 2026-11-02, both tires, both networks; it changes which trial samples the GN step uses):** `--skip-launch 0.5`: each trial counts
  only from the first sample whose speed error is below 0.5 m/s (the ILC trial starts on the plan). Both commands are below.

| tire | square2fast | fig8R100 | peanut (held out) | driven peak in sim |
|---|---|---|---|---|
| plastic | `_a45` | `_a33` | `_a33` (peanutM, clockwise) | square2fast 2.37-2.69 m/s |
| rubber | `_x120` | **`_x120` (practised, the only plan)** | `_x120` (peanutM, clockwise) | square2fast 3.49-3.54 m/s |

Networks (seed 0): PPO+DR `runs/freePPON_s0/policy_final.npz`; RMA `runs/rmaN_s0/policy_final.npz` (+ `rma_adapt.npz`).
Steering +-0.34 rad in the model, the runner and the car. Stage: **3 iterations x 2-lap trials = 6 laps** per tire.

## Simulation evidence (envelope 0.5 m; seeds 0 / 1; score nerr, lower is better; crash % where non-zero)

**Rubber, the design above, simulated WITH the car's launch (2026-11-01; 3 network seeds x 2 rubber cars = 6 cells, 4 runs x 5
laps from spread starts, evaluation seed 701; % nerr vs own zero-shot after 6 laps, bootstrap 95% CI over cells; runs/screen*/L120*,
SL50x120*):**

| rubber, fig8 `_x120` practice | peanut CW `_x120` (held out) | peanut CW `_x110` (held out) | fig8R100 `_x120` (practised) |
|---|---|---|---|
| **PPO+DR + stage, `--skip-launch 0.5`** | **-5.3 % [-6.5, -4.0]** (6/6 cells) | **-11.6 % [-15.4, -8.0]** (6/6) | **-21.5 % [-25.6, -18.0]** |
| **RMA + stage, `--skip-launch 0.5`** | **-4.2 % [-6.6, -1.9]** (6/6 cells) | +3.6 % [-1.3, +10.0] (n.s.) | **-19.0 % [-21.7, -16.4]** |
| PPO+DR + stage, no mask | **-2.6 % [-4.2, -1.4]** | **-10.8 % [-14.9, -6.9]** | -20.6 % |
| RMA + stage, no mask | -1.9 % [-4.2, +0.4] (n.s.) | +2.5 % (n.s.) | -11.8 % |

Each cell = mean over 6 (mask) / 4 (no mask) stage replicates (stage seeds 9001-9006). Without the launch in the simulation
(the earlier screens) the numbers differ: always simulate the launch from now on (`--sim-launch 1`).

**Plastic, simulated WITH the launch (square2fast `_a45` practice, 12 plastic cars, seeds 0 / 1):** held-out peanut `_a33`
PPO+DR -14 % / -13 %, RMA -4 % / -8 % (22 of 24 car cells better); square2fast PPO+DR -11 % / -11 %, RMA -3 % / -4 %.
With `--skip-launch 0.5` (seeds 0 / 1): PPO+DR peanut **-22.8 % / -26.9 %** (10/12 each), square2fast -16.2 % / -17.6 %; RMA peanut **-9.1 % / -11.0 %** (24/24 car cells better), square2fast -3.0 % / -6.5 %.

Superseded: practising `_x115` looked best in simulations that started at the plan speed (RMA -4.8 % [-7.9, -1.3]); with the
car's launch it gives RMA -2 % and PPO+DR -2 %.

Rubber, practised on the figure-eight `_x120` only (the earlier design; 2 rubber cars, 4 runs x 5 laps):

| | fig8R100 (practised) | square2fast | squareH (sim only) |
|---|---|---|---|
| PPO+DR zero-shot | 0.72 / 0.75 | 0.50 / 0.81 | 0.64 / 0.78 |
| **PPO+DR + stage** | **0.45 / 0.53** | **0.29 / 0.58** | 0.63 / 0.81 |
| RMA zero-shot | 0.90 / 0.81 | 0.39 / 0.35 | 0.53 / 0.73 |
| RMA + stage | 0.73 / 0.43 | 0.38 / 0.30 | 0.52 / 0.82 |

Stage rho > 0: PPO+DR 100 % / 100 %, RMA 67 % / 100 %; no stage crashes.

Held-out peanut (`_x120`; the same stage networks, scored 2026-10-31 23:10): PPO+DR 0.30 / 0.34 -> **0.28 / 0.28**, RMA 0.64 / 0.47 ->
**0.61 / 0.37** (paired change vs own zero-shot -14 % [-23, -6] and -13 % [-21, -4]); no crashes.

Plastic, practised on square2fast (12 plastic cars, 4 runs x 5 laps):

| | square2fast (practised) | fig8R100 (`_a33`) | squareH (sim only) |
|---|---|---|---|
| PPO+DR zero-shot | 1.99 / 2.06 (4 %) | 1.37 / 1.62 (4 %) | 1.63 / 1.80 |
| **PPO+DR + stage** | **1.70 / 1.79** (6 %) | 1.32 / 1.56 (4 %) | **1.26 / 1.43** |
| RMA zero-shot | 1.61 / 1.65 (2 %) | 1.33 / 1.47 (4 %) | 1.58 / 1.70 |
| RMA + stage | 1.50 / 1.47 (2 %) | 1.30 / 1.43 (4 %) | 1.36 / 1.41 |

Stage rho > 0: 91-100 % of iterations, 0-2 rollbacks per run.

Held-out peanut counter-clockwise (`_a33`): PPO+DR 1.44 / 1.63 -> 1.42 / 1.62, RMA 1.55 / 1.82 -> 1.54 / 1.77 (-1 % [-2, -0],
-3 % [-5, -0]; 2 % crashes either way): no plastic gain driven counter-clockwise (left turns; square2fast turns right only).
**Held-out peanut clockwise (`mocap_peanutM_b0_free_a33`, plastic's held-out from 2026-11-01)**, the same stage networks, % change
vs own zero-shot (seeds 0 + 1, 12 cars, 4 runs x 5 laps, 95 % CI) after 2 / 4 / 6 laps: PPO+DR -6 / -12 / **-21 % [-29, -11]**
(pace 0.885 -> 0.914), RMA -7 / -13 / **-19 % [-23, -14]** (pace 0.908 -> 0.923). The zero-shot networks are not mirror-symmetric:
their clockwise zero-shot is better than the counter-clockwise one (pace 0.885 / 0.908 vs 0.858 / 0.843).
(Tried and dropped 2026-11-01: plastic practising square2fast + fig8R100 (8 laps) left counter-clockwise peanut unchanged (+1 %,
PPO+DR s1): the fig8R100 `_a33` plan is at the steering limit (0.340 rad) in its corners and its steps lowered the current on the
straights, on fig8 and on peanut alike.)

Improvement over each network's own zero-shot (stage minus zero-shot, seeds 0 / 1; negative = better):

| | practised track | other tracks |
|---|---|---|
| rubber PPO+DR | fig8 -0.27 / -0.22 | square2fast -0.21 / -0.23; **peanut -0.02 / -0.06**; squareH -0.01 / +0.03 |
| rubber RMA | fig8 -0.17 / -0.38 | square2fast -0.01 / -0.05; **peanut -0.03 / -0.10**; squareH -0.01 / +0.09 |
| plastic PPO+DR | square2fast -0.29 / -0.27 | fig8 -0.05 / -0.06; **peanut -0.02 / -0.01**; squareH -0.37 / -0.37 |
| plastic RMA | square2fast -0.11 / -0.18 | fig8 -0.03 / -0.04; **peanut -0.01 / -0.05**; squareH -0.22 / -0.29 |

The stage improves both networks on the practised track for both tires and both seeds, and on every other track except
rubber squareH. Known limits:
- rubber squareH: its tight corners (curvature 1.28 1/m) are beyond the steering at the `_x120` speeds (steering pinned
  ~80 % of the lap, checked 2026-10-31); the square2fast-practised stage raised pace and the car ran wider there.
- plastic on the figure-eight: practising it was not reliably positive (rho > 0 37-53 % at R 1.2 m); it is scored only.

## 0. Before the session

1. Copy to the car / workstation: `plans_nom06/` (all of it; LQR caches `*_K.npy` are made on first load), the track files
   `../lifted_linear_tire_20261009/tracks_tight/mocap_fig8R100.npz`, `mocap_peanut.npz` and `mocap_peanutM.npz`, `runs/freePPON_s0/`, `runs/rmaN_s0/`, this ilc2real tree.
2. **Lay out the figure-eight and peanut** from `HW_TRACK_fig8R100_centerline.csv` / `HW_TRACK_peanut_centerline.csv` (room
   frame; peanut occupies the old 2.9 x 3.9 m area, x -2.18..0.68, y -2.24..1.64; plastic drives the same tape clockwise from
   `HW_TRACK_peanutM_centerline.csv`'s start). The figure-eight: If it goes elsewhere in the
   room, regenerate the track file with that offset (make_fig8_track.py) and re-solve its plans before the session.
3. **Steering check**: command +0.34 and -0.34 rad and confirm the servo is not clipped (`/commands/servo/position` vs the
   VESC's `servo_min` / `servo_max`; with the repo's `vesc.yaml` the driver would clip at +0.313 / -0.263 rad).
4. **Crash envelope 0.5 m**: `net_runner.py` defaults `safety_ey` to 0.5 (was 1.0); rebuild on the car
   (`colcon build --packages-select ilc_f1tenth_hw`). `env_tight.sh` sets `F1T_SAFETY_EY=0.5`.
5. Measure the plastic tire's friction (tilt or pull test) for the write-up.

## 1. Every terminal
```bash
cd ~/ilc_ws/src/ilc_f1tenth/ilc2real && source env_tight.sh
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06       # AFTER env_tight.sh: it sets the old mu-0.2 bank
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_priority_fusion=false NUMBA_NUM_THREADS=1
S=~/car_sessions/$(date +%m%d)_<tire>             # plastic / rubber
# plastic:
declare -A P=([square2fast]=mocap_square2fast_b0_free_a45 [fig8R100]=mocap_fig8R100_b0_free_a33 [peanut]=mocap_peanutM_b0_free_a33)   # peanut CLOCKWISE
G=mocap_square2fast_b0_free_a45; A="mocap_square2fast_b0_free_a45 mocap_figfast_b0_free_a45"
# rubber:
declare -A P=([square2fast]=mocap_square2fast_b0_free_x120 [fig8R100]=mocap_fig8R100_b0_free_x120 [peanut]=mocap_peanutM_b0_free_x120)   # peanut CLOCKWISE
G=mocap_fig8R100_b0_free_x120   # ONE practice plan; every trial starts at the start line (the left lobe first, as in sim)
A="mocap_square2fast_b0_free_x120 mocap_fig8R100_b0_free_x120"
```
(G = the practised plan; A = the stage's anchor plans, simulation-only regularisation, exactly as in the simulated runs.)

## 2. On the car: the runner (keep it up)
```bash
source <ilc2real>/env_tight.sh && export F1T_MU=0.6 F1T_PLANS=<ilc2real>/plans_nom06 && \
  ros2 run ilc_f1tenth_hw net_runner.py --ros-args -p session_dir:=$S
```

## 3. Zero-shot baselines (all three tracks; 2 runs x 5 laps each)
```bash
for T in square2fast fig8R100 peanut; do
python hw_eval_requests.py --session-dir $S --policy runs/rmaN_s0/policy_final.npz --tag rma_zs_<tire>_$T --plan ${P[$T]} --laps 5 --runs 2
python hw_eval_requests.py --session-dir $S --policy runs/freePPON_s0/policy_final.npz --tag ppodr_zs_<tire>_$T --plan ${P[$T]} --laps 5 --runs 2
done
```

## 4. Our stage on each network: 3 iterations x 2-lap trials = 6 laps on the practised track
```bash
for NET in freePPON rmaN; do
python -u hw_stage.py --backend manual --session-dir $S --cars real --policy runs/${NET}_s0/policy_final.npz \
  --out runs/hw/real_${NET}_<tire>_stage --goals $G --iters 3 --objective task --goal-cycle 1 \
  --trial-laps 2 --anchor-w 20 --anchor-plans $A --fit-realize 0.9 \
  --fit-max-steps 8000 --seed 9001 --launch 1.0 8.0 --save-its 1 --beta 0.5 --delta 0.05 --cap 0.1 --slope-w 1 --tr 1 \
  --err-scale 0.10 1e9 1e9 1e9 1e9 0.24 --eval-runs 0 --skip-launch 0.5
done
```
`--skip-launch 0.5` is the launch mask (approved 2026-11-02, see above): keep it for BOTH tires and both networks.
The car flies each request as it appears; the update between trials takes ~1-3 min. A trial lasts 2.05 plan lap times
(the car covers ~1.5-1.8 laps). RMA: the stage trains the actor only; `$S/rma_adapt.npz` holds the frozen adaptation module.

## 5. Evaluate the adapted networks (same protocol, all three tracks)
```bash
for NET in freePPON rmaN; do for T in square2fast fig8R100 peanut; do
python hw_eval_requests.py --session-dir $S --policy runs/hw/real_${NET}_<tire>_stage/real_policy.npz \
  --tag ${NET}_stage_<tire>_$T --plan ${P[$T]} --laps 5 --runs 2
done; done
```

## 6. Score
```bash
for T in square2fast fig8R100 peanut; do for t in rma_zs ppodr_zs freePPON_stage rmaN_stage; do
python hw_eval_requests.py --session-dir $S --tag ${t}_<tire>_$T --score; done; done
```
nerr = max(rms e_y / 10 cm, (1 - pace) / 0.1), crash = 5. Lower is better.

Desk-tested 2026-10-31 through `--backend manual` with `car_session_sim.py`: RMA + stage on plastic `_a45` square2fast and
PPO+DR + stage on rubber `_x120` square2fast (requests served, trials flown, the step ran, `real_policy.npz` written). The
fig8R100 plans load through the same plan bank as the simulated runs; they were not desk-tested through the manual backend.
