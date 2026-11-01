# Car hardware, nominal-mu-0.6 pipeline (2026-10-31, updated 18:10)

Arms (user, 2026-10-31): **RMA zero-shot, PPO+DR zero-shot, then our stage on both** (RMA + stage, PPO+DR + stage).
Tires: plastic and rubber; one session folder per tire. Supersedes HW_RUNBOOK_1031.md.

**Practise on square2fast; score all three tracks** (square2fast practised, figfast and squareH not practised). figfast
is not practised: 37 % of its centre line is tighter than the car can steer (min radius 0.58 m vs wheelbase / tan(0.34) =
0.93 m), so its plans keep the steering pinned for 54-58 % of the lap and the stage has no authority there (in simulation
the figfast-practised stage made figfast worse for every arm). square2fast's plans never reach the steering limit.

The planner and the sim never see the tires: every plan is solved for the nominal car (mu 0.6, plans_nom06). Plastic gets
lower planned speeds only: the `_a45` plans, a hard 4.5 m/s^2 bound on the planned total acceleration (trajopt_drift.py
--a-max 4.5 --vstar 5.5; corners slower, straights unchanged). Rubber flies the nominal plans 1.2x faster: `_x120`, a uniform time
scaling of the `_free` plans (scale_plan.py --k 1.2: same path and steering, speeds x 1.2; no friction model).

| tire | plan suffix `X` | plans (`mocap_<track>_b0_free$X`) | driven peak in sim |
|---|---|---|---|
| plastic | `_a45` | square2fast, figfast, squareH | square2fast 2.37-2.69 m/s, figfast 2.31-2.68 m/s |
| rubber | `_x120` | square2fast, figfast, squareH | square2fast 3.49-3.54 m/s, figfast 4.47-4.65 m/s |

Networks (seed 0): PPO+DR `runs/freePPON_s0/policy_final.npz`; RMA `runs/rmaN_s0/policy_final.npz` (its
`rma_adapt.npz` must sit next to it). Steering: +-0.34 rad in the model, the runner and the car.

Stage budget: **3 iterations x 2-lap trials = 6 laps** per tire (the user's cap: at most 6 laps).

## Simulation evidence (6-lap stage on square2fast, seeds 0 / 1; lower is better; no crashes unless noted)

Plastic, `_a45` plans (12 plastic cars, 4 runs x 5 laps each):

| | figfast | square2fast (practised) | squareH (held-out) |
|---|---|---|---|
| PPO+DR zero-shot | 1.94 / 2.56 (2 % crash) | 1.89 / 1.97 | 1.63 / 1.80 |
| **PPO+DR + stage** | **1.66 / 2.16** | **1.54 / 1.64** | **1.26 / 1.43** |
| RMA zero-shot | 2.00 / 2.80 | 1.56 / 1.61 | 1.58 / 1.70 |
| **RMA + stage** | **1.74 / 2.42** | **1.45 / 1.43** | **1.36 / 1.41** |

Stage positivity: rho > 0 on 91-100 % of iterations, 0-2 rollbacks per run. The stage improves both networks on all three
tracks; PPO+DR + stage beats RMA zero-shot on figfast and squareH and ties it on square2fast.

Rubber, `_x120` plans (2 rubber cars, 4 runs x 5 laps each):

| | figfast | square2fast (practised) | squareH (held-out) |
|---|---|---|---|
| PPO+DR zero-shot | 1.54 / 1.19 | 0.50 / 0.81 | 0.64 / 0.78 |
| **PPO+DR + stage** | 1.45 / 1.26 | **0.16 / 0.22** | 0.73 / 0.88 |
| RMA zero-shot | 1.23 / 0.94 | 0.39 / 0.35 | 0.53 / 0.73 |
| **RMA + stage** | 1.21 / 0.87 | **0.20 / 0.16** | 0.67 / 0.78 |

Stage positivity: rho > 0 on 100 % of iterations, no rollbacks. On the practised track PPO+DR + stage beats RMA
zero-shot for both seeds. Known limit (rubber): no transfer -- figfast is mixed and held-out squareH gets 0.05-0.14 worse
for both networks; RMA zero-shot stays best there. Faster rubber plans were worse for us: x1.35 practised 1.01 / 1.39 vs RMA
zero-shot 1.22 / 1.17; x1.50 the PPO+DR stage no longer improves (seed 1: rho < 0, 2 rollbacks).

## 0. Before the session

1. Copy to the car / workstation: `plans_nom06/` (all of it; the LQR caches `*_K.npy` are made on first load),
   `runs/freePPON_s0/`, `runs/rmaN_s0/` (with `rma_adapt.npz`), and this ilc2real tree.
2. **Steering check**: command +0.34 and -0.34 rad and confirm the servo is not clipped (`/commands/servo/position` vs
   the VESC's `servo_min` / `servo_max`). With the repo's `vesc.yaml` (gain -1.2135, offset 0.5304, servo 0.15-0.85), the
   driver would clip at +0.313 / -0.263 rad.
3. Measure the plastic tire's friction (tilt or pull test) for the write-up.
4. **Crash envelope 0.5 m** (user 2026-10-31: tightened back from 1.0): `net_runner.py` now defaults `safety_ey` to 0.5 and
   `env_tight.sh` sets `F1T_SAFETY_EY=0.5`. Rebuild `ilc_f1tenth_hw` on the car (`colcon build --packages-select
   ilc_f1tenth_hw`) so the runner picks it up. The simulation tables above were run with the old 1.0 m envelope; the reruns
   at 0.5 m are in progress.

## 1. Every terminal
```bash
cd ~/ilc_ws/src/ilc_f1tenth/ilc2real && source env_tight.sh
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06       # AFTER env_tight.sh: it sets the old mu-0.2 bank
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_priority_fusion=false NUMBA_NUM_THREADS=1
S=~/car_sessions/$(date +%m%d)_<tire>             # plastic / rubber
X=_a45                                            # plastic: _a45;  rubber: X=_x120
```

## 2. On the car: the runner (keep it up)
```bash
source <ilc2real>/env_tight.sh && export F1T_MU=0.6 F1T_PLANS=<ilc2real>/plans_nom06 && \
  ros2 run ilc_f1tenth_hw net_runner.py --ros-args -p session_dir:=$S
```

## 3. Zero-shot baselines (all three tracks; 2 runs x 5 laps each)
```bash
for T in square2fast figfast squareH; do
python hw_eval_requests.py --session-dir $S --policy runs/rmaN_s0/policy_final.npz --tag rma_zs_<tire>_$T --plan mocap_${T}_b0_free$X --laps 5 --runs 2
python hw_eval_requests.py --session-dir $S --policy runs/freePPON_s0/policy_final.npz --tag ppodr_zs_<tire>_$T --plan mocap_${T}_b0_free$X --laps 5 --runs 2
done
```

## 4. Our stage on each network, practised on square2fast: 3 iterations x 2-lap trials = 6 laps
```bash
for NET in freePPON rmaN; do
python -u hw_stage.py --backend manual --session-dir $S --cars real --policy runs/${NET}_s0/policy_final.npz \
  --out runs/hw/real_${NET}_<tire>_stage --goals mocap_square2fast_b0_free$X --iters 3 --objective task --goal-cycle 1 \
  --trial-laps 2 --anchor-w 20 --anchor-plans mocap_square2fast_b0_free$X mocap_figfast_b0_free$X --fit-realize 0.9 \
  --fit-max-steps 8000 --seed 9001 --launch 1.0 8.0 --save-its 1 --beta 0.5 --delta 0.05 --cap 0.1 --slope-w 1 --tr 1 \
  --err-scale 0.10 1e9 1e9 1e9 1e9 0.24 --eval-runs 0
done
```
(The simulated stage's flags.) The car flies each request as it appears; the update between trials takes ~1-3 min.
RMA: the stage trains the actor only; the adaptation module stays frozen and is written to `$S/rma_adapt.npz`.

## 5. Evaluate the adapted networks (same protocol as the baselines, all three tracks)
```bash
for NET in freePPON rmaN; do for T in square2fast figfast squareH; do
python hw_eval_requests.py --session-dir $S --policy runs/hw/real_${NET}_<tire>_stage/real_policy.npz \
  --tag ${NET}_stage_<tire>_$T --plan mocap_${T}_b0_free$X --laps 5 --runs 2
done; done
```
(RMA: `real_policy.npz` has `mode rma` and the stage folder holds `rma_adapt.npz`.)

## 6. Score
```bash
for T in square2fast figfast squareH; do for t in rma_zs ppodr_zs freePPON_stage rmaN_stage; do
python hw_eval_requests.py --session-dir $S --tag ${t}_<tire>_$T --score; done; done
```
nerr = max(rms e_y / 10 cm, (1 - pace) / 0.1), crash = 5. Lower is better.

Real laps per tire: stage 6 (the only adaptation data); evaluation 4 networks x 3 tracks x 10 laps = 120 (scoring only).

Desk-tested 2026-10-31: RMA + stage through `--backend manual` with `car_session_sim.py` (2 iterations, plastic
square2fast): requests carry the RMA actor + `rma_adapt.npz`, trials flown, the stage's step and rollback ran. The `_a45`
plans were not desk-tested through the manual backend; they load through the same plan bank as the simulated runs.
