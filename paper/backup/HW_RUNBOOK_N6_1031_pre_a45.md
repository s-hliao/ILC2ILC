# Car hardware, nominal-mu-0.6 pipeline (2026-10-31)

Arms (user, 2026-10-31): **RMA zero-shot, PPO+DR zero-shot, then our stage on both** (RMA + stage, PPO+DR + stage).
Tracks: figfast (the main track) and square2fast. Tires: plastic and rubber; one session folder per tire.
Supersedes HW_RUNBOOK_1031.md (the old mu-0.2 setup and its networks are not used).

The planner and the sim never see the tires: every plan is solved for the nominal car (mu 0.6, plans_nom06). Plastic gets
a lower planned speed only (the `_v70` plans, x 0.7); rubber flies the full-speed plans.

| tire | figfast plan | square2fast plan | driven peak in sim |
|---|---|---|---|
| plastic | `mocap_figfast_b0_free_v70` | `mocap_square2fast_b0_free_v70` | figfast ~2.9-3.1 m/s, square2fast ~2.0-2.1 m/s |
| rubber | `mocap_figfast_b0_free` | `mocap_square2fast_b0_free` | figfast ~4.0 m/s, square2fast ~3.2-3.6 m/s |

Networks (seed 0): PPO+DR `runs/freePPON_s0/policy_final.npz`; RMA `runs/rmaN_s0/policy_final.npz` (its
`rma_adapt.npz` must sit next to it).

Stage budget: **3 iterations x 2-lap trials = 6 laps** per tire per track (the user's cap: at most 6 laps).

## Simulation evidence (6-lap stage, seeds 0 / 1; score on the practised track, lower is better)

| practised | RMA zero-shot | RMA + stage | PPO+DR zero-shot | PPO+DR + stage |
|---|---|---|---|---|
| plastic square2fast | 0.45 / 0.40 | **0.38 / 0.32** | 0.80 / 0.87 | 0.60 / 0.63 |
| plastic figfast | 1.95 / 2.04 | 1.98 / 2.03 | 3.66 / 3.44 | 3.52 / 3.50 |
| rubber square2fast | 0.24 / 0.78 | 0.26 / 0.40 | 0.80 / 0.80 | 0.50 / 0.59 |
| rubber figfast | 0.90 / 0.92 | 0.92 / 0.86 | 1.08 / 1.08 | 1.01 / 1.00 |

Known limits: on plastic figfast no arm improves (the stage's positivity fails in the spin after the braking corner,
s ~ 10-12 m); PPO+DR + stage stays behind RMA zero-shot everywhere.

## 0. Before the session

1. Copy to the car / workstation: `plans_nom06/` (all of it: the LQR caches `*_K.npy` are made on first load),
   `runs/freePPON_s0/`, `runs/rmaN_s0/` (with `rma_adapt.npz`), and this ilc2real tree (hw_stage.py has the RMA
   manual-backend change of 2026-10-31).
2. Measure the plastic tire's friction (tilt or pull test) for the write-up.

## 1. Every terminal
```bash
cd ~/ilc_ws/src/ilc_f1tenth/ilc2real && source env_tight.sh
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06       # AFTER env_tight.sh: it sets the old mu-0.2 bank
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_priority_fusion=false NUMBA_NUM_THREADS=1
S=~/car_sessions/$(date +%m%d)_<tire>             # plastic / rubber
P=<plan>                                          # from the table above, per tire and track
```

## 2. On the car: the runner (keep it up)
```bash
source <ilc2real>/env_tight.sh && export F1T_MU=0.6 F1T_PLANS=<ilc2real>/plans_nom06 && \
  ros2 run ilc_f1tenth_hw net_runner.py --ros-args -p session_dir:=$S
```

## 3. Zero-shot baselines (per track; 2 runs x 5 laps each)
```bash
python hw_eval_requests.py --session-dir $S --policy runs/rmaN_s0/policy_final.npz --tag rma_zs_<tire>_<track> --plan $P --laps 5 --runs 2
python hw_eval_requests.py --session-dir $S --policy runs/freePPON_s0/policy_final.npz --tag ppodr_zs_<tire>_<track> --plan $P --laps 5 --runs 2
```

## 4. Our stage on each network (per track): 3 iterations x 2-lap trials = 6 laps
```bash
A=_v70    # plastic: the anchor plans at the tire's speed;  rubber: A=   (the full-speed plans)
for NET in freePPON rmaN; do
python -u hw_stage.py --backend manual --session-dir $S --cars real --policy runs/${NET}_s0/policy_final.npz \
  --out runs/hw/real_${NET}_<tire>_<track>_stage --goals $P --iters 3 --objective task --goal-cycle 1 --trial-laps 2 \
  --anchor-w 20 --anchor-plans mocap_square2fast_b0_free$A mocap_figfast_b0_free$A --fit-realize 0.9 --fit-max-steps 8000 \
  --seed 9001 --launch 1.0 8.0 --save-its 1 --beta 0.5 --delta 0.05 --cap 0.1 --slope-w 1 --tr 1 \
  --err-scale 0.10 1e9 1e9 1e9 1e9 0.24 --eval-runs 0
done
```
(These are the simulated stage's flags, per tire.)
The car flies each request as it appears; the update between trials takes ~1-3 min (8000 fit steps max).
RMA: the stage trains the actor only; the adaptation module stays frozen and is written to `$S/rma_adapt.npz`.

## 5. Evaluate the adapted networks (same protocol as the baselines)
```bash
for NET in freePPON rmaN; do
python hw_eval_requests.py --session-dir $S --policy runs/hw/real_${NET}_<tire>_<track>_stage/real_policy.npz \
  --tag ${NET}_stage_<tire>_<track> --plan $P --laps 5 --runs 2
done
```
(RMA: `real_policy.npz` has `mode rma` and the stage folder holds `rma_adapt.npz`.)

## 6. Score
```bash
for t in rma_zs ppodr_zs freePPON_stage rmaN_stage; do python hw_eval_requests.py --session-dir $S --tag ${t}_<tire>_<track> --score; done
```
nerr = max(rms e_y / 10 cm, (1 - pace) / 0.1), crash = 5. Lower is better.

Desk-tested 2026-10-31: RMA + stage through `--backend manual` with `car_session_sim.py` (2 iterations, plastic
square2fast): requests carry the RMA actor + `rma_adapt.npz`, trials flown, the stage's step and rollback ran.
