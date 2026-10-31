# Car hardware, 2026-10-31

Arms on hardware (the user: only a couple of forms on hardware, everything else in simulation):
1. PPO + DR + our stage. Its iteration 0 is PPO + DR zero-shot.
2. RMA zero-shot.
Network: seed 0 (`runs/freePPOV_s0`, `runs/rmaV_s0`). Tracks square2fast and figfast, plastic and rubber tires.

Stage budget: 2 iterations x 2-lap trials = **4 laps** per car per track (the user's deployed budget; at most 6 laps). Flags: trust
region (`--tr 1`, the paper's rule iv) and the score's tolerances as error scales (`--err-scale 0.10 ... 0.13`). Both still need the
user's OK. Desk-tested through `--backend manual` with the desk stand-in (2026-10-31 01:00, `car_session_sim.py`).

Simulation at this budget (3 seeds; plastic at mu 0.4 and rubber; paired 95 % CI):
- plastic 0.4: vs PPO+DR zero-shot -0.05 [-0.06, -0.03] practised, tie held-out; RMA better (+0.06 practised, +0.08 held-out).
  Positivity holds (rho > 0 on 94 % of figfast steps, 69 % of square2fast steps).
- rubber: vs PPO+DR zero-shot -0.05 practised, -0.06 held-out; vs RMA -0.02 practised, -0.09 held-out.

## 0. Every terminal
```bash
cd ~/ilc_ws/src/ilc_f1tenth/ilc2real && source env_tight.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_priority_fusion=false NUMBA_NUM_THREADS=1
S=~/car_sessions/$(date +%m%d)_<tire>      # one session folder per tire: plastic / rubber
```

## 1. On the car: the runner (keep it up)
```bash
source <ilc2real>/env_tight.sh && ros2 run ilc_f1tenth_hw net_runner.py --ros-args -p session_dir:=$S
```

## 2. Baseline: RMA zero-shot (per track; 2 runs x 5 laps)
```bash
python hw_eval_requests.py --session-dir $S --policy runs/rmaV_s0/policy_final.npz --tag rma_<tire>_<track> \
  --plan mocap_<track>_b0_free_v08 --laps 5 --runs 2
python hw_eval_requests.py --session-dir $S --policy runs/freePPOV_s0/policy_final.npz --tag ppodr_zs_<tire>_<track> \
  --plan mocap_<track>_b0_free_v08 --laps 5 --runs 2       # PPO+DR zero-shot, scored like the others
```

## 3. Our stage (per track): 2 iterations x 2-lap trials = 4 laps
```bash
python -u hw_stage.py --backend manual --session-dir $S --cars real --policy runs/freePPOV_s0/policy_final.npz \
  --out runs/hw/real_<tire>_<track>_stage --goals mocap_<track>_b0_free_v08 --iters 2 --objective task --goal-cycle 1 \
  --trial-laps 2 --anchor-w 20 --anchor-plans mocap_square2fast_b0_free_v08 mocap_figfast_b0_free_v08 --fit-realize 0.9 \
  --fit-max-steps 3000 --seed 9001 --launch 1.0 8.0 --save-its 1 --beta 0.5 --delta 0.05 --cap 0.1 --slope-w 1 \
  --tr 1 --err-scale 0.10 1e9 1e9 1e9 1e9 0.13
```
The car flies each request as it appears. The update between trials takes about 30-60 s.

## 4. Evaluate the adapted network (same protocol as the baselines)
```bash
python hw_eval_requests.py --session-dir $S --policy runs/hw/real_<tire>_<track>_stage/real_policy.npz \
  --tag stage_<tire>_<track> --plan mocap_<track>_b0_free_v08 --laps 5 --runs 2
```

## 5. Score
```bash
for t in rma ppodr_zs stage; do python hw_eval_requests.py --session-dir $S --tag ${t}_<tire>_<track> --score; done
```
nerr = max(rms e_y / 10 cm, (1 - pace) / 0.1), crash = 5. Lower is better.

Also measure the plastic tire's friction (e.g. tilt or pull test): the plastic claim uses mu 0.4.
