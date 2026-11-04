# Quad hardware stage (2026-10-31 23:20; retrained network, two arms)

Previous version (old network base_ekf/ppo_dr/it75, one arm): paper/backup/QUAD_DEPLOY_FR_1031_pre_ppo8.md.

Network: PPO + DR retrained on the hardware goal range (bank_mid8_ekf, 40-65 cm forward x 0-20 cm boxes):
`mid8/ppo_dr/it50` (member `policy`; it50 = the best of its own DR evaluation).

Stage: FR = the rollback rule (iv). A fall rolls the goal back with no clearance step (`--clear 0`); a trial clearly worse than
the goal's best (1.5x and +0.5) rolls back and halves its step (`--rollback 1.5 0.5`); neighbour targets within 12.5 cm
(`--nbr-radius 0.125`). No gate (`--gate 1000 1000 1000`, as the simulation; user 2026-10-31).

## Arms (user 2026-10-31)

| arm | practised goals | jumps | starts |
|---|---|---|---|
| PPO+DR zero-shot | none | eval only | nominal; and the perturbed stance for the perturbed comparison |
| **A: PPO+DR + stage** | 0.45, 0.60 flat; boxes (0.50, 0.05), (0.60, 0.10), (0.50, 0.15), (0.575, 0.15) | 6 goals x 6 it = 36 | nominal |
| **B: PPO+DR + perturbed starts + stage** | **boxes <= 10 cm only** (user 2026-10-31): 0.45, 0.60, (0.60, 0.10), (0.50, 0.05) | 4 goals x 6 it = 24 | each goal its OWN fixed stance (drawn once, kept every iteration; 2026-11-01) + the stance anchor |

## Simulation evidence (stage_xfix/, 3 seeds x 4 robots; held-out planeval goals; nerr = max(|ex| / 5 cm, |ez| / 3 cm), fall = 5)

Arm A, final (12 of 12 runs, 2026-11-01 04:29; paired per robot, seed, goal; 95 % CI):

| held-out goals | PPO+DR zero-shot | + stage A (Jacobian along the nominal sim rollout) | + stage, Jacobian along the measured trajectory |
|---|---|---|---|
| **flat + boxes <= 10 cm (the hardware set)** | 1.35 | **1.05 (-0.30 [-0.55, -0.09])** | 1.23 (-0.12 [-0.32, +0.10]) |
| flat | 1.30 | **1.04 (-0.26 [-0.44, -0.07])** | 1.27 |
| boxes <= 10 cm | 1.41 | 1.05 (-0.36 [-0.86, +0.03]) | 1.20 |
| boxes 10-20 cm (not on hardware) | 1.37 | 1.56 (+0.19 [-0.06, +0.46]), falls 10 % | 1.53 |
| positivity (median cos, frac > 0) | | +0.74, 83 % | +0.26, 59 % |

Practised goals: the last accepted step's J is 61 % below iteration 0 (median). Steps accepted 60 %, runaway-rolled-back 37 %,
fell 2 %. Weak spot: the 15 cm practised box (0.50, 0.15) accepts 48 % of its steps, and the held-out boxes near it got more falls.

Arm B (2026-11-01 update; user: "don't only practice crouches"). Each practised goal is flown from its own stance, drawn once
from the simulation's range (front / rear leg pair bent by s x (hip +0.1, knee -0.2) rad, s ~ U(-1, 1), seed 9001) and kept for
every iteration (`--train-starts 1.0 --starts-fixed`), plus the stance anchor (`--anchor-pert`: the refit also holds the network
on 16 perturbed-stance rollouts in the nominal sim). Simulation (stage_xfix/ppo8_FRPAL_*, 3 seeds x 4 robots; held-out goals
<= 10 cm; change of nerr vs the zero-shot from the same perturbation, paired, 95 % CI):

| start perturbation (evaluation) | zero-shot nerr | change, + stance anchor |
|---|---|---|
| front-feet blocks 1.5 / 2 cm | 2.73 | **-0.17 [-0.30, -0.05]** |
| half nose-up stance | 2.07 | **-0.20 [-0.31, -0.10]** |
| half tall stance | 1.73 | -0.25 [-0.56, +0.03] |
| crouch | 1.94 | -0.08 [-0.30, +0.20] |
| nose-down | 1.47 | +0.04 [-0.17, +0.37] |
| rear-feet blocks 1.5 / 2 cm | 1.44 | **+0.16 [+0.07, +0.25]** (worse: spread of the forward error grows 6.9 -> 8.3 cm) |
| all of the above | 1.88 | **-0.07 [-0.14, -0.00]** |

Positivity (the GN's predicted landing change vs the measured one between consecutive trials): median cos +0.51, 74 % > 0.

Which linearization (user 2026-11-01: "doesn't the jacobian linearize at each point along the trajectory"; Nguyen et al. linearize
along the executed trial). Offline, on the SAME arm-A trial pairs (consecutive iterations of a goal, no fall; $CLAUDE_JOB_DIR/tmp/
quad_lin_compare.py), the closed-loop landing sensitivity built three ways, cos(predicted, measured landing change), median:

| trial pairs from | MuJoCo along the nominal sim rollout (deployed) | MuJoCo at the measured trial states (--jac-at real) | SRB along the measured trial (Nguyen-style) |
|---|---|---|---|
| arm A runs (nominal Jacobian), 284 pairs | +0.44 | +0.44 | +0.08 (magnitude 7 x off) |
| --jac-at real runs, 278 pairs | +0.07 | +0.10 | +0.05 (3 x off) |

The measured-trajectory MuJoCo linearization predicts exactly as well as the nominal rollout on the same pairs; the SRB along the
trial predicts poorly. The --jac-at real runs took steps whose effect neither predicts and ended worse on held-out goals (-0.12 vs
-0.30); their steps' difference was not traced further. Deployed: the nominal rollout (equal prediction, better outcome, and no
dependence on mocap-derived velocities on the robot).
The Jacobian along the nominal sim rollout stays: along the measured trajectory (`--jac-at real`) or from the measured start
stance (`--sim-start measured`) was less positive in simulation (median +0.32 / +0.42) and no better on held-out goals.

## Commands (parse-checked with deploy.py's own parser, 2026-10-31 23:20)

On the GPU machine (`cd ~/ilc_ws/src/ilc_mjx`):
```bash
PYQ=~/miniconda3/envs/ilcmjx/bin/python
COMMON="--backend manual --robots go1 --iters 6 --policy /home/henry/ilc_ws/log/dilc/plane/mid8/ppo_dr/it50/policy --member policy \
  --update gn --est-window 0.025 --target-base own --gn-beta 0.5 --gn-delta 0.02 --step-rms 0.1 --clip-mode weighted \
  --nbr-radius 0.125 --clear 0 --rollback 1.5 0.5 --gate 1000 1000 1000 --seed 9001 --eval none"
# arm A
$PYQ scripts/deploy.py $COMMON --out <RUN_A> --goals 0.45 0.60 0.50,0.15 0.60,0.10 0.50,0.05 0.575,0.15
# arm B (<= 10 cm; each goal its own stance + the stance anchor; dry-run checked 2026-11-01 04:30: REQUEST.json carries "starts",
# REQUEST.md each goal's stand_offset)
$PYQ scripts/deploy.py $COMMON --out <RUN_B> --goals 0.45 0.60 0.60,0.10 0.50,0.05 --train-starts 1.0 --starts-fixed --sim-start nominal --anchor-pert
```
On the robot workstation (HARDWARE.md section 4 for the connection; `/imu` must be up: `ros2 topic hz /imu`):
```bash
python3 hw_session.py <RUN_A>/go1 --pose-topic <topic> --pose-type pose --pose-latency <measured> --mocap-offset x y z
python3 hw_session.py <RUN_B>/go1 --pose-topic <topic> --pose-type pose --pose-latency <measured> --mocap-offset x y z
```
Arm B: hw_session.py (updated 2026-11-01; copy src/ilc_quad/scripts/hw_session.py to the robot workstation and rebuild / re-source)
takes each goal's stance from REQUEST.json ("starts") and passes it as the node's stand_offset, exactly as the simulated arm; no
--start needed (it is the fallback for requests without starts). The stance is software only (joint offsets from the home pose).

## Evaluation (HARDWARE.md section 5 protocol, >= 4 jumps per goal per policy, alternate the policies)

Policies: zero-shot `mid8/ppo_dr/it50/policy`; arm A `<RUN_A>/go1/it6/policy`; arm B `<RUN_B>/go1/it6/policy`.
Held-out goals (the simulation's planeval set, never practised):
- primary, boxes <= 10 cm: 0.4375 flat, 0.5625 flat, (0.4625, 0.06), (0.6125, 0.07);
- then if the session allows: (0.53, 0.11), (0.56, 0.12), (0.51, 0.13), (0.52, 0.16) -- arm A's weak spot in simulation.
Arm A and zero-shot from the nominal stance. Arm B and zero-shot again from the perturbed starts, alternating policies, same
perturbations for both: front-feet block 1.5 cm (physical, under both front feet), `--start noseuph` (half nose-up, stand_offset
[0, 0, 0.05, -0.1]), `--start tallh` (half tall, [-0.05, 0.1, -0.05, 0.1]), and `--start crouch`. The simulation's numbers per
perturbation are in the arm-B table above (rear-feet blocks made it worse there; evaluate them only if time allows).
Score per jump: nerr = max(|ex| / 5 cm, |ez| / 3 cm), a fall = 5; report each arm's change against the zero-shot.
