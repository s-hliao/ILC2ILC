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
| **B: PPO+DR + perturbed starts + stage** | **boxes <= 10 cm only** (user 2026-10-31): 0.45, 0.60, (0.60, 0.10), (0.50, 0.05) | 4 goals x 6 it = 24 | one perturbed stance, repeated every jump |

## Simulation evidence (stage_xfix/, 3 seeds x 4 robots; held-out planeval goals; nerr = max(|ex| / 5 cm, |ez| / 3 cm), fall = 5)

Arm A, interim (8 of 12 runs finished before the quad was stopped 2026-10-31 23:05; paired per robot, seed, goal):

| held-out goals | PPO+DR zero-shot | + stage A | falls |
|---|---|---|---|
| flat | 1.32 | **1.12** | 0 -> 0 % |
| boxes <= 10 cm | 1.43 | **0.98** | 0 -> 0 % |
| boxes 10-20 cm | 1.43 | 1.53 | 7 -> 8 % |

Practised goals: the last accepted step's J is 61 % below iteration 0 (median). Steps accepted 60 %, runaway-rolled-back 37 %,
fell 2 %. Weak spot: the 15 cm practised box (0.50, 0.15) accepts 48 % of its steps, and the held-out boxes near it got more falls.

Arm B: the previous version practised all six goals from perturbed stances and fell on 87 % of its (0.575, 0.15) practice jumps
(18-20 % on the other >= 10 cm boxes). The <= 10 cm version (`LOWGOALS=1`, stage_xfix/ppo8_FRPL_*) is running in simulation now.

## Commands (parse-checked with deploy.py's own parser, 2026-10-31 23:20)

On the GPU machine (`cd ~/ilc_ws/src/ilc_mjx`):
```bash
PYQ=~/miniconda3/envs/ilcmjx/bin/python
COMMON="--backend manual --robots go1 --iters 6 --policy /home/henry/ilc_ws/log/dilc/plane/mid8/ppo_dr/it50/policy --member policy \
  --update gn --est-window 0.025 --target-base own --gn-beta 0.5 --gn-delta 0.02 --step-rms 0.1 --clip-mode weighted \
  --nbr-radius 0.125 --clear 0 --rollback 1.5 0.5 --gate 1000 1000 1000 --seed 9001 --eval none"
# arm A
$PYQ scripts/deploy.py $COMMON --out <RUN_A> --goals 0.45 0.60 0.50,0.15 0.60,0.10 0.50,0.05 0.575,0.15
# arm B (<= 10 cm)
$PYQ scripts/deploy.py $COMMON --out <RUN_B> --goals 0.45 0.60 0.60,0.10 0.50,0.05
```
On the robot workstation (HARDWARE.md section 4 for the connection; `/imu` must be up: `ros2 topic hz /imu`):
```bash
python3 hw_session.py <RUN_A>/go1 --pose-topic <topic> --pose-type pose --pose-latency <measured> --mocap-offset x y z
python3 hw_session.py <RUN_B>/go1 --pose-topic <topic> --pose-type pose --pose-latency <measured> --mocap-offset x y z --start crouch
```
`--start crouch` = joint offsets [0.1, -0.2, 0.1, -0.2] from the home pose before every jump (the evaluation's crouch posture, the
edge of the simulated arm's start range). Difference to the simulation: there, each goal drew its own fixed stance inside that
range; on the robot one named stance is repeated for every goal.

## Evaluation (HARDWARE.md section 5 protocol, >= 4 jumps per goal per policy, alternate the policies)

Policies: zero-shot `mid8/ppo_dr/it50/policy`; arm A `<RUN_A>/go1/it6/policy`; arm B `<RUN_B>/go1/it6/policy`.
Held-out goals (the simulation's planeval set, never practised):
- primary, boxes <= 10 cm: 0.4375 flat, 0.5625 flat, (0.4625, 0.06), (0.6125, 0.07);
- then if the session allows: (0.53, 0.11), (0.56, 0.12), (0.51, 0.13), (0.52, 0.16) -- arm A's weak spot in simulation.
Arm A and zero-shot from the nominal stance; arm B and zero-shot again from `--start crouch`.
Score per jump: nerr = max(|ex| / 5 cm, |ez| / 3 cm), a fall = 5; report each arm's change against the zero-shot.
