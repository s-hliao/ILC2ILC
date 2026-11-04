# Quad hardware stage with the FR fix (2026-10-31; user: "do the quad fix")

The fix: FR = the theory's rollback rule (iv) in deploy.py. A fall rolls the goal back with no clearance step (`--clear 0`). A
trial clearly worse than the goal's best (1.5x and +0.5) is rolled back and its step halved (`--rollback 1.5 0.5`). Neighbour
targets come from within 12.5 cm (`--nbr-radius 0.125`). Config `FR` in stage_xfix.sh.

Network: PPO + DR (base_ekf/ppo_dr/it75, member `policy`), the same network the car deploys.

## Simulation evidence (planeval held-out goals, 28 robot-goal pairs, 3 seeds x 4 robots; stage_xfix/)

| arm | held-out nerr | vs PPO+DR zero-shot | vs RMA zero-shot | vs FADA + adaptation |
|---|---|---|---|---|
| PPO+DR + stage FR | 1.19 | +0.06 [-0.11, +0.27] | -0.34 [-0.72, -0.04] | -0.95 [-1.41, -0.52] |
| PPO+DR + stage before the fix (I2 / J1) | 1.29 / 1.61 | +0.16 / +0.48 | | |
| PPO+DR zero-shot | 1.13 | | -0.40 | -1.01 |
| RMA zero-shot | 1.53 | | | -0.61 |
| FADA + adaptation | 2.14 | | | |

Positivity at the goal that failed it before the fix, s1 (0.60, 0.00): median cos -0.76 (I2) -> +0.92 (FR). n = 3, because
rollbacks cut the runs of consecutive trials.

## Command (parse-checked with deploy.py's own argument parser, 2026-10-31)

On the GPU machine:
```bash
cd ~/ilc_ws/src/ilc_mjx
~/miniconda3/envs/ilcmjx/bin/python scripts/deploy.py --backend manual --robots go1 --out <RUN> --iters 6 \
  --policy /home/henry/ilc_ws/log/dilc/plane/base_ekf/ppo_dr/it75/policy --member policy \
  --update gn --est-window 0.025 --target-base own --gn-beta 0.5 --gn-delta 0.02 --step-rms 0.1 --clip-mode weighted \
  --nbr-radius 0.125 --clear 0 --rollback 1.5 0.5 \
  --gate 1000 1000 1000 \
  --goals 0.45 0.60 0.60,0.10 0.50,0.05 --seed 9001 --eval none
```
`--gate 1000 1000 1000` = no gate (user 2026-10-31: "match the sim results"). `--backend manual` turns on a safety gate by default
(`0 1.5 0.3`); the simulation results flew with none. In the desk tests the default gate rejected every update: on the action change
(0.64-0.81 > 0.3), and with that cap lifted (`0 1.5 10`, desk test 4) on the sim score (13.70 -> 23.58 > 1.5 x 13.70 + 0.5). Either
way the robot would have flown the zero-shot policy every iteration.

Goals: at most 10 cm boxes (the 15 cm boxes wait for the re-solved plans, plans_mid8).

IMU: the policy is an EKF policy and needs `/imu` (policy_jump_go1.launch.py's default; the Go1 bridge carries it). Check
`ros2 topic hz /imu` before the first jump. Without it the 0.45 m jump landed 33 cm short in the desk stand-in (2026-10-31: the
stand-in published no IMU; fixed in sim_node.py, and the ROS path then matched the container sim within 1-2 cm).
This is 4 goals x 6 iterations = 24 jumps (the simulation runs: 6 goals x 6 = 36, including the 15 cm boxes). On the robot workstation: `hw_session.py <RUN>/go1 ...`
as HARDWARE.md section 4. The connection, measurement and abort steps there are unchanged.
