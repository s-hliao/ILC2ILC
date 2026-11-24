# ilc_dogball — DistILC on a Go1 carrying a soccer ball in a flat box

Go1 Edu in Unitree's high-level walking mode (the policy sends velocity set points vx, vy, wz at 20 Hz) carries a
size-5 soccer ball loose in a flat box on its back (45 x 35 cm inside, 7 cm padded walls) through a sequence of goals
in the mocap room; the next goal starts when the trunk is within 0.2 m (switch on reach, drone soccer's protocol).
The ball is coupled continuously to the trunk's acceleration (no ballistic phase): ILC has authority all the time
except while the ball touches a wall (masked).

| file | what |
|---|---|
| `ilc_dogball/model.py` | reduced model: command filter + rate-limited velocity response, rolling ball (k = 0.6) in the moving box, padded walls; `NOMINAL`, `PERTURBED` (sim stand-in for the real robot) |
| `ilc_dogball/sim.py` | batched goal-sequence rollouts, switch on reach, ball escape (rim vault ~1 m/s impact) |
| `ilc_dogball/ppo.py` | PPO (nominal) / `--dr` PPO+DR (box covers PERTURBED); validation seeds 200-215 |
| `ilc_dogball/stage.py` | rows: ball beyond 60 % of its travel (hinge) + approach lateral miss; nominal lifted G; GN step |
| `scripts/stage_run.py` | stage vs PERTURBED, refit every 6 goals; `--positivity N` step-0 check |

    python -m ilc_dogball.ppo --out runs/ppo_nominal_s0 --seed 0
    python scripts/stage_run.py --policy runs/ppo_nominal_s0/policy_it0800.npz --out runs/stage_nom_s0 --goals 216

Design notes (2026-11-16): a constant box tilt was removed from PERTURBED (0.3 deg levelling error kept): no policy can
hold the ball off the downhill wall at rest, and it made up most of the first version's gap. Zero-offset ball rows were
replaced by the margin hinge: every acceleration moves the ball, so zero-offset rows only ask the dog to walk slower.
Hardware open item: measuring the ball (retroreflective-taped ball in mocap, or a camera over the box).
