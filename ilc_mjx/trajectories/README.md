# Recorded Go1 jumps (ILC2Real, 2D goal plane, sim-to-sim on the CPU "real" robots)

Ground-truth trajectories as the CPU robots (ilc_quad's async lockstep MuJoCo sims: `real_r1`, `real_s1`,
`real_r4`, `real_r5`, `real_r4m`) recorded them, for replay. Each `<name>.npz` holds every jump of one source run, and
`<name>.json` is its human-readable index.

| file | what |
|---|---|
| `gate_loose` | **our hardware stage, the deployment recipe**: v10_s2 it20, nominal Gauss-Newton ILC, 6 fixed goals x 4 iterations (24 real jumps per robot), safety gate `1 1.5 0.4`; robots r1, s1, r4, r5 |
| `hwstage_tp_<cond>` | our hardware stage trained under one constant perturbation `<cond>` (nose down, blocks, ...); the GPU sim stays nominal |
| `hwstage_rma_crosstrial_calibration` | the cross-trial RMA baseline's "hardware stage": its 6 calibration jumps per robot |
| `hwstage_fada_lora` | FADA's on-robot LoRA adaptation (8 iterations x 6 goals) |
| `eval_<method>` | **sim-to-sim transfer**: every method on the 8 reserved test goals x 4 robots, 1 episode at the evaluation seed (701). Methods: `ours_zeroshot`, `ours_24jumps`, `learner_dr`, `ppo_dr`, `rma`, `rma_crosstrial`, `fada` |

These are single episodes for *replay*; the reported success rates come from the full evaluations (4 episodes per
goal, plus the 8 perturbations) in `log/dilc/plane/`.

## Contents of each npz

- `t` (T,): seconds, 2 ms ticks (T = 1200, 2.4 s)
- `pos` (n, T, 3): trunk position (world frame; the robot starts at the origin, facing +x)
- `quat` (n, T, 4): trunk orientation, w x y z
- `q` (n, T, 12): joint angles in go1.xml qpos order, FR FL RR RL x (hip, thigh, calf)
- `contacts` (n, T, 4): foot contact flags, FR FL RR RL
- `meta`: a JSON list, per jump: robot, cond (robot[+perturbation]), iteration (-1 for evaluations), n_ticks (the
  recording's length: shorter after a fall; the arrays repeat the last pose after it), goal (x, h) m, box_x_front and
  box_height (the box as the robot's scene placed it: 1 m long and wide, front face at box_x_front), ex / ez / eth (landing
  error: x m, z m, pitch rad), fell, J (the trial's cost), train_cond, file (the source episode)

## Replay

```bash
# MuJoCo viewer (needs a display and a mujoco_menagerie checkout: $MUJOCO_MENAGERIE_PATH or ~/mujoco_menagerie)
python scripts/replay_mujoco.py trajectories/gate_loose.npz --robot real_s1 --goal 0.575 0.15   # its 4 batches
python scripts/replay_mujoco.py trajectories/eval_ppo_dr.npz --robot real_r4 --goal 0.54 0.14 --speed 0.5
python scripts/replay_mujoco.py trajectories/gate_loose.npz --robot real_r1 --check             # headless sanity check
python scripts/replay_mujoco.py trajectories/gate_loose.npz --robot real_s1 --goal 0.575 0.15 --record s1_box.mp4  # rendered video (needs OpenGL)

# side-view animations (matplotlib; no rendering backend needed) -- all of them at once:
python scripts/make_animations.py                       # GIFs, small enough to commit; --format mp4 needs ffmpeg
python scripts/animate_hw_stage.py hwstage trajectories/gate_loose.npz --robot real_s1
python scripts/animate_hw_stage.py compare trajectories/eval_ours_24jumps.npz trajectories/eval_ppo_dr.npz \
    --labels "ours, 24 real jumps" "PPO + DR" --robot real_r4

# re-export from the raw episode files
python scripts/export_trajectories.py log/dilc/plane/gate_loose
```

Rendered animations: `figures/ilc2real/anim/` (`hwstage_*`: the hardware stage batch by batch, one panel per goal;
`compare_transfer_*`: all methods side by side on each test goal).
