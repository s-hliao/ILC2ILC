# Handoff 2026-11-04: hardware deployment (car + quad)

For the agent taking over deployment. Supersedes HANDOFF_2026-10-27.md for everything about hardware. Nothing has flown on real hardware yet;
every number below is simulation, the change of each network's error after the stage vs the SAME network's zero-shot (negative = better),
95 % bootstrap CI over runs. Nothing is running on matcha or ube at handoff.

## 1. The three experiments (user, 2026-11-03)

| experiment | hardware | runbook |
|---|---|---|
| car, PPO+DR + stage | rubber AND plastic tire: zero-shot, 3-iteration stage (2-lap trials = 6 laps), evaluation | `src/ilc_f1tenth/ilc2real/HW_RUNBOOK_N6_1031.md` |
| car, RMA + stage | same, both tires (adaptation module frozen) | same |
| quad, PPO+DR + stage (arm A) | one Go1: damping test, zero-shot, 36-jump stage, evaluation; also evaluate both policies from 1-2 perturbed stances | `log/dilc/plane/QUAD_DEPLOY_FL_1102.md` |

Perturbed-start PRACTICE on the quad is a simulation study only (user 2026-11-03). Schedule: car hardware Nov 2-8, quad Nov 9-12, paper by Nov 30.

## 2. Car (ready)

- Sim result (final; runbook "FINAL simulation table"): held-out peanut clockwise, rubber `_x120` PPO+DR -5.3 % [-6.9, -4.0] (18/18 runs),
  RMA -4.2 % [-6.0, -2.6] (16/18); plastic `_a33` PPO+DR -27.3 % [-29.3, -25.2] (9/9), RMA -13.4 % [-15.4, -11.6] (9/9). Practised tracks
  -7.9 % to -21.5 %. Reproduce: `python3 runs/screen_final/car_final_score.py runs/screen_final/rubber runs/screen_final/plastic`.
- Decisions made by the user: `--skip-launch 0.5` (launch mask) on both tires and networks; ONE held-out speed (rubber `_x120`); do not
  discuss why RMA's `_x110` was not flown. Lap budget <= 6 real laps per tire for the stage.
- Desk rehearsal of the runbook's own commands (sections 1, 3-6) passed for both tires and all four arms (car_session_sim.py as the car).
  Rule found there: `hw_eval_requests.py` only queues; score (section 6) only after every req has a res, every tag "2 flights".
- Before the session: steering check +-0.34 rad (vesc.yaml clips otherwise), crash envelope 0.5 m (`net_runner.py safety_ey`).

## 3. Quad (ready except one decision)

- First: the damping test (QUAD_DEPLOY_FR_1031.md, section "First session: measure the joint damping"; damping_flights.py + fit_damping.py;
  it reads high: sim truth 2 reads 3.0, truth 0.1 reads 0.3-0.5). The reading picks the network (QUAD_DEPLOY_FL_1102.md section 1):
  >= 1.5: `fl2s/ppo_dr/it50` (plans_fl2s); 0.6-1.4: `fl1s/ppo_dr_cc/it100` + anchor (plans_fl1s); <= 0.5: `fl01z/ppo_dr/it75` (plans_fl01s).
- Arm A = the section-4 command WITH `--anchor-pert` (adopted 2026-11-02 by a rule fixed before the result). d=2 sim: nominal held-out
  -0.33 [-0.45, -0.20], from perturbed starts -0.12 [-0.20, -0.05], no falls.
- **OPEN DECISION (user):** d=2 network. Cost-cap retrain `fl2s/ppo_dr_cc/it50` is the stronger baseline (zero-shot 1.02 vs 1.26); arm A +
  anchor on it -0.23 [-0.40, -0.08] -> 0.79. Recommended (never weaken baselines). Do not switch without the user's OK.
- Robot workstation: copy `log/dilc/plane/deploy_bundle_1102/` (121 files), `tar xzf quad_deploy_1102.tgz && bash verify.sh .` must print
  "ALL 121 FILES OK"; colcon build ilc_quad; overlay first on PYTHONPATH. GPU side needs the env line in section 4 (else target z off 13 cm).
- Evaluation: `src/ilc_quad/scripts/quad_eval_session.py` (section 6), served through the unchanged `hw_session.py`; `--start crouch` etc.
  for the perturbed-stance evaluation. Tested on a sim robot.
- Every hardware path was tested on a sim robot (stage_fl/hwsm2..5); NOT tested: Go1 bridge, real mocap, IMU. First jumps on foam / catch rig
  (landing: calf 0.3 rad from its stop at ~2 m/s).
- Perturbed-start sim study (for the paper; stage_fl/ARMB_VARIANTS_1102.md): one constant perturbation per stage, 12 runs each, all five
  beat zero-shot under their own condition: block 1.5 cm -0.82, crouch -0.28, nose-up -2.57, nose-down -0.42, tall -0.20; nose-up costs
  +1.38 at the nominal stance (measured stance leak).
- Unsolved, causes measured: d=0.1 held-out transfer (sim's predicted landing change cos +0.23); r5 at 20 cm (feet hit the box edge every
  trial; needs an r5-specific plan = system id, not allowed; r5feas/README.md); 20 cm practice interference (not in the hardware set).

## 4. Standing rules (user)

No git commits / pushes. Fair experiments only, never weaken baselines, selection rules written BEFORE results (and reported even when they
fail). Ask before method changes while the user is awake. Always inspect trajectories; never state an unchecked cause. Lead with continuous
errors + CIs. Car plans carry no friction info; no adaptive sysid. Keep Go1 and 500 Hz. When the user is asleep: decide and act, don't ask.

## 5. Machines and pitfalls

- matcha: 4 x 2080 Ti, 36 CPUs shared with other users: keep load <= 34 (`log/matcha/gpusched.py JOBS.json`, run with `GS_LOAD_MAX=28`;
  per-job GPU reservation 3500 MiB after one out-of-memory crash). GPUs <= 88 C. Docker container `ilc_quad` runs the sim robots.
- `/home` is SHARED and nearly full (6.5 GB free at handoff; it hit 0 once). Cleared so far: pip cache, JAX compilation cache (regenerable).
- ube (ssh host in `$CLAUDE_JOB_DIR/tmp/fada_car_eval.sh`, var U): 36 CPUs, 3 GPUs, ONE job per GPU, python env `f1t` (not f1t_ube).
- Stop runs with a script whose patterns live in the file (no pgrep self-match), kill container processes via docker exec too, remove a
  stopped run's partial folder before relaunching. Never edit a bash script that is running (monitors included: write a new file).

## 6. Where things are

Reports: `log/MORNING_REPORT_2026-11-02.md` (with the 2026-11-02 21:00 update). Arm / variant rules and results:
`log/dilc/plane/stage_fl/ARMB_VARIANTS_1102.md`. Scorers: `stage_fl/stB.py NET CFG plain|rob`, `stage_fl/stT.py` (constant-perturbation).
Memory index: `~/.claude/projects/-home-henry-ilc-ws/memory/MEMORY.md` (project_quad_deploy_fl_1102.md, project_deploy_decisions.md).

## 7. Added 2026-11-05: what the car's improvement is (measured)
The car table's error is max(rms e_y / 0.1 m, (1 - pace) / 0.1). On plastic the stage raises lap pace (11 of 12 cars; nominal plastic car
0.920 -> 0.953) while lateral rms e_y stays the same or worsens (harsh_combo 7.5 -> 10.4 cm); the lowest-grip car (ax:mu=0.40) is unchanged
(50 % crashes before and after). Rubber: lateral 8.1 -> 7.4 cm. The lateral flow in the 0.5 m tube already converges zero-shot
(src/paper/final/car_flow/flow_car.py, car_flow.png). Write the car claim as pace, not tracking.
