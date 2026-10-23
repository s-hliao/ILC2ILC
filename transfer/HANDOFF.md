# Handoff: ILC2Real (deep ILC sim → hardware; formerly "SD2"), Go1 jumping + F1TENTH drifting

> **Newest handoff: `src/paper/HANDOFF_2026-10-22.md`** (quad timing-DR fix, flow analysis, transfer to a bigger machine). Read it first, then `src/paper/HANDOFF_2026-10-18.md`.

Written 2026-10-05 for an agent continuing this work on a new machine. Read this first. Then read
`log/dilc/RESULTS_SUMMARY.md` (every result in one table) and `log/dilc/deploy/NOTES.md` (the detailed lab
notebook, chronological). To set up the new machine, see `transfer.sh` (packing on the old machine, restoring on the
new one).

## 1. The project in one paragraph

- **Paper:** "ILC2Real: Sample-Efficient Policy Learning and Adaptation with Iterative Learning Control" (renamed
  2026-10-08 from "Structure in Simulation, Details on Deployment" (SD2): no explicit structure is named or shown to
  matter -- value gradients don't help, and the learned feedback structure does not predict transfer).
- **Thesis:** learn a goal-conditioned policy in a cheap nominal simulator, with no domain randomization; then
  converge it to the real dynamics with a few hardware trials, using the same iterative learning control (ILC) update
  grounded in measured outcomes (the model only supplies the direction; domain randomization instead depends on its
  family covering the deployment dynamics).
- **Starting point:** Gurumurthy et al.'s Deep Off-Policy ILC (VG-SAC). Their learned critic failed here (see
  `memory/project_deep_ilc_findings.md`). What works is a closed-loop co-state ILC update applied directly to a policy
  network.
- **Task 1, done in sim-to-sim:** a Go1 quadruped jumping to goal distances (0.40–0.60 m).
- **Task 2, F1TENTH car drifting:** in the abstract, but barely started.
- **"Hardware" so far:** CPU-simulated "real-like" robots with async timing, sensing noise and perturbed dynamics.
  The real-Go1 path is built and tested in sim but has not been flown.

## 2. Machine layout (keep these paths: about 80 scripts hardcode `/home/henry/...`)

| Path | What |
|---|---|
| `~/ilc_ws/src/` | **git repo** `git@github.com:s-hliao/ILC2ILC.git`, branch `quad` (pushed; `main` is merged in) |
| `src/ilc_quad/` | ROS 2 package. CPU MuJoCo sim, JumpILC controller (Nguyen et al.), robot bridges, and the deep-ILC tools in `scripts/` |
| `src/ilc_mjx/` | GPU sim (MuJoCo MJX / JAX): batched Go1 jump, the hardware-stage driver `scripts/deploy.py`, baselines |
| `src/ilc_f1tenth/` | F1TENTH car: Fiala-tire dynamics, iLQR, ILC loop (plain Python + ROS) |
| `src/f1tenth_gym/` | git submodule (`s-hliao/f1tenth_gym`, branch `dev-humble`) |
| `~/ilc_ws/install/` | colcon install of `ilc_quad`, built **inside the container**; scripts are often copied here by hand after edits (see §7) |
| `~/ilc_ws/log/dilc/` | **every deep-ILC experiment** (not in git): runs, results, queue scripts |
| `~/mujoco_menagerie` | Go1 model (google-deepmind/mujoco_menagerie, commit `da76818`) |
| `~/miniconda3/envs/dilc` | host env, Python 3.11, torch 2.14 (cu126): `dilc_train.py` (the sim stage learner) |
| `~/miniconda3/envs/ilcmjx` | host env, JAX 0.9.0.1, MuJoCo 3.12.0 (MJX), CasADi: `deploy.py`, everything GPU, the F1TENTH code |
| Docker container `ilc_quad` (image `ilc_quad:humble`) | ROS 2 Humble + MuJoCo 3.12 CPU sim. Mounts `~/ilc_ws → /ilc_ws` and `~/mujoco_menagerie → /mujoco_menagerie`; `--cpus 28`; PID 1 `sleep infinity`; no GPU |

Commands run in the container as
`sg docker -c "docker exec ilc_quad bash -lc 'cd /ilc_ws && source install/setup.bash && ...'"`.

## 3. The Go1 pipeline

### Policy

- **Inputs (31):** the single-rigid-body error (6), time (1), goal (2), the 3 past errors (18), the previous action
  (4).
- **Network:** MLP with 2 × 256 ReLU layers and a tanh mean.
- **Outputs (4):** corrections to the front/rear contact forces. The forces are the plan's forces (interpolated from a
  bank of 5 trajectory-optimization plans) plus `a_scale` (25 N for fx, 50 N for fz) × the action, over the 60
  contact samples.
- **Landing:** a PD controller.

### Sim stage (GPU, nominal, no domain randomization)

- **Command:** `log/dilc/vgpre.sh coilc <gpu>` with env switches. The final config ("ex1") is
  `GB=128 COREC=128 UTD=0.25 EPS=1200 TE=-13 FALL=2 FRESH=1 ALR=1e-4 SCHED="0.2 0.5" FLOOR=0.1 GN=1 EXPL="0.08 0.15"`
  (also `log/dilc/holdout/FROZEN.md`). It calls `src/ilc_quad/scripts/dilc_train.py`.
- **Update:**
  - per rollout, closed-loop co-states from finite-difference sim Jacobians;
  - a Gauss–Newton step on the landing error in Stage III;
  - the actor regresses onto the stepped actions.
- **Exploration:** random stance offsets and action offsets.
- **Outputs:** `log/dilc/pre_coilc<TAG>/policy/g0_coilc.s0_e1200.npz`.
  - Seeds: `ex1`, `ex1s1` … `ex1s5`.
  - `scr*`: from scratch (`INIT=none`).
  - `w{32,64,128}s*`: width ablation.
  - `drtrue`: DR over the true family on CPU robots.
- **Init:** `log/dilc/off10/` (a network trained on JumpILC trials) is the default init. From scratch is worse
  (3.50 vs 2.67 on the reserved test).

### Hardware stage (`src/ilc_mjx/scripts/deploy.py`)

- **Method ("E3"):**
  `--update gn --gn-beta 0.3 --step-rms 0.05 --target-base own --bold 0.3 --stall 3`, 10 iterations × goals
  0.425 / 0.5 / 0.575 = 30 jumps per robot.
- **Each iteration:**
  1. fly one jump per goal on the robot;
  2. take a Gauss–Newton ILC step on the measured landing error, through the nominal sim's closed-loop sensitivity
     with the policy's feedback at the real states;
  3. refit the network to the stepped actions, anchored to its previous outputs;
  4. optionally run the safety gate (`--gate 0 1.5 0.3`).
- **Backends:**
  - `--backend container` (default): the CPU sim robots, used for sim-to-sim;
  - `--backend manual`: writes `REQUEST.json` / `REQUEST.md` for a real robot and waits for its episodes (see §6).
- **Options added recently:**
  - `--train-starts S`: each hardware jump starts from a random posture. The front and rear leg pairs are bent
    s ~ U(−S, S) × (hip +0.1, knee −0.2) rad.
  - `--revisit`: every other iteration, each goal re-flies its worst start so far.
  - `--reg-jac`, `--anchor-pert`: the flow-style regularization (a negative result).
  - `--jac-dyn`, `--jac-flip`: the model-wrongness sweep.
  - `--eval holdout`, `--perturbed`, `--reps`, `--control random`.

### Robots and evaluation (the CPU sim in the container; `src/ilc_quad/scripts/dilc_execute.py`)

- **Robots:** `real_r0` … `real_r5` and `real_s1` are drawn from `sample_condition` with seed 2026. They vary mass,
  CoM, motors (scale and torque–speed curve), friction, joint friction, ground compliance and payload, plus async
  effects: actuation delay and jitter, command drops, mocap rate/latency/noise, joint noise, foot-sensor bias.
- **Fixed-sensor twins:** `real_r0f/r2f/r3f` keep the foot-sensor bias fixed per robot (`--sensor-seed`).
- **Evaluation sets** (`deploy.py` `EVAL`):

| set | robots | goals | seeds | use |
|---|---|---|---|---|
| `val` | r1, s1, r4, r5 | 0.45 / 0.4625 / 0.5375 / 0.55 | from 301 | tuning |
| `final` ("reserved") | r1, s1, r4, r5 | 0.4375 … 0.5625 | from 701 | now a second validation set |
| `holdout` (frozen test) | r0, r2, r3 | 6 goals | from 1301 | the pre-registered final test (`holdout/FROZEN.md`); done once, do not retune on it |

- **Perturbed set:** 8 perturbations per robot (`log/dilc/final701/rob_<robot>.txt`): front-foot blocks of 1.5 / 2 cm,
  crouch, tall, nose-up, nose-down, bad mocap, and a 10 ms delay.
- **Score:** 0.01 × landing cost + 20 × fall. Lower is better.
- **Stats:** `log/dilc/deploy/stats.py` (bootstrap over seeds then jumps; paired; sign test); tables via
  `log/dilc/deploy/tab.py RUN...`.

## 4. Where the results stand (details: `RESULTS_SUMMARY.md`, `deploy/NOTES.md`)

**Strong:**
- **Nominal sim beats domain randomization at transfer.** Every DR / RMA / FADA / PPO baseline is significantly worse,
  even zero-shot. PPO is the best in the GPU sim and the worst at transfer.
- **The ILC direction is causal.** Random steps of the same size do much worse (12/12 cells).
- **Very few jumps are needed.** One iteration (3 jumps) gets nearly all of the nominal gain.
- **Sign-correct is enough.** A model with thrust-to-weight wrong by ±30% still works; a flipped sensitivity is worse
  than not adapting. The measured S_true·L has two positive eigenvalues and one ≈ 0.
- **Frozen held-out test:** ours 1.32 vs zero-shot 1.60, p = 0.0075.
- **Snapshot selection without the robot:** pick the sim snapshot with the best perturbed-stance GPU-sim score.

**Weak or negative:**
- The hardware stage from nominal starts does not improve perturbed starts.
- Per-goal JumpILC is more precise on easy robots. This is real: the fairness check is in `fair/`.
- The safety gate can block good updates (reserved test: 2.99 with it vs 2.67 without).
- Flow-style regularization does not help.
- A narrower network is worse.
- Oracle DR over the true family is more robust (6.77 perturbed) but needs the true family.

**Newest:**
- Flying the hardware stage **from varied starts** (`--train-starts 1`) cuts perturbed error by 25%
  (10.76 → 8.08, 10/12 cells) with the same precision, and halves the front-foot-block error though blocks are never
  flown.
- Nose-down was not fixed. The cause is one heavy robot (`real_r5`): a sim-learned pitch reflex cuts rear thrust,
  and the ILC step cap (rms 0.05) pins.
- **In flight at handoff:** `log/dilc/vstart/{vs1_cap10, vs1_rev, vs1_cap10_rev}_{ex1,ex1s1,ex1s2}`, testing a step
  cap of 0.10 and `--revisit`. When they finish, the scheduler writes `log/dilc/vstart/RESULTS.txt`.
  - If you are reading this on the new machine, these runs either finished before the transfer (check the file) or
    were cut off: rerun via the scheduler.
  - Check real_r5's nose-down by goal, nominal precision, and the other perturbations.

**Recommended hardware budget:** 3–9 jumps for nominal starts. For varied starts, 30 works at S = 1; the checkpoint
evaluation to see whether fewer suffice hasn't been run.

## 5. Experiment infrastructure (use it; it encodes hard-won fixes)

| Tool | What it does |
|---|---|
| `log/dilc/runall/make_jobs.py` → `jobs.json` | the job list (name, command, CPU weight, GPU slot, dependencies, done-file). Re-read live by the scheduler. A job counts as done when its done-file exists. |
| `log/dilc/runall/runall.py` | the scheduler: container CPU budget about 34 weight units, ≤ 4 jobs per GPU on GPUs **0–2** (GPU 3 is reserved for others), retries. Run it as `systemd-run --user --unit runall ...`; state in `runall/state.json`, `status.txt`, `logs/`. |
| `log/dilc/runall/validate.py` | integrity check for every `deploy.py` run: no skipped jumps, no missing evals, eval files the right length. **Run it before trusting any result.** |
| `log/dilc/runall/shmclean.sh` (unit `shmclean`) | every minute, cleans stale Fast DDS segments in the container's 64 MB `/dev/shm` and kills orphaned sim processes |
| `log/dilc/runall/watch.py N` | exits on a job failure or retry, `/dev/shm` > 70%, a run losing jumps, all jobs done, or after N minutes. Run it in the background. |
| `log/dilc/refly.py` | re-flies one missing evaluation and patches `summary.json` |

**Lessons that cost a day (2026-10-04); see "Infrastructure incident" in `deploy/NOTES.md`:**
- **Node-creation deadlock:** Fast DDS 2.6 ROS node creation deadlocks on `/dev/shm` port locks left by killed
  processes.
  - Fix: `ilc_jump_lockstep.py` uses UDP-on-loopback only (`fastdds_lockstep_udp.xml`, with `ROS_LOCALHOST_ONLY`
    unset for that process). Results are bit-identical.
  - Do not remove it.
- **Skipped jumps:** `deploy.py` now re-flies missing goals, or aborts the run, instead of silently skipping them.
  Twenty-five runs had been corrupted and were rerun.
- **Worker pools:** `dilc_execute.py` pools use `forkserver`. The fork-based pool could deadlock.
- **Hang diagnostics:** sim workers dump stack traces to `runall/hangs/` if a jump runs past 240 s.
- **Compile cache:** the JAX persistent compile cache is at `~/.cache/jax_ilc` (set per job in `make_jobs.py`).

## 6. The real-robot path (built, tested in sim, not yet flown)

- **Runbook:** `src/ilc_mjx/HARDWARE.md`.
- **Pieces:**
  - `src/ilc_quad/scripts/policy_jump_node.py`: flies the policy with IlcJumpNode's estimator and safety.
  - `src/ilc_quad/scripts/hw_session.py`: the operator's session runner. It follows `deploy.py --backend manual`;
    on the robot, press Enter per jump.
  - Launch files `policy_jump_go1.launch.py`, `policy_jump_sim.launch.py`, `go1_bridge.launch.py`.
- **Tested end to end:** `hw_session.py --backend sim` (`log/dilc/hwtest/session`).
- **The user's physical to-do:**
  - bring up `go1_bridge`;
  - measure the mocap latency (`pose_latency`);
  - share the episode folder between machines;
  - fly about 9–12 jumps plus the evaluation;
  - fly the hardware baselines the abstract claims (DR and per-goal ILC).

## 7. F1TENTH (the paper's second task; mostly open)

- **`src/ilc_f1tenth/`:**
  - `dynamics.py`: a 9-state model (x, y, φ, vx, vy, ω, wheel speed, motor current, steer) with a combined-slip
    Fiala brush tire. It is identical to LLA-MPC's `nmpc_gen_fiala_fixed.py`.
  - `iLQR.py`: rewritten on 2026-10-04 to match that NMPC. RK4 with 10 substeps (the wheel mode is stiff,
    ≈ −200 s⁻¹), its least-squares cost with the "FAST" weights, input box constraints via a box QP, soft state
    bounds, and Levenberg–Marquardt. It also provides `solve()`, `rollout()` and `alpha_search()`.
  - `ILC.py`: an ILC loop on the model.
  - `f1tenth_ilc.py`: the ILC loop against the ROS F1TENTH simulator.
  - `trajectory_analysis.py`: plots every trial. Runs save to `~/ilc_ws/log/f1tenth/*.npz`; plot with
    `python trajectory_analysis.py`.
- **Known issues:**
  - `ILC.py`'s plant still integrates with one RK4 step per dt (unstable for this model); point it at
    `iLQR.rollout`.
  - The FAST weights barely track speed (`w_vx` = 0.01).
  - `trajectory_generator.py`'s vy reference has a suspicious sign.
  - The ROS bridge `f1tenth_gym_ros` is not set up (`~/LLA-MPC-online/f1tenth_gym_ros` is an empty submodule).
  - No drifting yet.
- **Design discussion with the user:**
  - trajectory optimization with their Fiala variant;
  - for LQR, linearize that same model along the trajectory (time-varying LQR from iLQR's gains), not a global
    linear tire model: the rear tire is saturated in a drift;
  - the GP-residual drifting paper (arXiv 2503.11083) is a candidate baseline.

## 8. Working with this user (from the memory files in `memory/`)

- **Resources:** container ≤ 28 CPUs; GPUs 0–2 only (GPU 3 free for others); host Python envs `dilc` (torch) and
  `ilcmjx` (JAX); long jobs via `systemd-run --user`.
- **Method constraints:** NO domain randomization in our method (allowed for baselines); no ensembles; a single
  network; no residual policies; no task-specific `critic_sens`.
- **Git:** commit only when asked. The user commits and pushes themselves; branch `quad`.
- **Deletions:** only when the user asks. Move bad runs aside (e.g. `*_invalid_shm`) instead.
- **Experiment queues:** maximal parallelism and automatic failure fixing, no confirmation needed. Answer status
  questions first, before launching new work.
- **Batches:** small and fast (≤ 10 min per iteration, ~36-run batches, no exhaustive grids).
- **Writing:** for paper text, edit in the user's style and change as little as needed.

## 9. Open threads, in priority order

1. Read the varied-start nose-down results (`vstart/RESULTS.txt`), and fold them into `RESULTS_SUMMARY.md` and
   `deploy/NOTES.md`.
2. Varied-start budget: evaluate the `it<k>` checkpoints of `vstart/vs1_*` on the perturbed set (as in
   `budget/budget_pert.sh`). Does 9–15 jumps suffice?
3. Decide the hardware method: varied starts (S = 1) plus whichever nose-down fix wins; gate on or off.
4. F1TENTH drifting: everything (§7).
5. Real Go1 runs (§6).
