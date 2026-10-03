# Reproducing the Go1 experiments on a new machine

This rebuilds the whole setup from scratch in a fresh Docker container and conda
environment, then reruns the experiments:
- the 8-task Go1 validation suite;
- the sim-to-real robustness grid;
- the Stage III and stance-damping comparisons.

Everything runs in simulation. No robot is needed.

## What you get from the repo

| Path | What it is |
|---|---|
| `docker/Dockerfile` | the container: ROS 2 Humble + MuJoCo 3.12.0 + CasADi 3.8.0 |
| `environment.yml` | host conda env `mj_quad` (no ROS), for the viewer and `check_model` |
| `experiments/go1/tasks.txt` | the 8 suite tasks: name, plan margin, jump, box, plan file, transfer source |
| `experiments/go1/plans/` | the 8 trajectory-optimization plans the tasks fly |
| `experiments/go1/env.sh` | shared settings: paths, parallelism, threads |
| `experiments/go1/validate.sh` | the suite, then `suite_report.py` |
| `experiments/go1/robust/grid.sh`, `job.sh` | tasks × reality conditions, then `robust_report.py` |
| `experiments/go1/robust/conds*.txt` | condition sets, one `name\|arguments` per line |
| `experiments/go1/robust/*_check.py`, `sens_time.py` | model-sensitivity diagnostics (see below) |

The plans are committed rather than re-solved, because IPOPT can land on slightly
different plans on another CPU or BLAS. If a plan file is missing, the node solves it
with its current defaults (the settings the committed plans were made with) and saves
it at that path.

## 1. Code and model

```bash
mkdir -p ~/ilc_ws/src ~/ilc_ws/log && cd ~/ilc_ws/src
git clone git@github.com:s-hliao/ILC2ILC.git ilc_quad

# MuJoCo Menagerie, pinned: the controller reads the robot's masses, geometry and
# limits from its Go1 model, and the plans were made against this version
cd ~ && git clone https://github.com/google-deepmind/mujoco_menagerie.git
git -C ~/mujoco_menagerie checkout da76818e269b82289eba39808e2fb91d679d6994
```

The rest of this guide assumes this layout:
- the workspace at `~/ilc_ws`, with the repo at `~/ilc_ws/src/ilc_quad`;
- the menagerie at `~/mujoco_menagerie`.

## 2. Container

```bash
cd ~/ilc_ws/src/ilc_quad
docker build -t ilc_quad:humble docker/

docker run -d --name ilc_quad \
  -v ~/ilc_ws:/ilc_ws \
  -v ~/mujoco_menagerie:/mujoco_menagerie:ro \
  ilc_quad:humble
```

To use the MuJoCo viewer from inside the container instead of the host, add these
flags and run `xhost +local:docker` first:

```bash
--net=host -e DISPLAY=$DISPLAY -e MUJOCO_GL=glx -v /tmp/.X11-unix:/tmp/.X11-unix:rw
```

The real robot also needs `--net=host`; the experiments don't.

Build the package inside the container:

```bash
docker exec -it ilc_quad bash
cd /ilc_ws && source /opt/ros/humble/setup.bash
colcon build --packages-select ilc_quad
```

> **Rebuild after every edit.** `colcon build` copies the scripts and the Python
> package into `install/`, and the experiments run that copy. An edit under `src/`
> does nothing until you rebuild. Conversely, rebuilding while a grid runs changes the
> code under every run the grid starts afterwards.

The container runs as root, so everything it writes under `~/ilc_ws` is root-owned
on the host. Reclaim it with `sudo chown -R $USER: ~/ilc_ws/log`, or read the files
through `docker exec`.

## 3. Host environment (optional: viewer, replays)

```bash
conda env create -f ~/ilc_ws/src/ilc_quad/environment.yml
conda activate mj_quad
export MUJOCO_MENAGERIE_PATH=~/mujoco_menagerie
```

Humble's `rclpy` is tied to the container's Python 3.10, so this Python 3.12 env
can't run the ROS nodes. It runs the tools that import the package from the source
tree:

```bash
cd ~/ilc_ws/src/ilc_quad
python -m ilc_quad.check_model
python scripts/replay_trials.py ~/ilc_ws/log/final2/b50_20_m90 --trials 1 last --speed 0.25 --loop
```

Replays show the recorded (measured) trunk pose. Under the grid's mocap-noise
conditions, that pose is what the controller saw, not what happened.

## 4. Settings: paths, parallelism, threads

Every script sources `experiments/go1/env.sh`. Override any setting in the
environment:

| Variable | Default | Meaning |
|---|---|---|
| `JOBS` | 3 | runs at once. Each run is one process on about one core. |
| `THREADS` | 2 | BLAS/OpenMP threads per run |
| `LOG` | `/ilc_ws/log` | where runs are written |
| `SUITE` | `$LOG/final2` | the nominal suite run the grid transfers from |
| `WS` | `/ilc_ws` | the colcon workspace |
| `EXTRA` | (empty) | extra `--param NAME:=VALUE ...` passed to every run |

3 × 2 was the limit on the original machine. On a bigger one, set `JOBS` to about
cores ÷ `THREADS`, for example `JOBS=16 THREADS=2` on 32 cores. Runs are
independent, and each is deterministic on a given machine.

Timing on the original machine: one 20-trial run takes about 1 minute of one core.
A task that converges early stops sooner (7 trials on most boxes).

## 5. Smoke test (about 1 minute)

Inside the container:

```bash
cd /ilc_ws/src/ilc_quad/experiments/go1
robust/job.sh /ilc_ws/log/smoke b50_10_m85 nominal ""
python3 robust/robust_report.py /ilc_ws/log/smoke
```

On the original machine this reads:

```
b50_10_m85__nominal  OK   n= 8 falls=[] | ... | last  -0.2 cm  +1.2 cm  -0.5° | ...
```

Another CPU can differ in the last digits. A different verdict means the setup
differs: check the menagerie commit and the package versions.

## 6. The experiments

All commands run inside the container from `/ilc_ws/src/ilc_quad/experiments/go1`.
Every run writes to `OUT/<run>/` with meta.json, reference.npz and one
`trial_NNN.npz` per trial, plus `OUT/<run>.json` (the per-trial summary) and
`OUT/<run>.log`.

### 6.1 Nominal suite: run it first

```bash
./validate.sh /ilc_ws/log/final2
```

This runs 8 tasks in three waves. The harder boxes transfer from easier runs in the
same folder. The robustness grid also transfers from this folder (`SUITE`), so it
has to exist before the grid runs.

Pass is the paper's Table I at trial 20: no falls, |ex| ≤ 1 cm, |ey| ≤ 2 cm and
|eθ| ≤ 2°, measured on the real landing state at the plan's end.

### 6.2 Robustness grid (sim-to-real)

```bash
T="f60_m85 b50_10_m85 b60_10_m90 b50_20_m90"
robust/grid.sh /ilc_ws/log/robust/g2s "$T"                   # Stage III safeguard (default)
EXTRA="--param stage3_safeguard:=false" \
  robust/grid.sh /ilc_ws/log/robust/g2p "$T"                 # the paper's Stage III law
```

`conds.txt` holds 21 conditions, so that's 84 runs per grid. The lockstep runner
applies each condition to the simulated robot only; the controller is never told.
`ilc_jump_lockstep.py --help` lists every option under "reality gap".

| Condition | Change |
|---|---|
| nominal | none |
| delay4, delay10 | 4 / 10 ms extra command-to-motor delay |
| mocap | 240 Hz frames, 6 ms latency, 0.5 mm / 0.003 rad noise; latency compensated (`pose_latency`) |
| mocapnolat | the same, latency **not** compensated |
| mocapbad | 120 Hz, 15 ms, 2 mm / 0.01 rad, compensated |
| enc | encoder noise 0.002 rad, 0.3 rad/s |
| heavy15, light10 | robot mass and inertia +15% / −10% |
| comfwd, comback | trunk CoM ±2 cm |
| weak85 | torque limits −15% |
| curve, curvesag | Go1 torque-speed envelope; plus 15% lower no-load speed |
| mu05 | ground μ 0.5 (the plans assume 0.6) |
| mu05plan | μ 0.5 with plans made for μ 0.45 (solved into `$LOG/plans_mu045/` on first use) |
| jfric | +0.3 N·m joint Coulomb friction |
| hardgnd | the paper's hard ground (2e4 N/m, 3e3 N·s/m) |
| footsens | raw Go1 foot sensors: per-foot bias up to 40, gain ±30%, noise 5 |
| real_s1, real_s2 | combinations of the above, 2 seeds |

Other condition sets:
- **`conds_baseline.txt`:** the first grid, run before the mocap and foot-sensor fixes.
  It has no latency compensation.
- **`conds_mocap.txt`:** the mocap conditions only.
- **`conds_kd.txt`:** stance-leg damping `contact_kd` 0 / 0.3, and 0 with the paper's
  Stage III law.

`robust_report.py` scores every run on the **true** state (`rec_true_*`, recorded
next to what the controller measured). Its columns, per task and condition:
- **verdict:** OK or FAIL, with the true falls (tilt past 0.6 rad after the jump
  window);
- **trial 1 and last-trial landing errors:** ex, ey [cm], eθ [deg];
- **landing quality** over all trials: the lowest nose-down pitch, the highest rear
  foot above the surface, and the drift once holding.

### 6.3 Diagnostics

Run them in the container:

```bash
python3 landing_check.py RUN_DIR [trial ...]     # touchdown pitch, dive, rear feet, creep
python3 robust/ratio_check.py RUN_DIR [...]      # model gain actual/predicted: theta, x, z
python3 robust/sens_check.py RUN_DIR plans/ref_s7_go1_f60_m85.npz 0.6 0.0
python3 robust/sens_time.py RUN_DIR plans/ref_s7_go1_f60_m85.npz 5      # f60 only
python3 robust/secant_check.py RUN_DIR plans/ref_s7_go1_f60_m85.npz     # f60 only
```

## 7. Where things stood (2026-09-29)

> **Defaults changed on 2026-09-30** (section 9), so the numbers below were made with the
> old defaults. To reproduce them, pass the old values:
> `EXTRA="--param ilc_gain:=1.0 --param ilc_mdc:=legacy --param level_feet_front_lever:=false --param bal_w_lever:=0.1"`.

These are the numbers a rerun should reproduce. They come from the code in this
commit.

**Before the fixes.** In the baseline grid (`conds_baseline.txt`), four sensing
problems broke the ILC:
- **Mocap frames at 120–240 Hz:** the ILC saw 20–40 cm landing errors that weren't
  there.
- **Mocap latency:** not accounted for.
- **Raw foot sensors:** ended the flight mid-air.
- **Friction below μ ≈ 0.55:** slips at push-off.

The first three are fixed: fresh-frame timing with `pose_latency`, fitted
velocities, a ballistic flight fit, and an in-flight foot-sensor tare. See the
README's lockstep section.

**Grid `g2s` (safeguard, all fixes).** Each cell shows the verdict and the last
trial's ex / ey [cm] / eθ [deg]. `F<n>` means n trials fell; `x` means it failed
without falls.

| condition | f60_m85 | b50_10_m85 | b60_10_m90 | b50_20_m90 |
|---|---|---|---|---|
| nominal | x −1.0/−0.1/+1.9 | ok −0.1/+1.4/+0.2 | ok +0.3/−0.4/−0.1 | ok +0.2/−0.2/+0.4 |
| delay4 | x −1.0/+0.1/+2.1 | ok +0.1/+1.4/−0.3 | ok +0.9/−0.6/−0.0 | ok +0.2/−0.2/−0.0 |
| delay10 | x −1.1/+0.7/+2.6 | ok +0.4/+1.6/+0.4 | ok +0.4/−0.3/+0.6 | ok +0.2/+0.1/−0.4 |
| mocap | x −2.1/+0.2/+2.3 | ok +0.3/+1.1/−0.1 | ok +0.7/−0.6/−0.1 | ok +0.4/−0.3/+0.4 |
| mocapnolat | x −1.2/−0.4/+1.2 | x +2.0/+0.1/−0.4 | x +1.8/−1.3/−1.0 | x +1.2/−0.7/−0.8 |
| mocapbad | x −1.3/+0.1/+2.9 | ok +0.7/+1.4/+1.8 | ok +0.6/−0.3/+1.2 | ok +0.5/−0.2/+1.4 |
| enc | x −1.0/−0.1/+1.9 | ok −0.1/+1.4/+0.3 | ok +0.3/−0.3/+0.1 | ok +0.1/−0.2/+0.4 |
| footsens | x −1.0/−0.1/+1.8 | ok −0.1/+1.4/+0.2 | ok +0.3/−0.4/−0.1 | ok +0.2/−0.2/+0.4 |
| hardgnd | x −0.4/+0.6/+2.6 | ok −0.3/+1.8/−0.4 | ok +0.3/+1.3/+0.1 | ok +0.2/+0.7/−1.1 |
| heavy15 | x −1.5/−1.2/−0.7 | x −1.3/−1.2/−3.4 | x −1.8/−1.8/−3.4 | F10 −6.1/−0.7/−1.4 |
| light10 | x +0.0/+0.1/+4.2 | x −0.4/+3.8/+6.4 | x +2.9/+0.7/+7.2 | F1 +0.1/−0.7/+2.0 |
| comfwd | ok −0.9/−0.9/−1.2 | ok +0.7/+0.8/−0.6 | ok +0.3/−0.6/−0.4 | F4 +0.1/−0.2/−0.9 |
| comback | x −0.8/+0.3/+4.2 | ok −1.0/+1.8/+0.9 | ok −0.7/+0.1/+0.5 | x +0.2/−0.1/+3.4 |
| weak85 | x −1.0/−0.5/+1.9 | ok −0.5/−0.1/−0.6 | x −1.3/−1.0/−1.3 | F20 −0.6/−1.2/+4.3 |
| curve | x −1.1/−0.4/+1.6 | ok −0.2/+0.5/+0.5 | ok −0.8/−1.8/−1.9 | F1 −0.2/−0.8/+0.4 |
| curvesag | x −1.3/−1.1/+0.6 | ok −0.2/−1.0/−0.2 | x −2.6/−1.4/−1.6 | F20 −0.4/−0.1/+0.4 |
| jfric | x −1.2/−1.1/−1.5 | ok +0.1/+0.6/+0.3 | ok +0.0/−0.9/−0.8 | F3 +0.0/−0.6/−0.3 |
| mu05 | x −36.2/+1.2/−1.1 | x −24.6/−6.9/+9.9 | x −28.4/−6.9/+7.0 | F1 −17.3/−19.8/−0.5 |
| mu05plan | x −3.8/+2.2/+8.2 | x −1.4/−1.3/+1.8 | F1 −5.5/−0.9/+3.8 | F1 −10.6/−0.9/+31.2 |
| real_s1 | x −1.3/−1.0/−1.2 | x +0.1/−1.2/−3.3 | x −2.1/−1.7/−2.8 | F19 +0.0/−1.0/+0.3 |
| real_s2 | x −0.4/−0.2/+2.3 | ok +0.6/+0.5/+0.9 | x +0.9/−1.2/−2.0 | F3 +1.2/−0.8/+1.6 |

The README's "Go1 validation" table predates the log-pipeline changes. With this code,
f60 nominal misses narrowly (−1.0 cm, +1.9°). The other three nominal tasks pass.

**Why f60 and the mass and CoM cases stall.** The cause is Stage III pitch:
- **The model is wrong about pitch.** `ratio_check.py` gives an actual/predicted
  landing-pitch gain of −0.2 to +0.05, against 0.25–1.0 for x. The SRB model's
  pitch inertia is exact (0.394 kg·m², the same as MuJoCo's). So the commanded 1–3 N
  force steps simply don't reach the ground as commanded.
- **The prime suspect is stance-leg joint damping** (`contact_kd` = 1). With the feet
  planted, trunk pitch turns the thigh joints, which damps pitch rate with a time
  constant of about 0.1 s.
- **The safeguard then freezes Stage III.** Rejected steps push its step weight to
  10⁵–10⁸.

**Damping test (`conds_kd.txt`, f60).**
- `contact_kd` 0 with the paper's law: **passes** at +0.2 / −1.0 / +0.4°.
- `contact_kd` 0 with the safeguard: −1.4 cm, fails.
- `contact_kd` 0.3 with the safeguard: −1.1 cm, fails.

**Next to run** is the whole grid and suite with that combination:

```bash
EXTRA="--param contact_kd:=0.0 --param stage3_safeguard:=false" ./validate.sh /ilc_ws/log/final_kd0
EXTRA="--param contact_kd:=0.0 --param stage3_safeguard:=false" \
  SUITE=/ilc_ws/log/final_kd0 robust/grid.sh /ilc_ws/log/robust/g4 "$T"
```

These exist as options, off by default:
- **`stage3_secant`:** Broyden correction of the landing sensitivities. Offline it
  gave mixed predictions, so it's untested in flight.
- **`mu`:** the friction the plan assumes, for plans made for a measured floor.

## 8. Carrying over old results (optional)

The repo holds no run data; the original `log/` is several GB. To take runs along,
for example the nominal suite as a transfer source, copy them into the new `LOG`:

```bash
rsync -a old-machine:~/docker_workspaces/ilc_ws/log/final2 ~/ilc_ws/log/
```

## Troubleshooting

- **`rclpy` import errors after installing something:** numpy was upgraded past
  1.21.5. Reinstall it: `pip3 install numpy==1.21.5`.
- **Rendering errors with `--video`:** set `MUJOCO_GL=egl` (GPU) or `osmesa`
  (CPU only).
- **A run's summary JSON is missing:** read its `.log`. `ILC update failed` there
  stops the run at that trial.
- **A reference "made for another task or TO setting":** the plan file doesn't match
  the task's settings, and the node re-solves and overwrites it. Check the margin and
  task columns in `tasks.txt`.

## 9. Fall-recovery sweep and new defaults (2026-09-30)

Tools: `experiments/go1/sweep/` (`sweep.sh OUT JOBSFILE` runs `run|task|scratch-or-transfer|args`
lines on every core; `score.py`, `stick.py` and `compare.py` score and compare them).

**Findings.**
- **The ILC does learn out of falls.** With learning frozen (`qu`, `qu_stage3` 1e3), every
  hard cell fell on all 30 trials. With learning on, most stopped falling after 3–12 trials.
- **Recovery was slow.** The robot realizes only part of each model step, so `ilc_gain` 2
  halved the fall streaks.
- **The ILC asked speed-limited knees for torque they couldn't make.** Its legacy A1 motor
  constraint never binds.
- **Most falls were landing failures after the plan's end.** On the b50_20 plans a slightly
  early touchdown put the front feet too far back to brake the forward momentum.

**New defaults:**
- `ilc_gain` 2.0;
- `ilc_mdc` go1 (the Go1 torque-speed envelope in the ILC's QP);
- `level_feet_front_lever` true (in the last `level_feet_samples` of flight, the front feet
  are never aimed behind the plan's landing lever);
- `bal_w_lever` 0.3.

Over the 8 tasks × 21 conditions from scratch:

| | old defaults | new defaults |
|---|---|---|
| falls | 253 | 98 |
| falls in the last 5 trials | 39 (16 runs) | 16 (5 runs) |
| Table I passes | 93 | 99 |
| runs with a foot unloaded > 300 ms | 15 | 5 |

What got worse is low friction (mu05plan). `level_feet_hold_x` (both pairs held at the
landing offsets) fixed the box plans but made the 60 cm jumps fall, so it stays off.

**Flight time.** Plans with 200–400 ms of flight (`phases:=[30,30,Nfl]`, Nfl 20–40; plans
`plans/ref_x_go1_<task>_fl<Nfl>.npz`, tasks in `sweep/tasks_extra.txt`) were compared with
the current 300 ms:
- **400 ms:** only b50_20_m100, f40 and f60 plan exactly.
- **350 ms:** b50_20_m90 does not plan exactly.
- **200 ms:** four tasks don't either.
- **On the cells that plan, under the new defaults and 7 conditions:**
  - 200 ms: 115 vs 48 falls;
  - 250 ms: 72 vs 69;
  - 350 ms: 173 vs 50;
  - 400 ms: 145 vs 34.
- **300 ms stays.** 250 ms had fewer late falls (2 vs 11) but fewer Table I passes (19 vs
  27), and it is worse on the b50_20_m90 gaps.

## 10. Rear feet, capture-point plans and the async sim-to-real test (2026-09-30)

**Rear feet lifting at landing.** The rear feet lifted for 100–200 ms after most box
touchdowns: the trunk pitched about the front feet. Whether they lift is decided by the
landing's capture-point margin, the front foot x minus (CoM x + vx √(h/g)) when all four feet
are down (`sweep/capture.py`).
- **The old plans sat right at the limit.** They land every leg in the home pose, so the
  front feet land ~0.2 m ahead at ~1.35 m/s.
- **The margin predicts the lifting.** Over 2600 landings the correlation is −0.78: below
  zero, 130–180 ms unloaded; above +4 cm, almost none.
- **Landing-controller gains only traded rear contact for falls.** That includes stiffer or
  pushier legs, rear-only push, braking, and a pitch-moment floor.

**The fix is in the plan.** `landing_capture_margin` (node and `init_trajopt`, default 0.06 m)
frees the front legs at the landing sample and requires the front feet ahead of the capture
point by that margin. The suite now flies these plans (`plans/ref_s8_go1_<task>.npz`, in
`tasks.txt`). Pass `--param landing_capture_margin:=-1.0` with the old `s7`/`s4` plans.
On the 8 × 21 grid, last 5 trials:

| | before | capture plans |
|---|---|---|
| rear feet never unloaded | 25% | 88% |
| trials with rear unloaded > 100 ms | 52% | 4% |

**Asynchrony is standard in the sim-to-real test (`robust/conds_async.txt`).** New
`ilc_jump_lockstep.py` options:
- `--act-jitter`: command latency jitter, kept in order;
- `--cmd-drop`: dropped commands, the last one held;
- `--state-delay` / `--state-jitter`: the age of the joint readings.

Every condition runs on the same async base:
- 2 ms + U(0, 3 ms) command latency, 2% of commands dropped;
- joint readings 0–4 ms old, encoder noise;
- 240 Hz mocap with 6 ms latency (compensated) and noise;
- raw foot sensors.

On that base the grid varies mass (+15%, −10%), CoM ±2 cm, weak or speed-limited motors,
μ 0.5, joint friction, hard and soft ground, a 2 kg payload, 10 ms delay, bad mocap, two
combined "real" cases and 3 seeds. Run it with:

    grep -v '^#' robust/conds_async.txt | while IFS='|' read c a; do for t in $(awk '{print $1}' tasks.txt); do
      echo "${t}__D__$c|$t|scratch|$a"; done; done > /tmp/async.jobs
    sweep/sweep.sh $LOG/async /tmp/async.jobs
    python3 sweep/report.py $LOG/async D robust/conds_async.txt

**Stage III back-off (new default `stage3_backoff` alpha).** Under asynchrony the safeguard's
best trial is a lucky one, so every retry was rejected and its step weight grew to 1e3–1e9:
Stage III froze in 123 of 152 runs. Alpha instead halves the step on each rejection.

| async grid, 152 runs | old back-off | alpha |
|---|---|---|
| Table I passes | 53 | 68 |
| falls | 121 | 115 |
| median unloaded time | 6 ms | 8 ms |

**Tried under async and dropped:**
- best-trial averaging (`stage3_best_refresh` with tolerance);
- a step-weight cap;
- x/z-only acceptance;
- `stage3_theta_mode:=hold`, `stage3_secant`, a Stage III gain of 1, a longer Stage II;
- the paper's law (15/32 but late falls);
- flight attitude feedback: both signs are worse by 7–9° of pitch, because the ILC's
  ballistic tail then ignores the feedback's own correction;
- `level_feet_samples` 5 or 0, and `level_feet_front_lever` off.

**Still open on the async grid:**
- **Pitch bias.** f60 settles 2–5° nose-up and f40 about 2° nose-down. The ILC's
  extrapolated landing pitch correlates only 0.1–0.6 with the true one.
- **Capacity limits.** Heavy15 leaves the 90% boxes 5–6 cm short; b50_15 falls late under
  weak85, soft ground and the payload.
- **Remaining rear unloading.** b50_20_m90 weak85 (~1 s), b60 mocapbad and b55 delay10.

## 11. Measured landing state (2026-09-30)

**The problem.** The ILC's landing sample (N) was a free-flight extrapolation from 10 samples
before N. Under asynchrony its pitch correlated only 0.1–0.6 with the pitch the robot landed
with, which is what Table I scores.

**New defaults.**
- `ilc_landing_state` all: the ILC's landing x, z and pitch come from a line through the
  mocap frames within `ilc_landing_window` (15 ms) of N.
- `ilc_landing_fell_tail` true: a trial that fell keeps the extrapolation. Its early, low
  touchdown made the measured landing read short and low, and the ILC pushed harder every
  trial.

Both runs below use the stage3_backoff alpha default. A run "ends falling" if it falls in its
last 5 trials.

| | extrapolated | measured x/z/pitch + fallback |
|---|---|---|
| async grid (152 runs), Table I passes | 68 | 78 (+20 / −10) |
| async grid, median eθ | 1.4° | 1.0° |
| async grid, falls | 115 | 120 |
| async grid, runs ending falling | 4 | 8 |
| suite (`validate.sh`), converged to Table I on the last trial | 6 of 8 | 7 of 8 |

**Where it costs.** The 8 runs that end falling are all b50_15 and b50_20_m90 under
capacity-limited conditions: weak85, curvesag, payload, soft ground, μ 0.5 and real_s1.

**Suite details.**
- It fails `validate.sh`'s no-falls rule on b55 (falls on trials 2–3), b50_20_m100 (trial 2)
  and b60 (trial 3).
- b60 ends 1.6 cm short.
- The capture plans' landings dip 5–12° nose-down after touchdown, with the rear feet down.

**Tried and dropped.**
- **Measuring pitch only, or x and pitch only:** the measured z is what earns the passes.
- **No rollback at all (`stage3_safeguard` false):** 19/32 on the subset, but more runs end
  falling, and in the suite b60 tips over on trial 20 and several tasks never converge.

## 12. Deep-ILC: a goal-conditioned policy from a few ILC'd goals (2026-10-01)

`scripts/dilc_train.py` (host: torch, 4 GPUs) and `scripts/dilc_execute.py` (container)
extract one state-feedback, goal-conditioned force policy from ILC runs on a few goals. It
then jumps zero-shot to goals between them, without re-running the ILC. Everything runs on
the asynchronous reality model. Training sees only the nominal robot, with no domain
randomization; the hard robots are deployment targets only.

**Pipeline.**
1. **ILC on the nominal robot:** 5 goals (f40 to f60) × 2 seeds, 91 jumps (`--stages ilc`).
2. **Dataset:** every trial becomes an episode twice, for the goal it aimed at and in
   hindsight for where it landed. Each episode carries labels from its own linearization (its
   A_t, B_t, as `JumpILC._step` builds them) and the ILC's Stage III Q's:
   - the Stage III step S;
   - the Riccati gains K_t;
   - the outcome metric G_N;
   - closed-loop co-states.
3. **Offline training,** many members batched per GPU:
   - Sobolev actor terms (value through G_N, dπ/de = −K_t, dπ/dg), held over a
     neighbourhood of each trial by its own gain;
   - a critic fitted to the trials' closed-loop co-states;
   - a SAC step on that critic, bounded by the ILC QP's curvature R + BᵀPB.
4. **Adapting to a new robot** (`--stages explore`, then fine-tune). The critics predict which
   goals will land worst. Short Stage III ILC runs explore those goals on the new robot,
   warm-started from the policy's jumps. Their trials become labelled episodes, and the critic's
   co-states use B corrected by the secant measured on the new robot.

Driver scripts are in `log/dilc/`: `explore_cond.sh`, `ctl.sh` and `cond_report.py`.

```bash
P=~/miniconda3/envs/dilc/bin/python; T=src/ilc_quad/scripts/dilc_train.py
$P $T log/dilc/cov5 --stages bank,ilc,dataset --goals f40_m85,f45_m85,f50_m85,f55_m85,f60_m85 \
   --ilc-conds nominal --ilc-seeds 1 2
$P $T log/dilc/off10 --stages train,eval --online-episodes 0 --pretrain-updates 15000 \
   --rl-start 0.6667 --pretrain-snapshots 0.6667 --seeds-per-group 4 --variants "$VARIANTS"
ACQ=critic TAG=_lab SECANT_RUNS=heavy15 log/dilc/explore_cond.sh dilc_cl_h3 heavy15 1 log/dilc/off10
```

**Bugs found along the way (all fixed):**
- **Reward:** a plan-tracking floor outweighed the landing cost 5–35× on converged trials.
- **One-step value-gradient bootstrap (paper eq. 14), offline:** it drifts from the trials' A, B
  chain until early-stance action gradients point against the ILC's (cosine −0.7). The trials'
  co-states replace it.
- **Open-loop co-states paired with closed-loop curvature:** every sample's step corrected the
  whole landing error. Closed-loop co-states, iLQR's V_x, fixed it.
- **Hidden behaviour-cloning term:** members with `rl=0` silently cloned the best trials.
- **Concurrent evaluations deadlocked:** ROS domain IDs above 100 collide with ephemeral ports.
  Domains are now below 100, with per-jump timeouts.

**Offline results.** The held-out goals are 0.425, 0.475, 0.525 and 0.575, under 8 async hard
conditions. Lower is better.

| controller | score |
|---|---|
| Deep-ILC (closed-loop co-states) | 4.55 |
| Sobolev network | 4.65 |
| ILC + LQR | 4.2–4.4 |
| ILC forces replayed | 11.5–12.3 |

Offline, RL neither helps nor hurts: the critic has no information beyond the labels.

**Adapting to a hard robot, zero-shot at the held-out goals.**
- **Setup:** 2 rounds × 8 goals × ≤4 ILC trials = 80 jumps on that robot.
- **Fresh robots:** seed base 301; no seed used for any choice.
- **Statistics:** 16 seeds × 8 jumps per row.
- **ILC rows:** the same robots and goals, 20 trials each.

| robot | not fine-tuned | ILC from nearest ILC'd goal, best of 20 (converged) | ILC from scratch, best of 20 | Sobolev + its own exploration | Deep-ILC (value-guided) |
|---|---|---|---|---|---|
| heavy15 | 5.25 | 5.80 (0/8) | 3.19 | 1.57 | **0.93** (Table I 22 %) |
| payload2 | 6.29 | 3.92 (0/8) | 1.94 | 2.34 | **1.26–1.53** |
| real_s1 | 3.56 | 6.48 (0/8) | 1.86 | 2.06 | 2.17 (median 1.60 vs 1.50) |

- **Deep-ILC beats:**
  - its own not-fine-tuned policy, 2–6× on all three robots;
  - the ILC re-converging from the nearest ILC'd goal: that ILC never matches it within 20
    trials in 74–98 % of jumps, and never converges itself.
- **Against Sobolev:** better on the mass perturbations (heavy15, payload2), and level on
  real_s1, which mixes motor curve, joint friction, CoM offset and +5 % mass.
- **Where the edge comes from:** exploration, not the learner. Given the same exploration
  data, the Sobolev learner equals or beats the Deep-ILC learner on all three robots (`ctl_*`).
  - Deep-ILC's critic predicts where the policy will land worst. After round 1 it sends the
    ILC to the untested low goals, while Sobolev's ensemble keeps exploring only the upper half.
  - The secant-corrected critic (pitch error 9.6° → 1.7° on heavy15) and the trust region
    anchored on the labels narrowed the learner gap on real_s1 (1.85 → 1.70 on equal data),
    but did not close it.

**Closing the real_s1 gap: a critic of the deployed robot (supersedes the attribution above).**
The critic was fitted to every transition in the replay. About 290 of the roughly 400
episodes came from the nominal robot. At a jump's start the state is the same on every
robot, so the critic learned a mixture of robots. On real_s1 it ranked the goals backwards:
it predicted the highest cost at 0.40, where real_s1 actually lands best. That inverted
ranking sent value-guided exploration to the near goals. It also pulled the SAC step toward
the nominal robot. Two changes, both in the paper's spirit of fine-tuning in the target
environment:
- **`crit_t=1`:** the critic's TD and co-state losses use only the new robot's
  transitions (`--extra-data`).
- **`--target-frac 0.5`:** half of every batch, and half of the trials for the value term,
  come from the new robot.

The Sobolev actor terms still see all the data. On fixed data (Deep-ILC's own real_s1
exploration, rounds 1–2), this moves Deep-ILC from 2.11 to 1.90, while Sobolev stays at
2.12 → 2.09 (`ctl_tf0_real_s1`, `ctl_tf5_real_s1`). The critic's start-of-jump prediction is
still nearly flat over the goals, so the acquisition is now the members' disagreement,
exactly as Sobolev's.

To rerun: `log/dilc/dct_chain.sh COND`, which runs `explore_cond2.sh` for 3 rounds × 8 goals
× ≤4 trials, ≤96 jumps on the robot. Sobolev gets the same budget through `explore_cond.sh`.
`explore_cond2.sh` adds `TF`, `EXPLORE_VAR` and `EX_SECANT`.

Fresh robots (seed base 301), 16 seeds × 8 jumps per row. Mean, with the median in brackets.

| robot | not fine-tuned | ILC from nearest, best of 20 | ILC from scratch, best of 20 | Sobolev r3 | old Deep-ILC r2 | **Deep-ILC r3** |
|---|---|---|---|---|---|---|
| real_s1 | 3.56 | 6.48 | 1.86 | 1.88 (1.17) | 2.52 | **1.54 (0.82)** |
| heavy15 | 5.23 | 5.80 | 3.19 | 1.53 (1.16) | 0.95–1.07 | 1.20 (0.89) |
| payload2 | 6.24 | 3.92 | 1.94 | 2.21 (1.75) | 1.26 | **0.87 (0.56)** |

- **Deep-ILC r3 against Sobolev r3,** over 16 member seeds per arm, as a 95 % bootstrap
  interval for (Sobolev − Deep-ILC): real_s1 [0.19, 0.50], heavy15 [0.16, 0.49], payload2
  [1.19, 1.50]. Deep-ILC is better on all three robots.
- **It also beats:**
  - the not-fine-tuned policy, 2.3–7×;
  - ILC from the nearest ILC'd goal, which does not match it within 20 trials in 77–100 % of
    jumps;
  - the ILC's best of 20 trials from scratch, on every robot.
- **heavy15:** the earlier pipeline (critic acquisition, mixed-robot critic) remains better,
  at 0.95 against 1.20.
- **Exploration ILC with a pooled secant prior** (`stage3_secant_prior`, `--secant-prior`,
  `EX_SECANT=1`): Stage III plans with G_N + C. C is a ridge fit over this robot's earlier
  exploration trial pairs. On real_s1 the best score by trial 4 improves from 1.36 to 1.16
  (`log/dilc/sectest`). A prior fitted on nominal data does not help early (1.52). It was
  not used in the table above.
- **What is left on real_s1** is a landing pitch bias of about −2 to −7°, nose-down, growing
  with distance. The ILC itself removes little of it within 8 trials, even with the secant
  prior.

## 13. A critic that carries the problem's structure: learned landing sensitivity (2026-10-01)

> **Superseded (2026-10-03).** This structured critic was trained on measured, task-specific landing
> sensitivities, so it doesn't carry across robots and tasks. It was dropped, and its scripts
> (`critic_sens.py`, `grad_ilc.py`, `critic_*`) were removed. See section 14 for the general pipeline.

Section 12's critics did not encode the jump's local structure. We tested that directly.
`scripts/value_bench.py` flies the policy's own jumps again with every action pushed along a
direction d, at ±ε with common random numbers. The resulting dJ/dε is what −Σ_k ∇_aQ·d_k
should predict. Repeat measurements agree at r = 0.99.

On 204 fresh nominal points, the gradient models score r = 0.20 (SRB closed-loop co-states)
and r = 0.16 (the Deep-ILC critic). Pooling the measured gradients of other seeds at nearby
goals already reaches r = 0.72. The structure is there to learn; the models get it wrong.

**Fitting a plain critic's action gradients to the measurements (`critic_truegrad.py`).**
- **Nominal test:** r = 0.64 on the reserved test set, against 0.23 for the SRB.
- **Transfer:** it fails on the real-like robots (r = 0.11 pooled, −0.28 on real_r5).
- **Why:** Q(s, a) has to predict the landing error from the state alone. On a mismatched robot
  it predicts the nominal one.

**The structured critic (`critic_sens.py`).** The landing cost is quadratic in the landing
error, so the critic is

    Q(s, a) = −c μ(s, a)ᵀ Qe μ(s, a),     dJ/da_k = 2c (Qe μ)ᵀ ∂μ/∂a_k.

- μ is the landing error the jump will end with.
- The sim teaches ∂μ/∂a, the landing's sensitivity to the forces: the local structure.
- On a robot, μ is replaced by the landing error the robot actually measured. That is the part
  of the actual dynamics hardware supplies.

`value_bench.py` now also stores the landing error vectors (e0, E±). `∂μ/∂a` is trained on
(E₊ − E₋)/2ε, six numbers per flown pair.

State coverage comes from the nominal sim only, with no domain randomization: `--offset S` flies
the policy plus a smooth random action offset, so landing errors reach the size a mismatched
robot produces. The training sets are biga, bigd, offa, offb and offc, with weights selected on
offv. Generate them with `log/dilc/vbench/gen_e.sh`.

r of predicted against measured dJ/dε, random directions only:

| set | structured critic (measured e) | SRB | sensitivity r (x / z / pitch) |
|---|---|---|---|
| nominal reserved test (tst) | **0.94** | 0.23 | 0.98 / 0.94 / 0.89 |
| real_r1 | **0.83** | 0.40 | 0.96 / 0.94 / 0.77 |
| real_s1 | **0.89** | 0.22 | 0.95 / 0.81 / 0.89 |
| real_r4 | 0.35 | 0.40 | 0.71 / 0.54 / 0.23 |
| real_r5 | **0.44** | 0.28 | 0.91 / 0.60 / 0.64 |

real_r4's measurements are reliable (half-differences agree at r = 0.98–1.00), so its
sensitivity genuinely differs from nominal. Likely causes are its 14 % weak motors and 15 ms
pose delay. Sim structure cannot know either.

**ILC with the learned sensitivity (`grad_ilc.py`).**
- **Setup:** 3 goals, 6 trials, a fresh seed every trial, so only the robot's systematic error
  can be learned.
- **Gauss–Newton step on the measured landing error:** ff ← ff − β Sᵀ(SSᵀ + δI)⁻¹ e, with S the
  critic's ∂μ/∂a over x, z and pitch.
- **Safeguards:** backtracking (`--backtrack 1.5`) and a damped Broyden secant correction of S
  from the robot's own trial pairs (`--broyden 0.5`). Without them, goal 0.60 on real_r5
  diverged.

Mean landing score over the last 2 trials and 3 goals (`log/dilc/vbench/gi`, `gi_sum.py`):

| robot | none | SRB gradient | JumpILC | learned-S Gauss–Newton |
|---|---|---|---|---|
| real_r1 | 3.73 | 2.44 | 2.77 | **0.16–0.23** |
| real_s1 | 1.39 | 0.96 | 1.50 | **0.25–0.29** |
| real_r4 | 6.34 | 4.37 | 2.45 | **1.45–1.57** |
| real_r5 | 5.09 | 3.51 | 11.5 | **1.39** (safeguarded) |

**Converge then correct, ≤ 30 real jumps, one network** (`log/dilc/ctc_sens.sh`;
the baseline is `ctc.sh`, now single-network).
- **Round 1:** ILC at 0.40 / 0.50 / 0.60 × 5 from the sim policy.
- **Policy update:** `ilc_policy_update.py` moves the policy toward the best trials.
- **Rounds 2 and 3:** at 0.45 / 0.55 × 4 and × 3, the `cov` variant. The default re-runs
  0.40 / 0.50 / 0.60.
- **Selection:** the variant was chosen on the validation midpoints (seed 301).
- **Final numbers:** from the reserved set, goals 0.4375 / 0.4875 / 0.5125 / 0.5625 × seeds from
  701, 16 jumps per cell. Mean ± s.e. (median); the last column is the 95 % CI of
  structured − JumpILC, paired by jump:

| robot | sim policy | JumpILC + update | structured critic + update | difference |
|---|---|---|---|---|
| real_r1 | 2.50 (2.23) | 0.65 (0.48) | **0.15 ± 0.03 (0.09)** | [−0.77, −0.23] |
| real_s1 | 3.29 (2.99) | 2.97 (1.26) | **0.44 ± 0.18 (0.07)** | [−3.89, −1.36] |
| real_r4 | 8.77 (8.72) | 2.66 (2.46) | 2.02 ± 0.48 (1.42) | [−1.60, +0.34] |
| real_r5 | 8.05 (8.05) | 6.31 (5.79) | **1.10 ± 0.36 (0.28)** | [−6.36, −4.06] |
| mean | 5.65 | 3.15 | **0.93** | |

No falls in any cell. On validation the default goal schedule also beat the baseline, mean 1.23
against 2.78, so the gain is not only the coverage schedule.

**What did not work, and caveats.**
- **Stepping the network in nominal sim along any gradient** did nothing measurable
  (`critic_policy_step.py`, `vbench/ps`). In nominal sim the remaining error is per-seed noise.
- **real_r4:** the learned sensitivity transfers worst there, and on the validation goals the
  JumpILC pipeline beat it, 0.26 against 1.72.
- **One pipeline run per robot:** the update's own seed-to-seed spread is not measured.

```bash
cd log/dilc/vbench && ./gen_e.sh        # gradient sets with landing errors (~11k sim jumps)
P=~/miniconda3/envs/dilc/bin/python; S=src/ilc_quad/scripts
$P $S/critic_sens.py cs/s2.pt --data biga bigd offa offb offc --val offv --steps 6000   # also writes cs/s2_np.npz
$P $S/critic_sens.py x.pt --data offb --load cs/s2.pt --eval tst x_real_r1 x_real_s1 x_real_r4 x_real_r5
cd .. && TAG=cov G2="0.45 0.55" N2=4 G3="0.45 0.55" N3=3 ./ctc_sens.sh real_s1
./final_test.sh                          # the reserved test, final701/
```

## 14. Deep ILC: structure in simulation, specifics on hardware (2026-10-03)

This is the general pipeline: no task-specific critic and no domain randomization. Working notes, every
sweep and the numbers are in `log/dilc/deploy/NOTES.md`. The GPU simulator is `src/ilc_mjx` (see its
README), and the tools for the gradient audit, descent test and hardware stage are in `src/ilc_mjx/scripts`.

**Why the VG-SAC critic was replaced.** Audited against exact closed-loop gradients on the
deterministic GPU sim (`audit_truth.py`, `audit_critic.py`):
- **Wrong target.** VG with a′ held (the paper's released code) targets the *open-loop* gradient. That
  gradient is uncorrelated with the truth for a state-feedback policy (r = 0.00).
- **Right ingredients.** The simulator's finite-difference Jacobians, chained with the policy's own
  feedback (the ILC's closed-loop co-states), reach r ≈ 0.6.
- **Unstable fix.** Teaching the critic those co-states diverges: the targets depend on the actor's
  gains, and the actor follows the critic.

**What replaced it: the ILC's own update, lifted into the network.** Each trial takes the ILC step,
computed from its closed-loop co-states, and the network regresses onto the stepped actions:
- In Stage III the step is Gauss–Newton on the landing error through the closed-loop landing
  sensitivity; the earlier stages use the normalized gradient step (Nguyen et al.'s stages as a
  curriculum).
- Each batch of rollouts is one ILC iteration; the network amortizes the steps across goals.
- Trainer: `dilc_train.py`, variant `co_fd=1, co_ilc=1, co_gn=1, rl=0`.

**Sim stage (GPU, nominal, no domain randomization).** `log/dilc/vgpre.sh coilc`, environment:
```
GPUR=<gpu> GB=128 COREC=128 UTD=0.25 EPS=1200 TE=-13 FALL=2 FRESH=1 ALR=1e-4 \
SCHED="0.2 0.5" FLOOR=0.1 GN=1 EXPL="0.08 0.15" TAG=ex1 SEED=0 ./vgpre.sh coilc <gpu>
```
`EXPL` is exploration, not randomization: each jump starts from its own stance (planar joint offsets
~ N(0, 0.08) rad) and gets smooth action offsets of rms 0.15 during stance. Without it the policy
never sees states off its nominal trajectory: it transfers poorly and has no robustness (perturbed
~40 vs ~10).

**Hardware stage.** Few-shot ILC on the robot, here the async real-like robots in the container:
```
cd src/ilc_mjx && ~/miniconda3/envs/ilcmjx/bin/python scripts/deploy.py --policy <run>/policy \
  --member g0_coilc.s0_e1200 --robots real_r1 real_s1 real_r4 real_r5 --out <dir> \
  --update gn --gn-beta 0.5 --step-rms 0.08 --target-base own --bold 0.3 [--eval final --perturbed]
```
- **Budget:** 10 iterations × 3 goals = 30 jumps per robot.
- **Each trial:** a Gauss–Newton step on its measured landing error, through the nominal GPU sim's
  closed-loop landing sensitivity, with the policy's feedback taken at the real states.
- **Safeguard:** a step-size safeguard with a noise margin (`--bold`).
- **Regression:** the network regresses onto every trial's targets, with weights halving with age,
  anchored to its own previous behaviour.
- **Noise:** single-trial landings are noisy (std 0.5–3 at a fixed goal). Never size steps or roll
  back on one trial's cost, and evaluate with replicates.

Reserved test (mean of `real_r1`, `real_s1`, `real_r4`, `real_r5`; 0.01 × landing cost + 20 × fall):

| | unperturbed | perturbed (8 perturbations) |
|---|---|---|
| JumpILC (3 repeats) | 3.16 | 28.6 |
| VG-SAC-FD zero-shot | 3.57 | 10.7 |
| Deep ILC, sim without exploration → hardware ILC | 1.62 | 41.8 |
| Deep ILC, sim with exploration → hardware ILC (2 sim seeds, no stall guard) | 2.24 | 10.4 |
| the same with the stall guard (3 sim seeds) | 2.67 (seeds 2.09 / 2.49 / 3.43) | 10.8 |

Add `--stall 3` to the hardware stage when a goal may be out of the robot's reach.
