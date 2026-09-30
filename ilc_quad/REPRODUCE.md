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
