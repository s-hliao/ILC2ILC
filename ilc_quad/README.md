# ilc_quad

MuJoCo simulation of the Menagerie Unitree **Go2** and **Go1**, set up for
iterative learning control of jumping, plus the ILC jumping controller itself: a
full-body trajectory optimization for the reference and the paper's 3-stage
force-based ILC, flown against the simulator or a real Go2 (see
[ILC jumping](#ilc-jumping)).

## What's here

| | |
|---|---|
| `ilc_quad/sim_quad_model.py` | The model layer. Normalizes both robots to one torque interface in one joint order. |
| `scripts/sim_node.py` | ROS 2 node owning the physics. Subscribes torques, publishes state. |
| `ilc_quad/check_model.py` | No-ROS sanity check of the setup. Run it first. |
| `launch/sim.launch.py`, `config/{go2,go1}.yaml` | Bring-up. |
| `ilc_quad/trial_log.py` | `TrialRecorder`: one folder per ILC run, one self-contained npz per trial, to resume or transfer from any trial. |
| `ilc_quad/ilc_gen.py` | Full-body TO (`PlanarQuadModel`, `init_trajopt`) and the ILC (`JumpILC`). `python -m ilc_quad.ilc_gen` runs an offline demo. |
| `scripts/ilc_jump_sim.py` | ROS 2 node flying ILC jump trials, on the sim or a real Go2. |
| `scripts/ilc_jump_lockstep.py` | The same ILC node against in-process MuJoCo, lockstep and faster than realtime, for tests and sweeps. |
| `scripts/go1_bridge.py` | unitree_legged_sdk (v3.8, UDP) bridge making a real Go1 look like `sim_node`. |
| `scripts/go2_bridge.py`, `ilc_quad/go2_lowcmd.py` | unitree_ros2 bridge making a real Go2 look like `sim_node`. |
| `launch/ilc_jump_{sim,go1,go2}.launch.py` | ILC bring-up for each backend. |

`ament_cmake`: the `ilc_quad/` module is installed by `ament_python_install_package`, and
`scripts/*.py` are installed as executables with the `.py` stripped — so the node is
`ros2 run ilc_quad sim_node`, while `check_model` stays a module (`python3 -m`, below).

## The interface you write against

```
subscribes  joint_torque_cmd   std_msgs/Float64MultiArray   12 torques, canonical order
            joint_cmd          std_msgs/Float64MultiArray   60: [q, dq, kp, kd, tau] x 12
publishes   joint_states       sensor_msgs/JointState       position, velocity, effort
            base_odom          nav_msgs/Odometry            twist in the base frame
            foot_contacts      std_msgs/Float64MultiArray   4 normal forces, FL FR RL RR
            /clock             rosgraph_msgs/Clock          monotonic sim time
service     reset_trial        std_srvs/Trigger             -> sim time of the reset
```

Canonical order is `FL, FR, RL, RR` x `hip, thigh, calf` — see
`sim_quad_model.CANONICAL_JOINT_NAMES`, and `JointState.name` carries it on every
message. Run your controller with `use_sim_time:=True`.

Four things worth knowing before you write the control:

**`effort` is the applied torque, not your command.** It is read back from MuJoCo
after clamping, so where a motor saturated it differs from what you sent. For ILC
that distinction matters: the applied torque is the input that produced the
trajectory, and a learning law fed its own intended torque is learning from a
trial that did not happen. The knee saturates readily on a jump — 45.43 N·m on
Go2, 35.55 on Go1.

**The last command is held.** A dropped or late message means the previous torque
is reapplied, not zero. Paired with `effort`, a lossy trial is still honestly
recorded rather than silently wrong.

**`/clock` is monotonic across resets.** `reset_trial` returns the sim time the
reset happened at, as a string in `Trigger.Response.message`; subtract it from a
message's `header.stamp` to get run-relative time. Nothing resets the clock to
zero, so no `use_sim_time` subscriber in the graph ever sees time run backwards.

**Sample the reference on the control grid.** The sim publishes at
`control_rate_hz` (250 Hz default = 4 ms), not the 2 ms physics rate. One
reference sample per control tick means the sequence the ILC updates is exactly
the sequence that gets applied.

## Go2 vs Go1

Both are `nq=19 nv=18 nu=12` with a 2 ms timestep and a `home` keyframe at 0.27 m,
but they differ in three ways that `QuadModel` hides by reading the compiled model
— there are no per-robot constants in this package:

| | Go2 | Go1 |
|---|---|---|
| Actuators | `motor`, direct torque; limits in `ctrlrange` | `position` servo kp=100; limits in `forcerange` |
| Actuator order | FL, FR, RL, RR → `[0..11]` | FR, FL, RR, RL → `[3,4,5,0,1,2,9,10,11,6,7,8]` |
| Base body | `base` | `trunk` |
| Knee torque | 45.43 N·m | 35.55 N·m |
| Knee range | `[-2.723, -0.838]` | `[-2.818, -0.888]` |
| Mass | 15.21 kg | 12.74 kg |

Go1's position servos are rewritten into torque motors on the compiled `MjModel`
at load (the Menagerie checkout is never modified), so both robots take identical
torque commands. **A servo left in the loop would be a second controller fighting
yours**, which is why this is not optional.

The tighter joint limits are the thing most likely to bite you: a trajectory
written as fixed angles for Go2 drives Go1 past its knee limit, where it collapses
on landing. Clip against `QuadModel.joint_range`.

`control_rate_hz` must divide the 2 ms timestep evenly — 250, 125, 100 or 500 Hz.
**200 Hz is rejected** (2.5 steps), because a control period that isn't a whole
number of physics steps lets the reference index and the simulation drift apart
over a run.

## Setup

To rebuild everything on a new machine and rerun the Go1 experiments, follow
[REPRODUCE.md](REPRODUCE.md). It covers `docker/Dockerfile`, `environment.yml` and
`experiments/go1/`.

No ROS on the host, so this runs in a container off the `ros:humble-ros-base-jammy`
image. On the host:

```bash
xhost +local:docker     # only if you want the viewer

docker run -it --name ilc_quad --net=host \
  -e DISPLAY=$DISPLAY -e ROS_LOCALHOST_ONLY=1 -e MUJOCO_GL=glx \
  -e MUJOCO_MENAGERIE_PATH=/mujoco_menagerie \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /home/henry/docker_workspaces/ilc_ws:/ilc_ws \
  -v /home/henry/docker_workspaces/mujoco_menagerie:/mujoco_menagerie:ro \
  ros:humble-ros-base-jammy bash
```

Inside — the image ships neither pip nor colcon:

```bash
apt update && apt install -y python3-pip python3-colcon-common-extensions \
    libgl1 libglx-mesa0 libglfw3 libosmesa6
pip3 install "mujoco==3.12.0" "numpy==1.21.5"
```

> **Pin numpy.** `mujoco` declares `numpy` unpinned, so a bare `pip3 install
> mujoco` pulls numpy 2.x over the 1.21.5 that all 191 `ros-humble-*` packages are
> built against, and `rclpy` breaks. Verified working: mujoco 3.12.0 imports fine
> against numpy 1.21.5 with `rclpy` intact. If a future wheel refuses 1.21, use
> `python3 -m venv --system-site-packages` so `rclpy` stays visible.

The py3.12 `mj_quad` conda env **cannot** be used in the container — Humble's
`rclpy` is tied to system py3.10. Use that env for `check_model` on the host.

## Running

```bash
# host, no ROS
conda run -n mj_quad python -m ilc_quad.check_model

# container
cd /ilc_ws && source /opt/ros/humble/setup.bash
colcon build --packages-select ilc_quad && source install/setup.bash

ros2 launch ilc_quad sim.launch.py robot:=go2 viewer:=true
```

The robot sits on the floor until something publishes `joint_torque_cmd` — with
zero torque it has no way to hold itself up. That's the idle state, not a fault.

Sanity checks against a running sim:

```bash
ros2 topic hz /joint_states        # ~250
ros2 topic echo /foot_contacts     # sums to ~149 N (Go2) / ~125 N (Go1) standing
ros2 service call /reset_trial std_srvs/srv/Trigger
```

## Notes

`nav_msgs/Odometry` carries the twist in the **base frame**. MuJoCo keeps a free
joint's linear velocity in the world frame and its angular velocity in the body
frame, so `QuadModel.base_velocity_body` rotates the linear half and leaves the
angular half alone. Rotating both is the easy mistake; it yields a twist that
looks plausible until the robot pitches.

Everything here uses stock messages. Now that the package is `ament_cmake`, a
typed trial message is possible in place — add `rosidl_generate_interfaces` to
`CMakeLists.txt` — rather than needing a separate `ilc_quad_msgs` package.

`joint_cmd` is a per-joint PD target plus feedforward, field-major:
`[q_des(12), dq_des(12), kp(12), kd(12), tau_ff(12)]`, executed as
`tau_ff + kp (q_des - q) + kd (dq_des - dq)` at every 2 ms physics step — the law
Go2's motor drivers run on a LowCmd, so a controller written against it runs
unchanged on hardware. `sim_quad_model.pack_joint_cmd` builds one. `sim_node` also
takes `box_x_front` / `box_height` (> 0 adds a box to the scene) for jumping onto.

## ILC jumping

`ilc_jump_sim.py` flies the jump trial after trial and learns the contact forces
between them. At startup it solves the full-body TO, or loads it from
`reference_file`. Each trial then runs stand → jump → land → ILC update. It speaks
only the topics above, so the simulator and the real robot are the same to it.

```bash
# MuJoCo, box ahead, trials start and reset on their own (the sim runs at 500 Hz)
ros2 launch ilc_quad ilc_jump_sim.launch.py viewer:=true \
    reference_file:=/ilc_ws/log/ref.npz log_dir:=/ilc_ws/log/run1

# MuJoCo Go1 (plan it for Go1: the reference file records the robot)
ros2 launch ilc_quad ilc_jump_sim.launch.py robot:=go1 box:=false jump_dx:=0.4 jump_dz:=0.0 \
    reference_file:=/ilc_ws/log/ref_go1_f40.npz viewer:=true

# real Go1: unitree_legged_sdk's Python wrapper built, robot in low-level mode
# (L2+A, L2+B, L1+L2+Start), OptiTrack trunk pose published
ros2 launch ilc_quad ilc_jump_go1.launch.py pose_topic:=/optitrack/go1/pose pose_type:=pose \
    sdk_path:=$HOME/unitree_legged_sdk/lib/python/amd64 box_height:=0.0 jump_dx:=0.4 jump_dz:=0.0

# real Go2: unitree_ros2 sourced, sport mode released, OptiTrack trunk pose published
ros2 launch ilc_quad ilc_jump_go2.launch.py pose_topic:=/optitrack/go2/pose pose_type:=pose
ros2 service call /start_trial std_srvs/srv/Trigger     # each trial, operator-started
ros2 service call /damp std_srvs/srv/Trigger            # any time: go limp
```

The controller during the jump:

- **Feedforward:** the TO's joint torque, plus the ILC's force correction
  `J(q)ᵀR(θ)ᵀ(U − u_TO)`.
- **Legs in contact:** force-controlled, with damping only (`contact_kp`/`contact_kd`).
- **Swing and flight legs:** a joint PD tracks the TO's joint trajectory.

Measured trials are resampled onto the 10 ms TO grid, expressed relative to where
the CoM stood at takeoff, and handed to `JumpILC.update`. The TO is solved once;
set `reference_file` to reuse it, since it takes tens of seconds. `log_dir` gets one
npz per trial, and `resume_file` continues learning from one.

On the real robot there are two layers of safety:

- **`go1_bridge` / `go2_bridge`:** damps if `joint_cmd` stops for 50 ms, latches damping past a
  joint limit (`/clear_fault` releases it), and clips torques and targets to the
  MJCF limits.
- **`ilc_jump_sim.py`:** damps on stale state or pose, or on excessive tilt before
  the jump. A failed landing damps the robot but still learns from the jump.

### Runs, resuming and transferring

With `log_dir` set, each launch writes a run folder `<log_dir>/<run_name>/`
(`run_name` defaults to a timestamp):

- `meta.json` holds the task, the weights, every node parameter, the sim conditions
  (lockstep runs) and the parent trial, if any.
- `reference.npz` is the plan the run flew.
- `trial_NNN.npz` is one file per trial. Each file stands alone: the forces flown and
  the ILC's next forces, the result and the history so far, the reference, the trial
  on the TO grid, and the raw recording. `TrialRecorder.load(path, trial=k)` reads
  one back.

**Landing.** A box jump touches down moving forward at about 1.5 m/s. If the feet
simply stop, that momentum pivots the trunk over the front feet. Braking at the feet
pitches it nose-down as well, because the backward push acts below the CoM. The old
landing held the home pose with a joint PD. With it, the trunk rolled 8–19° nose-down
and the rear feet lifted for 250–450 ms. The node now handles touchdown in two parts.

**Last 0.1 s of flight** (`level_feet_samples`, default 10). The legs aim the feet
where the plan has them relative to the trunk, as if the trunk were at its planned
pitch. They take up the pitch error, so all four feet touch down together.

**From touchdown** (`landing_controller:=balance`, the default), a 4-force QP runs
every tick.

- **Hard constraints:**
  - every pair pushes down at least `bal_fmin` and stays in its friction cone;
  - every motor stays within 85% of its torque limit;
  - total push is at most `bal_fz_max` body weights (3.5);
  - with `bal_fx_guard`, the net fore-aft push never drives the CoM further forward
    while it is ahead of its place or moving forward.
- **Objectives, in order:**
  - the pitch moment that levels the trunk;
  - strong braking (`bal_brake`, 60/s);
  - the CoM height.
- **Why each constraint is there:**
  - The moment includes the braking forces' own lever, so the QP loads the front legs
    to cancel it. Braking has to be strong: at 10–20/s the CoM reaches the front feet
    before it stops.
  - Without the fore-aft guard, the QP buys nose-up moment by pushing forward with the
    front feet, and that carries the CoM past them.
  - At 2 body weights the push cap was too tight to brake a 20 cm box landing.
  - When a solve reports failure, the controller re-solves with bounds and friction
    cones only.

**Legs during the landing.** The joint targets come from IK that keeps each foot
anchored on the landing surface. The surface height is the standing foot height plus
the box. While the trunk still moves faster than `bal_hold_speed` (0.2 m/s):

- the target pose follows the trunk, so the legs give with the impact;
- joint damping stays light (`bal_kd_absorb`), because damping toward zero speed
  stiffened the legs against it.

Below that speed the target moves over `bal_hold_tau` to a level standing pose over
the feet, and the legs hold it there. With targets that kept following the trunk, the
robot crept forward at 3–16 cm/s and never stopped. The legs compress 2–3 cm.
`landing_controller:=pd` restores the old hold.

**The ILC does not learn from the landing controllers.** They act from 10 samples
before the end of the plan (the level-feet legs), or from touchdown if that comes
earlier. With `flight_att_kp` / `flight_att_kd` set, flight attitude feedback acts from
takeoff. It is off by default; it swings the thighs against the pitch error. From
that sample on, the ILC's measured state is replaced by the error at that sample
carried on as in free flight:

- position and pitch errors grow by the velocity and pitch-rate errors;
- those velocity and pitch-rate errors stay constant.

That is exact for the CoM, which is ballistic whatever the legs do, and it is how the
ILC's model propagates errors after takeoff. Each trial's result records the sample in
`measured_until`.

**Stage III is safeguarded** (`JumpILC.update`). Stage III steps on the SRB model's
sensitivities of the landing state alone. Once corrections are small, the model
misjudges them, sometimes even in sign: a step predicted to add 2 cm/s of takeoff
speed took 1 cm/s away. Unchecked, the landing then drifted by 1–10 cm over 10 trials.
The safeguard works like this:

- Every trial is scored by the landing cost. A trial that fell never counts as best.
- In Stage III, a trial worse than the best so far sends the next trial back to the
  best trial's forces, with a 4× more heavily weighted step.
- The best trial can come from Stages I–II too.

With it, Stage III can use a small step penalty (`qu_stage3` 1e-5). The ILC's own
schedule is now 2 trials of Stage I and 3 of Stage II (`n_stage1`, `n_stage2`). Stage I,
which tracks the contact phase, did little for landing.

**Watching a run.** `replay_trials.py` plays the recorded MuJoCo states back in the
viewer, in the scene the run flew in (box, payload). It is kinematic, with no
physics, so what you see is what was flown:

```bash
ros2 run ilc_quad replay_trials.py /ilc_ws/log/go1_fp/b50_20_m100               # all trials
ros2 run ilc_quad replay_trials.py <run> --trials 1 last --speed 0.25 --loop    # slow motion
ros2 run ilc_quad replay_trials.py <run> --trials 1 20 --video jump.mp4         # no display
```

In the viewer, space pauses, the right and left arrows skip between trials, and
backspace restarts the current trial.

Any of those files can seed a new run:

- **`resume_from:=<trial file or run folder>`** continues the same task from that
  trial. Continuing from trial 3 reproduces the original trials 4–6 exactly.
- **`transfer_from:=<file or folder>`** starts a *different* task from what another
  task learned, in one of two ways:
  - `transfer_mode:=retarget` (default): this task's own plan, plus the correction
    the other task learned on top of its plan (resampled onto this task's contact
    phases).
  - `transfer_mode:=paper`: the paper's Sec. II-D3. It keeps the other task's plan
    and learned forces, aims them at this goal, and learns in Stage III only, after
    one Stage III step before the first trial.

In sim on Go1, neither transfer helped the 40 cm → (60, 10) cm box jump. The box
plan (with the edge clearance below) already lands within 2.5 cm on trial 1 from its
own TO forces. The 40 cm correction overshoots it by 10–20 cm, and the 40 cm joint
profile (paper mode) clips the box edge.

**Box plans keep the feet off the box's front edge.** `box_clearance` (default 4 cm)
and `box_setback` (3 cm) keep any moving foot at least that far above the box top
whenever it is within `box_setback` of the front face. That includes the front foot
swinging during rear-leg contact, which the plan used to route right across the edge,
so that a slightly late push clipped it and tumbled. With it, Go1 plans (60, 10) and
(50, 10) and lands on them on trial 1.

**Box plans start from a single-rigid-body plan** (`to_guess:=srb`, the default).
The robot is first planned as one rigid body with its stance feet fixed, the legs'
reach limits, force limits and friction, and hips high enough for a tucked foot to
clear the box. That plan is turned into a full-body starting point by IK, with smooth
swing-foot paths over the box edge. Then the full-body TO runs once. From the old
straight-line start, IPOPT failed every tall-box plan. From this start, one solve
found exact Go1 (50, 20), (50, 15), (60, 10) and A1 (50, 20) plans in 11–28 s.
`to_steps:=4` (continuation in box height) also works, at 2–4× the time.

**Plans keep the legs apart, clear of the box, and fly a steady spin.** On top of the
box-edge clearance, `init_trajopt` has:

- **`leg_clearance`** (0.10 m): along the trunk, every point of the front leg (hip,
  knee, foot) stays this far ahead of every point of the rear leg. Without it, plans
  crossed the calves or tucked the front feet into the rear hips.
- **Calf midpoints above the terrain.**
- **`landing_clearance`** (0.06 m): over the last 15 samples, the feet stay this high
  over the surface they land on, then come straight down. The flown front feet run
  3–7 cm below the plan late in flight, so plans that skimmed the box landed on the
  front feet early.
- **`flight_min_pitch`** (0°): never nose-down in flight. Nose-up is free.
- **`flight_spin_change`** (20°/s): every flight pitch rate lies in one narrow band.
  Otherwise plans swing the legs as a reaction wheel to brake a 150–220°/s nose-down
  spin. The flown legs lag that swing and land 9° nose-down.
- **`landing_pitch_rate`** (90°/s at touchdown): with the narrow spin band, this lets
  plans take off about 30° nose-up.

The initial guess matches these: swing feet start moving with their hips and hover
above the landing spot. Every plan in the validation below converged in a single
solve from that guess. The three hardest at 85% needed a raised cap: up to 1266
iterations and 78 s, above the 500 default.

### Go1 validation (2026-09-30)

`/ilc_ws/log/final_validate.sh OUT` runs eight tasks with the node's defaults. Scoring
is `/ilc_ws/log/suite_report.py`, on the real landing state at the plan's end.

- **Easy tasks** learn from scratch on plans at 85% of the limits.
- **Harder boxes** fly plans at 90% or 100% and start by transfer (retarget) from an
  easier run that converged. From scratch they fell in their first 1–10 trials; the
  learned correction (weak push, extra spin) carries over.
- **Pass** is the paper's Table I at trial 20: no falls, |ex| ≤ 1 cm, |ey| ≤ 2 cm and
  |eθ| ≤ 2°.

| Task (dx × box height) | Plan | Start | Falls | Trial 1 ex, ey, eθ | Last ex, ey, eθ |
|---|---|---|---|---|---|
| 40 cm flat | 85% | scratch | 0 | −2.8, +0.4 cm, +3.3° | +0.3, −0.7 cm, +0.8° (trial 20) |
| 60 cm flat | 85% | scratch | 0 | −5.6, +1.4 cm, +2.4° | −0.7, −0.3 cm, +1.6° (20) |
| 50 × 10 cm box | 85% | scratch | 0 | −2.1, +0.8 cm, −3.9° | −0.1, +1.3 cm, +0.1° (converged, 7) |
| 55 × 10 cm box | 85% | scratch | 0 | −1.9, −0.1 cm, −5.1° | −0.1, +0.4 cm, −0.2° (converged, 7) |
| 50 × 15 cm box | 90% | from 50 × 10 | 0 | +3.1, −1.7 cm, +2.0° | +0.1, −0.9 cm, +1.4° (20) |
| 60 × 10 cm box | 90% | from 55 × 10 | 0 | +1.3, −1.5 cm, −4.3° | +0.3, −0.3 cm, +0.2° (converged, 10) |
| 50 × 20 cm box | 100% | from 50 × 10 | 0 | +1.4, −1.1 cm, −0.3° | −0.0, −0.3 cm, +0.6° (converged, 7) |
| 50 × 20 cm box | 90% | from 50 × 20 @100% | 0 | +2.3, −0.8 cm, −0.6° | +0.1, −0.2 cm, +0.3° (converged, 7) |

Landings on the last trials:

- trunk dips 0.8–3.3° nose-down after touchdown, then settles level;
- rear feet rise at most 1.7 cm;
- legs compress 2–3 cm;
- the trunk creeps under 2 cm/s once holding.

Two box tasks are out of reach:

- **60 × 15 cm at 90%:** plans only with a raised cap and lands 45 cm off, past Go1 here.
- **40 × 10 cm:** impossible for any face distance. The front feet stand past the face
  a 40 cm landing needs.

### Tests and weight sweeps without ROS transport

`ilc_jump_lockstep.py` runs the node's own code, calling its callbacks directly
and stepping MuJoCo only once each command is in. It is deterministic, runs about
5× realtime (≈1 s per trial), and several copies can run in parallel. The sim
alone can be given uncertainties the controller is not told about:

- **`--ground KP KD`:** per-foot contact stiffness and damping, the paper's hard
  (2e4, 3e3) or soft (2e3, 5e2) ground. The MJCF's own foot contact is close to
  the soft one.
- **`--payload KG`:** a mass welded on the trunk.

`sim_node` takes the same settings as `ground_kp`/`ground_kd`/`payload_mass`.

The lockstep runner also models the rest of the gap to the real Go1 (`Reality` in
`ilc_jump_lockstep.py`). None of it is told to the controller:

| Option | What it models |
|---|---|
| `--mass-scale S` | every robot body's mass and inertia × S |
| `--com-offset DX DZ` | trunk CoM shifted, m |
| `--friction MU` | ground friction |
| `--joint-friction NM` | Coulomb joint friction added |
| `--act-delay S` | command → motors, on top of the one control tick the loop always has |
| `--motor-scale S` | torque limits × S |
| `--motor-curve` | Go1 torque-speed envelope (full torque to half the no-load speed, 30.1 / 20.06 rad/s, then linear to zero); `--motor-speed-scale` for battery sag |
| `--pose-rate HZ`, `--pose-delay S`, `--pose-noise M RAD` | mocap frames: rate (held between frames), latency, per-frame noise |
| `--joint-noise RAD RAD_S` | encoder noise |
| `--foot-sensor BIAS SPREAD NOISE` | Go1's raw `footForce`: per-foot offset U(0, BIAS), gain 1 ± SPREAD, noise |

With any of these, each trial also records the true trunk pose, joints and foot forces
(`rec_true_*`), and runs are scored on those. What the controller measured is no
longer the truth.

Three controller settings exist for the real robot's sensing:
- **`pose_latency`:** the mocap's capture-to-receipt delay (Motive reports it). The
  ILC log dates every frame this much earlier. Left at 0, the ~1.5 m/s landing puts
  the measured landing about 1.5 mm short per ms of latency, and the ILC learns to
  overshoot by that much.
- **`log_vel_window`:** the ILC log fits velocities by a line through ±0.01 s of pose
  frames, and fits the flight's CoM as ballistic (see `_free_flight_tail`).
  Differencing held or noisy mocap frames had the ILC seeing 20–40 cm landing errors
  that weren't there.
- **`touchdown_force`:** counted above each foot's own reading in the first half of the
  flight (a per-trial tare). Raw Go1 foot sensors read tens of counts unloaded, which
  used to end the flight mid-air.

```bash
python3 install/ilc_quad/lib/ilc_quad/ilc_jump_lockstep.py --jump 0.6 0.0 \
    --ground 2e3 5e2 --reference-file /ilc_ws/log/ref_f60.npz --summary /ilc_ws/log/soft.json
python3 install/ilc_quad/lib/ilc_quad/ilc_jump_lockstep.py --jump 0.6 0.1 --box 0.35 0.1 \
    --param "phases:=[30,30,30]" --viewer --realtime
```

The ILC weights are `qe` (Qe on [x, z, θ, vx, vz, ω], default
diag(3, 3, 3, .01, .01, .01)), `qu` (Qu in Stages I–II, default 3e-5) and
`qu_stage3` (Qu in Stage III, default 3e-4). They come from sweeps over the paper's
tasks (Sec. III: 40 and 60 cm jumps, hard/soft ground, a 2 kg payload, boxes) on
both robots:

- **Stage III needs about 10× the step penalty of Stages I–II.** With the paper's
  single Qu the landing gets close by trial 10, then drifts off by trial 20. Stage
  III weighs only the landing state, and its steps overshoot where the SRB model
  mispredicts pitch.
- **Qe weighting x like z beats the paper's diag(1, 3, 3).** The paper's prices a
  degree of pitch like 9 cm of distance. The exception is Go2's (60, 10) cm box,
  where the paper's Qe did better.

What Go1 can do in sim with these weights:

- **40 cm:** reached (1 cm, 1°) at trial 11.
- **60 cm:** needs `margin:=0.9`; at 0.8 the plan lands 3.9 cm short. Then it
  reaches 1 cm and 1° by trial 15–17. On hard or soft ground, or carrying 2 kg,
  it gets within about 1 cm but keeps 3–6° of landing pitch.
- **Boxes:** none of (60, 10), (50, 20) or (60, 30) cm are learned.

Neither bridge has run against a robot yet, only offline: `go2_bridge` against
unitree_sdk2's LowCmd layout and CRC, `go1_bridge` against a mock of
unitree_legged_sdk's Python wrapper. Go1's bridge also runs the SDK's
`PowerProtect` at `power_level` (default 10, full power, which a jump needs); use
a lower level for a first stand test, hanging in a harness.

