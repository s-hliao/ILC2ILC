# ilc_f1tenth_hw: the F1TENTH hardware layer for ILC2Real

State estimation and actuation for the real car, ported from
[LLA-Control/LLA-MPC-onboard](https://github.com/LLA-Control/LLA-MPC-onboard) (commit `f24fb84`, by the same lab). The
car, its Fiala model and its current-controlled drive are the ones ILC2Real's networks are trained for (`../ilc_f1tenth/ilc2real`).

## What is where

| | |
|---|---|
| `../f1tenth_system` | submodule: upstream [f1tenth/f1tenth_system](https://github.com/f1tenth/f1tenth_system) `humble-devel` (with its own `vesc`, `ackermann_mux`, `teleop_tools` submodules) |
| `patches/*.patch` | LLA-MPC-onboard's modifications of it, as patches on the pinned upstream (below) |
| `scripts/apply_hw_patches.sh` | applies them (idempotent; `--revert` undoes them) |
| `../natnet_ros2` | submodule: [L2S-lab/natnet_ros2](https://github.com/L2S-lab/natnet_ros2), pinned at LLA's tested commit `883b095`; the OptiTrack NatNet client |
| `scripts/optitrack_node.py` | NatNet rigid-body pose -> `/optitrack/odom` (pose + filtered finite-difference body twist) and the `map -> base_link` TF |
| `scripts/imu_zupt_prep.py` | VESC IMU -> SI units + covariances on `/sensors/imu/data`; a zero-velocity update on `/zupt` when stopped |
| `config/mocap.yaml` | the robot_localization EKF: mocap pose + twist, IMU yaw rate and accelerations, ZUPT -> `/odometry/filtered` (body frame `cg`) |
| `scripts/drive_relay.py` | `/mpc/drive` -> `/drive` with a watchdog: brakes (zero duty) if commands stop for `timeout_s` (0.12 s) |
| `launch/opti_launch.py` | f1tenth stack + natnet + ZUPT + EKF + optitrack node |
| `scripts/net_runner.py` | **flies the network** at 40 Hz (EKF + VESC in, current + steering out on `/mpc/drive`), the **0.5 m safety stop**, trial requests / records for the hardware stage (below) |
| `scripts/car_sim_node.py` | desk stand-in for the car + EKF (a simulated multi-body car on the same topics), for testing `net_runner.py` without hardware |
| `scripts/desk_test_runner.sh` | the desk test: `car_sim_node.py` + `net_runner.py`, a trained network for 2 laps and an envelope violation |

## The f1tenth_system patches

- **`vesc.patch`**:
  - `vesc_ackermann/ackermann_to_vesc` gains LLA's actuation-mode gate on `AckermannDriveStamped.drive.jerk`:
    - `jerk = 2.0`: `drive.acceleration` is the **motor current in A**, published to `commands/motor/current` and clamped to
      `current_command_min/max`;
    - `jerk = 1.0`: `drive.acceleration` is the **duty cycle**, clamped to `duty_cycle_command_min/max` (+-0.95);
    - anything else: the usual `drive.speed` -> ERPM, or ERPM 0 when stopped.

    `vesc_driver` applies no default clipping to current commands, so this clamp is the safety layer. The servo position
    is clamped to [0.1, 0.9].
  - `vesc_to_odom`: the wheel-speed sign is flipped (this car's motor turns the other way).
- **`ackermann_mux.patch`**: the mux's output topic is `ackermann_drive_out`, remapped to `ackermann_cmd` in the bringup.
- **`f1tenth_system.patch`** (`f1tenth_stack`):
  - `vesc.yaml`:
    - VESC duty limits +-1;
    - **`current_command_min/max: -25 / 50 A`**: the current range ILC2Real's networks are trained for
      (`car_model.NOMINAL`), tighter than LLA's +-55 A default;
  - `mux.yaml`: timeouts 0.5 s;
  - `joy_teleop.yaml`: LLA's joystick mapping, including a **stop deadman on axis 4 that outranks the controller** in
    the mux;
  - `bringup_launch.py`: static TFs `base_link -> cg` (0.15 m forward: the EKF's body frame) and
    `base_link -> imu_link`;
  - `tf_publisher.py`.

## Build and run

```bash
cd ~/ilc_ws
git -C src submodule update --init --recursive        # f1tenth_system (+ vesc, ackermann_mux, teleop_tools), natnet_ros2
bash src/ilc_f1tenth_hw/scripts/apply_hw_patches.sh   # LLA's f1tenth_system modifications
sudo apt install ros-humble-robot-localization ros-humble-ackermann-msgs ros-humble-tf2*   # + the f1tenth_system deps
colcon build --symlink-install --packages-up-to ilc_f1tenth_hw natnet_ros2   # natnet downloads the NatNet SDK on its first build
source install/setup.bash
ros2 launch ilc_f1tenth_hw opti_launch.py server_ip:=<Motive PC> client_ip:=<this car>
ros2 run ilc_f1tenth_hw drive_relay.py                # on the car
```

## OptiTrack setup (from LLA's README; this is where a new room takes work)

- **Rigid body.** `natnet_ros2` publishes each rigid body on `/<name>/pose`. The default topic is `/f1tenth/pose`, so the
  car's rigid body in Motive is `f1tenth`, or pass `mocap_topic:=/<name>/pose`.
- **Up axis.** `optitrack_node` expects Motive to stream **Y-up**. It remaps the position to `(-x, z, y)` and rotates the
  orientation by `B @ P @ R`. natnet_ros2's own README says to select Z-up; for this node keep Y-up.
- **Heading.** `yaw_offset` (default -pi/2, LLA's room) is found empirically. If the position is right but the heading is
  off by 90 or 180 deg, change it. If the car moves mirrored, change the signs in `P` and in the position remapping
  together. Check by pushing the car forward by hand and watching it move along its own x axis in RViz.
- **Velocities** are filtered finite differences: a causal EMA plus a windowed spike clamp (`vel_ema_tau`,
  `vel_spike_pct`, `vel_spike_window`). Keep `vel_ema_tau` small; its lag adds to the EKF's.
- **Covariances:**
  - pose 1e-6 for position and 1e-4 for orientation (mocap is accurate);
  - twist moderate (a filtered finite difference).

  Tune them together with `odom0_config` in `config/mocap.yaml`.
- **Track frame.** ILC2Real's tracks are LLA's `mocap_*` tracks, in this mocap frame. The tight tracks
  (`make_tight_tracks.py`) are shortened about their own centre, so they sit inside the same space.

## Remote running (CycloneDDS, as LLA-MPC-f1tenth `llampc/utils/cyclonedds{car,laptop}.xml`)

The car runs the stack, state estimation and the brake watchdog. The laptop (the GPU machine) runs the network
(`net_runner.py`) and the hardware stage. They talk over ROS 2 on CycloneDDS with unicast discovery (multicast off: the
lab WiFi drops it).

| file | |
|---|---|
| `config/cyclonedds/cyclonedds_car.xml` | car side: interface 172.26.112.71 (the car, as `opti_launch.py`'s `client_ip`), peers car + laptop 172.26.100.199 (LLA's pair) |
| `config/cyclonedds/cyclonedds_laptop.xml` | laptop side: interface 172.26.100.199, the same peers |
| `config/cyclonedds/cyclonedds_local.xml` | one machine, loopback (desk tests) |
| `scripts/dds_env.sh` | `source` it: `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`, `CYCLONEDDS_URI`, `ROS_DOMAIN_ID` (default 0, same on both), restarts the ros2 daemon; with `--car-ip A --laptop-ip B [--peer C]` it writes `~/.ros/cyclonedds_<role>.xml` for other addresses and warns if this machine does not own its interface address |

Both machines need `ros-humble-rmw-cyclonedds-cpp`. It is installed in `docker/f1t/Dockerfile` (and in the running `f1t`
container). On the car: `sudo apt install ros-humble-rmw-cyclonedds-cpp`.

```bash
# car
source scripts/dds_env.sh car            # or: car --car-ip <car> --laptop-ip <laptop>
ros2 launch ilc_f1tenth_hw opti_launch.py server_ip:=<Motive PC> client_ip:=<car>   # includes drive_relay.py (the watchdog)
# laptop
source scripts/dds_env.sh laptop         # or: laptop --car-ip <car> --laptop-ip <laptop>
ros2 topic hz /odometry/filtered         # discovery check: the car's EKF at its rate
source ../ilc_f1tenth/ilc2real/env_tight.sh
ros2 run ilc_f1tenth_hw net_runner.py --ros-args -p session_dir:=$HOME/car_session
python ../ilc_f1tenth/ilc2real/hw_stage.py --backend manual --session-dir $HOME/car_session ...   # no shared mount needed
```

- **The watchdog stays on the car** (`opti_launch.py drive_relay:=true`, the default). If the WiFi drops, the car stops
  receiving `/mpc/drive`. After 0.12 s, `drive_relay.py` brakes on the car side.
- **On the laptop, `net_runner.py` stops a trial as `stale`** when the EKF or VESC state is older than 0.1 s.
- **The network then sees the WiFi delay** (one way, typically 2–10 ms) on top of the EKF's.
  - The simulated real cars were trained and scored with an actuation delay (`real_car.py`).
  - Check the measured round trip before the first trial: `ros2 topic delay /odometry/filtered` on the laptop, with
    synchronized clocks (chrony).
  - If it is regularly > 15 ms, run `net_runner.py` on the car instead. The session dir then needs the shared mount, as
    in the hardware-stage section above.
- **Desk test over CycloneDDS:** `DDS_ROLE=local bash scripts/desk_test_runner.sh real_nom <policy>`. Result (2026-10-17):
  - trial A: 2.05 laps, RMS e_y 4.1 cm, peak 9.3 cm, identical to the FastDDS run;
  - trial B (the network steering away): stopped and braking at a peak e_y of 0.48 m. The heading check (`spin`, 1.2 rad)
    fired just before the 0.5 m wall check.

## ROS 2 Foxy (Jetson Nano / Ubuntu 20.04)

The package runs on Foxy as well as Humble. The ROS distro picks the f1tenth_system branch and its patch set.

| | Humble | Foxy |
|---|---|---|
| `../f1tenth_system` | `humble-devel` (the submodule's pin) | `foxy-devel` (vesc `foxy`, ackermann_mux and teleop_tools `foxy-devel`) |
| patches | `patches/{f1tenth_system,vesc,ackermann_mux}.patch` | `patches/foxy/{f1tenth_system,vesc}.patch` |
| CycloneDDS | `ros-humble-rmw-cyclonedds-cpp` | `ros-foxy-rmw-cyclonedds-cpp` |

- **The Foxy set is the same LLA modification**, ported to Foxy's sources:
  - the VESC current / duty-cycle modes with their clamps;
  - the odometry sign flip;
  - the -25 / 50 A current range;
  - the mux timeouts and the joystick deadman;
  - the base_link -> cg / imu_link static TFs.
- **Foxy's `ackermann_mux` already publishes `ackermann_drive_out`, and Foxy already ships `tf_publisher.py`**, so the
  Foxy set has no mux patch and adds no file.
- **The Python nodes, launch file and configs are shared.** The runtime the network uses (`policy_runtime.py`,
  `bank.py`, `car_model.py`) needs only numpy (>= 1.17, Foxy's). It parses under Python 3.6, and CasADi is only needed
  if a plan's LQR gains are not cached (all of `plans_mu02_tight` are).

```bash
source /opt/ros/foxy/setup.bash
cd ~/ilc_ws/src
git -C f1tenth_system fetch origin foxy-devel && git -C f1tenth_system checkout foxy-devel
git -C f1tenth_system submodule update --init --recursive
bash ilc_f1tenth_hw/scripts/apply_hw_patches.sh          # picks patches/foxy from $ROS_DISTRO (or --distro foxy)
sudo apt install ros-foxy-ackermann-msgs ros-foxy-robot-localization ros-foxy-rmw-cyclonedds-cpp ros-foxy-serial-driver \
                 ros-foxy-diagnostic-updater python3-numpy
cd ~/ilc_ws && rosdep install -y -r --from-paths src/f1tenth_system --ignore-src --rosdistro foxy
colcon build --symlink-install --packages-up-to ilc_f1tenth_hw natnet_ros2
```

- **On a Jetson Nano with JetPack 4.x (Ubuntu 18.04)** there are no Foxy apt packages. Run the above inside an L4T
  Foxy container (e.g. dustynv's `ros:foxy-ros-base-l4t-r32.7.1`), or build Foxy from source.
- **Do not mix distros over DDS.** Foxy and Humble nodes discover each other, but message and service compatibility is
  not guaranteed. With the network on the laptop (remote running), run the laptop side in Foxy too, e.g. a
  `ros:foxy-ros-base` container with `~/ilc_ws` mounted. Or run `net_runner.py` on the car.
- **Tested (2026-10-17) in a `ros:foxy-ros-base` container** (Python 3.8, numpy 1.17):
  - the patched foxy-devel stack (vesc_msgs, vesc_driver, vesc_ackermann, ackermann_mux, f1tenth_stack) and this package
    build;
  - plans and networks load;
  - the desk test over CycloneDDS (`ROS_DISTRO=foxy DDS_ROLE=local bash scripts/desk_test_runner.sh real_nom <policy> <foxy ws>/install/setup.bash`;
    the stand-in car needs `pip3 install numba==0.53.1`, the real car does not):
    - trial A: 2.05 laps, RMS e_y 4.1 cm, peak 9.3 cm, max current 36.1 A, identical to Humble;
    - trial B: stopped `wall` at 0.506 m, braking.
  - natnet_ros2 (its README lists Foxy) was not built here: it downloads the NatNet SDK on its first build.

## The interface the ILC2Real network uses (as LLA-MPC's `llampc_fiala_fixed.py`)

The network's state is the Fiala car's `[X, Y, psi, vx, vy, r, omega_w, I, delta]`, at 40 Hz (`DT = 0.025`, as LLA-MPC).

| state | source |
|---|---|
| X, Y, psi, vx, vy, r | `/odometry/filtered` (EKF, body frame `cg`; psi unwrapped) |
| omega_w | `/sensors/core` (`vesc_msgs/VescStateStamped`): `state.speed` (ERPM) / 2 pole pairs -> motor rpm -> rad/s, / gear ratio 11.82 |
| I | `/sensors/core` `state.avg_iq`, the measured motor current |
| delta | the last steering command |

The network's actions are set points for steering (rad) and motor current (A): `a * A_SCALE + A_OFF`. They go out as
`AckermannDriveStamped` on `/mpc/drive` with `drive.jerk = 2.0`, `drive.acceleration = current`,
`drive.steering_angle = steer`, `drive.speed = 0`, and `drive_relay.py` forwards them to `/drive`. LLA-MPC uses duty-cycle
pure pursuit (`jerk = 1.0`) below 0.1 m/s and current above.

## The network runner (`scripts/net_runner.py`)

**What it runs.** Each control period goes through `../ilc_f1tenth/ilc2real/policy_runtime.Controller`: the controller step
of `real_car.run`, i.e. exactly what every simulated result was measured with. `test_policy_runtime.py` replays simulated
trials through it with identical actions, observations and plan indices, for networks, the slow-start launch and FADA. It
covers the plan projection, the observation features, the network, and the launch controller: below `v_hand` (1.0 m/s) it
drives a fixed current (8 A) on the plan's steering plus path feedback, as `real_car.run`'s launch does, then hands over
to the network.

**Topics.**

| | |
|---|---|
| in | `/odometry/filtered` (EKF), `/sensors/core` (VESC) |
| out | `/mpc/drive`: `jerk = 2.0`, `acceleration` = current A (clipped to -25..50), `steering_angle` (clipped to +-0.34); `drive_relay.py` forwards it to `/drive` |
| services | `/net_runner/start`, `/net_runner/stop` (`std_srvs/Trigger`) |

**Safety: a trial stops when any of these happens.**
- `wall`: **|e_y| > `safety_ey` = 0.5 m** (the room's envelope, on the measured EKF state);
- `spin`: |e_psi| > 1.2 rad;
- `stall`: vx < 0.3 m/s after the launch;
- `stale`: the odometry or the VESC state is older than 0.1 s;
- `stopped`: `/net_runner/stop` or Ctrl-C.

On a stop it sends duty-cycle 0 (`jerk = 1.0`, the VESC brakes) for 1 s, then nothing, so `drive_relay.py`'s watchdog
keeps braking. A trial also ends normally after its laps. (The simulations score crashes at the stricter 0.3 m.)

**Trials (the session protocol).** The runner watches `session_dir` for requests:
- request: `req_<k>.json`, `{"plan": <bank plan>, "policy": <npz>, "laps": 2.05, "launch": {"v_hand": 1.0, "current": 8.0}}`;
- it logs each pending request. Place the car at the plan's start, at rest, and press **Enter** in its terminal (or call
  `/net_runner/start`);
- result: `res_<k>.npz` (`real_car.run`'s dict: `x, obs, mu, a, err, idx, applied, laps, failed, crash, handover, ...`; the
  "true" entries are the measured ones) and `res_<k>.json` (summary).

Plans come from the same bank as the simulations: `source ../ilc_f1tenth/ilc2real/env_tight.sh` first (`F1T_PLANS`), with
the bank synced to the car. Run it with:
```bash
source ../ilc_f1tenth/ilc2real/env_tight.sh
ros2 run ilc_f1tenth_hw net_runner.py --ros-args -p session_dir:=$HOME/car_session   # (-p safety_ey:=0.5 is the default)
```

**The hardware stage on the real car.** The 10-lap ladder recipe of the low-friction fix runs as
`hw_stage.py --backend manual`, on the machine with the GPU, sharing the session dir with the car (a shared mount):
```bash
cd ../ilc_f1tenth/ilc2real && source env_tight.sh
python hw_stage.py --backend manual --session-dir <shared>/car_session --cars real --policy runs/v3drBtL_s0/policy_final.npz \
  --out runs/hw/real_car_ladder --iters 5 --goals mocap_square2fast_b25 mocap_figfast_b25 --goal-cycle 1 --goal-order worst \
  --fallback-suffix _mu80,_mu60 --fallback-on band --fallback-after 1 --fallback-revert 1 --eval-runs 1 --eval-laps 5
```
- Each trial (and each evaluation run) becomes a request; the operator starts it at the car.
- The launch phase is cut from the data the ILC learns from (`--launch V_HAND CURRENT`, default 1.0 m/s / 8 A).
- The ladder and the ILC update are unchanged.
- Desk test without the car: `python car_session_sim.py --session-dir <dir> --car real_mu` serves the same requests on a
  simulated car. With that serving, a 2-iteration manual stage switched both tracks to `_mu80` (as the sim backend does)
  and scored square b25 `_mu80` at 2.8 cm.

**Desk test of the runner over ROS 2 (no car).**
`bash scripts/desk_test_runner.sh real_nom ../ilc_f1tenth/ilc2real/runs/v3drBtL_s0/policy_final.npz` runs
`car_sim_node.py` + `net_runner.py`, needing `vesc_msgs` built. Results (2026-10-12):
- the trained network on square b25: 2.05 laps, RMS e_y 4.1 cm, peak 9.3 cm;
- a car started 0.45 m off the path and steered away: stopped `wall` at 0.506 m, braking.
