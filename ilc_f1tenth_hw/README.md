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

The room's safety envelope (ILC2Real's `F1T_SAFETY_EY = 0.3` m) has to be enforced on the car as well: stop (zero
current / brake) once the lateral offset from the plan exceeds it. That belongs in the network runner, which is not
written yet.
