# ilc_go1_hw: the Go1 hardware layer in ROS 1

Low-level I/O for a real Go1, on the robot's own computer (ROS 1: melodic or noetic). This is the
robot-side counterpart of the car's `ilc_f1tenth_hw`:
- a low-level interface with the simulation's safety layer;
- 500 Hz joint, IMU, contact, leg-odometry, and zero-velocity data for the workstation estimator.

The car's unicast `natnet_ros2` receiver and `robot_localization` EKF run on the ROS 2 workstation, so the roughly
500 Hz mocap stream never makes an unnecessary round trip through the Go1's Wi-Fi link.

On top of the car's layer it uses what a legged robot has and a car does not:
- **joint angles and rates (leg odometry):** the stance feet's kinematics give the trunk velocity;
- **foot-force contacts:** they decide which feet count as stance feet;
- **a zero-velocity update** while the robot stands still.

The jump controllers (`ilc_quad`, ROS 2) see the robot exactly as they saw the simulated one: the same topics, joint
order, units and safety rules as `ilc_quad/scripts/go1_bridge.py`.

## What is where

| | |
|---|---|
| `include/ilc_go1_hw/go1_core.h` | the robot-independent core (no ROS, no SDK; unit-tested off the robot): joint order, names, MJCF limits; leg kinematics and Jacobian; leg odometry; contact hysteresis; zero-velocity detector; quaternion-order detection; command clipping |
| `src/go1_lowlevel_node.cpp` | `unitree_legged_sdk` 3.8 low-level UDP <-> ROS topics (below), 500 Hz |
| `scripts/mocap_odom_node.py`, `config/ekf.yaml` | optional onboard-estimation fallback; the default estimator is in `ilc_quad` on the workstation |
| `config/go1_hw.yaml` | every parameter (safety values = `go1_bridge.py`'s) |
| `launch/go1_hw.launch` | low-level sensor/actuator node + IMU frame; `use_onboard_ekf:=true` enables the fallback estimator |
| `scripts/joint_cmd_relay.py`, `scripts/ros1_bridge.sh` | the link to the ROS 2 controller (workstation side) |
| `test/` | `test_go1_core.cpp` (kinematics vs MuJoCo, Jacobian, leg odometry, quaternion order on recorded Go1 data, commands, contacts) and `test_mocap_processor.py` |
| `scripts/gen_fk_reference.py` | regenerates `test/fk_reference.h` from the Menagerie `go1.xml` (the controllers' model) |

## Topics (all in the robot's namespace)

### Published by `go1_lowlevel_node`

| topic | type | content |
|---|---|---|
| `joint_states` | `sensor_msgs/JointState` | 12 joints in canonical order `FL_hip_joint, FL_thigh_joint, FL_calf_joint, FR_…, RL_…, RR_…` (the controllers' order; Unitree's motor slots are FR, FL, RR, RL and are re-ordered here); position rad, velocity rad/s, effort = `tauEst` N m |
| `foot_contacts` | `std_msgs/Float64MultiArray` | raw foot forces FL FR RL RR (as `go1_bridge.py`) |
| `contact_state` | `std_msgs/UInt8MultiArray` | contact flags FL FR RL RR (hysteresis on the raw forces) |
| `imu` | `sensor_msgs/Imu` (frame `imu_link` = the trunk origin) | orientation in ROS order (x, y, z, w); gyro rad/s; accelerometer m/s² raw (gravity included: +9.7 on z standing) |
| `leg_odom` | `geometry_msgs/TwistWithCovarianceStamped` (frame `trunk`) | trunk velocity from the joint angles and rates (see below) |
| `zupt` | `geometry_msgs/TwistWithCovarianceStamped` | zero velocity, while all four feet are down and the joints and trunk are still for 0.1 s |

### Subscribed, and services

| topic / service | type | content |
|---|---|---|
| `joint_cmd` (remapped to `joint_cmd_r`) | `std_msgs/Float64MultiArray` | `[q, dq, kp, kd, tau] x 12`, field-major, canonical; executed on the motor drivers as `tau + kp (q - q_meas) + kd (dq - dq_meas)` |
| `clear_fault` | `std_srvs/Trigger` | release a latched joint-limit fault |

### Published on the workstation by `ilc_quad`'s NatNet processor and EKF

| topic | type | content |
|---|---|---|
| `mocap/pose` | `geometry_msgs/PoseStamped` | the tracked point, z-up world: **what the controller reads** |
| `mocap/odom` | `nav_msgs/Odometry` | the trunk origin and its body twist (EKF input) |
| `trunk/odometry` | `nav_msgs/Odometry` | the EKF's fused trunk state (and TF `world -> trunk`) |

TF `trunk -> imu_link` is the identity (the MJCF's IMU site).

**Safety** (independent of the controller, identical to `go1_bridge.py`):
- **watchdog:** no `joint_cmd` for 50 ms → damping;
- **joint limits:** a joint more than 0.20 rad past its MJCF range → damping, latched until `clear_fault`
  while out of range only pure damping (kp = τ = 0, dq = 0) or a position target back into the range is accepted
  (a folded Go1's hips sit at about ±1.13 rad), anything else latches;
- **clipping:** q to the range, kp to [0, 100], kd to [0, 10], τ to 23.7 / 23.7 / 35.55 N m. kp = 0 sends Unitree's "no
  position target", likewise kd = 0;
- **power:** the SDK's `PowerProtect` at level 10;
- **startup:** the idle command (passive motors) until the first state arrives.

## The joint angles in the estimate (leg odometry)

The Go1 reports every joint's angle and rate. A foot in contact is at rest in the world, so the trunk's velocity in its
own frame follows from that leg alone:

    v = -( J(q) dq + w x p_foot(q) )

where:
- `p_foot`, `J`: the foot position and Jacobian from the MJCF's geometry (hips at (±0.1881, ±0.04675, 0), thigh offset
  ±0.08, thigh and calf 0.213 m; checked against MuJoCo to 1e-9 m);
- `w`: the gyro.

The node averages the stance feet (contact from the foot forces, with hysteresis). The covariance grows with how much the
feet disagree, which is what a slipping foot looks like. Below `min_contacts` stance feet (in flight) it publishes
nothing, and the EKF carries on with mocap and the IMU.

The joint angles also give the whole-body CoM. The controller computes that itself from `joint_states`, as in simulation.

## Build (on the Go1's computer)

```bash
ls /opt/ros                       # melodic (Ubuntu 18.04, Python 2.7) or noetic (20.04): both supported
sudo apt install ros-$ROS_DISTRO-robot-localization
# unitree_legged_sdk v3.8.x built (its lib/cpp/<arch>/libunitree_legged_sdk.a); the Pi / Nano is arm64
mkdir -p ~/go1_ws/src && cp -r ilc_go1_hw ~/go1_ws/src/ && rm ~/go1_ws/src/ilc_go1_hw/COLCON_IGNORE
cd ~/go1_ws && catkin_make -DUNITREE_SDK_DIR=$HOME/Unitree/sdk/unitree_legged_sdk && source devel/setup.bash
```

Off-robot test (any machine, no ROS):
`g++ -std=c++14 -Itest -Iinclude test/test_go1_core.cpp -o t && ./t`, and `python test/test_mocap_processor.py`.

## Run

**Robot.** Put it in low-level mode: from standing, L2+A (lie down), L2+B (damp), L1+L2+Start. Then start the ROS 1
hardware publisher:
```bash
roslaunch ilc_go1_hw go1_hw.launch
```

**Workstation (ROS 2 controller and EKF), once per session.** Start the bidirectional ROS bridge, then the same unicast
NatNet client used by the car together with the local EKF. `client_ip` is this workstation's address on the Motive
network, not the Go1 Ethernet address:
```bash
git submodule update --init natnet_ros2
bash scripts/ros1_bridge.sh <go1 ip> <workstation ip>      # ros1_bridge in docker (noetic + foxy)
ros2 launch ilc_quad natnet_go1.launch.py server_ip:=172.26.30.22 client_ip:=<workstation mocap IP>
python3 scripts/joint_cmd_relay.py                          # in the ilc_quad container (ROS 2)
```
`natnet_ros2` publishes `/<Motive rigid-body name>/pose` at every Motive frame; neither the launch nor the bridge
throttles it. With Motive configured near 500 Hz, check `ros2 topic hz /dl_go1/pose` on the workstation. The Go1
publishes `joint_states`, `imu`, and joint-derived `leg_odom` at 500 Hz; the bridge carries those to the workstation.
`go1_sensor_restamp.py` puts the bridged sensor stamps in the workstation clock domain before the 500 Hz EKF consumes
them. Mocap remains local and its configured `pose_latency` is subtracted from the measurement timestamp.

Then run the controller as in `src/ilc_mjx/HARDWARE.md`, with
`pose_topic:=/mocap/pose pose_type:=pose pose_latency:=<END-TO-END latency> mocap_offset:=[x, y, z] imu_topic:=/imu`.

**Why the controller runs its own EKF on raw mocap + IMU, not `trunk/odometry`.** The current networks (EKF banks, e.g.
`ekfft_s2`, since 2026-10-26) were trained with the controller's OWN planar trunk EKF (`ilc_quad/scripts/planar_ekf.py`,
`dilc_execute.EkfEstimator`): the IMU predicts, each raw mocap frame (dated by `pose_latency`) corrects, one step per
10 ms control sample -- the same filter in the JAX training env and on the robot. So the controller reads raw
`mocap/pose` and `imu` and fuses them itself; feeding it the workstation EKF would change what the policy sees. The
workstation EKF's `trunk/odometry` stays for monitoring and logging. (Pre-10-26 networks, banks without `"estimator":
"ekf"`, use the line estimator on raw mocap alone.)

## Calibrate once

1. **IMU quaternion order.** The SDK header says (w, x, y, z). A recorded Go1 streamed (x, y, z, w): standing at yaw
   -0.72 it read (-0.010, -0.008, -0.352, 0.936), and only xyzw reproduces the SDK's own rpy (error 0.002 rad vs 3.16).
   - With `quat_order: auto` the node decides from the first ~50 samples and logs the result. Pin it in
     `go1_hw.yaml` afterwards.
   - **Tilt test:** pitch the nose down by hand. `rostopic echo /imu` should show the orientation's pitch positive
     (y-axis rotation, ROS convention), matching `rpy[1]`.
2. **Foot contact thresholds.** `rostopic echo /foot_contacts`:
   - with the robot hanging (feet unloaded), then standing;
   - set `contact_off` above the unloaded readings and `contact_on` well below the standing ones. The defaults 60 / 40
     are placeholders.
3. **Mocap axes and offset.**
   - Stream Z-up from Motive (`up_axis: z`), or keep Motive's Y-up with `up_axis: y`, which is the car's conversion.
   - Push the robot forward by hand: x must grow in `mocap/pose`, and the EKF's `trunk/odometry` x velocity must be
     positive.
   - Measure the rigid body's origin in the trunk frame, then set it as `mocap_offset` in `go1_hw.yaml` (for the EKF)
     **and** as the controller's `mocap_offset` (it reads the raw tracked point).
4. **Latency.** Rotate the robot fast by hand and record the mocap pitch and the IMU gyro (`rosbag record /mocap/pose /imu`).
   - The lag that best aligns the differentiated mocap pitch with gyro y is the mocap latency. Set it as `pose_latency`.
   - For the controller, measure the same thing on the WORKSTATION after the bridge. Its `pose_latency` is end-to-end,
     and the simulated robots assumed 4-10 ms.
5. **Clocks.** The workstation restamps bridged Go1 sensors on receipt, while local NatNet frames use the workstation
   clock. Tight clock synchronization with the robot is therefore not required for the EKF; network jitter still
   appears as sensor timing noise, so use wired Ethernet when possible.

## What is tested and what is not

- **Tested here, off the robot:**
  - the core: kinematics against MuJoCo's Go1 (worst 6e-10 m), the Jacobian (exact), leg odometry (recovers a moving
    trunk exactly), quaternion-order detection on the recorded data, clipping and sentinels, joint limits, contacts,
    ZUPT;
  - the mocap processor: trunk origin, body twist, and the Y-up conversion on a synthetic spinning trunk;
  - the node compiles against stand-ins of the ROS 1 and SDK 3.8 APIs (`-fsyntax-only`);
  - launch and YAML files parse.
- **Not tested:** the workstation EKF on live robot/mocap data, the bridge's latency, and the contact thresholds. Do the
  calibration above before the first flight, with `power_level` low for a first stand.
