# ilc_quad

MuJoCo simulation of the Menagerie Unitree **Go2** and **Go1**, set up for
iterative learning control of jumping. This package is the simulator and the model
layer only — no control law, no trajectory, no learning. You publish torques, it
tells you what happened.

## What's here

| | |
|---|---|
| `ilc_quad/sim_quad_model.py` | The model layer. Normalizes both robots to one torque interface in one joint order. |
| `scripts/sim_node.py` | ROS 2 node owning the physics. Subscribes torques, publishes state. |
| `ilc_quad/check_model.py` | No-ROS sanity check of the setup. Run it first. |
| `launch/sim.launch.py`, `config/{go2,go1}.yaml` | Bring-up. |

`ament_cmake`: the `ilc_quad/` module is installed by `ament_python_install_package`, and
`scripts/*.py` are installed as executables with the `.py` stripped — so the node is
`ros2 run ilc_quad sim_node`, while `check_model` stays a module (`python3 -m`, below).

## The interface you write against

```
subscribes  joint_torque_cmd   std_msgs/Float64MultiArray   12 torques, canonical order
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
