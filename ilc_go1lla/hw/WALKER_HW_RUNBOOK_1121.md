# Go1 goal walker on hardware: runbook (2026-11-21, written from matcha; nothing here has run on the real robot yet)

The Playground joystick walker walks to a goal 2.5 m straight ahead and stands; DistILC stage corrects the one network from real
walks. Code: `src/ilc_go1lla/hw/` (matcha) = `~/go1_walk/hw/` (workstation).

## What is tested, and what is not
| test | result |
|---|---|
| numpy policy vs the JAX one (`export_policy.py`) | max action error 3e-6 (nominal, full DR, early) |
| command law, joint orders, actions vs the MJX walker (`test_walker_core.py`) | exact (1e-6) |
| observation math vs MuJoCo's sensors at the same state (`test_obs_sensors.py`) | 1e-7 (MJX's own observation lags 4 ms: one physics substep) |
| closed loop on CPU MuJoCo with mocap offset / body yaw / 7 ms latency / noise and the workstation's MocapProcessor (`sim_closed_loop.py`) | nominal net, nominal robot: 6/6 arrive, 3.36 s (MJX 3.40), path 1.4 cm; 4 kg: falls (MJX zero-shot falls too); full DR: 4 kg 6/6 in 3.37 s, foam 6/6 in 3.45 s |
| stage through the hardware pipeline on that robot (`stage_goal_hw.py --backend cpusim`, nominal net, 3 kg @ 1.2) | 5 iterations x 4 walks, 0 falls: time 3.36 -> 3.07 s (nominal 3.15), path 16.6 -> 4.7 cm |
| ROS 2: `walker_node.py` against `ilc_quad` sim_node on the workstation (isolated domain 77) | stand up, 2 walks, logs; 50 Hz held (20.0 ms, max 22.9 ms) |
| two machines: matcha request -> rsync -> walker_node (sim_node) -> logs back -> G / refit -> next request | one full cycle (`runs/hw/remote_test`): 2 walks, refit, the new network walked; its walk tipped (46 deg) 0.04 s after the hand-over to the standing controller, on sim_node's menagerie Go1 (not the training model) -- see the hand-over note |
| **real Go1** | **not run**: the IMU publishing (`go1_bridge_imu.py`), the gyro axes, tauEst's convention, the walker's sim-to-real |

## 0. Before the session
- The robot: charged; a tether / hand on it for the first walks; the mocap rigid body `dl_go1` as for the jumps
  (calibration in `~/go1_runs/quad_env_1102.sh`: `--mocap-offset 0.010 0 -0.0025 --mocap-body-yaw-deg 89.07`, latency 7 ms).
- The floor: a straight lane of **3.5 m** inside the tracked area (`--bounds`, default x [-2.39, 1.92], y [-2.56, 2.50] = the car's
  measured area minus 0.3 m). Mark the start: the robot always walks 2.5 m straight ahead of where it stands.
- Old processes: the 11-10 `natnet_go1.launch.py` and `joint_cmd_relay.py` are still running on the workstation (default ROS domain).
  `joint_cmd_relay.py` forwards `joint_cmd` to the ROS 1 robot layer: **stop it** before using `go1_bridge_imu.py` (only one
  thing may command the motors).

## 1. Terminals on the workstation
```bash
source /opt/ros/humble/setup.bash; source ~/ros2_ws/install/setup.bash; cd ~/go1_walk/hw
# T1 mocap (if not already up): ros2 launch ilc_quad natnet_go1.launch.py   -> /dl_go1/pose
# T2 bridge (robot in low-level mode: L2+A, L2+B, L1+L2+Start; wired 192.168.123.x):
python3 go1_bridge_imu.py --ros-args -p sdk_path:=$HOME/unitree_legged_sdk/lib/python/amd64 -p power_level:=10 -p menagerie_root:=$HOME/mujoco_menagerie
ros2 topic hz /imu /joint_states /dl_go1/pose      # ~500, ~500, mocap rate
```

## 2. First contact (robot hanging or lying, nominal network, no perturbation)
```bash
python3 walker_node.py eval ~/go1_walk/eval_nominal_zs --policy policies/nominal.npz --walks 6 --speed 1.0
#   i  (fresh joints / imu / mocap; tilt ~0; mocap yaw sensible)   s  (stand up, 2.5 s)   w  (walk)   d  damp at any time
```
Then on matcha: `rsync -a nsh1609@172.26.79.206:go1_walk/eval_nominal_zs/ runs/hw/eval_nominal_zs/` and
`~/venvs/pg0/bin/python hw/check_logs.py runs/hw/eval_nominal_zs/walks --G` -> the gyro must correlate with the mocap angular
velocity on all three axes (> 0.8, positive scale ~1); `tau +KD dq` must match `Kp(q*-q)` better than `raw` (else tauEst
excludes the damping: set the logs' `tau_includes_kd` False). **If the nominal network does not walk on the nominal robot, stop:
that is the sim-to-real gap itself, and it decides what we can claim.**

## 3. Zero-shot evaluations (6 walks each, per perturbation: the payload offset, the foam)
`walker_node.py eval ~/go1_walk/eval_<net>_<case>_zs --policy policies/<net>.npz --walks 6` for `nominal`, `fulldr_foamlow`,
`early_lightdr65M`.

## 4. Stage (matcha drives; 10 iterations x 4 walks = 40 real walks, the sim arms' settings)
```bash
# matcha:
~/venvs/pg0/bin/python hw/stage_goal_hw.py --params <base params> --backend files --speeds 1.0 --iters 10 --practice 4 --cap 0.03 \
    --out runs/hw/stage_<net>_<case> --remote nsh1609@172.26.79.206:/home/nsh1609/go1_walk/stage_<net>_<case>
# workstation:
python3 walker_node.py stage ~/go1_walk/stage_<net>_<case>      # s, then w per walk; the update takes ~30-60 s between iterations
```
Bases: nominal `runs/walker_nom_s0/params.pkl`; early `runs/walker_lightdr2_s0/ck/params_00065536000.pkl`; full DR
`runs/walker_fulldr5_foamlow/ck/params_00255590400.pkl`. A walk that went wrong for an outside reason (cable, a person): `v` voids
it and it is walked again. `touch ~/go1_walk/stage_<...>/ABORT` stops the stage. The final network: `runs/hw/stage_.../policy_final.npz`
-> copy to the workstation and evaluate as in 3.

## Notes
- **Hand-over risk (measured on sim_node, 2026-11-21):** the task hands the walker to the standing controller (joint PD to home,
  a = 0) 0.5 s after the reference reaches the goal, as in every simulated evaluation (MJX: 61 falls after the hand-over in 4620
  walks). On sim_node's menagerie Go1 a walk that arrived faster (x 2.40 m at the hand-over vs 2.17) tipped 11 -> 46 deg within
  0.04 s of it. Expect this on the real robot too; keep a hand near it at the goal. Changing the hand-over (e.g. the policy with a
  zero command) would change the task the sim results used: decide before the session.
- The walk frame turns the start pose to face the goal, as every simulated walk starts; physically every walk is 2.5 m straight.
- Safety in `walker_core.WalkController`: tilt > 45 deg, trunk 12 cm below its start height (relative: foam), mocap older than
  0.1 s, outside `--bounds` -> damping; the bridge's joint-limit latch and 50 ms watchdog stay.
- Hardware tauEst includes the motor PD's -Kd dq; MuJoCo's actuator force does not (passive joint damping). The stage adds Kd dq
  back (log meta `tau_includes_kd`), verified on sim_node's PD (0.11 vs 0.71 N m rms).
