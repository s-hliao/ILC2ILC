# Deep ILC on a real Go1: runbook

This covers the hardware stage on a real Go1 for the headline quadruped model, **DR (B) + 24 real jumps**: our learner
trained under domain randomization (`log/dilc/plane/drB_s2/policy`, member `g0_plane_it30`), then 24 real jumps of the
hardware stage, then the evaluation. It replaces the 2026-10-07 recipe, which flew the nominal-sim network
`v10_s2/snap20 g0_plane_it20`.

**What it did in simulation** (the CPU "real" robots r1 / s1 / r5, reserved test goals; `log/dilc/plane/fixbox/drB_hw24`):
- **43.8 %** test success after 24 jumps per robot. Per robot: r1 46.9 %, s1 62.5 %, r5 21.9 %.
- The snapshot curve: 41.7 / 42.7 / 45.8 / 45.8 / 43.8 % at 0 / 6 / 12 / 18 / 24 jumps
  (`log/dilc/plane/match/drb_snap`, CI ± 9.7).
- Most of the level is the DR-trained network itself. The stage mostly fixes near-misses, and r5's far / box goals stay
  out of reach.

**What is tested and what is not.**
- **Tested end to end in simulation:** the hardware path (`policy_jump_node.py` → `deploy.py --backend manual` →
  ILC update → safety gate → next flight). `sim_node` stood in for the Go1 bridge on the same ROS topics
  (`scripts/sim_operator.py`, `log/dilc/hwtest/loop`; `hw_session.py --backend sim`, `log/dilc/hwtest/session`).
- **Checked for this recipe (2026-10-12):**
  - the deploy command below parses with `deploy.py`'s own argument parser;
  - `policy_jump_node.py prepare` builds the DR-B policy's plans, both flat (0.45 m) and onto a box (0.50 m / 0.15 m:
    box front 0.265 m ahead of the standing CoM).
- **Not tested here:** either robot connection below on a real Go1, the real mocap latency and the real contact.

## 0. What to bring

| Item | Where |
|---|---|
| The policy | `log/dilc/plane/drB_s2/policy`, member `g0_plane_it30` (DR (B) sim stage, seed 2, it30) |
| Boxes | 5, 10 and 15 cm tall, firm and non-sliding: three of the six training goals jump onto a box (placement: `prepare` writes `ref_..._box.json`) |
| Workstation with ROS 2 Humble + `ilc_quad` | the `ilc_quad` container on this machine works if it can reach the robot's network |
| GPU machine with the `ilcmjx` env | runs `deploy.py`: the ILC update needs the GPU sim's Jacobians and the gate needs its rollouts. It can be this server, with the episode folder shared (see 3). |
| Go1 low-level setup | low-level mode (from standing: L2+A, L2+B, L1+L2+Start); one of the two connections in section 1 |
| OptiTrack | the trunk's rigid body, z-up world frame |

## 1. Connecting the robot: two options

**(a) ROS 2 on the workstation: `go1_bridge.py` (simplest).**
- The workstation is wired to the robot (192.168.123.x) and talks to its motor board directly over UDP with the
  `unitree_legged_sdk` 3.8 Python wrapper: `ros2 launch ilc_quad go1_bridge.launch.py sdk_path:=<sdk>`.
- There is no ROS 1 and no bridge; the controller gets `joint_states` / `foot_contacts` and sends `joint_cmd` on the same
  machine.
- The trunk pose comes from your ROS 2 mocap client (e.g. `natnet_ros2`).

**(b) ROS 1 on the Go1's own computer: `src/ilc_go1_hw`** (the ROS 1 hardware layer; see its README).
- `roslaunch ilc_go1_hw go1_hw.launch mocap_server:=<Motive PC> mocap_body:=go1` on the robot runs:
  - the SDK low-level node: the same topics and safety as `go1_bridge.py`, plus IMU, leg odometry, ZUPT;
  - the VRPN client and `mocap/pose`;
  - the EKF.
- On the workstation: `bash src/ilc_go1_hw/scripts/ros1_bridge.sh <go1 ip> <workstation ip>` (`ros1_bridge`, noetic +
  foxy, in docker), and `python3 src/ilc_go1_hw/scripts/joint_cmd_relay.py` in the `ilc_quad` container.
- **The relay is required.** The controller publishes `joint_cmd` BEST_EFFORT and `ros1_bridge` subscribes RELIABLE.
  DDS never matches those, so without the relay no command reaches the robot. The relay republishes it RELIABLE as
  `joint_cmd_r`, which the robot's node subscribes to.
- Controller pose input: `pose_topic:=/mocap/pose pose_type:=pose` (`hw_session.py --pose-topic /mocap/pose --pose-type pose`).
  This is the RAW tracked point, as in simulation. The EKF's `trunk/odometry` is for monitoring.
- `pose_latency` must be measured END TO END on the workstation, through the bridge (section 2).

## 2. Measure once, then the checklist before the first jump

1. **Mocap latency** (`pose_latency`). The policy's state estimator dates every mocap frame by it, and the sim robots
   assumed 4–10 ms. A few ms of error biases the velocities the policy reacts to.
   - Measure it with a fast hand rotation of the robot: mocap pitch against the IMU gyro or the encoders.
   - With connection (b), measure on the workstation, after the bridge.
2. **Mocap offset** (`mocap_offset`): the tracked point in the base frame.
3. **Flat floor, clear landing zone** ≥ 0.8 m ahead of the standing CoM, and the boxes placed per `ref_..._box.json`.

**Before the first jump** (with (b), the details are in `src/ilc_go1_hw/README.md`, "Calibrate once"):
- [ ] **IMU quaternion order** pinned (`quat_order` in `go1_hw.yaml`). The recorded Go1 streamed (x, y, z, w), not the
      SDK header's (w, x, y, z). The node auto-detects it and logs the result. Tilt test: nose down gives a positive
      pitch.
- [ ] **Foot-contact thresholds** (`contact_on` / `contact_off`) set from `foot_contacts` with the robot hanging and
      standing.
- [ ] **Mocap axes:** push the robot forward by hand, and x must grow in the pose the controller reads. Set the mocap
      offset in the controller (and in `go1_hw.yaml` for the EKF).
- [ ] **End-to-end `pose_latency`** set in `hw_session.py`.
- [ ] **Stand test at low power:** `power_level` 3–5 first. A stand and a damp (`/damp`, `a` in `hw_session.py`), then the
      jump power (10: a jump needs the motors' full power).
- [ ] **The joint-limit margin is 0.20 rad**, not 0.05: rear calves over-extend at takeoff.

## 3. Pre-flight in simulation (5 minutes, same ROS stack)

```bash
P=/ilc_ws/log/dilc/plane/drB_s2/policy
python3 install/ilc_quad/lib/ilc_quad/policy_jump_node.py prepare --policy $P --goal 0.45 0 --out /tmp/ref_0.450.npz
ros2 launch ilc_quad policy_jump_sim.launch.py policy_dir:=$P policy_member:=g0_plane_it30 \
    reference_file:=/tmp/ref_0.450.npz jump_dx:=0.45 episode_dir:=/tmp/pre max_trials:=2
# a box goal: prepare --goal 0.50 0.15 -> box:=true box_x_front:=0.265 box_height:=0.15 jump_dx:=0.50 jump_dz:=0.15
```
- Expect a landing error of a few cm and no fall.
- If anything errors here, it will error on the robot.
- If ROS nodes stop seeing each other after many launches, clean stale shared memory: `fastdds shm clean`
  (`hw_session.py` does this at start).
- On a machine where other experiments use the same ROS graph, set a separate `ROS_DOMAIN_ID`.

## 4. The hardware stage (24 jumps)

**Budget.** 6 training goals × 4 iterations = 24 jumps, the headline arm's. Two of the goals are flat, four are onto
boxes.

If the robot allows more, 96 jumps did better on the sim robots: 49.0 % test (`fixbox/drB_nobold_hw96`: the bold driver
off; `drB_nogate_hw96`: the gate off), against 43.8 % at 24. Those arms changed a safety knob too, so keep the
24-jump recipe for the first session.

**On the GPU machine** (exactly `fixbox/drB_hw24`'s arguments, with the manual backend):
```bash
cd src/ilc_mjx
~/miniconda3/envs/ilcmjx/bin/python scripts/deploy.py --backend manual --robots go1 --out <RUN> --iters 4 \
  --policy /home/henry/ilc_ws/log/dilc/plane/drB_s2/policy --member g0_plane_it30 \
  --update gn --gn-beta 0.3 --gn-delta 0.1 --step-rms 0.05 --est-window 0.025 --clear 0.03 \
  --stall 3 --stall-sat 0.2 --sat-project --bold 0.3 --target-base own --gate 1 1.5 0.4 \
  --goals 0.45 0.60 0.50,0.15 0.60,0.10 0.50,0.05 0.575,0.15 --seed 9001 --eval none
```
The first iteration's update takes ~2.5 min (one-time compilation), so start `deploy.py` while you set up. Later updates
take ~30–60 s, while the robot is reset.

**The gate: nominal, loose (`--gate 1 1.5 0.4`), no `--gate-dr`.**
- The 2026-10-10 test (`fixbox/drb_gate_test.sh`, 24 jumps per robot) checked the gate under DR draws (`--gate-dr`),
  2 jumps per goal (`--reps 2`), and both:
  - 43.8 / 42.7 / 43.8 %, against 43.8 % for this recipe;
  - per robot within ±6 points, which is trial noise.
- Neither changes the outcome, and `--gate-dr` costs extra GPU rollouts per update, so the recipe keeps the headline
  arm's nominal gate.

What the gate checks before a refit may fly:
- in the nominal GPU sim over the goal range, nominal and from perturbed stances, the update must not fall more,
  nor score worse than 1.5× the current policy;
- its actions on the robot's own recent states must not move more than 0.4 (normalized).
- Otherwise the update is rejected: the current policy flies again with halved steps.
- On the sim robots it rejected 1–2 of 4 updates per robot.

**On the robot workstation, once per session:**
```bash
# (a): ros2 launch ilc_quad go1_bridge.launch.py sdk_path:=<sdk>           -- kept up all session
# (b): go1_hw.launch on the robot + ros1_bridge.sh + joint_cmd_relay.py   -- section 1
python3 hw_session.py <RUN>/go1 --pose-topic <topic> --pose-type pose --pose-latency <measured, end to end> \
    --mocap-offset x y z [--path-map <GPU machine's ws>=<this machine's ws>]
```
`hw_session.py` follows `deploy.py` for you:
- for each iteration's request, goal by goal, it prepares the plan (and the box placement) and launches the policy node
  for the jumps owed;
- **at each prompt**, reset the robot at the start mark (and the box), then press **Enter** to jump (`d` = damp,
  `a` = abort);
- it prints each landing error, then waits while `deploy.py` takes the ILC step, refits and runs the gate;
- it stops when `deploy.py` is done.

The episodes land in `<RUN>/go1/it<k>/real`, which both machines must see (shared mount, or `--path-map`).

**Manual alternative.** Without `hw_session.py`, each `<RUN>/go1/it<k>/REQUEST.md` lists the exact commands:
`policy_jump_node.py prepare`, `policy_jump_go1.launch.py`, `/start_trial`.

**Safety, as on the sim robots:**
- **Steps are small:** Gauss–Newton β 0.3, rms ≤ 0.05.
- **Bold driver:** halves a goal's step after it gets worse.
- **Stall guard:** freezes a goal that stopped improving (with saturation projection), so an unreachable goal cannot drag
  its neighbours. On the sim robots, r5's far / box goals are such goals.
- **Falls:** a fall rolls that goal back to its best jump. Keep the fall's episode; it is data.
- **Bridge layer:** the watchdog (50 ms without `joint_cmd` → damping), the joint limits (0.20 rad margin, latched until
  `/clear_fault`) and `/damp` are unchanged in both connections.

**Abort:** `touch <RUN>/go1/it<k>/real/ABORT`. Restarting `deploy.py` with the same `--out` keeps the jumps already
flown for that iteration.

**Stop and investigate:**
- two falls in a row;
- a landing error over 10 cm on a flat goal;
- the gate rejecting twice in a row;
- any motor at its torque limit through the whole stance (the node logs peak torques).

## 5. Evaluation

Fly the start policy (`it0`) and the final one (`it4`) with the same node on the reserved TEST goals:
- the goals: 0.4875, 0.6125, (0.52, 0.12), (0.54, 0.14), (0.57, 0.11), (0.51, 0.17), (0.4875, 0.035), (0.6375, 0.09);
- this is the sim's `planefinal` set; never train on it;
- alternate the two policies, ≥ 4 jumps per goal per policy.

Success = no fall, |e_x| ≤ 5 cm, |e_z| ≤ 3 cm. `info` in the saved episodes holds the landing error, and `rc[:, 2]` sums
to the landing cost.

**Baselines on the same robot:**
- JumpILC (`ilc_jump_go1.launch.py`, same goals, same jump budget);
- the start policy zero-shot.

## 6. If it does not transfer

Check, in order:
1. **`pose_latency`:** the estimator's velocities. Compare the episode's `X` rates with the mocap derivative, and with (b)
   with the EKF's `trunk/odometry` / `leg_odom`.
2. **Contact timing:** `rec_contacts` against the plan's phases.
3. **Torque saturation:** `rec_tau_total` against the limits.

The hardware stage corrects first-order mismatch. A sim that is badly off fails: in sim-to-sim, mass −15% with
motors +15% failed, and half of that worked.
