# Deep ILC on a real Go1: runbook

This covers the hardware stage of "structure in simulation, specifics on hardware" on a real Go1. You need a policy
from the sim stage (GPU), then 30 real jumps of few-shot ILC, then the evaluation.

**What is tested and what is not.**
- **Tested end to end in simulation:** the hardware path (`policy_jump_node.py` → `deploy.py --backend manual` →
  ILC update → safety gate → next flight). `sim_node` stood in for the Go1 bridge on the same ROS topics
  (`scripts/sim_operator.py`, `log/dilc/hwtest/loop`).
- **Not tested here:** the `go1_bridge` with this node on a real robot, the real mocap latency and the real contact.

## 0. What to bring

| Item | Where |
|---|---|
| The policy | `log/dilc/pre_coilc<seed>/policy`, member `g0_coilc.s0_e1200`. Use the seed the frozen test (`log/dilc/holdout/RESULTS.txt`) and val favour; `ex1` until then. |
| Workstation with ROS 2 Humble + `ilc_quad` | the `ilc_quad` container on this machine works if it can reach the robot's network |
| GPU machine with the `ilcmjx` env | runs `deploy.py`: the ILC update needs the GPU sim's Jacobians and the gate needs its rollouts. It can be this server, with the episode folder shared (see 3). |
| Go1 low-level setup | `unitree_legged_sdk` 3.8 Python wrapper, wired 192.168.123.x, low-level mode (L2+A, L2+B, L1+L2+Start): see `ilc_quad/launch/ilc_jump_go1.launch.py` |
| OptiTrack | trunk pose, z-up world frame |

## 1. Measure, once

1. **Mocap latency** (`pose_latency`). The policy's state estimator dates every mocap frame by it. The sim robots
   assumed 4–10 ms, and an error of a few ms biases the velocity estimate the policy reacts to.
   - Measure it: e.g. a fast hand rotation of the robot, mocap pitch against IMU or encoder timing.
2. **Mocap offset** (`mocap_offset`): the tracked point in the base frame.
3. **Flat floor, clear landing zone** ≥ 0.8 m ahead of the standing CoM.

## 2. Pre-flight in simulation (5 minutes, same ROS stack)

```bash
P=log/dilc/pre_coilcex1/policy   # (exported form: deploy.py writes OUT/<robot>/it0/policy; or use that)
python3 install/ilc_quad/lib/ilc_quad/policy_jump_node.py prepare --policy $P --goal 0.425 0 --out /tmp/ref.npz
ros2 launch ilc_quad policy_jump_sim.launch.py policy_dir:=$P policy_member:=g0_coilc.s0_e1200 \
    reference_file:=/tmp/ref.npz jump_dx:=0.425 episode_dir:=/tmp/pre max_trials:=2
```
- Expect a landing error of a few cm and no fall.
- If anything errors here, it will error on the robot.

## 3. The hardware stage (30 jumps)

**On the GPU machine:**
```bash
cd src/ilc_mjx
~/miniconda3/envs/ilcmjx/bin/python scripts/deploy.py --backend manual --robots go1 --out <RUN> \
  --policy log/dilc/pre_coilcex1/policy --member g0_coilc.s0_e1200 \
  --update gn --gn-beta 0.3 --step-rms 0.05 --target-base own --bold 0.3 --stall 3 --eval none
```
The safety gate is on by default with `--backend manual` (`--gate 0 1.5 0.3`).

**Each iteration** (3 jumps, one per goal 0.425 / 0.5 / 0.575):
- `deploy.py` writes `<RUN>/go1/it<k>/REQUEST.md` with the exact commands and waits for the episodes.
- **On the robot workstation**, per goal:
  ```bash
  python3 policy_jump_node.py prepare --policy <it k policy> --goal <g> 0 --out <episode_dir>/ref_<g>.npz
  ros2 launch ilc_quad policy_jump_go1.launch.py pose_topic:=<topic> pose_latency:=<measured> \
      mocap_offset:="[x, y, z]" policy_dir:=<it k policy> reference_file:=<episode_dir>/ref_<g>.npz \
      jump_dx:=<g> jump_dz:=0.0 episode_dir:=<episode_dir> episode_tag:=go1_it<k>
  ros2 service call /start_trial std_srvs/srv/Trigger      # one jump; the node logs the landing error
  ```
- The episode lands in `<episode_dir>` = `<RUN>/go1/it<k>/real`. Both machines must see it (shared mount or rsync).
- `deploy.py` then takes the ILC step and refits.
- **The gate checks the refit before it may fly:**
  - in the nominal GPU sim over the goal range, nominal and from perturbed stances, the update must not fall more,
    nor score worse than 1.5× the current policy (+0.5);
  - its actions on the robot's own recent states must not move more than 0.3 (normalized).
  - Otherwise the update is rejected: the current policy flies again with halved steps.

**Safety, as on the sim robots:**
- **Steps are small:** Gauss–Newton β 0.3, rms ≤ 0.05.
- **Bold driver:** halves a goal's step after it gets worse.
- **Stall guard:** freezes a goal that stopped improving, so an unreachable goal cannot drag its neighbours.
- **Falls:** a fall rolls that goal back to its best jump. Keep the fall's episode; it is data.
- **Bridge layer:** the bridge's watchdog, joint limits and `/damp` are unchanged.

**Abort:** `touch <RUN>/go1/it<k>/real/ABORT`. Restarting `deploy.py` with the same `--out` keeps the jumps already
flown for that iteration.

**Stop and investigate:**
- two falls in a row;
- a landing error over 10 cm;
- the gate rejecting twice in a row;
- any motor at its torque limit through the whole stance (the node logs peak torques).

## 4. Evaluation

Fly the start policy (`it0`) and the final one (`it10`) with the same node, at goals between the training goals
(e.g. 0.4375 / 0.4875 / 0.5125 / 0.5625), alternating the two policies, ≥ 4 jumps per goal per policy. Score each
with `0.01 × landing_cost + 20 × fall`: `info` in the saved episodes holds the landing error, and `rc[:, 2]` sums to
the landing cost.

**Baselines on the same robot:**
- JumpILC (`ilc_jump_go1.launch.py`, same goals, same jump budget);
- the start policy zero-shot.

## 5. If it does not transfer

Check, in order:
1. **`pose_latency`:** the estimator's velocities. Compare the episode's `X` rates with the mocap derivative.
2. **Contact timing:** `rec_contacts` against the plan's phases.
3. **Torque saturation:** `rec_tau_total` against the limits.

The hardware stage corrects first-order mismatch. A sim that is badly off fails: in sim-to-sim, mass −15% with
motors +15% failed, and half of that worked.
